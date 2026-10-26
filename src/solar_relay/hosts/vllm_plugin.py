"""vLLM host: `vllm.general_plugins` entry point.

vLLM 은 이 플러그인을 API 서버뿐 아니라 EngineCore·워커 프로세스에서도 부른다. 훅은 템플릿을 렌더하는
API 서버 프로세스에서만 건다. 이 모듈은 최상단에서 vLLM 을 import 하지 않는다 — vLLM 은 플러그인
import 실패를 로그만 남기고 삼키므로, import 가 깨지면 훅이 조용히 빠진다. 대신 register() 안의
자가 점검이 실패하면 예외를 올려 기동을 멈춘다 (vLLM 은 register() 의 예외는 삼키지 않는다).
"""

from __future__ import annotations

import functools
import inspect
import logging
import multiprocessing
import os
from collections.abc import Callable, Mapping
from typing import Any

from solar_relay.channels import EFFORT_ORDER
from solar_relay.relay import Patch, RequestFields, relay, responses_text_format_to_chat
from solar_relay.switches import Switches, disabled_by_header

# vLLM 로거 아래에 두어야 vLLM 의 핸들러·포맷으로 출력된다.
logger = logging.getLogger("vllm.plugins.solar_relay")

_WRAPPED_MARK = "__solar_relay_wrapped__"
_EFFORT_ORDER = EFFORT_ORDER["vllm"]

# 훅이 기대하는 vLLM 쪽 이름. 버전을 올려 하나라도 사라지면 register() 가 기동을 멈춘다.
_CHAT_REQUEST_FIELDS = (
    "reasoning_effort",
    "chat_template_kwargs",
    "vllm_xargs",
    "response_format",
    "parallel_tool_calls",
    "tools",
    "messages",
)
_RESPONSES_REQUEST_FIELDS = (
    "reasoning",
    "text",
    "chat_template_kwargs",
    "vllm_xargs",
    "parallel_tool_calls",
    "tools",
)


def register() -> None:
    """entry point. API 서버 프로세스에서만 chat(·messages) 와 responses 서빙에 훅을 건다."""
    if not _is_api_server_process():
        return
    if not Switches.from_env().enabled:
        logger.info("solar-relay: disabled by SOLAR_RELAY, hooks not installed")
        return

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

    install(
        chat_serving=OpenAIServingChat,
        chat_request=ChatCompletionRequest,
        responses_serving=OpenAIServingResponses,
        responses_request=ResponsesRequest,
    )


def install(
    *,
    chat_serving: type,
    chat_request: type,
    responses_serving: type,
    responses_request: type,
) -> None:
    """자가 점검 후 훅을 건다. 점검이 실패하면 RuntimeError 로 기동을 멈춘다."""
    check_target(
        serving=chat_serving,
        method="create_chat_completion",
        server_defaults_param="default_chat_template_kwargs",
        request_model=chat_request,
        request_fields=_CHAT_REQUEST_FIELDS,
    )
    check_target(
        serving=responses_serving,
        method="create_responses",
        server_defaults_param="default_chat_template_kwargs",
        request_model=responses_request,
        request_fields=_RESPONSES_REQUEST_FIELDS,
    )
    # chat 서빙은 생성자 인자를 default_chat_template_kwargs 로, responses 서빙은 chat_template_kwargs 로 저장한다.
    wrap(
        serving=chat_serving,
        method="create_chat_completion",
        server_defaults_attr="default_chat_template_kwargs",
        read_fields=chat_fields,
        apply_patch=apply_chat_patch,
    )
    wrap(
        serving=responses_serving,
        method="create_responses",
        server_defaults_attr="chat_template_kwargs",
        read_fields=responses_fields,
        apply_patch=apply_responses_patch,
    )
    logger.info(
        "solar-relay: hooked %s.create_chat_completion (chat, messages) and "
        "%s.create_responses (pid %d)",
        chat_serving.__name__,
        responses_serving.__name__,
        os.getpid(),
    )


def check_target(
    *,
    serving: type,
    method: str,
    server_defaults_param: str,
    request_model: type,
    request_fields: tuple[str, ...],
) -> None:
    target = getattr(serving, method, None)
    if target is None or not inspect.iscoroutinefunction(target):
        raise RuntimeError(f"solar-relay: {serving.__name__}.{method} is missing or not async")
    params = list(inspect.signature(target).parameters)
    if params[:3] != ["self", "request", "raw_request"]:
        raise RuntimeError(
            f"solar-relay: {serving.__name__}.{method} takes {params}, "
            "expected (self, request, raw_request, ...)"
        )
    if server_defaults_param not in inspect.signature(serving.__init__).parameters:
        raise RuntimeError(
            f"solar-relay: {serving.__name__}.__init__ has no {server_defaults_param}"
        )
    missing = [name for name in request_fields if name not in request_model.model_fields]
    if missing:
        raise RuntimeError(f"solar-relay: {request_model.__name__} has no fields {missing}")


def wrap(
    *,
    serving: type,
    method: str,
    server_defaults_attr: str,
    read_fields: Callable[[Any, Mapping[str, Any] | None], RequestFields],
    apply_patch: Callable[[Any, Patch], None],
) -> None:
    original = getattr(serving, method)
    if getattr(original, _WRAPPED_MARK, False):
        return
    # 운영자 스위치는 기동 때 정해지므로 훅을 걸 때 한 번만 읽는다.
    switches = Switches.from_env()

    @functools.wraps(original)
    async def hooked(self, request, raw_request=None, *args, **kwargs):
        try:
            _relay_one(
                request=request,
                raw_request=raw_request,
                server_defaults=getattr(self, server_defaults_attr, None),
                read_fields=read_fields,
                apply_patch=apply_patch,
                switches=switches,
            )
        except Exception:
            # 요청을 막지 않는다. 대신 매번 ERROR 로 남겨 조용히 빠지지 않게 한다.
            logger.exception("solar-relay: request passed through unchanged after an error")
        return await original(self, request, raw_request, *args, **kwargs)

    setattr(hooked, _WRAPPED_MARK, True)
    setattr(serving, method, hooked)


def _relay_one(*, request, raw_request, server_defaults, read_fields, apply_patch, switches) -> None:
    headers = getattr(raw_request, "headers", None) or {}
    if disabled_by_header(headers):
        return
    patch = relay(
        read_fields(request, server_defaults),
        effort_order=_EFFORT_ORDER,
        switches=switches,
    )
    if patch:
        apply_patch(request, patch)
        logger.info("solar-relay applied=%s", ",".join(patch.applied))


# ---------------------------------------------------------------------------
# 요청 객체 ↔ RequestFields / Patch
# ---------------------------------------------------------------------------
def chat_fields(request, server_defaults: Mapping[str, Any] | None) -> RequestFields:
    return RequestFields(
        reasoning_effort=request.reasoning_effort,
        chat_template_kwargs=request.chat_template_kwargs,
        server_template_defaults=server_defaults,
        engine_args=request.vllm_xargs,
        response_format=_as_dict(request.response_format),
        parallel_tool_calls=_sent(request, "parallel_tool_calls"),
        has_tools=bool(request.tools),
        messages=request.messages,
    )


def responses_fields(request, server_defaults: Mapping[str, Any] | None) -> RequestFields:
    reasoning = request.reasoning
    text = request.text
    return RequestFields(
        reasoning_effort=None if reasoning is None else reasoning.effort,
        chat_template_kwargs=request.chat_template_kwargs,
        server_template_defaults=server_defaults,
        engine_args=request.vllm_xargs,
        response_format=responses_text_format_to_chat(
            _as_dict(None if text is None else text.format)
        ),
        parallel_tool_calls=_sent(request, "parallel_tool_calls"),
        has_tools=bool(request.tools),
    )


def apply_chat_patch(request, patch: Patch) -> None:
    _merge_channels(request, patch)
    for index, (content, reasoning) in patch.reasoning_moves.items():
        message = request.messages[index]
        message["content"] = content
        if reasoning:
            message["reasoning"] = reasoning


def apply_responses_patch(request, patch: Patch) -> None:
    _merge_channels(request, patch)


def _merge_channels(request, patch: Patch) -> None:
    # 요청에 이미 있는 키가 이긴다. relay() 가 이미 걸렀지만 host 에서도 덮지 않는다.
    if patch.template_kwargs:
        request.chat_template_kwargs = {
            **patch.template_kwargs,
            **(request.chat_template_kwargs or {}),
        }
    if patch.engine_args:
        request.vllm_xargs = {**patch.engine_args, **(request.vllm_xargs or {})}


def _sent(request, name: str) -> Any:
    # 클라이언트가 보낸 필드만 값으로 본다. pydantic 기본값(parallel_tool_calls=True)은 외부 값이 아니다.
    return getattr(request, name) if name in request.model_fields_set else None


def _as_dict(value) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    # vLLM 의 JsonSchemaResponseFormat 은 schema 를 json_schema 필드(alias "schema")에 둔다.
    return value.model_dump(mode="json", by_alias=True, exclude_none=True)


def _is_api_server_process() -> bool:
    # API 서버가 하나면 MainProcess, 여럿이면 vllm.v1.utils 가 ApiServer_{i} 로 띄운다.
    name = multiprocessing.current_process().name
    return name == "MainProcess" or name.startswith("ApiServer_")
