"""vLLM host.

- vLLM API 서버: `vllm.general_plugins` entry point(register)로 chat(·messages)·responses 서빙 메서드를 감싼다.
- 요청 변환(relay_chat)은 Dynamo host 도 쓴다. Dynamo 앞단도 vLLM 의 ChatCompletionRequest 를 쓰기 때문이다.

vLLM 은 이 플러그인을 프로세스마다 한 번, API 서버뿐 아니라 EngineCore·워커 프로세스에서도 부른다. 훅은 템플릿을
렌더하는 API 서버 프로세스에서만 건다. 이 모듈은 최상단에서 vLLM 을 import 하지 않는다 — vLLM 은 플러그인
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

from solar_relay.relay import Patch, relay, relay_enabled

# vLLM 로거 아래에 두어야 vLLM 의 핸들러·포맷으로 출력된다.
logger = logging.getLogger("vllm.plugins.solar_relay")

# 훅이 읽고 쓰는 vLLM 요청 필드. 버전을 올려 하나라도 사라지면 register() 가 기동을 멈춘다.
_CHAT_REQUEST_FIELDS = (
    "chat_template_kwargs",
    "vllm_xargs",
    "response_format",
    "parallel_tool_calls",
    "messages",
)
_RESPONSES_REQUEST_FIELDS = ("chat_template_kwargs", "vllm_xargs", "text", "parallel_tool_calls")


def register() -> None:
    """entry point. API 서버 프로세스에서만 chat(·messages) 와 responses 서빙에 훅을 건다."""
    if not _is_api_server_process():
        return
    if not relay_enabled():
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


def install(*, chat_serving: type, chat_request: type, responses_serving: type, responses_request: type) -> None:
    """자가 점검을 모두 통과한 뒤에 훅을 건다. 점검이 실패하면 RuntimeError 로 기동을 멈춘다."""
    check_target(chat_serving, "create_chat_completion", chat_request, _CHAT_REQUEST_FIELDS)
    check_target(responses_serving, "create_responses", responses_request, _RESPONSES_REQUEST_FIELDS)
    _wrap(chat_serving, "create_chat_completion", relay_chat)
    _wrap(responses_serving, "create_responses", relay_responses)
    logger.info(
        "solar-relay: hooked %s.create_chat_completion (chat, messages) and %s.create_responses (pid %d)",
        chat_serving.__name__,
        responses_serving.__name__,
        os.getpid(),
    )


def check_target(serving: type, method: str, request_model: type, fields: tuple[str, ...]) -> None:
    target = getattr(serving, method, None)
    if not inspect.iscoroutinefunction(target):
        raise RuntimeError(f"solar-relay: {serving.__name__}.{method} is missing or not async")
    params = list(inspect.signature(target).parameters)
    if params[:2] != ["self", "request"]:
        raise RuntimeError(f"solar-relay: {serving.__name__}.{method} takes {params}, expected (self, request, ...)")
    if not callable(getattr(serving, "_effective_chat_template_kwargs", None)):
        raise RuntimeError(f"solar-relay: {serving.__name__} has no _effective_chat_template_kwargs")
    missing = [name for name in fields if name not in request_model.model_fields]
    if missing:
        raise RuntimeError(f"solar-relay: {request_model.__name__} has no fields {missing}")


def _wrap(serving: type, method: str, relay_request: Callable[[Any, Mapping[str, Any]], Patch]) -> None:
    original = getattr(serving, method)

    @functools.wraps(original)
    async def hooked(self, request, *args, **kwargs):
        try:
            # 템플릿이 받을 kwargs 는 vLLM 에게 묻는다 (최상위 effort, 요청 kwargs, 서버 기본값을 vLLM 규칙대로 합친 값).
            patch = relay_request(request, self._effective_chat_template_kwargs(request))
            if patch:
                logger.debug("solar-relay applied=%s", patch.applied)
        except Exception:
            # 요청을 막지 않는다. 대신 매번 ERROR 로 남겨 조용히 빠지지 않게 한다.
            logger.exception("solar-relay: request passed through unchanged after an error")
        return await original(self, request, *args, **kwargs)

    setattr(serving, method, hooked)


def _is_api_server_process() -> bool:
    # API 서버가 하나면 MainProcess, 여럿이면 vllm.v1.utils 가 ApiServer_{i} 로 띄운다.
    name = multiprocessing.current_process().name
    return name == "MainProcess" or name.startswith("ApiServer_")


# ---------------------------------------------------------------------------
# vLLM 요청 객체 ↔ relay(). vLLM 을 import 하지 않고 요청 필드 이름만 안다.
# ---------------------------------------------------------------------------
def relay_chat(request, template_kwargs: Mapping[str, Any], *, template_target: dict[str, Any] | None = None) -> Patch:
    """ChatCompletionRequest 에 relay() 를 적용한다.

    template_target 을 주면 템플릿 쪽 값은 요청 대신 그 dict 에 쓴다. Dynamo 는 요청에서 kwargs 를 이미 꺼내
    확정했으므로, 확정된 dict 에 써야 렌더에 닿는다.
    """
    patch = relay(
        template_kwargs=template_kwargs,
        lp_args=request.vllm_xargs,
        response_format=_as_dict(request.response_format),
        parallel_tool_calls=_sent(request, "parallel_tool_calls"),
        messages=request.messages,
    )
    _apply(request, patch, template_target)
    for index, (content, reasoning) in patch.reasoning_moves.items():
        message = request.messages[index]
        message["content"] = content
        if reasoning:
            # vLLM 렌더러(chat_utils)는 assistant 의 생각을 이 키로만 읽는다.
            message["reasoning"] = reasoning
    return patch


def relay_responses(request, template_kwargs: Mapping[str, Any]) -> Patch:
    """ResponsesRequest 에 relay() 를 적용한다. 이전 턴 메시지는 Responses 의 input 이라 다루지 않는다."""
    text = request.text
    patch = relay(
        template_kwargs=template_kwargs,
        lp_args=request.vllm_xargs,
        response_format=_text_format_to_chat(_as_dict(text and text.format)),
        parallel_tool_calls=_sent(request, "parallel_tool_calls"),
    )
    _apply(request, patch, None)
    return patch


def _apply(request, patch: Patch, template_target: dict[str, Any] | None) -> None:
    # relay() 가 빈 자리만 골랐으므로 그대로 얹는다.
    if patch.template_kwargs:
        if template_target is None:
            request.chat_template_kwargs = {**(request.chat_template_kwargs or {}), **patch.template_kwargs}
        else:
            template_target.update(patch.template_kwargs)
    if patch.lp_args:
        request.vllm_xargs = {**(request.vllm_xargs or {}), **patch.lp_args}


def _sent(request, name: str) -> Any:
    # 클라이언트가 보낸 필드만 값으로 본다. pydantic 기본값(parallel_tool_calls=True)은 외부 값이 아니다.
    return getattr(request, name) if name in request.model_fields_set else None


def _text_format_to_chat(text_format: dict[str, Any] | None) -> dict[str, Any] | None:
    # Responses 의 text.format 은 schema 를 바깥에 평평하게 둔다. 템플릿은 Chat 모양(json_schema.schema)만 읽는다.
    if text_format and text_format.get("type") == "json_schema":
        inner = {key: value for key, value in text_format.items() if key != "type"}
        return {"type": "json_schema", "json_schema": inner}
    return text_format


def _as_dict(value) -> dict[str, Any] | None:
    # vLLM 의 JsonSchemaResponseFormat 은 schema 를 json_schema 필드(alias "schema")에 둔다.
    return None if value is None else value.model_dump(by_alias=True, exclude_none=True)
