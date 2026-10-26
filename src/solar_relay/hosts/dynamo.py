"""Dynamo host: 파이썬 chat processor(`--dyn-chat-processor vllm|sglang`) 입구에서 부르는 함수.

Dynamo 에는 요청 훅 장치가 없어서, processor 의 generator(request: dict) 첫 줄에서 이 함수를 부른다.
이 시점의 request 는 Rust 검증을 마친 Chat Completions dict 다 (messages·responses 도 chat 으로 바뀐 뒤).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from solar_relay.channels import EFFORT_ORDER, LP_CHANNEL_FIELD
from solar_relay.relay import Patch, RequestFields, relay
from solar_relay.switches import Switches

logger = logging.getLogger("dynamo.solar_relay")

# Rust 의 pythonize 는 serde alias 를 거꾸로 쓰지 않아서 클라이언트의 chat_template_kwargs 가
# chat_template_args 로 올 수 있다 (dynamo/frontend/prepost.py 와 같은 이유).
_TEMPLATE_KWARGS_KEYS = ("chat_template_kwargs", "chat_template_args")


def relay_request(
    request: dict[str, Any],
    *,
    server_template_defaults: Mapping[str, Any] | None = None,
    engine: str = "vllm",
    switches: Switches | None = None,
) -> Patch:
    """request 를 제자리에서 고치고, 적용한 변경분을 돌려준다. 오류가 나도 request 를 막지 않는다."""
    try:
        patch = relay(
            _fields(request, server_template_defaults, engine),
            effort_order=EFFORT_ORDER["dynamo"],
            switches=switches,
        )
        if patch:
            _apply(request, patch, engine)
            logger.info("solar-relay applied=%s", ",".join(patch.applied))
        return patch
    except Exception:
        logger.exception("solar-relay: request passed through unchanged after an error")
        return Patch()


def _template_kwargs(request: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    # 둘 다 오면 chat_template_kwargs 가 이긴다 (prepost 가 그쪽을 먼저 읽는다).
    for key in reversed(_TEMPLATE_KWARGS_KEYS):
        value = request.get(key)
        if isinstance(value, Mapping):
            merged.update(value)
    return merged


def _fields(
    request: Mapping[str, Any],
    server_template_defaults: Mapping[str, Any] | None,
    engine: str,
) -> RequestFields:
    parallel = request.get("parallel_tool_calls")
    return RequestFields(
        reasoning_effort=request.get("reasoning_effort"),
        chat_template_kwargs=_template_kwargs(request),
        server_template_defaults=server_template_defaults,
        engine_args=request.get(LP_CHANNEL_FIELD[engine]),
        response_format=request.get("response_format"),
        parallel_tool_calls=parallel if isinstance(parallel, bool) else None,
        has_tools=bool(request.get("tools")),
        messages=request.get("messages") or (),
    )


def _apply(request: dict[str, Any], patch: Patch, engine: str) -> None:
    if patch.template_kwargs:
        # 클라이언트 값을 아래 깔고 쓴다. chat_template_args 로 온 값도 잃지 않게 합쳐서 chat_template_kwargs 에 둔다.
        request["chat_template_kwargs"] = {**patch.template_kwargs, **_template_kwargs(request)}
    if patch.engine_args:
        field = LP_CHANNEL_FIELD[engine]
        request[field] = {**patch.engine_args, **(request.get(field) or {})}
    messages = request.get("messages") or []
    for index, (content, reasoning) in patch.reasoning_moves.items():
        messages[index]["content"] = content
        if reasoning:
            # vLLM 렌더러(chat_utils)는 assistant 의 생각을 "reasoning" 키로만 읽는다.
            messages[index]["reasoning"] = reasoning
