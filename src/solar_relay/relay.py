from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from solar_relay.channels import (
    LP_PARALLEL_TOOL_CALLS,
    LP_REASONING_EFFORT,
    TEMPLATE_REASONING_EFFORT,
    TEMPLATE_RESPONSE_FORMAT,
)
from solar_relay.switches import Switches

# vLLM merge_kwargs 가 "설정 안 됨"으로 보는 값. 템플릿이 보는 effort 와 맞추려고 같은 규칙을 쓴다.
_UNSET = (None, "auto")
_JSON_RESPONSE_FORMATS = ("json_object", "json_schema")
_LEADING_THINK = re.compile(r"\A\s*<think>(.*?)</think>\s*", re.S)


@dataclass(frozen=True)
class RequestFields:
    """host 가 요청 하나에서 읽어 넘기는 값. Chat Completions 모양으로 맞춘다."""

    reasoning_effort: Any = None
    chat_template_kwargs: Mapping[str, Any] | None = None
    server_template_defaults: Mapping[str, Any] | None = None
    engine_args: Mapping[str, Any] | None = None
    response_format: Mapping[str, Any] | None = None
    # None 은 "요청이 보내지 않았다". 프레임워크 기본값(True)을 넣으면 안 된다.
    parallel_tool_calls: bool | None = None
    has_tools: bool = False
    messages: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class Patch:
    """요청에 더할 값. 전부 요청에 비어 있던 자리만 담는다."""

    template_kwargs: dict[str, Any] = field(default_factory=dict)
    engine_args: dict[str, Any] = field(default_factory=dict)
    # 메시지 인덱스 -> (think 블록을 뺀 content, reasoning). reasoning 이 빈 문자열이면 태그만 지운다.
    reasoning_moves: dict[int, tuple[str, str]] = field(default_factory=dict)
    applied: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.template_kwargs or self.engine_args or self.reasoning_moves)


def relay(
    fields: RequestFields,
    *,
    effort_order: Sequence[str],
    switches: Switches | None = None,
) -> Patch:
    """요청·운영자가 이미 준 값을 템플릿과 LP 통로로 옮긴다. 값을 새로 만들지 않는다."""
    switches = Switches.from_env() if switches is None else switches
    if not switches.enabled:
        return Patch()

    template_kwargs: dict[str, Any] = {}
    engine_args: dict[str, Any] = {}
    applied: list[str] = []

    if switches.reasoning_effort:
        effort = _relay_reasoning_effort(fields, effort_order)
        if effort is not None:
            engine_args[LP_REASONING_EFFORT] = effort
            applied.append("reasoning_effort")

    if switches.response_format:
        response_format = _relay_response_format(fields)
        if response_format is not None:
            template_kwargs[TEMPLATE_RESPONSE_FORMAT] = response_format
            applied.append("response_format")

    if switches.parallel_tool_calls:
        flag = _relay_parallel_tool_calls(fields)
        if flag is not None:
            engine_args[LP_PARALLEL_TOOL_CALLS] = flag
            applied.append("parallel_tool_calls")

    reasoning_moves: dict[int, tuple[str, str]] = {}
    if switches.think_history:
        reasoning_moves = _relay_think_history(fields.messages)
        if reasoning_moves:
            applied.append("think_history")

    return Patch(
        template_kwargs=template_kwargs,
        engine_args=engine_args,
        reasoning_moves=reasoning_moves,
        applied=tuple(applied),
    )


def _relay_reasoning_effort(fields: RequestFields, effort_order: Sequence[str]) -> str | None:
    # R1: 템플릿이 보게 될 effort 를 LP 에도 준다. 요청 LP 통로에 이미 있으면 그 값이 이긴다.
    if LP_REASONING_EFFORT in (fields.engine_args or {}):
        return None
    sources = {
        "top_level": fields.reasoning_effort,
        "template_kwargs": (fields.chat_template_kwargs or {}).get(TEMPLATE_REASONING_EFFORT),
        "server_defaults": (fields.server_template_defaults or {}).get(TEMPLATE_REASONING_EFFORT),
    }
    for source in effort_order:
        value = sources[source]
        if value in _UNSET:
            continue
        # 문자열이 아닌 값은 LP 가 거절하므로 옮기지 않는다. 템플릿 쪽은 원래 값을 그대로 받는다.
        return value if isinstance(value, str) else None
    return None


def _relay_response_format(fields: RequestFields) -> dict[str, Any] | None:
    # R2: json_object / json_schema 만 템플릿의 출력 형식 블록으로 간다. text 는 블록이 없다.
    if TEMPLATE_RESPONSE_FORMAT in (fields.chat_template_kwargs or {}):
        return None
    response_format = fields.response_format
    if not isinstance(response_format, Mapping):
        return None
    if response_format.get("type") not in _JSON_RESPONSE_FORMATS:
        return None
    return dict(response_format)


def _relay_parallel_tool_calls(fields: RequestFields) -> int | None:
    # R3: 요청이 명시한 값만 옮긴다. tools 가 없으면 LP 가 막을 툴콜이 없다.
    if not fields.has_tools or not isinstance(fields.parallel_tool_calls, bool):
        return None
    if LP_PARALLEL_TOOL_CALLS in (fields.engine_args or {}):
        return None
    # vllm_xargs 는 str/int/float 만 실어 나르므로 bool 대신 int 로 둔다.
    return int(fields.parallel_tool_calls)


def _relay_think_history(messages: Sequence[Mapping[str, Any]]) -> dict[int, tuple[str, str]]:
    # (a): 이전 턴 assistant content 앞에 붙은 <think>…</think> 를 reasoning 필드로 옮긴다.
    # 클라이언트가 reasoning 을 따로 보낸 메시지는 건드리지 않는다.
    moves: dict[int, tuple[str, str]] = {}
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if message.get("reasoning") or message.get("reasoning_content"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = _LEADING_THINK.match(content)
        if match is None:
            continue
        moves[index] = (content[match.end() :], match.group(1).strip())
    return moves


def responses_text_format_to_chat(text_format: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Responses API 의 `text.format`(평평한 모양)을 Chat Completions `response_format` 모양으로 바꾼다."""
    if not isinstance(text_format, Mapping):
        return None
    format_type = text_format.get("type")
    if format_type == "json_object":
        return {"type": "json_object"}
    if format_type == "json_schema":
        inner = {
            key: text_format[key]
            for key in ("name", "schema", "strict", "description")
            if key in text_format
        }
        return {"type": "json_schema", "json_schema": inner}
    return None
