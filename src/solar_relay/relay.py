"""solar-relay core. 요청에 이미 있는 값을 Solar 템플릿과 logits processor(LP) 통로로 옮길 변경분을 정한다.

값을 새로 만들지 않는다. 요청이나 운영자 설정이 준 값만 옮기고, 없으면 엔진·템플릿 기본값이 쓰이게 둔다.
엔진을 import 하지 않는다.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Solar LP(vllm/v1/sample/logits_processor/solar_open2.py)가 extra_args 에서 읽는 키.
# 이름이 어긋나면 LP 가 값을 못 보고 조용히 기본값으로 돌아가므로 LP 상수와 같아야 한다.
LP_REASONING_EFFORT = "reasoning_effort"
LP_PARALLEL_TOOL_CALLS = "parallel_tool_calls"
# Solar 템플릿이 읽는 변수 이름.
TEMPLATE_REASONING_EFFORT = "reasoning_effort"
TEMPLATE_RESPONSE_FORMAT = "response_format"

_JSON_RESPONSE_FORMATS = ("json_object", "json_schema")
# 여는 태그는 앞 공백만 허용하고, 닫는 태그는 str.find 로 찾는다 (정규식으로 생각 본문을 훑으면 3–4배 느리다).
_THINK_OPEN = re.compile(r"\s*<think>")
_THINK_CLOSE = "</think>"


def relay_enabled() -> bool:
    """`SOLAR_RELAY=0` 이면 패키지 전체를 끈다. 이미지를 다시 만들지 않고 되돌리는 비상용이다."""
    return os.environ.get("SOLAR_RELAY", "1").strip().lower() not in ("0", "false", "off", "no")


@dataclass(frozen=True)
class Patch:
    """요청에 더할 값. 전부 요청에 비어 있던 자리만 담는다."""

    template_kwargs: dict[str, Any] = field(default_factory=dict)
    lp_args: dict[str, Any] = field(default_factory=dict)
    # 메시지 인덱스 -> (think 블록을 뺀 content, reasoning). reasoning 이 빈 문자열이면 태그만 지운다.
    reasoning_moves: dict[int, tuple[str, str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.template_kwargs or self.lp_args or self.reasoning_moves)

    @property
    def applied(self) -> list[str]:
        """로그용. 채운 키 이름과, 메시지를 옮겼으면 think_history."""
        moved = ["think_history"] if self.reasoning_moves else []
        return [*self.lp_args, *self.template_kwargs, *moved]


def relay(
    *,
    template_kwargs: Mapping[str, Any],
    lp_args: Mapping[str, Any] | None,
    response_format: Mapping[str, Any] | None,
    parallel_tool_calls: bool | None,
    messages: Sequence[Mapping[str, Any]] = (),
) -> Patch:
    """요청을 고치지 않고 변경분만 돌려준다.

    template_kwargs: 템플릿이 실제로 받을 kwargs. 렌더러마다 병합 규칙이 달라서 흉내 내지 않고,
        host 가 렌더러에게 묻거나 렌더러가 확정한 값을 넘긴다.
    lp_args: 요청의 LP 통로(vllm_xargs). 여기 이미 있는 키는 건드리지 않는다.
    parallel_tool_calls: 요청이 보낸 값. 보내지 않았으면 None (프레임워크 기본값 True 를 넣으면 안 된다).
    """
    lp_args = lp_args or {}
    new_lp: dict[str, Any] = {}
    # R1: 템플릿이 받을 effort 를 LP 에도 준다. 문자열이 아니면(None 포함) LP 가 거절하므로 옮기지 않는다.
    effort = template_kwargs.get(TEMPLATE_REASONING_EFFORT)
    if isinstance(effort, str) and LP_REASONING_EFFORT not in lp_args:
        new_lp[LP_REASONING_EFFORT] = effort
    # R3: false 면 LP 가 툴콜을 하나로 막게 한다(LP 값 0). true 는 LP 기본값(여러 개 허용)과 같다.
    if parallel_tool_calls is False and LP_PARALLEL_TOOL_CALLS not in lp_args:
        new_lp[LP_PARALLEL_TOOL_CALLS] = 0

    new_template: dict[str, Any] = {}
    # R2: json_object / json_schema 만 템플릿의 출력 형식 블록으로 간다. text 는 블록이 없다.
    if (
        response_format is not None
        and response_format.get("type") in _JSON_RESPONSE_FORMATS
        and TEMPLATE_RESPONSE_FORMAT not in template_kwargs
    ):
        new_template[TEMPLATE_RESPONSE_FORMAT] = response_format

    return Patch(
        template_kwargs=new_template,
        lp_args=new_lp,
        reasoning_moves=_think_history_moves(messages),
    )


def _think_history_moves(messages: Sequence[Mapping[str, Any]]) -> dict[int, tuple[str, str]]:
    # 이전 턴 assistant content 앞에 글자로 붙어 온 <think>…</think> 는 원래 reasoning 필드에 있어야 할 값이다.
    # 클라이언트가 reasoning 을 따로 보낸 메시지는 빈 문자열이어도 건드리지 않는다.
    moves: dict[int, tuple[str, str]] = {}
    for index, message in enumerate(messages):
        content = message.get("content")
        if (
            message.get("role") != "assistant"
            or message.get("reasoning") is not None
            or message.get("reasoning_content") is not None
            or not isinstance(content, str)
        ):
            continue
        opened = _THINK_OPEN.match(content)
        if opened is None:
            continue
        closed = content.find(_THINK_CLOSE, opened.end())
        if closed >= 0:
            answer = content[closed + len(_THINK_CLOSE) :].lstrip()
            moves[index] = (answer, content[opened.end() : closed].strip())
    return moves
