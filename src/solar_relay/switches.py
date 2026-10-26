from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

REQUEST_HEADER = "x-solar-relay"

_OFF_VALUES = frozenset({"0", "false", "off", "no"})


def _on(env: Mapping[str, str], name: str) -> bool:
    value = env.get(name)
    return value is None or value.strip().lower() not in _OFF_VALUES


@dataclass(frozen=True)
class Switches:
    """운영자가 규칙을 끄는 스위치. 요청 값을 정하지 않고, 전달을 할지 말지만 정한다."""

    enabled: bool = True
    reasoning_effort: bool = True
    response_format: bool = True
    parallel_tool_calls: bool = True
    think_history: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Switches:
        env = os.environ if env is None else env
        return cls(
            enabled=_on(env, "SOLAR_RELAY"),
            reasoning_effort=_on(env, "SOLAR_RELAY_REASONING_EFFORT"),
            response_format=_on(env, "SOLAR_RELAY_RESPONSE_FORMAT"),
            parallel_tool_calls=_on(env, "SOLAR_RELAY_PARALLEL_TOOL_CALLS"),
            think_history=_on(env, "SOLAR_RELAY_THINK_HISTORY"),
        )


def disabled_by_header(headers: Mapping[str, str]) -> bool:
    """요청 헤더 `x-solar-relay: off` 면 그 요청은 건드리지 않는다."""
    for key, value in headers.items():
        if key.lower() == REQUEST_HEADER:
            return value.strip().lower() in _OFF_VALUES
    return False
