from __future__ import annotations

import pytest

from solar_relay import Switches
from solar_relay.hosts import dynamo

JSON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}}
TOOLS = [{"type": "function", "function": {"name": "f"}}]


def run(request, **kwargs):
    kwargs.setdefault("switches", Switches())
    dynamo.relay_request(request, **kwargs)
    return request


def test_values_reach_both_channels():
    request = run({
        "reasoning_effort": "low",
        "response_format": JSON_SCHEMA,
        "parallel_tool_calls": False,
        "tools": TOOLS,
        "messages": [],
    })
    assert request["vllm_xargs"] == {"reasoning_effort": "low", "parallel_tool_calls": 0}
    assert request["chat_template_kwargs"] == {"response_format": JSON_SCHEMA}


def test_operator_default_effort_is_relayed():
    request = run({"messages": []}, server_template_defaults={"reasoning_effort": "none"})
    assert request["vllm_xargs"] == {"reasoning_effort": "none"}


def test_top_level_effort_beats_request_kwargs_like_dynamo_prepost():
    request = run({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}})
    assert request["vllm_xargs"] == {"reasoning_effort": "low"}


def test_chat_template_args_from_rust_are_read_and_kept():
    request = run({
        "chat_template_args": {"reasoning_effort": "max", "think_render_option": "preserved"},
        "response_format": JSON_SCHEMA,
    })
    assert request["vllm_xargs"] == {"reasoning_effort": "max"}
    assert request["chat_template_kwargs"] == {
        "response_format": JSON_SCHEMA,
        "reasoning_effort": "max",
        "think_render_option": "preserved",
    }


def test_existing_values_win():
    request = run({
        "reasoning_effort": "low",
        "vllm_xargs": {"reasoning_effort": "high"},
        "chat_template_kwargs": {"response_format": None},
        "response_format": JSON_SCHEMA,
    })
    assert request["vllm_xargs"] == {"reasoning_effort": "high"}
    assert request["chat_template_kwargs"] == {"response_format": None}


def test_null_parallel_tool_calls_from_rust_is_not_relayed():
    request = run({"parallel_tool_calls": None, "tools": TOOLS})
    assert "vllm_xargs" not in request


def test_think_history_moves_to_reasoning_key():
    request = run({"messages": [{"role": "assistant", "content": "<think>t</think>a"}]})
    assert request["messages"][0] == {"role": "assistant", "content": "a", "reasoning": "t"}


def test_sglang_engine_uses_custom_params():
    request = run({"reasoning_effort": "low"}, engine="sglang")
    assert request["custom_params"] == {"reasoning_effort": "low"}


def test_errors_leave_the_request_alone(monkeypatch):
    def boom(*_args, **_kwargs):
        raise ValueError("broken")

    monkeypatch.setattr(dynamo, "relay", boom)
    request = {"reasoning_effort": "low"}
    assert not dynamo.relay_request(request, switches=Switches())
    assert request == {"reasoning_effort": "low"}


@pytest.mark.parametrize("request_body", [{}, {"messages": None}])
def test_empty_request(request_body):
    assert run(dict(request_body)) == request_body
