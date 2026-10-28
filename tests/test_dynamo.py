from __future__ import annotations

from fakes import JSON_SCHEMA, ChatRequest, DumpOnly

from solar_relay.hosts import dynamo


def run(request, template_kwargs=None):
    """Dynamo 가 확정한 템플릿 kwargs 와 함께 부른다."""
    template_kwargs = {} if template_kwargs is None else template_kwargs
    patch = dynamo.relay_chat_request(request, template_kwargs)
    return request, template_kwargs, patch


def test_values_reach_both_channels():
    request, template_kwargs, _ = run(
        ChatRequest(response_format=DumpOnly(JSON_SCHEMA), parallel_tool_calls=False),
        {"reasoning_effort": "low"},
    )
    assert request.vllm_xargs == {"reasoning_effort": "low", "parallel_tool_calls": 0}
    # 템플릿 쪽 값은 확정된 dict 에 들어가야 렌더에 닿는다. 요청 쪽 kwargs 는 이미 읽힌 뒤라 쓰지 않는다.
    assert template_kwargs == {"reasoning_effort": "low", "response_format": JSON_SCHEMA}
    assert request.chat_template_kwargs is None


def test_env_switch_turns_everything_off(monkeypatch):
    monkeypatch.setenv("SOLAR_RELAY", "0")
    request, template_kwargs, patch = run(ChatRequest(response_format=DumpOnly(JSON_SCHEMA)), {"reasoning_effort": "low"})
    assert not patch
    assert request.vllm_xargs is None
    assert template_kwargs == {"reasoning_effort": "low"}


def test_errors_leave_the_request_alone(monkeypatch):
    def boom(*_args, **_kwargs):
        raise ValueError("broken")

    monkeypatch.setattr(dynamo, "relay_chat", boom)
    request, template_kwargs, patch = run(ChatRequest(), {"reasoning_effort": "low"})
    assert not patch
    assert request.vllm_xargs is None
    assert template_kwargs == {"reasoning_effort": "low"}
