from __future__ import annotations

import pytest

from solar_relay.relay import Patch, relay, relay_enabled

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "r", "schema": {"type": "object"}},
}


def run(**kwargs) -> Patch:
    values = {"template_kwargs": {}, "lp_args": None, "response_format": None, "parallel_tool_calls": None}
    return relay(**{**values, **kwargs})


# ---------------------------------------------------------------------------
# R1 — 템플릿이 받을 effort 를 LP 통로로
# ---------------------------------------------------------------------------
class TestReasoningEffort:
    def test_template_effort_reaches_lp(self):
        assert run(template_kwargs={"reasoning_effort": "low"}).lp_args == {"reasoning_effort": "low"}

    def test_explicit_lp_value_wins(self):
        patch = run(template_kwargs={"reasoning_effort": "low"}, lp_args={"reasoning_effort": "high"})
        assert patch.lp_args == {}

    @pytest.mark.parametrize("value", [None, 3])
    def test_non_string_value_is_not_relayed(self, value):
        assert run(template_kwargs={"reasoning_effort": value}).lp_args == {}


# ---------------------------------------------------------------------------
# R2 — response_format 을 템플릿 변수로
# ---------------------------------------------------------------------------
class TestResponseFormat:
    @pytest.mark.parametrize("response_format", [JSON_SCHEMA, {"type": "json_object"}])
    def test_json_formats_reach_the_template(self, response_format):
        patch = run(response_format=response_format)
        assert patch.template_kwargs == {"response_format": response_format}

    @pytest.mark.parametrize("response_format", [None, {"type": "text"}, {"type": "regex"}])
    def test_other_formats_are_left_alone(self, response_format):
        assert run(response_format=response_format).template_kwargs == {}

    def test_template_value_wins(self):
        patch = run(response_format=JSON_SCHEMA, template_kwargs={"response_format": {"type": "json_object"}})
        assert patch.template_kwargs == {}


# ---------------------------------------------------------------------------
# R3 — parallel_tool_calls=false 를 LP 통로로
# ---------------------------------------------------------------------------
class TestParallelToolCalls:
    def test_false_limits_lp_to_one_call(self):
        assert run(parallel_tool_calls=False).lp_args == {"parallel_tool_calls": 0}

    @pytest.mark.parametrize("value", [True, None])
    def test_true_or_unsent_is_the_lp_default(self, value):
        assert run(parallel_tool_calls=value).lp_args == {}

    def test_explicit_lp_value_wins(self):
        assert run(parallel_tool_calls=False, lp_args={"parallel_tool_calls": 1}).lp_args == {}


# ---------------------------------------------------------------------------
# think 이력 — content 에 섞인 <think> 를 reasoning 으로
# ---------------------------------------------------------------------------
class TestThinkHistory:
    def test_leading_think_block_moves_to_reasoning(self):
        messages = [{"role": "assistant", "content": "<think>plan A</think>\n\nanswer"}]
        assert run(messages=messages).reasoning_moves == {0: ("answer", "plan A")}

    def test_empty_think_block_only_strips_tags(self):
        messages = [{"role": "assistant", "content": "<think></think>answer"}]
        assert run(messages=messages).reasoning_moves == {0: ("answer", "")}

    def test_null_reasoning_counts_as_not_sent(self):
        messages = [{"role": "assistant", "content": "<think>t</think>a", "reasoning": None}]
        assert run(messages=messages).reasoning_moves == {0: ("a", "t")}

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "assistant", "content": "<think>x</think>y", "reasoning": "kept"},
            {"role": "assistant", "content": "<think>x</think>y", "reasoning": ""},
            {"role": "assistant", "content": "<think>x</think>y", "reasoning_content": "kept"},
            {"role": "assistant", "content": "<think>x</think>y", "reasoning_content": ""},
            {"role": "user", "content": "<think>x</think>y"},
            {"role": "assistant", "content": "answer <think>x</think>"},
            {"role": "assistant", "content": "<think>never closed"},
            {"role": "assistant", "content": [{"type": "text", "text": "<think>x</think>y"}]},
            {"role": "assistant", "content": None},
        ],
    )
    def test_other_messages_are_left_alone(self, message):
        assert run(messages=[message]).reasoning_moves == {}

    def test_reports_the_index_of_each_moved_message(self):
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>t1</think>a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "<think>t2</think>a2"},
        ]
        assert run(messages=messages).reasoning_moves == {1: ("a1", "t1"), 3: ("a2", "t2")}


# ---------------------------------------------------------------------------
# 공통 — 전체 결과, 전체 끄기
# ---------------------------------------------------------------------------
def test_full_request_patch():
    patch = run(
        template_kwargs={"reasoning_effort": "low"},
        response_format=JSON_SCHEMA,
        parallel_tool_calls=False,
        messages=[{"role": "assistant", "content": "<think>t</think>a"}],
    )
    assert patch.lp_args == {"reasoning_effort": "low", "parallel_tool_calls": 0}
    assert patch.template_kwargs == {"response_format": JSON_SCHEMA}
    assert patch.reasoning_moves == {0: ("a", "t")}
    assert patch.applied == ["reasoning_effort", "parallel_tool_calls", "response_format", "think_history"]


def test_empty_request_gives_empty_patch():
    patch = run()
    assert not patch
    assert patch.applied == []


@pytest.mark.parametrize(("value", "enabled"), [(None, True), ("1", True), ("0", False), (" OFF ", False), ("no", False)])
def test_env_switch(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv("SOLAR_RELAY", raising=False)
    else:
        monkeypatch.setenv("SOLAR_RELAY", value)
    assert relay_enabled() is enabled


# ---------------------------------------------------------------------------
# 계약 — 키 이름이 Solar LP 상수와 같아야 한다 (vLLM 포크가 있는 환경에서만)
# ---------------------------------------------------------------------------
def test_lp_keys_match_solar_logits_processor():
    solar_lp = pytest.importorskip("vllm.v1.sample.logits_processor.solar_open2")
    from solar_relay.relay import LP_PARALLEL_TOOL_CALLS, LP_REASONING_EFFORT

    assert LP_REASONING_EFFORT == solar_lp.REASONING_EFFORT_EXTRA_ARG
    assert LP_PARALLEL_TOOL_CALLS == solar_lp.PARALLEL_TOOL_CALLS_EXTRA_ARG
