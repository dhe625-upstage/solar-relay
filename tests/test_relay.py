from __future__ import annotations

from dataclasses import replace

import pytest

from solar_relay import (
    EFFORT_ORDER,
    Patch,
    RequestFields,
    Switches,
    disabled_by_header,
    relay,
    responses_text_format_to_chat,
)

ALL_ON = Switches()
VLLM = EFFORT_ORDER["vllm"]
SGLANG = EFFORT_ORDER["sglang"]

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "r", "schema": {"type": "object"}},
}
TOOLS_ON = {"has_tools": True}


def run(order=VLLM, switches=ALL_ON, **kwargs) -> Patch:
    return relay(RequestFields(**kwargs), effort_order=order, switches=switches)


def apply(fields: RequestFields, patch: Patch) -> RequestFields:
    """host 가 하는 일을 흉내 낸다: 비어 있는 자리에만 더한다."""
    template_kwargs = {**patch.template_kwargs, **(fields.chat_template_kwargs or {})}
    engine_args = {**patch.engine_args, **(fields.engine_args or {})}
    messages = [dict(m) for m in fields.messages]
    for index, (content, reasoning) in patch.reasoning_moves.items():
        messages[index]["content"] = content
        if reasoning:
            messages[index]["reasoning"] = reasoning
    return replace(
        fields,
        chat_template_kwargs=template_kwargs,
        engine_args=engine_args,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# R1 — effort 를 LP 통로로
# ---------------------------------------------------------------------------
class TestReasoningEffort:
    def test_top_level_value_reaches_lp(self):
        assert run(reasoning_effort="low").engine_args == {"reasoning_effort": "low"}

    def test_request_template_kwarg_is_used_when_top_level_is_absent(self):
        patch = run(chat_template_kwargs={"reasoning_effort": "max"})
        assert patch.engine_args == {"reasoning_effort": "max"}

    def test_operator_default_is_used_when_request_has_none(self):
        patch = run(server_template_defaults={"reasoning_effort": "none"})
        assert patch.engine_args == {"reasoning_effort": "none"}

    def test_nothing_is_invented_when_no_one_set_effort(self):
        assert run().engine_args == {}

    def test_explicit_lp_value_wins(self):
        patch = run(reasoning_effort="low", engine_args={"reasoning_effort": "high"})
        assert patch.engine_args == {}

    @pytest.mark.parametrize("unset", [None, "auto"])
    def test_unset_values_fall_through_to_next_source(self, unset):
        patch = run(
            reasoning_effort=unset,
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        assert patch.engine_args == {"reasoning_effort": "medium"}

    @pytest.mark.parametrize(
        ("order", "expected"),
        [(VLLM, "low"), (EFFORT_ORDER["dynamo"], "low"), (SGLANG, "high")],
    )
    def test_follows_the_order_the_template_uses(self, order, expected):
        patch = run(
            order=order,
            reasoning_effort="low",
            chat_template_kwargs={"reasoning_effort": "high"},
        )
        assert patch.engine_args == {"reasoning_effort": expected}

    def test_non_string_value_is_not_relayed(self):
        assert run(chat_template_kwargs={"reasoning_effort": 3}).engine_args == {}


# ---------------------------------------------------------------------------
# R2 — response_format 을 템플릿 변수로
# ---------------------------------------------------------------------------
class TestResponseFormat:
    @pytest.mark.parametrize("response_format", [JSON_SCHEMA, {"type": "json_object"}])
    def test_json_formats_reach_the_template(self, response_format):
        patch = run(response_format=response_format)
        assert patch.template_kwargs == {"response_format": response_format}

    def test_relayed_value_is_a_copy(self):
        patch = run(response_format=JSON_SCHEMA)
        assert patch.template_kwargs["response_format"] is not JSON_SCHEMA

    @pytest.mark.parametrize("response_format", [None, {"type": "text"}, {"type": "regex"}])
    def test_other_formats_are_left_alone(self, response_format):
        assert run(response_format=response_format).template_kwargs == {}

    def test_explicit_template_kwarg_wins(self):
        patch = run(
            response_format=JSON_SCHEMA,
            chat_template_kwargs={"response_format": None},
        )
        assert patch.template_kwargs == {}


# ---------------------------------------------------------------------------
# R3 — parallel_tool_calls 를 LP 통로로
# ---------------------------------------------------------------------------
class TestParallelToolCalls:
    @pytest.mark.parametrize(("value", "flag"), [(False, 0), (True, 1)])
    def test_explicit_value_reaches_lp_as_int(self, value, flag):
        patch = run(parallel_tool_calls=value, **TOOLS_ON)
        assert patch.engine_args == {"parallel_tool_calls": flag}

    def test_not_relayed_without_tools(self):
        assert run(parallel_tool_calls=False).engine_args == {}

    def test_not_relayed_when_request_did_not_send_it(self):
        assert run(**TOOLS_ON).engine_args == {}

    def test_explicit_lp_value_wins(self):
        patch = run(
            parallel_tool_calls=False,
            engine_args={"parallel_tool_calls": 1},
            **TOOLS_ON,
        )
        assert patch.engine_args == {}


# ---------------------------------------------------------------------------
# (a) — content 에 섞인 <think> 를 reasoning 으로
# ---------------------------------------------------------------------------
class TestThinkHistory:
    def test_leading_think_block_moves_to_reasoning(self):
        messages = [{"role": "assistant", "content": "<think>plan A</think>\n\nanswer"}]
        assert run(messages=messages).reasoning_moves == {0: ("answer", "plan A")}

    def test_empty_think_block_only_strips_tags(self):
        messages = [{"role": "assistant", "content": "<think></think>answer"}]
        assert run(messages=messages).reasoning_moves == {0: ("answer", "")}

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "assistant", "content": "<think>x</think>y", "reasoning": "kept"},
            {"role": "assistant", "content": "<think>x</think>y", "reasoning_content": "kept"},
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
# 공통 — 스위치, 멱등성, Responses 모양 변환
# ---------------------------------------------------------------------------
FULL_REQUEST = {
    "reasoning_effort": "low",
    "response_format": JSON_SCHEMA,
    "parallel_tool_calls": False,
    "has_tools": True,
    "messages": [{"role": "assistant", "content": "<think>t</think>a"}],
}


class TestCommon:
    def test_full_request_patch(self):
        patch = run(**FULL_REQUEST)
        assert patch.engine_args == {"reasoning_effort": "low", "parallel_tool_calls": 0}
        assert patch.template_kwargs == {"response_format": JSON_SCHEMA}
        assert patch.reasoning_moves == {0: ("a", "t")}
        assert patch.applied == (
            "reasoning_effort",
            "response_format",
            "parallel_tool_calls",
            "think_history",
        )

    def test_applying_twice_changes_nothing_more(self):
        fields = RequestFields(**FULL_REQUEST)
        once = apply(fields, relay(fields, effort_order=VLLM, switches=ALL_ON))
        assert not relay(once, effort_order=VLLM, switches=ALL_ON)

    def test_empty_request_gives_empty_patch(self):
        patch = run()
        assert not patch
        assert patch.applied == ()

    def test_master_switch_disables_everything(self):
        assert not run(switches=Switches(enabled=False), **FULL_REQUEST)

    @pytest.mark.parametrize(
        ("switch", "rule"),
        [
            ("reasoning_effort", "reasoning_effort"),
            ("response_format", "response_format"),
            ("parallel_tool_calls", "parallel_tool_calls"),
            ("think_history", "think_history"),
        ],
    )
    def test_each_switch_disables_only_its_rule(self, switch, rule):
        patch = run(switches=replace(ALL_ON, **{switch: False}), **FULL_REQUEST)
        assert rule not in patch.applied
        assert len(patch.applied) == 3


class TestSwitches:
    def test_everything_is_on_when_env_is_empty(self):
        assert Switches.from_env({}) == Switches()

    @pytest.mark.parametrize("value", ["0", "false", "OFF", " no "])
    def test_off_values_turn_a_rule_off(self, value):
        switches = Switches.from_env({"SOLAR_RELAY_RESPONSE_FORMAT": value})
        assert switches.response_format is False
        assert switches.enabled is True

    def test_master_env_switch(self):
        assert Switches.from_env({"SOLAR_RELAY": "0"}).enabled is False

    @pytest.mark.parametrize(
        ("headers", "disabled"),
        [
            ({"x-solar-relay": "off"}, True),
            ({"X-Solar-Relay": "0"}, True),
            ({"x-solar-relay": "on"}, False),
            ({}, False),
        ],
    )
    def test_request_header(self, headers, disabled):
        assert disabled_by_header(headers) is disabled


class TestResponsesTextFormat:
    def test_json_schema_is_nested_like_chat(self):
        text_format = {"type": "json_schema", "name": "r", "schema": {"type": "object"}, "strict": True}
        assert responses_text_format_to_chat(text_format) == {
            "type": "json_schema",
            "json_schema": {"name": "r", "schema": {"type": "object"}, "strict": True},
        }

    def test_json_object(self):
        assert responses_text_format_to_chat({"type": "json_object"}) == {"type": "json_object"}

    @pytest.mark.parametrize("text_format", [None, {"type": "text"}])
    def test_text_has_no_response_format(self, text_format):
        assert responses_text_format_to_chat(text_format) is None


# ---------------------------------------------------------------------------
# 계약 — 키 이름이 Solar LP 상수와 같아야 한다 (vLLM 포크가 있는 환경에서만)
# ---------------------------------------------------------------------------
def test_lp_keys_match_solar_logits_processor():
    solar_lp = pytest.importorskip("vllm.v1.sample.logits_processor.solar_open2")
    from solar_relay.channels import LP_PARALLEL_TOOL_CALLS, LP_REASONING_EFFORT

    assert LP_REASONING_EFFORT == solar_lp.REASONING_EFFORT_EXTRA_ARG
    assert LP_PARALLEL_TOOL_CALLS == solar_lp.PARALLEL_TOOL_CALLS_EXTRA_ARG
