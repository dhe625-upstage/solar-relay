from __future__ import annotations

import asyncio
import multiprocessing
from types import SimpleNamespace

import pytest

from solar_relay.hosts import vllm_plugin

CHAT_FIELDS = vllm_plugin._CHAT_REQUEST_FIELDS
RESPONSES_FIELDS = vllm_plugin._RESPONSES_REQUEST_FIELDS


# ---------------------------------------------------------------------------
# vLLM 없이 돌리는 가짜 요청·서빙. 훅이 기대하는 이름과 모양만 흉내 낸다.
# ---------------------------------------------------------------------------
class FakeModel:
    model_fields: dict = {}

    def __init__(self, **values):
        defaults = {name: None for name in self.model_fields}
        defaults.update(values)
        self.__dict__.update(defaults)
        # pydantic 처럼, 생성 때 넘긴 필드만 "보낸 값"으로 친다.
        self.model_fields_set = set(values)


class ChatRequest(FakeModel):
    model_fields = {name: None for name in CHAT_FIELDS}

    def __init__(self, **values):
        values.setdefault("messages", [])
        super().__init__(**values)
        self.model_fields_set.discard("messages")
        if "parallel_tool_calls" not in self.model_fields_set:
            self.parallel_tool_calls = True  # vLLM 기본값


class ResponsesRequest(FakeModel):
    model_fields = {name: None for name in RESPONSES_FIELDS}


class DumpOnly:
    """pydantic 모델처럼 model_dump 만 있는 값."""

    def __init__(self, data):
        self._data = data

    def model_dump(self, **kwargs):
        assert kwargs["by_alias"] is True
        return dict(self._data)


def make_serving_classes():
    class ChatServing:
        def __init__(self, default_chat_template_kwargs=None):
            self.default_chat_template_kwargs = default_chat_template_kwargs or {}

        async def create_chat_completion(self, request, raw_request=None):
            return request

    class ResponsesServing:
        def __init__(self, default_chat_template_kwargs=None):
            self.chat_template_kwargs = default_chat_template_kwargs or {}

        async def create_responses(self, request, raw_request=None):
            return request

    return ChatServing, ResponsesServing


@pytest.fixture
def servings(monkeypatch):
    for name in ("SOLAR_RELAY", "SOLAR_RELAY_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    chat, responses = make_serving_classes()
    vllm_plugin.install(
        chat_serving=chat,
        chat_request=ChatRequest,
        responses_serving=responses,
        responses_request=ResponsesRequest,
    )
    return chat, responses


def call_chat(serving_cls, request, headers=None, server_defaults=None):
    raw = SimpleNamespace(headers=headers or {})
    serving = serving_cls(default_chat_template_kwargs=server_defaults)
    return asyncio.run(serving.create_chat_completion(request, raw))


def call_responses(serving_cls, request, server_defaults=None):
    serving = serving_cls(default_chat_template_kwargs=server_defaults)
    return asyncio.run(serving.create_responses(request, SimpleNamespace(headers={})))


JSON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}}
TOOLS = [{"type": "function", "function": {"name": "f"}}]


# ---------------------------------------------------------------------------
# chat (·messages) 훅
# ---------------------------------------------------------------------------
class TestChatHook:
    def test_values_reach_both_channels(self, servings):
        chat, _ = servings
        request = ChatRequest(
            reasoning_effort="low",
            response_format=DumpOnly(JSON_SCHEMA),
            parallel_tool_calls=False,
            tools=TOOLS,
        )
        out = call_chat(chat, request)
        assert out.vllm_xargs == {"reasoning_effort": "low", "parallel_tool_calls": 0}
        assert out.chat_template_kwargs == {"response_format": JSON_SCHEMA}

    def test_operator_default_effort_comes_from_the_serving_instance(self, servings):
        chat, _ = servings
        out = call_chat(chat, ChatRequest(), server_defaults={"reasoning_effort": "none"})
        assert out.vllm_xargs == {"reasoning_effort": "none"}

    def test_framework_default_parallel_tool_calls_is_not_relayed(self, servings):
        chat, _ = servings
        out = call_chat(chat, ChatRequest(tools=TOOLS))
        assert out.vllm_xargs is None

    def test_existing_request_values_are_kept(self, servings):
        chat, _ = servings
        request = ChatRequest(
            reasoning_effort="low",
            vllm_xargs={"reasoning_effort": "high", "other": 1},
            chat_template_kwargs={"think_render_option": "preserved"},
            response_format=JSON_SCHEMA,
        )
        out = call_chat(chat, request)
        assert out.vllm_xargs == {"reasoning_effort": "high", "other": 1}
        assert out.chat_template_kwargs == {
            "think_render_option": "preserved",
            "response_format": JSON_SCHEMA,
        }

    def test_think_history_is_moved_in_place(self, servings):
        chat, _ = servings
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>t</think>a"},
            {"role": "user", "content": "q2"},
        ]
        out = call_chat(chat, ChatRequest(messages=messages))
        assert out.messages[1] == {"role": "assistant", "content": "a", "reasoning": "t"}

    def test_header_off_leaves_request_untouched(self, servings):
        chat, _ = servings
        out = call_chat(chat, ChatRequest(reasoning_effort="low"), headers={"X-Solar-Relay": "off"})
        assert out.vllm_xargs is None

    def test_an_error_does_not_block_the_request(self, servings, monkeypatch):
        chat, _ = servings

        def boom(*_args, **_kwargs):
            raise ValueError("broken")

        monkeypatch.setattr(vllm_plugin, "relay", boom)
        out = call_chat(chat, ChatRequest(reasoning_effort="low"))
        assert out.vllm_xargs is None

    def test_wrapping_twice_installs_one_hook(self, servings):
        chat, responses = servings
        hooked = chat.create_chat_completion
        vllm_plugin.install(
            chat_serving=chat,
            chat_request=ChatRequest,
            responses_serving=responses,
            responses_request=ResponsesRequest,
        )
        assert chat.create_chat_completion is hooked


# ---------------------------------------------------------------------------
# responses 훅
# ---------------------------------------------------------------------------
class TestResponsesHook:
    def test_reasoning_effort_and_text_format(self, servings):
        _, responses = servings
        request = ResponsesRequest(
            reasoning=SimpleNamespace(effort="low"),
            text=SimpleNamespace(
                format=DumpOnly({"type": "json_schema", "name": "r", "schema": {"type": "object"}})
            ),
        )
        out = call_responses(responses, request)
        assert out.vllm_xargs == {"reasoning_effort": "low"}
        assert out.chat_template_kwargs == {"response_format": JSON_SCHEMA}

    def test_operator_default_comes_from_chat_template_kwargs_attr(self, servings):
        _, responses = servings
        out = call_responses(responses, ResponsesRequest(), server_defaults={"reasoning_effort": "none"})
        assert out.vllm_xargs == {"reasoning_effort": "none"}


# ---------------------------------------------------------------------------
# 자가 점검 — 대상이 바뀌면 기동을 멈춘다
# ---------------------------------------------------------------------------
class TestSelfCheck:
    def check(self, serving, request_model=ChatRequest, fields=CHAT_FIELDS):
        vllm_plugin.check_target(
            serving=serving,
            method="create_chat_completion",
            server_defaults_param="default_chat_template_kwargs",
            request_model=request_model,
            request_fields=fields,
        )

    def test_current_shape_passes(self):
        chat, _ = make_serving_classes()
        self.check(chat)

    def test_missing_method(self):
        class Serving:
            def __init__(self, default_chat_template_kwargs=None):
                pass

        with pytest.raises(RuntimeError, match="missing or not async"):
            self.check(Serving)

    def test_sync_method(self):
        class Serving:
            def __init__(self, default_chat_template_kwargs=None):
                pass

            def create_chat_completion(self, request, raw_request=None):
                pass

        with pytest.raises(RuntimeError, match="missing or not async"):
            self.check(Serving)

    def test_changed_signature(self):
        class Serving:
            def __init__(self, default_chat_template_kwargs=None):
                pass

            async def create_chat_completion(self, req):
                pass

        with pytest.raises(RuntimeError, match="expected"):
            self.check(Serving)

    def test_missing_server_defaults_param(self):
        class Serving:
            def __init__(self):
                pass

            async def create_chat_completion(self, request, raw_request=None):
                pass

        with pytest.raises(RuntimeError, match="default_chat_template_kwargs"):
            self.check(Serving)

    def test_missing_request_field(self):
        chat, _ = make_serving_classes()

        class OldRequest(FakeModel):
            model_fields = {name: None for name in CHAT_FIELDS if name != "vllm_xargs"}

        with pytest.raises(RuntimeError, match="vllm_xargs"):
            self.check(chat, request_model=OldRequest)


# ---------------------------------------------------------------------------
# 프로세스 구분 — 훅은 API 서버 프로세스에서만
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "is_api_server"),
    [
        ("MainProcess", True),
        ("ApiServer_0", True),
        ("EngineCore_DP0", False),
        ("VllmWorker-0", False),
    ],
)
def test_api_server_process_detection(monkeypatch, name, is_api_server):
    monkeypatch.setattr(multiprocessing, "current_process", lambda: SimpleNamespace(name=name))
    assert vllm_plugin._is_api_server_process() is is_api_server


def test_register_does_nothing_outside_the_api_server(monkeypatch):
    monkeypatch.setattr(vllm_plugin, "_is_api_server_process", lambda: False)
    monkeypatch.setattr(vllm_plugin, "install", lambda **_: pytest.fail("install called"))
    vllm_plugin.register()


def test_register_respects_the_master_switch(monkeypatch):
    monkeypatch.setattr(vllm_plugin, "_is_api_server_process", lambda: True)
    monkeypatch.setattr(vllm_plugin, "install", lambda **_: pytest.fail("install called"))
    monkeypatch.setenv("SOLAR_RELAY", "0")
    vllm_plugin.register()
