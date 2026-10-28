from __future__ import annotations

import asyncio
import copy
import multiprocessing
from types import SimpleNamespace

import pytest
from fakes import JSON_SCHEMA, ChatRequest, DumpOnly, FakeModel, ResponsesRequest

from solar_relay.hosts import vllm as vllm_host

CHAT_FIELDS = vllm_host._CHAT_REQUEST_FIELDS


def make_serving_classes():
    """vLLM 서빙 흉내. 템플릿이 받을 kwargs 는 vLLM 이 정하므로 테스트가 그 답(effective)을 준다.

    vLLM 처럼 요청의 chat_template_kwargs 는 답에 얹힌다.
    """

    class Serving:
        def __init__(self, effective=None):
            self.effective = effective or {}

        def _effective_chat_template_kwargs(self, request):
            return {**self.effective, **(request.chat_template_kwargs or {})}

    class ChatServing(Serving):
        async def create_chat_completion(self, request, raw_request=None):
            return request

    class ResponsesServing(Serving):
        async def create_responses(self, request, raw_request=None):
            return request

    return ChatServing, ResponsesServing


@pytest.fixture
def servings():
    chat, responses = make_serving_classes()
    vllm_host.install(
        chat_serving=chat,
        chat_request=ChatRequest,
        responses_serving=responses,
        responses_request=ResponsesRequest,
    )
    return chat, responses


def call_chat(serving_cls, request, effective=None):
    return asyncio.run(serving_cls(effective).create_chat_completion(request, None))


def call_responses(serving_cls, request, effective=None):
    return asyncio.run(serving_cls(effective).create_responses(request, None))


# ---------------------------------------------------------------------------
# chat (·messages) 훅
# ---------------------------------------------------------------------------
class TestChatHook:
    def test_values_reach_both_channels(self, servings):
        chat, _ = servings
        request = ChatRequest(response_format=DumpOnly(JSON_SCHEMA), parallel_tool_calls=False)
        out = call_chat(chat, request, effective={"reasoning_effort": "low"})
        assert out.vllm_xargs == {"reasoning_effort": "low", "parallel_tool_calls": 0}
        assert out.chat_template_kwargs == {"response_format": JSON_SCHEMA}

    def test_framework_default_parallel_tool_calls_is_not_relayed(self, servings):
        chat, _ = servings
        assert call_chat(chat, ChatRequest()).vllm_xargs is None

    def test_existing_request_values_are_kept(self, servings):
        chat, _ = servings
        request = ChatRequest(
            vllm_xargs={"reasoning_effort": "high", "other": 1},
            chat_template_kwargs={"think_render_option": "preserved"},
            response_format=DumpOnly(JSON_SCHEMA),
        )
        out = call_chat(chat, request, effective={"reasoning_effort": "low"})
        assert out.vllm_xargs == {"reasoning_effort": "high", "other": 1}
        assert out.chat_template_kwargs == {"think_render_option": "preserved", "response_format": JSON_SCHEMA}

    def test_think_history_is_moved_in_place(self, servings):
        chat, _ = servings
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>t</think>a"},
            {"role": "user", "content": "q2"},
        ]
        out = call_chat(chat, ChatRequest(messages=messages))
        assert out.messages[1] == {"role": "assistant", "content": "a", "reasoning": "t"}

    def test_applying_twice_changes_nothing_more(self, servings, monkeypatch):
        chat, _ = servings
        patches = []
        real_relay = vllm_host.relay

        def recording_relay(**kwargs):
            patches.append(real_relay(**kwargs))
            return patches[-1]

        monkeypatch.setattr(vllm_host, "relay", recording_relay)
        request = ChatRequest(
            response_format=DumpOnly(JSON_SCHEMA),
            parallel_tool_calls=False,
            messages=[{"role": "assistant", "content": "<think>t</think>a"}],
        )
        def written(out):
            # solar-relay 가 쓰는 세 필드만 본다.
            return copy.deepcopy((out.chat_template_kwargs, out.vllm_xargs, out.messages))

        once = written(call_chat(chat, request, effective={"reasoning_effort": "low"}))
        twice = written(call_chat(chat, request, effective={"reasoning_effort": "low"}))
        assert patches[0]
        assert not patches[1]
        assert twice == once

    def test_an_error_does_not_block_the_request(self, servings, monkeypatch):
        chat, _ = servings

        def boom(**_kwargs):
            raise ValueError("broken")

        monkeypatch.setattr(vllm_host, "relay", boom)
        out = call_chat(chat, ChatRequest(), effective={"reasoning_effort": "low"})
        assert out.vllm_xargs is None


# ---------------------------------------------------------------------------
# responses 훅
# ---------------------------------------------------------------------------
class TestResponsesHook:
    def test_reasoning_effort_and_text_format(self, servings):
        _, responses = servings
        text_format = {"type": "json_schema", "name": "r", "schema": {"type": "object"}}
        request = ResponsesRequest(text=SimpleNamespace(format=DumpOnly(text_format)))
        out = call_responses(responses, request, effective={"reasoning_effort": "low"})
        assert out.vllm_xargs == {"reasoning_effort": "low"}
        assert out.chat_template_kwargs == {"response_format": JSON_SCHEMA}

    @pytest.mark.parametrize(
        ("text_format", "expected"),
        [({"type": "json_object"}, {"response_format": {"type": "json_object"}}), ({"type": "text"}, None)],
    )
    def test_other_text_formats(self, servings, text_format, expected):
        _, responses = servings
        request = ResponsesRequest(text=SimpleNamespace(format=DumpOnly(text_format)))
        assert call_responses(responses, request).chat_template_kwargs == expected


# ---------------------------------------------------------------------------
# 자가 점검 — 대상이 바뀌면 기동을 멈춘다
# ---------------------------------------------------------------------------
class TestSelfCheck:
    def check(self, serving, request_model=ChatRequest):
        vllm_host.check_target(serving, "create_chat_completion", request_model, CHAT_FIELDS)

    def test_current_shape_passes(self):
        chat, _ = make_serving_classes()
        self.check(chat)

    def test_missing_method(self):
        class Serving:
            pass

        with pytest.raises(RuntimeError, match="missing or not async"):
            self.check(Serving)

    def test_sync_method(self):
        class Serving:
            def create_chat_completion(self, request, raw_request=None):
                pass

        with pytest.raises(RuntimeError, match="missing or not async"):
            self.check(Serving)

    def test_changed_signature(self):
        class Serving:
            async def create_chat_completion(self, req):
                pass

        with pytest.raises(RuntimeError, match="expected"):
            self.check(Serving)

    def test_missing_effective_template_kwargs(self):
        class Serving:
            async def create_chat_completion(self, request, raw_request=None):
                pass

        with pytest.raises(RuntimeError, match="_effective_chat_template_kwargs"):
            self.check(Serving)

    def test_missing_request_field(self):
        chat, _ = make_serving_classes()

        class OldRequest(FakeModel):
            model_fields = {name: None for name in CHAT_FIELDS if name != "vllm_xargs"}

        with pytest.raises(RuntimeError, match="vllm_xargs"):
            self.check(chat, request_model=OldRequest)


# ---------------------------------------------------------------------------
# 실제 vLLM 클래스 — 포크가 설치된 환경에서만
# ---------------------------------------------------------------------------
def test_real_vllm_resolves_the_effort_the_lp_gets():
    serving_module = pytest.importorskip("vllm.entrypoints.openai.chat_completion.serving")
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    serving = SimpleNamespace(
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs={"reasoning_effort": "none"},
    )
    effective = serving_module.OpenAIServingChat._effective_chat_template_kwargs

    def lp_effort(**body):
        request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "q"}], **body)
        return vllm_host.relay_chat(request, effective(serving, request)).lp_args.get("reasoning_effort")

    assert lp_effort() == "none"
    assert lp_effort(reasoning_effort="low") == "low"
    assert lp_effort(chat_template_kwargs={"reasoning_effort": "max"}) == "max"
    assert lp_effort(reasoning_effort="low", chat_template_kwargs={"reasoning_effort": "max"}) == "low"


def test_real_vllm_response_format_reaches_the_template_in_its_shape():
    pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "q"}],
        response_format={"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}},
    )
    vllm_host.relay_chat(request, {})
    # 템플릿은 response_format.type 과 response_format.json_schema.schema 를 읽는다.
    assert request.chat_template_kwargs == {"response_format": JSON_SCHEMA}


# ---------------------------------------------------------------------------
# 프로세스 구분 — 훅은 API 서버 프로세스에서만
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "is_api_server"),
    [("MainProcess", True), ("ApiServer_0", True), ("EngineCore_DP0", False), ("VllmWorker-0", False)],
)
def test_api_server_process_detection(monkeypatch, name, is_api_server):
    monkeypatch.setattr(multiprocessing, "current_process", lambda: SimpleNamespace(name=name))
    assert vllm_host._is_api_server_process() is is_api_server


def test_register_does_nothing_outside_the_api_server(monkeypatch):
    monkeypatch.setattr(vllm_host, "_is_api_server_process", lambda: False)
    monkeypatch.setattr(vllm_host, "install", lambda **_: pytest.fail("install called"))
    vllm_host.register()


def test_register_respects_the_env_switch(monkeypatch):
    monkeypatch.setattr(vllm_host, "_is_api_server_process", lambda: True)
    monkeypatch.setattr(vllm_host, "install", lambda **_: pytest.fail("install called"))
    monkeypatch.setenv("SOLAR_RELAY", "0")
    vllm_host.register()
