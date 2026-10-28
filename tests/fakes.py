"""vLLM 없이 돌리는 가짜 요청. host 가 기대하는 이름과 모양만 흉내 낸다."""

from __future__ import annotations

from solar_relay.hosts.vllm import _CHAT_REQUEST_FIELDS, _RESPONSES_REQUEST_FIELDS

JSON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}}


class FakeModel:
    model_fields: dict = {}

    def __init__(self, **values):
        defaults = {name: None for name in self.model_fields}
        defaults.update(values)
        self.__dict__.update(defaults)
        # pydantic 처럼, 생성 때 넘긴 필드만 "보낸 값"으로 친다.
        self.model_fields_set = set(values)


class ChatRequest(FakeModel):
    model_fields = {name: None for name in _CHAT_REQUEST_FIELDS}

    def __init__(self, **values):
        values.setdefault("messages", [])
        super().__init__(**values)
        if "parallel_tool_calls" not in self.model_fields_set:
            self.parallel_tool_calls = True  # vLLM 기본값


class ResponsesRequest(FakeModel):
    model_fields = {name: None for name in _RESPONSES_REQUEST_FIELDS}


class DumpOnly:
    """pydantic 모델처럼 model_dump 만 있는 값."""

    def __init__(self, data):
        self._data = data

    def model_dump(self, **kwargs):
        assert kwargs["by_alias"] is True
        return dict(self._data)
