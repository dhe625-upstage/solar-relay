from __future__ import annotations

# Solar LP(vllm/v1/sample/logits_processor/solar_open2.py)가 extra_args 에서 읽는 키.
# 이름이 어긋나면 LP 가 값을 못 보고 조용히 기본값으로 돌아가므로 LP 상수와 같아야 한다.
LP_REASONING_EFFORT = "reasoning_effort"
LP_PARALLEL_TOOL_CALLS = "parallel_tool_calls"

# Solar 템플릿이 읽는 변수 이름.
TEMPLATE_REASONING_EFFORT = "reasoning_effort"
TEMPLATE_RESPONSE_FORMAT = "response_format"

# 엔진별로 요청 단위 LP 인자를 싣는 요청 필드.
LP_CHANNEL_FIELD = {
    "vllm": "vllm_xargs",
    "sglang": "custom_params",
}

# 템플릿이 실제로 보게 되는 effort 의 우선순위 (앞이 이긴다). 렌더하는 쪽마다 다르다.
#   vllm   : build_chat_params 가 최상위 값을 요청 kwargs 위에 얹고, with_defaults 가 서버 기본값을 깐다.
#   dynamo : prepost 가 서버 기본값 + 요청 kwargs 를 합친 뒤 최상위 값을 덮어쓴다.
#   sglang : _convert_to_internal_request 가 kwargs 의 값을 꺼내 최상위 값을 덮어쓴다.
# R1 은 이 순서를 그대로 따라야 템플릿과 LP 가 같은 effort 를 본다.
EFFORT_ORDER = {
    "vllm": ("top_level", "template_kwargs", "server_defaults"),
    "dynamo": ("top_level", "template_kwargs", "server_defaults"),
    "sglang": ("template_kwargs", "top_level", "server_defaults"),
}
