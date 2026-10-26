# solar-relay

요청에 이미 있는 값을 Solar 채팅 템플릿과 logits processor(LP) 통로로 옮기는 패키지입니다.
템플릿을 렌더하는 프로세스(vLLM API 서버, Dynamo 프론트엔드, SGLang 서버)의 입구에서 요청마다 한 번 실행됩니다.

**원칙: 값을 만들지 않습니다.** 요청이나 운영자 설정이 준 값만 옮깁니다. 값이 없으면 엔진·템플릿 기본값이 그대로 쓰입니다.

## 규칙

| 규칙 | 입력 | 옮기는 곳 |
|---|---|---|
| R1 effort | 최상위 `reasoning_effort` → 요청 `chat_template_kwargs.reasoning_effort` → 운영자 `--default-chat-template-kwargs` (순서는 렌더하는 쪽마다 다름, `EFFORT_ORDER`) | LP 통로 `reasoning_effort` |
| R2 response_format | `response_format` (json_object, json_schema) | `chat_template_kwargs.response_format` |
| R3 툴콜 1개 | 요청이 명시한 `parallel_tool_calls` (tools가 있을 때) | LP 통로 `parallel_tool_calls` (0/1) |
| think 이력 | assistant content 앞의 `<think>…</think>` | 그 메시지의 `reasoning` |

요청의 같은 자리에 값이 이미 있으면 덮지 않습니다. 두 번 적용해도 결과가 같습니다.

## 끄기

| 범위 | 방법 |
|---|---|
| 전체 | `SOLAR_RELAY=0` |
| 규칙별 | `SOLAR_RELAY_REASONING_EFFORT`, `SOLAR_RELAY_RESPONSE_FORMAT`, `SOLAR_RELAY_PARALLEL_TOOL_CALLS`, `SOLAR_RELAY_THINK_HISTORY` = `0` |
| 요청별 | 헤더 `x-solar-relay: off` |

## host

| 프레임워크 | 켜는 법 | 덮는 경로 |
|---|---|---|
| vLLM | 패키지를 설치하면 `vllm.general_plugins` entry point로 자동 로드. API 서버 프로세스에서만 `OpenAIServingChat.create_chat_completion`과 `OpenAIServingResponses.create_responses`를 감싼다. 끄려면 `SOLAR_RELAY=0` 또는 `VLLM_PLUGINS` | `/v1/chat/completions`, `/v1/messages`, `/v1/responses` |
| Dynamo | 파이썬 chat processor의 `generator(request)` 첫 줄에서 `solar_relay.hosts.dynamo.relay_request(request, server_template_defaults=...)` 호출 | chat (Rust가 messages·responses를 chat으로 바꾼 뒤) |

vLLM host는 기동할 때 훅 대상(메서드 이름·시그니처, 생성자 인자, 요청 필드)을 점검하고, 맞지 않으면 기동을 멈춥니다. 요청 처리 중 오류는 그 요청만 그대로 통과시키고 ERROR 로그를 남깁니다.

## host를 직접 만들 때

```python
from solar_relay import EFFORT_ORDER, LP_CHANNEL_FIELD, RequestFields, relay

patch = relay(
    RequestFields(
        reasoning_effort=request.reasoning_effort,
        chat_template_kwargs=request.chat_template_kwargs,
        server_template_defaults=server_default_chat_template_kwargs,
        engine_args=request.vllm_xargs,
        response_format=response_format_as_dict,
        parallel_tool_calls=parallel_tool_calls_if_sent_else_none,
        has_tools=bool(request.tools),
        messages=messages_as_dicts,
    ),
    effort_order=EFFORT_ORDER["vllm"],
)
# patch.template_kwargs → chat_template_kwargs 의 빈 자리에
# patch.engine_args     → LP_CHANNEL_FIELD["vllm"] (= vllm_xargs) 의 빈 자리에
# patch.reasoning_moves → 해당 메시지의 content / reasoning 에
```

## 테스트

```bash
uv venv && uv pip install pytest -e .
python -m pytest -q
```

LP 키 이름 계약 테스트는 vLLM 포크가 설치된 환경에서만 돌고, 나머지는 건너뜁니다.
