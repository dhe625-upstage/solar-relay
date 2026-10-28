# solar-relay

요청에 이미 있는 값을 Solar 채팅 템플릿과 logits processor(LP) 통로로 옮기는 패키지입니다.
템플릿을 렌더하는 프로세스(vLLM API 서버, Dynamo 프론트엔드)에서 렌더 직전에 요청마다 한 번 실행됩니다. SGLang host는 아직 없습니다.

**원칙: 값을 만들지 않습니다.** 요청이나 운영자 설정이 준 값만 옮깁니다. 값이 없으면 엔진·템플릿 기본값이 그대로 쓰입니다.

## 규칙

| 규칙 | 입력 | 옮기는 곳 |
|---|---|---|
| R1 effort | 템플릿이 실제로 받을 `reasoning_effort` (최상위 값·요청 kwargs·운영자 `--default-chat-template-kwargs` 를 렌더러가 합친 결과. host 가 렌더러에게 묻는다) | LP 통로 `reasoning_effort` |
| R2 response_format | `response_format` (json_object, json_schema) | `chat_template_kwargs.response_format` |
| R3 툴콜 1개 | 요청이 보낸 `parallel_tool_calls: false` (`true`는 LP 기본값과 같아서 옮기지 않음) | LP 통로 `parallel_tool_calls: 0` |
| think 이력 | assistant content 앞의 `<think>…</think>` | 그 메시지의 `reasoning` |

요청의 같은 자리에 값이 이미 있으면 덮지 않습니다. 두 번 적용해도 결과가 같습니다.

규칙은 모두 요청 값을 제자리로 옮기는 것이라 따로 끄지 않습니다. 패키지 전체만 `SOLAR_RELAY=0`으로 끕니다. 이미지를 다시 만들지 않고 되돌리는 비상용입니다.

## 구성

| 파일 | 내용 |
|---|---|
| `relay.py` | 규칙. `relay()`는 요청을 고치지 않고 변경분(`Patch`)만 돌려준다. 엔진을 import 하지 않는다 |
| `hosts/vllm.py` | vLLM 플러그인과 vLLM 요청 객체 변환(`relay_chat`, `relay_responses`) |
| `hosts/dynamo.py` | Dynamo 앞단에서 부르는 함수. Dynamo도 vLLM의 `ChatCompletionRequest`를 쓰므로 `relay_chat`을 그대로 쓴다 |

| host | 붙는 법 | 덮는 경로 |
|---|---|---|
| vLLM | 패키지를 설치하면 `vllm.general_plugins` entry point로 자동 로드(플러그인). API 서버 프로세스에서만 `OpenAIServingChat.create_chat_completion`과 `OpenAIServingResponses.create_responses`를 감싼다. 템플릿이 받을 kwargs는 vLLM의 `_effective_chat_template_kwargs(request)`에 묻는다 | `/v1/chat/completions`, `/v1/messages`, `/v1/responses` |
| Dynamo | 플러그인이 아니라 함수 호출. vLLM chat processor의 `_prepare_request`(`dynamo/frontend/prepost.py`)에서 템플릿 kwargs가 확정된 직후, 렌더용 인자를 만들기 전에 `solar_relay.hosts.dynamo.relay_chat_request(request_for_sampling, chat_template_kwargs)` 호출 (solar-dynamo 패치 한 줄) | chat (Rust가 messages·responses를 chat으로 바꾼 뒤) |

지원하지 않는 경로: vLLM 전용 배치 엔드포인트 `/v1/chat/completions/batch`. `create_chat_completion`을 거치지 않아 훅이 닿지 않습니다. OpenAI 표준이 아니라 whale·고객 클라이언트가 쓰지 않아서 붙이지 않았습니다.

vLLM host는 기동할 때 훅 대상(메서드 이름·시그니처, `_effective_chat_template_kwargs`, 요청 필드)을 점검하고, 맞지 않으면 기동을 멈춥니다. 요청 처리 중 오류는 그 요청만 그대로 통과시키고 ERROR 로그를 남깁니다. 적용 내역은 DEBUG 로그로만 남깁니다.

## 테스트

```bash
uv venv && uv pip install pytest -e .
python -m pytest -q
```

실제 vLLM 클래스를 쓰는 테스트(LP 키 이름 계약, effort 해석)는 vLLM 포크가 설치된 환경에서만 돌고, 나머지 환경에서는 건너뜁니다.
