"""요청에 이미 있는 값을 Solar 템플릿과 logits processor 통로로 옮긴다.

- relay.py: 무엇을 어디로 옮길지 정하는 규칙 (엔진 import 없음)
- hosts/vllm.py: vLLM API 서버에 붙이는 플러그인, vLLM 요청 객체 변환
- hosts/dynamo.py: Dynamo 앞단에서 부르는 함수
"""
