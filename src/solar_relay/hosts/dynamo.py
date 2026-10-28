"""Dynamo host: vLLM chat processor 의 `_prepare_request`(dynamo/frontend/prepost.py)에서 부르는 함수.

Dynamo 에는 요청 훅 장치가 없어서 solar-dynamo 패치로 호출 한 줄을 넣는다. 자리는 템플릿 kwargs 가 확정된 직후,
메시지와 kwargs 를 렌더용으로 넘기기 전이다. 그래서 Dynamo 의 병합 규칙을 흉내 내지 않고 확정된 kwargs 를 읽는다.
요청은 vLLM 의 ChatCompletionRequest 라 vLLM host 의 변환을 그대로 쓴다.
"""

from __future__ import annotations

import logging
from typing import Any

from solar_relay.hosts.vllm import relay_chat
from solar_relay.relay import Patch, relay_enabled

logger = logging.getLogger("dynamo.solar_relay")


def relay_chat_request(request: Any, chat_template_kwargs: dict[str, Any]) -> Patch:
    """request 와 확정된 chat_template_kwargs 를 제자리에서 고친다. 오류가 나도 요청을 막지 않는다."""
    if not relay_enabled():
        return Patch()
    try:
        patch = relay_chat(request, chat_template_kwargs, template_target=chat_template_kwargs)
    except Exception:
        logger.exception("solar-relay: request passed through unchanged after an error")
        return Patch()
    if patch:
        logger.debug("solar-relay applied=%s", patch.applied)
    return patch
