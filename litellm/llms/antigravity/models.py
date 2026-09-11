from __future__ import annotations

import re
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue

MODEL_ALIASES: Final = MappingProxyType(
    {
        "gemini-3.7-flash": "gemini-3.7-flash-tiered",
        "gemini-3.7-flash-high": "gemini-3.7-flash-tiered",
        "gemini-3.7-flash-medium": "gemini-3.7-flash-tiered",
        "gemini-3.7-flash-low": "gemini-3.7-flash-tiered",
        "gemini-3.1-pro-high": "gemini-pro-agent",
        "gpt-oss-120b": "gpt-oss-120b-medium",
        "gemini-3-pro-image-preview": "gemini-3-pro-image",
        "gemini-claude-sonnet-4-5": "claude-sonnet-4-6",
        "gemini-claude-sonnet-4-5-thinking": "claude-sonnet-4-6",
        "gemini-claude-opus-4-5-thinking": "claude-opus-4-6-thinking",
    }
)
PUBLIC_MODELS: Final = (
    "gemini-3.7-flash-high",
    "gemini-3.7-flash-medium",
    "gemini-3.7-flash-low",
    "gemini-3.7-flash-tiered",
    "gemini-pro-agent",
    "gemini-3.1-pro-low",
    "gemini-3.1-flash-lite",
    "claude-opus-4-6-thinking",
    "claude-sonnet-4-6",
    "gpt-oss-120b-medium",
)

_NON_CHAT_MODEL_IDS: Final = frozenset(
    {
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-tts-preview",
        "gemini-2.5-flash-preview-tts",
        "tab_flash_lite_preview",
        "tab_jump_flash_lite_preview",
    }
)
_RETIRED_MODEL_IDS: Final = frozenset(
    {
        "gemini-3-pro-preview",
        "gemini-3.1-pro",
        "gemini-3.6-flash-high",
        "gemini-3.6-flash-medium",
        "gemini-3.6-flash-low",
        "gemini-3-flash-agent",
        "gemini-3.5-flash",
        "gemini-3.5-flash-extra-low",
        "gemini-3.5-flash-low",
        "gemini-3.5-flash-high",
        "gemini-3.5-flash-medium",
        "gemini-3.5-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash-thinking",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-computer-use-preview-10-2025",
    }
)
_NON_CHAT_MODEL_PATTERN: Final = re.compile(
    r"(?:^|[-_])(image|imagen|audio|tts|embedding|embed|video|veo)(?:[-_]|$)", re.I
)
_INTERNAL_MODEL_PATTERN: Final = re.compile(r"^chat_\d+$", re.I)


def resolve_model_id(model_id: str) -> str:
    return MODEL_ALIASES.get(model_id, model_id)


def is_discoverable_model(model_id: str, information: JsonValue) -> bool:
    return (
        bool(model_id)
        and not (isinstance(information, dict) and information.get("isInternal") is True)
        and model_id not in _NON_CHAT_MODEL_IDS
        and model_id not in _RETIRED_MODEL_IDS
        and _INTERNAL_MODEL_PATTERN.fullmatch(model_id) is None
        and _NON_CHAT_MODEL_PATTERN.search(model_id) is None
    )


def output_token_cap(model_id: str) -> int:
    if "claude" in model_id.lower():
        return 16384
    if model_id == "gpt-oss-120b-medium":
        return 32768
    if is_discoverable_model(model_id, None):
        return 65535 if model_id.startswith("gemini") else 65536
    return 16384
