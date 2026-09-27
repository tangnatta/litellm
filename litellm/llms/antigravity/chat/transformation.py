from __future__ import annotations

import json
import secrets
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared transport factory has legacy dictionary annotations
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared transport factory has legacy dictionary annotations
)
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.vertex_ai.gemini.transformation import (
    _transform_request_body,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # reuse the shared Gemini transformation; validate its result below
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import ModelResponseIterator
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import LlmProviders, ModelResponse

from ..authenticator import RUNTIME_URL, AntigravityError, Authenticator, content_headers, get_authenticator
from ..models import output_token_cap, resolve_model_id

_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class CodeAssistRequest(TypedDict):
    project: ReadOnly[str]
    model: ReadOnly[str]
    requestId: ReadOnly[str]
    request: ReadOnly[JsonValue]
    userAgent: ReadOnly[str]
    requestType: ReadOnly[str]


def _part_is_valid(part: JsonValue) -> bool:
    if not isinstance(part, dict):
        return False
    if part.get("text") == "":
        return False
    function_call: Final = part.get("functionCall", part.get("function_call"))
    return not (isinstance(function_call, dict) and not function_call.get("name"))


def _normalize_contents(contents: JsonValue, model: str) -> tuple[Mapping[str, JsonValue], ...]:
    if not isinstance(contents, list):
        return ()
    normalized: Final[list[Mapping[str, JsonValue]]] = []  # mutable-ok: ordered turn merge accumulator
    for content in contents:
        if not isinstance(content, dict):
            continue
        parts_value = content.get("parts")  # rebind-ok: loop-local value
        if not isinstance(parts_value, list):
            continue
        parts: tuple[JsonValue, ...] = tuple(  # rebind-ok: loop-local value
            part for part in parts_value if _part_is_valid(part)
        )
        if not parts:
            continue
        has_function_response = any(  # rebind-ok: loop-local classification
            isinstance(part, dict) and ("functionResponse" in part or "function_response" in part) for part in parts
        )
        role = "user" if has_function_response else str(content.get("role", "user"))  # rebind-ok: loop-local role
        entry: Mapping[str, JsonValue] = _OBJECT.validate_python(  # rebind-ok: loop-local normalized content
            {  # mutable-ok: validated immediately as provider JSON
                **content,
                "role": role,
                "parts": list(parts),  # mutable-ok: JSON arrays require a list at the provider boundary
            }  # mutable-ok: validated immediately as provider JSON
        )
        if normalized and normalized[-1].get("role") == role:
            previous: Mapping[str, JsonValue] = normalized[-1]  # rebind-ok: loop-local preceding content
            previous_parts: JsonValue = previous.get("parts")  # rebind-ok: loop-local preceding parts
            combined: tuple[JsonValue, ...] = (  # rebind-ok: loop-local merged parts
                (*previous_parts, *parts) if isinstance(previous_parts, list) else parts
            )
            merged: Mapping[str, JsonValue] = _OBJECT.validate_python(  # rebind-ok: loop-local merged content
                {  # mutable-ok: validated immediately as provider JSON
                    **previous,
                    "parts": list(combined),  # mutable-ok: JSON arrays require a list at the provider boundary
                }  # mutable-ok: validated immediately as provider JSON
            )
            normalized[-1] = merged
        else:
            normalized.append(entry)
    lower_model: Final = model.lower()
    strips_assistant: Final = "claude" in lower_model or (
        (lower_model.startswith("gemini-3") or lower_model == "gemini-pro-agent") and "image" not in lower_model
    )
    while strips_assistant and len(normalized) > 1 and normalized[-1].get("role") == "model":
        normalized.pop()
    return tuple(normalized)


def _normalize_generation_config(
    value: JsonValue, model: str
) -> dict[str, JsonValue]:  # mutable-ok: JSON dictionary is required by the provider transport
    source: Final[Mapping[str, JsonValue]] = (
        value
        if isinstance(value, dict)
        else MappingProxyType[str, JsonValue]({})  # mutable-ok: empty literal is frozen before use
    )
    thinking: Final = source.get("thinkingConfig", source.get("thinking_config"))
    allows_thinking: Final = "claude" not in model.lower() and not model.startswith("gpt-oss")
    canonical: Final = MappingProxyType(
        {
            key: item
            for key, item in source.items()
            if key not in ("top_k", "top_p", "max_output_tokens", "thinking_config")
        }
    )
    without_thinking: Final = (
        MappingProxyType({key: item for key, item in canonical.items() if key != "thinkingConfig"})
        if not allows_thinking
        else canonical
    )
    cap: Final = output_token_cap(model)
    requested_max: Final = source.get("maxOutputTokens", source.get("max_output_tokens"))
    thinking_budget: Final = thinking.get("thinkingBudget") if isinstance(thinking, dict) else None
    minimum_for_thinking: Final = thinking_budget + 1 if isinstance(thinking_budget, int) and thinking_budget > 0 else 0
    bounded_max: Final = (
        min(max(requested_max if isinstance(requested_max, int) else 0, minimum_for_thinking), cap)
        if isinstance(requested_max, int) or minimum_for_thinking > 0
        else None
    )
    return _OBJECT.validate_python(
        {  # mutable-ok: validated immediately as provider JSON
            **without_thinking,
            "topK": source.get("topK", source.get("top_k", 40)),
            "topP": source.get("topP", source.get("top_p", 1.0)),
            **(
                {"thinkingConfig": thinking}  # mutable-ok: conditional provider JSON field
                if allows_thinking and isinstance(thinking, dict)
                else {}  # mutable-ok: conditional provider JSON field
            ),
            **(
                {"maxOutputTokens": bounded_max}  # mutable-ok: conditional provider JSON field
                if bounded_max is not None
                else {}  # mutable-ok: conditional provider JSON field
            ),
        }
    )


def _normalize_request(
    request: Mapping[str, JsonValue], model: str
) -> dict[str, JsonValue]:  # mutable-ok: JSON dictionary is required by the provider transport
    contents: Final = _normalize_contents(request.get("contents"), model)
    safety: Final = request.get("safetySettings")
    safe_settings: Final = (
        tuple(
            setting
            for setting in safety
            if not (isinstance(setting, dict) and setting.get("category") == "HARM_CATEGORY_CIVIC_INTEGRITY")
        )
        if isinstance(safety, list)
        else None
    )
    tools: Final = request.get("tools")
    tool_config: Final = (
        {"functionCallingConfig": {"mode": "VALIDATED"}}  # mutable-ok: validated in final provider JSON
        if isinstance(tools, list) and tools
        else request.get("toolConfig")
    )
    return _OBJECT.validate_python(
        {  # mutable-ok: validated immediately as provider JSON
            **request,
            **(
                {"contents": list(contents)}  # mutable-ok: JSON arrays require a list at the provider boundary
                if contents
                else {}  # mutable-ok: conditional provider JSON field
            ),
            "generationConfig": _normalize_generation_config(request.get("generationConfig"), model),
            **(
                {"safetySettings": list(safe_settings)}  # mutable-ok: JSON array at provider boundary
                if safe_settings is not None
                else {}  # mutable-ok: conditional provider JSON field
            ),
            **(
                {"toolConfig": tool_config}  # mutable-ok: conditional provider JSON field
                if tool_config is not None
                else {}  # mutable-ok: conditional provider JSON field
            ),
        }
    )


def unwrap_event(event: str) -> str:
    try:
        envelope: Final = _OBJECT.validate_json(event)
    except ValueError:
        raise AntigravityError(status_code=502, message="Antigravity returned an invalid streaming event") from None
    payload: Final = envelope.get("response", envelope)
    if not isinstance(payload, dict):
        raise AntigravityError(status_code=502, message="Antigravity returned an invalid response envelope")
    return json.dumps(payload)


class EventDecoder:
    def __init__(self) -> None:
        self.parts: tuple[str, ...] = ()
        self.finished = False

    def feed(self, line: str) -> str | None:
        if self.finished:
            return None
        if line.startswith("data:"):
            self.parts = (*self.parts, line[5:].lstrip())
            return None
        return self.flush() if not line else None

    def flush(self) -> str | None:
        if not self.parts:
            return None
        event: Final = "\n".join(self.parts)
        self.parts = ()
        if event == "[DONE]":
            self.finished = True
            return None
        return unwrap_event(event)


def unwrap_lines(lines: Iterator[str]) -> Iterator[str]:
    decoder: Final = EventDecoder()
    for line in lines:
        if (event := decoder.feed(line)) is not None:
            yield event
        if decoder.finished:
            return
    if (last := decoder.flush()) is not None:
        yield last


async def unwrap_async_lines(response: httpx.Response) -> AsyncIterator[str]:
    decoder: Final = EventDecoder()
    try:
        async for line in response.aiter_lines():
            if (event := decoder.feed(line)) is not None:
                yield event
            if decoder.finished:
                return
        if (last := decoder.flush()) is not None:
            yield last
    finally:
        await response.aclose()


def unwrap_response(response: httpx.Response) -> Iterator[str]:
    try:
        yield from unwrap_lines(response.iter_lines())
    finally:
        response.close()


class AntigravityConfig(BaseConfig):
    def __init__(self, authenticator: Authenticator | None = None) -> None:
        self.authenticator = authenticator or get_authenticator()
        self.gemini = GoogleAIStudioGeminiConfig()

    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: BaseConfig requires a list return value
        return self.gemini.get_supported_openai_params(model)

    def map_openai_params(
        self,
        non_default_params: Mapping[str, JsonValue],
        optional_params: Mapping[str, JsonValue],
        model: str,
        drop_params: bool,
    ) -> dict[str, JsonValue]:  # mutable-ok: BaseConfig requires a JSON dictionary return value
        return _OBJECT.validate_python(
            self.gemini.map_openai_params(  # pyright: ignore[reportUnknownMemberType]  # legacy Gemini output is validated into JSON below
                MappingProxyType(non_default_params).copy(),
                MappingProxyType(optional_params).copy(),
                model,
                drop_params,
            )
        )

    def validate_environment(
        self,
        headers: Mapping[str, str] | None,
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseConfig requires a headers dictionary return value
        account: Final = str(litellm_params.get("antigravity_account", "default"))
        credentials: Final = self.authenticator.credentials(account=account)
        return MappingProxyType(
            {
                **(headers or MappingProxyType({})),
                **content_headers(credentials.access_token),
                "Accept": "text/event-stream",
            }
        ).copy()

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        stream: bool | None = None,
    ) -> str:
        return f"{(api_base or RUNTIME_URL).rstrip('/')}/v1internal:streamGenerateContent?alt=sse"

    def transform_request(
        self,
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        headers: Mapping[str, str],
    ) -> dict[str, JsonValue]:  # mutable-ok: BaseConfig requires a JSON dictionary return value
        account = str(litellm_params.get("antigravity_account", "default"))
        credentials: Final = self.authenticator.credentials(account=account)
        upstream_model: Final = resolve_model_id(model)
        request: Final = _OBJECT.validate_python(
            _transform_request_body(
                messages=deepcopy(
                    list(messages)  # mutable-ok: Gemini conversion requires a private mutable list
                ),
                model=upstream_model,
                optional_params=MappingProxyType(
                    {key: value for key, value in optional_params.items() if key != "stream"}
                ).copy(),
                custom_llm_provider="gemini",
                litellm_params=MappingProxyType(litellm_params).copy(),
                cached_content=None,
            )
        )
        normalized_keys: Final = MappingProxyType(
            {("systemInstruction" if key == "system_instruction" else key): value for key, value in request.items()}
        )
        normalized: Final = _normalize_request(normalized_keys, upstream_model)
        request_with_session: Final = MappingProxyType(
            {**normalized, "sessionId": f"-{secrets.randbelow(9_000_000_000_000_000_000)}"}
        ).copy()
        envelope: Final[CodeAssistRequest] = {
            "project": credentials.project_id,
            "model": upstream_model,
            "requestId": f"agent/{time.time_ns() // 1_000_000}/{secrets.token_hex(4)}",
            "request": request_with_session,
            "userAgent": "antigravity",
            "requestType": "agent",
        }
        return _OBJECT.validate_python(envelope)

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        return False

    @property
    def has_custom_stream_wrapper(self) -> bool:
        return True

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Logging,
        request_data: Mapping[str, JsonValue],
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        iterator: Final = ModelResponseIterator(unwrap_lines(raw_response.iter_lines()), True, logging_obj)
        chunks: Final = tuple(chunk for chunk in iterator if chunk is not None)
        result: Final = litellm.stream_chunk_builder(  # pyright: ignore[reportUnknownMemberType]  # shared assembler has legacy list annotations
            list(chunks),  # mutable-ok: stream_chunk_builder requires a list
            messages=list(messages),  # mutable-ok: stream_chunk_builder requires lists
        )
        if not isinstance(result, ModelResponse):
            raise AntigravityError(status_code=502, message="Antigravity returned an empty response")
        result.model = model
        logging_obj.post_call(  # pyright: ignore[reportUnknownMemberType]  # shared logging API accepts untyped payloads
            input=messages,
            api_key="",
            original_response=raw_response.text,
            additional_args=MappingProxyType({"complete_input_dict": request_data}),
        )
        return result

    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: Logging,
        api_base: str,
        headers: Mapping[str, str],
        data: Mapping[str, JsonValue],
        messages: Sequence[AllMessageValues],
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        json_mode: bool | None = None,
        signed_json_body: bytes | None = None,
    ) -> CustomStreamWrapper:
        transport: Final = client if isinstance(client, HTTPHandler) else _get_httpx_client()
        response: Final = transport.post(  # pyright: ignore[reportUnknownMemberType]  # shared HTTP handler has legacy dictionary annotations
            api_base,
            headers=MappingProxyType(headers).copy(),
            data=_OBJECT.dump_json(MappingProxyType(data).copy()),
            stream=True,
            logging_obj=logging_obj,
        )
        iterator: Final = ModelResponseIterator(unwrap_response(response), True, logging_obj, response=response)
        return CustomStreamWrapper(
            completion_stream=iter(iterator),
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )

    async def get_async_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: Logging,
        api_base: str,
        headers: Mapping[str, str],
        data: Mapping[str, JsonValue],
        messages: Sequence[AllMessageValues],
        client: AsyncHTTPHandler | None = None,
        json_mode: bool | None = None,
        signed_json_body: bytes | None = None,
    ) -> CustomStreamWrapper:
        transport: Final = client or get_async_httpx_client(llm_provider=LlmProviders.ANTIGRAVITY, params=None)
        response: Final = await transport.post(  # pyright: ignore[reportUnknownMemberType]  # shared HTTP handler has legacy dictionary annotations
            api_base,
            headers=MappingProxyType(headers).copy(),
            data=_OBJECT.dump_json(MappingProxyType(data).copy()),
            stream=True,
            logging_obj=logging_obj,
        )
        iterator: Final = ModelResponseIterator(unwrap_async_lines(response), False, logging_obj, response=response)
        return CustomStreamWrapper(
            completion_stream=iterator, model=model, custom_llm_provider=custom_llm_provider, logging_obj=logging_obj
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return AntigravityError(
            status_code=status_code,
            message=f"Antigravity request failed (HTTP {status_code}): {error_message}",
            headers=httpx.Headers(headers),
        )
