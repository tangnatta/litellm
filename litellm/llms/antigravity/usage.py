from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from .authenticator import (
    BOOTSTRAP_URL,
    DISCOVERY_URLS,
    RUNTIME_URL,
    Authenticator,
    bootstrap_metadata,
    content_headers,
)
from .models import is_discoverable_model

_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_RUNTIME_BASES: Final = (RUNTIME_URL, BOOTSTRAP_URL)
_CACHE_TTL_SECONDS: Final = 60
_CACHE_LOCK: Final = threading.RLock()


class Quota(BaseModel):
    id: str
    used: int
    total: int
    remaining: int
    remaining_percentage: float
    reset_at: str | None
    unlimited: bool
    source: str
    display_name: str | None = None

    model_config = ConfigDict(frozen=True)


class UsageResult(BaseModel):
    plan: str
    project_id: str
    quotas: tuple[Quota, ...]
    checked_at: str

    model_config = ConfigDict(frozen=True)


_usage_cache: tuple[str, float, UsageResult] | None = None  # rebind-ok: refreshed TTL cache entry


def _number(value: JsonValue, default: float = -1) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _reset_time(value: JsonValue) -> str | None:
    if isinstance(value, str) and value.strip():
        stripped: Final = value.strip()
        numeric: Final = _number(stripped)
        if numeric < 0:
            return stripped
        return (
            datetime.fromtimestamp(numeric / 1000 if numeric > 10_000_000_000 else numeric, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    seconds: Final = float(value) / 1000 if value > 10_000_000_000 else float(value)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _quota(
    identifier: str, data: Mapping[str, JsonValue], source: str, display_name: str | None = None
) -> Quota | None:
    fraction: Final = _number(data.get("remainingFraction"))
    if fraction < 0:
        return None
    remaining_fraction: Final = min(1.0, max(0.0, fraction))
    reset_at: Final = _reset_time(data.get("resetTime"))
    unlimited: Final = reset_at is None and remaining_fraction >= 1
    total: Final = 0 if unlimited else 1000
    remaining: Final = 0 if unlimited else round(total * remaining_fraction)
    return Quota(
        id=identifier,
        used=0 if unlimited else total - remaining,
        total=total,
        remaining=remaining,
        remaining_percentage=100.0 if unlimited else remaining_fraction * 100,
        reset_at=reset_at,
        unlimited=unlimited,
        source=source,
        display_name=display_name,
    )


def _post_first(
    client: httpx.Client, urls: tuple[str, ...], headers: Mapping[str, str], body: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue] | None:
    for url in urls:
        try:
            response = client.post(  # rebind-ok: each endpoint produces a new response
                url,
                headers=headers,
                json=dict(body),  # mutable-ok: httpx JSON boundary requires a dictionary
            )
            if response.is_success:
                return _OBJECT.validate_python(response.json())
            if response.status_code in (401, 403):
                return None
        except (httpx.HTTPError, ValueError):
            continue
    return None


def _tier_value(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(str(value.get(key, "")) for key in ("id", "name", "displayName"))
    return ""


def _plan(subscription: Mapping[str, JsonValue]) -> str:
    text: Final = " ".join(
        _tier_value(subscription.get(key)) for key in ("currentTier", "paidTier", "subscriptionTier")
    ).upper()
    for marker, label in (
        ("ULTRA", "Ultra"),
        ("PRO", "Pro"),
        ("PREMIUM", "Pro"),
        ("GOOGLE_ONE", "Pro"),
        ("ENTERPRISE", "Enterprise"),
        ("BUSINESS", "Business"),
        ("STANDARD", "Business"),
        ("PLUS", "Plus"),
        ("LITE", "Lite"),
    ):
        if marker in text:
            return label
    return "Free"


def _catalog_quotas(data: Mapping[str, JsonValue] | None) -> tuple[Quota, ...]:
    models: Final = data.get("models") if data else None
    if not isinstance(models, dict):
        return ()
    results: tuple[Quota, ...] = ()  # rebind-ok: functional result accumulator
    for model_id, information in models.items():
        if not is_discoverable_model(model_id, information) or not isinstance(information, dict):
            continue
        quota_information = information.get("quotaInfo")  # rebind-ok: loop-local catalog quota
        if not isinstance(quota_information, dict):
            continue
        quota = _quota(  # rebind-ok: loop-local normalized catalog quota
            model_id, _OBJECT.validate_python(quota_information), "fetchAvailableModels"
        )
        if quota is not None:
            results = (*results, quota)
    return results


def _live_quotas(data: Mapping[str, JsonValue] | None) -> tuple[Quota, ...]:
    buckets: Final = data.get("buckets") if data else None
    if not isinstance(buckets, list):
        return ()
    return tuple(
        quota
        for bucket in buckets
        if isinstance(bucket, dict)
        and isinstance(model_id := bucket.get("modelId"), str)
        and is_discoverable_model(model_id, None)
        and (quota := _quota(model_id, bucket, "retrieveUserQuota")) is not None
    )


def _weekly_quotas(data: Mapping[str, JsonValue] | None) -> tuple[Quota, ...]:
    nested: Final = data.get("quotaSummary") if data else None
    groups: Final = data.get("groups") if data else nested.get("groups") if isinstance(nested, dict) else None
    if not isinstance(groups, list):
        return ()
    results: tuple[Quota, ...] = ()  # rebind-ok: functional result accumulator
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_object = _OBJECT.validate_python(group)  # rebind-ok: loop-local weekly group
        buckets = group_object.get("buckets")  # rebind-ok: loop-local weekly buckets
        if not isinstance(buckets, list):
            continue
        display = str(group_object.get("displayName", "")).strip()  # rebind-ok: loop-local group label
        bucket = next(  # rebind-ok: loop-local weekly bucket
            (
                item
                for item in buckets
                if isinstance(item, dict)
                and "weekly" in f"{item.get('bucketId', '')} {item.get('displayName', '')}".lower()
                and item.get("disabled") is not True
            ),
            None,
        )
        identifier = re.sub(  # rebind-ok: loop-local quota id
            r"[^a-z0-9]+", "_", re.sub(r"\bmodels?\b|\band\b", " ", display.lower())
        ).strip("_")
        quota = (  # rebind-ok: loop-local normalized weekly quota
            _quota(
                f"{identifier}_weekly",
                _OBJECT.validate_python(bucket),
                "retrieveUserQuotaSummary",
                display,
            )
            if identifier and isinstance(bucket, dict)
            else None
        )
        if quota:
            results = (*results, quota)
    return results


def get_usage(authenticator: Authenticator, force_refresh: bool = False, account: str = "default") -> UsageResult:
    global _usage_cache
    credentials: Final = authenticator.credentials(account=account)
    cache_key: Final = hashlib.sha256(
        f"{account}:{credentials.access_token}:{credentials.project_id}".encode()
    ).hexdigest()
    with _CACHE_LOCK:
        if (
            not force_refresh
            and _usage_cache
            and _usage_cache[0] == cache_key
            and time.time() - _usage_cache[1] < _CACHE_TTL_SECONDS
        ):
            return _usage_cache[2]
        headers: Final = content_headers(credentials.access_token)
        project_body: Final = MappingProxyType({"project": credentials.project_id})
        subscription: Final = _post_first(
            authenticator.client,
            (f"{BOOTSTRAP_URL}/v1internal:loadCodeAssist",),
            headers,
            MappingProxyType(
                {"metadata": dict(bootstrap_metadata())}  # mutable-ok: nested JSON object sent through httpx
            ),
        ) or MappingProxyType({})
        catalog: Final = _post_first(authenticator.client, DISCOVERY_URLS[:3], headers, project_body)
        quota_data: Final = _post_first(
            authenticator.client,
            tuple(f"{base}/v1internal:retrieveUserQuota" for base in _RUNTIME_BASES),
            headers,
            project_body,
        )
        summary: Final = _post_first(
            authenticator.client,
            tuple(f"{base}/v1internal:retrieveUserQuotaSummary" for base in _RUNTIME_BASES),
            headers,
            project_body,
        )
        merged: Final[dict[str, Quota]] = {  # mutable-ok: keyed merge lets live quota override catalog data
            quota.id: quota for quota in _catalog_quotas(catalog)
        }
        merged.update(
            {quota.id: quota for quota in _live_quotas(quota_data)}  # mutable-ok: live values override catalog
        )
        merged.update(
            {quota.id: quota for quota in _weekly_quotas(summary)}  # mutable-ok: add family-level limits
        )
        result: Final = UsageResult(
            plan=_plan(subscription),
            project_id=credentials.project_id,
            quotas=tuple(sorted(merged.values(), key=lambda quota: quota.id)),
            checked_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        _usage_cache = (cache_key, time.time(), result)  # rebind-ok: refreshed TTL cache entry
        return result
