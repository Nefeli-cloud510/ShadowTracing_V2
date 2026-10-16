from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from threading import Lock
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from core.runtime_config import (
    get_dasyscope_api_key,
    get_dasyscope_base_url,
    get_default_llm_model,
    load_project_env,
)


SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


class GatewayStatistics:
    """Aggregate per-role latency / token / retry / cache statistics."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.calls: list[dict[str, Any]] = []

    def record(self, entry: dict[str, Any]) -> None:
        with self.lock:
            self.calls.append(entry)

    def summary(self) -> dict[str, Any]:
        with self.lock:
            entries = list(self.calls)
        if not entries:
            return {"calls": 0, "total_seconds": 0.0, "by_role": {}}
        total_seconds = sum(float(item.get("elapsed_seconds") or 0.0) for item in entries)
        by_role: dict[str, dict[str, Any]] = {}
        for entry in entries:
            role = str(entry.get("role") or entry.get("model") or "unknown")
            role_stats = by_role.setdefault(
                role,
                {
                    "calls": 0,
                    "cache_hits": 0,
                    "retry_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_seconds": 0.0,
                },
            )
            role_stats["calls"] += 1
            role_stats["cache_hits"] += int(entry.get("cache_hit") or 0)
            role_stats["retry_count"] += int(entry.get("retries") or 0)
            role_stats["input_tokens"] += int(entry.get("input_tokens") or 0)
            role_stats["output_tokens"] += int(entry.get("output_tokens") or 0)
            role_stats["elapsed_seconds"] += float(entry.get("elapsed_seconds") or 0.0)
        return {
            "calls": len(entries),
            "total_seconds": round(total_seconds, 3),
            "by_role": by_role,
        }


class LLMGateway:
    """Unified structured-output gateway for planner-side LLM roles."""

    _statistics = GatewayStatistics()
    _cache: dict[str, str] = {}
    _cache_lock = Lock()

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        client: Any | None = None,
        allow_fallback: bool = True,
        use_cache: bool = True,
        max_retries: int = 1,
        deadline_seconds: float | None = None,
        statistics_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        load_project_env()
        self.model = model or get_default_llm_model()
        self.api_key = api_key or get_dasyscope_api_key()
        self.base_url = base_url or get_dasyscope_base_url()
        self.temperature = temperature
        self.allow_fallback = allow_fallback
        self.use_cache = use_cache
        self.max_retries = max(0, int(max_retries))
        self.deadline_seconds = deadline_seconds
        self.client = client or self._build_client()
        self._statistics_observer = statistics_observer

    def is_available(self) -> bool:
        return self.client is not None

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[SchemaModelT],
        fallback_factory: Callable[[], SchemaModelT | dict[str, Any]] | None = None,
        payload_fixer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        temperature: float | None = None,
        image_paths: list[str] | None = None,
        role: str | None = None,
        use_cache: bool = True,
        deadline_seconds: float | None = None,
    ) -> SchemaModelT:
        started = time.perf_counter()

        def record(
            *,
            ok: bool = True,
            cache_hit: bool = False,
            retries: int = 0,
            input_tokens: int | None = None,
            output_tokens: int | None = None,
            error: str | None = None,
            used_fallback: bool = False,
        ) -> None:
            entry = {
                "model": self.model,
                "role": role or self.model,
                "ok": bool(ok),
                "cache_hit": bool(cache_hit),
                "retries": int(retries),
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "error": error,
                "used_fallback": bool(used_fallback),
            }
            self._statistics.record(entry)
            if self._statistics_observer is not None:
                self._statistics_observer(entry)

        if self.client is None:
            if fallback_factory is None or not self.allow_fallback:
                record(ok=False, error="client_unavailable")
                raise RuntimeError("LLM client is unavailable for the real workflow.")
            result = self._coerce_response(response_model, fallback_factory())
            record(ok=False, error="client_unavailable", used_fallback=True)
            return result

        try:
            user_content: Any = user_prompt
            image_blob = ""
            if image_paths:
                content_parts: list[dict[str, Any]] = [
                    {"type": "text", "text": user_prompt},
                ]
                for image_path in image_paths:
                    encoded, media_type = _encode_image(image_path)
                    image_blob += f"{media_type}:{encoded[:64]};"
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded}",
                            },
                        }
                    )
                user_content = content_parts
            effective_temperature = self.temperature if temperature is None else temperature
            cache_key = _cache_key(
                model=self.model,
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=effective_temperature,
                image_blob=image_blob,
            )
            use_cache = bool(use_cache and self.use_cache)
            if use_cache:
                cached = self._cache_get(cache_key)
                if cached is not None:
                    payload = _extract_json_payload(cached)
                    if payload_fixer is not None:
                        payload = payload_fixer(payload)
                    result = response_model.model_validate(payload)
                    record(cache_hit=True)
                    return result

            deadline = deadline_seconds if deadline_seconds is not None else self.deadline_seconds
            retries = 0
            while True:
                if deadline is not None and (time.perf_counter() - started) >= deadline:
                    record(
                        ok=False,
                        retries=retries,
                        error=f"timeout_after_{deadline:.0f}s",
                    )
                    raise TimeoutError(
                        f"LLM 调用超过统一截止时间 {deadline:.0f}s：model={self.model}"
                    )
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=effective_temperature,
                    )
                    content = response.choices[0].message.content or ""
                    payload = _extract_json_payload(content)
                    if payload_fixer is not None:
                        payload = payload_fixer(payload)
                    result = response_model.model_validate(payload)
                    usage = getattr(response, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", None)
                    output_tokens = getattr(usage, "completion_tokens", None)
                    record(
                        retries=retries,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    if use_cache:
                        self._cache_set(cache_key, content)
                    return result
                except (TimeoutError, ZeroDivisionError):
                    raise
                except Exception:
                    if retries >= self.max_retries:
                        raise
                    retries += 1
                    time.sleep(min(1.5 * retries, 5.0))
        except Exception:
            if fallback_factory is None or not self.allow_fallback:
                record(ok=False, error="llm_call_failed")
                raise
            try:
                fallback = self._coerce_response(response_model, fallback_factory())
            except Exception:
                record(ok=False, error="fallback_factory_failed")
                raise
            record(ok=False, error="llm_call_failed", used_fallback=True)
            return fallback

    def _cache_get(self, key: str) -> str | None:
        with self._cache_lock:
            return self._cache.get(key)

    def _cache_set(self, key: str, content: str) -> None:
        with self._cache_lock:
            self._cache[key] = content

    @classmethod
    def statistics_summary(cls) -> dict[str, Any]:
        return cls._statistics.summary()

    @classmethod
    def reset_statistics(cls) -> None:
        with cls._statistics.lock:
            cls._statistics.calls.clear()

    @classmethod
    def clear_cache(cls) -> None:
        with cls._cache_lock:
            cls._cache.clear()

    def _build_client(self) -> Any | None:
        if not self.api_key:
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=300,
            max_retries=1,
        )

    @staticmethod
    def _coerce_response(
        response_model: type[SchemaModelT],
        payload: SchemaModelT | dict[str, Any],
    ) -> SchemaModelT:
        if isinstance(payload, response_model):
            return payload
        return response_model.model_validate(payload)


def _encode_image(path: str) -> tuple[str, str]:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Report chart not found for LLM analysis: {path}")
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(image_path.suffix.lower(), "image/png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return encoded, media_type


def _cache_key(
    *,
    model: str,
    system_prompt: str,
    user_content: Any,
    temperature: float,
    image_blob: str,
) -> str:
    payload = {
        "model": model,
        "system_prompt": system_prompt,
        "user_content": user_content,
        "temperature": temperature,
        "image_blob": image_blob,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_json_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{[\s\S]*\}", stripped)
    if match is None:
        raise ValueError("LLM response does not contain a JSON object.")
    return json.loads(match.group(0))
