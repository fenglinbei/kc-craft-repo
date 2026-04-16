from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import CacheConfig, LLMConfig
from .utils import ensure_dir, stable_hash


@dataclass
class ChatResponse:
    content: str
    raw: Dict[str, Any]
    cached: bool = False


class OpenAICompatibleChatClient:
    def __init__(self, cfg: LLMConfig, cache_cfg: Optional[CacheConfig] = None, namespace: str = "default"):
        self.cfg = cfg
        self.cache_cfg = cache_cfg or CacheConfig(enabled=False)
        self.namespace = namespace
        self.session = requests.Session()
        self.cache_dir = ensure_dir(Path(self.cache_cfg.cache_dir) / namespace) if self.cache_cfg.enabled else None

    def _url(self) -> str:
        base = self.cfg.api_base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _cache_path(self, payload: Dict[str, Any]) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{stable_hash(payload)}.json"

    @staticmethod
    def _normalize_content(message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            parts: List[str] = []
            for item in message_content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return str(message_content)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }

        rf = response_format if response_format is not None else self.cfg.response_format
        if rf:
            payload["response_format"] = rf

        if self.cfg.extra_body:
            payload.update(self.cfg.extra_body)
        if extra_body:
            payload.update(extra_body)

        if self.cache_cfg.enabled and self.cache_dir is not None:
            cache_path = self._cache_path(payload)
            if cache_path.exists():
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                content = self._extract_content(raw)
                return ChatResponse(content=content, raw=raw, cached=True)

        headers = {
            "Content-Type": "application/json",
            **self.cfg.headers,
        }
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self.session.post(
                    self._url(),
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.timeout,
                )
                resp.raise_for_status()
                raw = resp.json()
                content = self._extract_content(raw)
                if self.cache_cfg.enabled and self.cache_dir is not None:
                    cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                return ChatResponse(content=content, raw=raw, cached=False)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_wait_seconds)
                else:
                    raise RuntimeError(
                        f"Chat completion failed after {self.cfg.max_retries} attempts: {exc}"
                    ) from exc

        raise RuntimeError(f"Unreachable state in chat(); last_error={last_error!r}")

    def _extract_content(self, raw: Dict[str, Any]) -> str:
        choices = raw.get("choices", [])
        if not choices:
            raise ValueError(f"No 'choices' in response: {raw}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        return self._normalize_content(content)
