from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import requests

from .config import CacheConfig, LLMConfig
from .utils import ensure_dir, stable_hash

LOGGER = logging.getLogger(__name__)
_LOCAL_VLLM_ENGINE_LOCK = Lock()
_LOCAL_VLLM_ENGINE_REGISTRY: dict[tuple[Any, ...], dict[str, Any]] = {}


def _normalize_backend(value: str) -> str:
    return value.strip().lower().replace("-", "_")


@dataclass
class ChatResponse:
    content: str
    raw: Dict[str, Any]
    cached: bool = False


class OpenAICompatibleChatClient:
    def __init__(
        self,
        cfg: LLMConfig,
        cache_cfg: Optional[CacheConfig] = None,
        namespace: str = "default",
        enable_messages_batch_api: bool = True,
        debug: bool = False,
        debug_preview_chars: int = 1200,
        debug_head_chars: int = 450,
        debug_tail_chars: int = 450,
    ):
        self.cfg = cfg
        self.cache_cfg = cache_cfg or CacheConfig(enabled=False)
        self.namespace = namespace
        self.enable_messages_batch_api = enable_messages_batch_api
        self.debug = debug
        self.debug_preview_chars = debug_preview_chars
        self.debug_head_chars = debug_head_chars
        self.debug_tail_chars = debug_tail_chars
        self.session = requests.Session()
        self.cache_dir = ensure_dir(Path(self.cache_cfg.cache_dir) / namespace) if self.cache_cfg.enabled else None
        self.backend = _normalize_backend(self.cfg.backend)
        self._local_llm: Any = None
        self._local_sampling_params_cls: Any = None
        self._local_tokenizer: Any = None
        if self.backend == "local_vllm":
            self._init_local_vllm()

    def _init_local_vllm(self) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "backend=local_vllm requires both 'vllm' and 'transformers' installed"
            ) from exc

        model_path = self.cfg.local_model_path or self.cfg.model
        if not model_path:
            raise ValueError("local_vllm backend requires local_model_path (or model) in config")

        engine_key = (
            model_path,
            max(1, int(self.cfg.local_tensor_parallel_size)),
            self.cfg.local_dtype,
            int(self.cfg.local_max_model_len),
            float(self.cfg.local_gpu_memory_utilization),
            int(self.cfg.local_max_num_batched_tokens),
            int(self.cfg.local_max_num_seqs),
            bool(self.cfg.local_async_scheduling),
            bool(self.cfg.local_trust_remote_code),
        )
        with _LOCAL_VLLM_ENGINE_LOCK:
            shared_engine = _LOCAL_VLLM_ENGINE_REGISTRY.get(engine_key)
            if shared_engine is None:
                shared_engine = {
                    "llm": LLM(
                        model=model_path,
                        tensor_parallel_size=max(1, int(self.cfg.local_tensor_parallel_size)),
                        dtype=self.cfg.local_dtype,
                        max_model_len=int(self.cfg.local_max_model_len),
                        gpu_memory_utilization=float(self.cfg.local_gpu_memory_utilization),
                        max_num_batched_tokens=int(self.cfg.local_max_num_batched_tokens),
                        max_num_seqs=int(self.cfg.local_max_num_seqs),
                        async_scheduling=bool(self.cfg.local_async_scheduling),
                        trust_remote_code=bool(self.cfg.local_trust_remote_code),
                    ),
                    "sampling_params_cls": SamplingParams,
                    "tokenizer": AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=bool(self.cfg.local_trust_remote_code),
                    ),
                }
                _LOCAL_VLLM_ENGINE_REGISTRY[engine_key] = shared_engine
                LOGGER.info("Initialized new shared local_vllm engine for namespace=%s", self.namespace)
            else:
                LOGGER.info("Reusing shared local_vllm engine for namespace=%s", self.namespace)

        self._local_llm = shared_engine["llm"]
        self._local_sampling_params_cls = shared_engine["sampling_params_cls"]
        self._local_tokenizer = shared_engine["tokenizer"]

    @staticmethod
    def _raw_from_content(content: str) -> Dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ]
        }

    def _build_local_sampling_params(self, extra_body: Optional[Dict[str, Any]] = None) -> Any:
        assert self._local_sampling_params_cls is not None
        params: Dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.extra_body:
            params.update(self.cfg.extra_body)
        if extra_body:
            params.update(extra_body)
        return self._local_sampling_params_cls(**params)

    def _messages_to_local_prompt(self, messages: List[Dict[str, Any]]) -> str:
        assert self._local_tokenizer is not None
        normalized_messages = []
        for message in messages:
            normalized_messages.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": self._normalize_content(message.get("content", "")),
                }
            )
        return self._local_tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _local_generate(self, prompts: List[str], extra_body: Optional[Dict[str, Any]] = None) -> List[str]:
        assert self._local_llm is not None
        sampling_params = self._build_local_sampling_params(extra_body=extra_body)
        outputs = self._local_llm.generate(prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text if output.outputs else "" for output in outputs]

    def _url(self) -> str:
        if self.backend != "openai_compatible":
            raise RuntimeError(f"_url() is only valid for openai_compatible backend, got {self.backend}")
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

    def _preview_text(self, text: str) -> str:
        if self.debug_preview_chars <= 0 or len(text) <= self.debug_preview_chars:
            return text
        head_chars = max(1, self.debug_head_chars)
        tail_chars = max(1, self.debug_tail_chars)
        if head_chars + tail_chars >= len(text):
            return text
        omitted = len(text) - head_chars - tail_chars
        return f"{text[:head_chars]}\n...<omitted {omitted} chars>...\n{text[-tail_chars:]}"

    def _debug_log_request(self, payload: Dict[str, Any]) -> None:
        if not self.debug:
            return
        messages = payload.get("messages", [])
        LOGGER.debug(
            "[debug][%s] request model=%s messages=%d max_tokens=%s temperature=%s",
            self.namespace,
            payload.get("model"),
            len(messages),
            payload.get("max_tokens"),
            payload.get("temperature"),
        )
        for idx, message in enumerate(messages, start=1):
            role = message.get("role", "unknown")
            normalized_content = self._normalize_content(message.get("content", ""))
            LOGGER.debug(
                "[debug][%s] request.message[%d] role=%s chars=%d content=\n%s",
                self.namespace,
                idx,
                role,
                len(normalized_content),
                self._preview_text(normalized_content),
            )

    def _debug_log_response(self, content: str, elapsed_seconds: float, cached: bool) -> None:
        if not self.debug:
            return
        LOGGER.debug(
            "[debug][%s] response cached=%s elapsed=%.3fs chars=%d content=\n%s",
            self.namespace,
            cached,
            elapsed_seconds,
            len(content),
            self._preview_text(content),
        )

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

        if self.backend == "local_vllm":
            if response_format is not None or self.cfg.response_format:
                LOGGER.warning("local_vllm backend ignores response_format; text output only")
            self._debug_log_request(payload)
            started = time.perf_counter()
            if self.cache_cfg.enabled and self.cache_dir is not None:
                cache_path = self._cache_path(payload)
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    content = self._extract_content(raw)
                    self._debug_log_response(content=content, elapsed_seconds=time.perf_counter() - started, cached=True)
                    return ChatResponse(content=content, raw=raw, cached=True)

            prompt = self._messages_to_local_prompt(messages)
            generated = self._local_generate([prompt], extra_body=extra_body)[0]
            raw = self._raw_from_content(generated)
            if self.cache_cfg.enabled and self.cache_dir is not None:
                cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            self._debug_log_response(content=generated, elapsed_seconds=time.perf_counter() - started, cached=False)
            return ChatResponse(content=generated, raw=raw, cached=False)

        self._debug_log_request(payload)
        started = time.perf_counter()
        if self.cache_cfg.enabled and self.cache_dir is not None:
            cache_path = self._cache_path(payload)
            if cache_path.exists():
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                content = self._extract_content(raw)
                self._debug_log_response(content=content, elapsed_seconds=time.perf_counter() - started, cached=True)
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
                self._debug_log_response(content=content, elapsed_seconds=time.perf_counter() - started, cached=False)
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

    def chat_batch(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> List[ChatResponse]:
        if not messages_batch:
            return []
        if not self.enable_messages_batch_api:
            return [
                self.chat(
                    messages=messages,
                    response_format=response_format,
                    extra_body=extra_body,
                )
                for messages in messages_batch
            ]

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages_batch,
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

        if self.backend == "local_vllm":
            if response_format is not None or self.cfg.response_format:
                LOGGER.warning("local_vllm backend ignores response_format; text output only")
            self._debug_log_request(payload)
            started = time.perf_counter()
            if self.cache_cfg.enabled and self.cache_dir is not None:
                cache_path = self._cache_path(payload)
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    contents = self._extract_batch_contents(raw)
                    self._debug_log_response(
                        content="\n\n".join(contents),
                        elapsed_seconds=time.perf_counter() - started,
                        cached=True,
                    )
                    return [ChatResponse(content=c, raw=raw, cached=True) for c in contents]

            prompts = [self._messages_to_local_prompt(messages) for messages in messages_batch]
            contents = self._local_generate(prompts, extra_body=extra_body)
            raw = {
                "choices": [
                    {
                        "index": idx,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                    }
                    for idx, content in enumerate(contents)
                ]
            }
            if self.cache_cfg.enabled and self.cache_dir is not None:
                cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            self._debug_log_response(
                content="\n\n".join(contents),
                elapsed_seconds=time.perf_counter() - started,
                cached=False,
            )
            return [ChatResponse(content=c, raw=raw, cached=False) for c in contents]

        self._debug_log_request(payload)
        started = time.perf_counter()
        if self.cache_cfg.enabled and self.cache_dir is not None:
            cache_path = self._cache_path(payload)
            if cache_path.exists():
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                contents = self._extract_batch_contents(raw)
                self._debug_log_response(
                    content="\n\n".join(contents),
                    elapsed_seconds=time.perf_counter() - started,
                    cached=True,
                )
                return [ChatResponse(content=c, raw=raw, cached=True) for c in contents]

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
                contents = self._extract_batch_contents(raw)
                if self.cache_cfg.enabled and self.cache_dir is not None:
                    cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                self._debug_log_response(
                    content="\n\n".join(contents),
                    elapsed_seconds=time.perf_counter() - started,
                    cached=False,
                )
                return [ChatResponse(content=c, raw=raw, cached=False) for c in contents]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_wait_seconds)
                else:
                    # Fallback for providers that do not support batched message arrays.
                    if len(messages_batch) > 1:
                        LOGGER.warning(
                            "Batch chat request failed; fallback to sequential chat calls. error=%s",
                            exc,
                        )
                        return [
                            self.chat(
                                messages=messages,
                                response_format=response_format,
                                extra_body=extra_body,
                            )
                            for messages in messages_batch
                        ]
                    raise RuntimeError(
                        f"Batch chat completion failed after {self.cfg.max_retries} attempts: {exc}"
                    ) from exc

        raise RuntimeError(f"Unreachable state in chat_batch(); last_error={last_error!r}")

    def _extract_content(self, raw: Dict[str, Any]) -> str:
        choices = raw.get("choices", [])
        if not choices:
            raise ValueError(f"No 'choices' in response: {raw}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        return self._normalize_content(content)

    def _extract_batch_contents(self, raw: Dict[str, Any]) -> List[str]:
        choices = raw.get("choices", [])
        if not choices:
            raise ValueError(f"No 'choices' in batch response: {raw}")
        contents: List[str] = []
        for choice in choices:
            message = choice.get("message", {})
            contents.append(self._normalize_content(message.get("content", "")))
        return contents
