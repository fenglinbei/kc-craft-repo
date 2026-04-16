from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import shlex
from typing import Any, Dict, Optional

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


@dataclass
class LLMConfig:
    api_key: str
    api_base: str
    model: str
    timeout: int = 120
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    max_retries: int = 3
    retry_wait_seconds: float = 2.0
    headers: Dict[str, str] = field(default_factory=dict)
    response_format: Optional[Dict[str, Any]] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    model_path: str = "./models/bge-small-zh-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


@dataclass
class PipelineConfig:
    max_contrastive_questions: int = 5
    mmr_lambda: float = 1.0
    max_context_chars_for_answers: int = 24000
    max_context_chars_for_verification: int = 20000
    deduplicate_questions: bool = True
    save_raw_api_responses: bool = True
    merge_entity_strategy: str = "normalized_name"
    question_generation_mode: str = "kg"


@dataclass
class VerificationConfig:
    labels: list[str] = field(default_factory=list)
    label_descriptions: Dict[str, str] = field(default_factory=dict)


@dataclass
class CacheConfig:
    enabled: bool = True
    cache_dir: str = "./cache"


@dataclass
class RunConfig:
    mode: str = "full"
    seed: int = 42
    limit: Optional[int] = None
    num_workers: int = 1
    verbose: bool = True
    show_sample_stage_progress: bool = True
    debug: bool = False
    debug_preview_chars: int = 1200
    debug_head_chars: int = 450
    debug_tail_chars: int = 450


@dataclass
class DataConfig:
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    id_field: str = "id"
    claim_field: str = "claim"
    reports_field: str = "reports"
    label_field: str = "label"


@dataclass
class PromptConfig:
    use_operationalized_kg_prompt: bool = True
    llm_question_examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AppConfig:
    run: RunConfig
    data: DataConfig
    cache: CacheConfig
    models: Dict[str, LLMConfig]
    embedding: EmbeddingConfig
    pipeline: PipelineConfig
    verification: VerificationConfig
    prompts: PromptConfig
    extras: Dict[str, Any] = field(default_factory=dict)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return os.environ.get(key, "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export "):].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue

            try:
                parsed = shlex.split(value, posix=True)
                if parsed:
                    value = parsed[0]
            except ValueError:
                pass
            os.environ[key] = value


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | os.PathLike[str]) -> AppConfig:
    path = Path(path)
    dotenv_candidates = [
        Path.cwd() / ".env",
        path.parent / ".env",
        path.parent.parent / ".env",
    ]
    for dotenv_path in dotenv_candidates:
        _load_dotenv(dotenv_path)
    raw = _load_yaml(path)

    if "extends" in raw and raw["extends"]:
        base_path = (path.parent / raw["extends"]).resolve()
        base_raw = _load_yaml(base_path)
        raw = _deep_merge(base_raw, {k: v for k, v in raw.items() if k != "extends"})

    raw = _expand_env(raw)

    models = {
        key: LLMConfig(**value)
        for key, value in raw.get("models", {}).items()
    }

    extras = {
        k: v for k, v in raw.items()
        if k not in {
            "run", "data", "cache", "models", "embedding", "pipeline", "verification", "prompts"
        }
    }

    return AppConfig(
        run=RunConfig(**raw.get("run", {})),
        data=DataConfig(**raw.get("data", {})),
        cache=CacheConfig(**raw.get("cache", {})),
        models=models,
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        verification=VerificationConfig(**raw.get("verification", {})),
        prompts=PromptConfig(**raw.get("prompts", {})),
        extras=extras,
    )
