"""Environment-backed configuration shared by every provider."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODAL_PIP_PACKAGES = ("nltk", "torch", "torchaudio", "torchcodec")


def _pretrained_model_urls() -> tuple[tuple[str, str], ...]:
    """Read a JSON object mapping image-relative paths to download URLs."""

    raw = os.getenv("HAVEN_PRETRAINED_MODEL_URLS", "").strip()
    if not raw:
        return ()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("HAVEN_PRETRAINED_MODEL_URLS must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("HAVEN_PRETRAINED_MODEL_URLS must be a JSON object")

    entries: list[tuple[str, str]] = []
    for relative_path, url in parsed.items():
        if not isinstance(relative_path, str) or not isinstance(url, str):
            raise ValueError("Model paths and URLs must both be strings")
        if not relative_path or not url:
            raise ValueError("Model paths and URLs cannot be empty")
        entries.append((relative_path, url))
    return tuple(entries)


def _modal_pip_packages() -> tuple[str, ...]:
    raw = os.getenv("HAVEN_MODAL_PIP_PACKAGES", "").strip()
    if not raw:
        return DEFAULT_MODAL_PIP_PACKAGES
    packages = tuple(item.strip() for item in raw.split(",") if item.strip())
    return packages or DEFAULT_MODAL_PIP_PACKAGES


@dataclass(frozen=True)
class Settings:
    """Runtime settings with conservative defaults for local development."""

    engine_backend: str
    model_dir: Path
    pretrained_model_urls: tuple[tuple[str, str], ...]
    modal_app_name: str
    modal_gpu: str
    modal_timeout: int
    modal_pip_packages: tuple[str, ...]
    http_host: str
    http_port: int


def load_settings() -> Settings:
    """Load settings at call time so provider processes see current env vars."""

    return Settings(
        engine_backend=os.getenv("HAVEN_ENGINE_BACKEND", "echo").strip().lower(),
        model_dir=Path(os.getenv("HAVEN_MODEL_DIR", "models")),
        pretrained_model_urls=_pretrained_model_urls(),
        modal_app_name=os.getenv("HAVEN_MODAL_APP_NAME", "haven-provider"),
        modal_gpu=os.getenv("HAVEN_MODAL_GPU", "T4"),
        modal_timeout=int(os.getenv("HAVEN_MODAL_TIMEOUT", "900")),
        modal_pip_packages=_modal_pip_packages(),
        http_host=os.getenv("HAVEN_HTTP_HOST", "127.0.0.1"),
        http_port=int(os.getenv("HAVEN_HTTP_PORT", "8000")),
    )
