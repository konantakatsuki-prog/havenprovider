"""Build-time model and NLP resource preparation for Modal images."""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

from .config import load_settings


def _safe_model_path(model_dir: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe model path: {relative_path!r}")
    target = (model_dir / candidate).resolve()
    root = model_dir.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Unsafe model path: {relative_path!r}")
    return target


def download_pretrained_models(
    model_dir: Path | None = None,
    model_urls: tuple[tuple[str, str], ...] | None = None,
) -> None:
    """Download configured model files while the Modal image is being built.

    The repository does not contain GPT-SoVITS weights, so URLs are deliberately
    supplied through HAVEN_PRETRAINED_MODEL_URLS instead of being guessed here.
    """

    settings = load_settings()
    destination = Path(model_dir or settings.model_dir)
    entries = tuple(model_urls or settings.pretrained_model_urls)
    if not entries:
        raise RuntimeError(
            "No pretrained models configured. Set HAVEN_PRETRAINED_MODEL_URLS "
            "to a JSON object of relative paths and URLs before building Modal."
        )

    destination.mkdir(parents=True, exist_ok=True)
    for relative_path, url in entries:
        target = _safe_model_path(destination, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            continue

        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            urllib.request.urlretrieve(url, temporary_path)
            if temporary_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded model is empty: {url}")
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)


def preload_nltk_resources() -> None:
    """Ensure the English POS tagger and cmudict are in the image cache."""

    try:
        import nltk
    except ImportError as exc:  # pragma: no cover - only reached in a bad image
        raise RuntimeError("nltk must be installed in the Modal image") from exc

    resources = (
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/cmudict", "cmudict"),
    )
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(package_name, quiet=True):
                raise RuntimeError(f"Unable to download NLTK resource: {package_name}")
            nltk.data.find(resource_path)
