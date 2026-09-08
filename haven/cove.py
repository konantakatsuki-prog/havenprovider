"""Cove GPT-SoVITS inference shared by hosted provider adapters."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COVE_REPO_ID = "MoYoYoTech/tone-models"
COVE_REPO_REVISION = "198ceeac1c4d5b75072b6995af2e185e4559ec1d"
COVE_GPT_PATH = "GPT_weights/maple_best_gpt.ckpt"
COVE_SOVITS_PATH = "SoVITS_weights/cove_best_sovits.pth"
COVE_REFERENCE_PATH = "ref_audios/cove_ref.wav"
COVE_PROMPT_TEXT = (
    "and he has a long career in the Senate representing Delaware. "
    "So both have had significant impacts on American politics and policies."
)
COVE_ASSET_PATTERNS = (
    COVE_GPT_PATH,
    COVE_SOVITS_PATH,
    COVE_REFERENCE_PATH,
    "chinese-hubert-base/**",
    "chinese-roberta-wwm-ext-large/**",
)
MAX_TEXT_LENGTH = 1000


@dataclass(frozen=True)
class CoveConfig:
    """Pinned model configuration for the tested Cove voice."""

    model_dir: Path
    repo_id: str = COVE_REPO_ID
    revision: str = COVE_REPO_REVISION
    gpt_path: str = COVE_GPT_PATH
    sovits_path: str = COVE_SOVITS_PATH
    reference_path: str = COVE_REFERENCE_PATH
    prompt_text: str = COVE_PROMPT_TEXT

    @classmethod
    def from_environment(cls) -> "CoveConfig":
        return cls(
            model_dir=Path(
                os.getenv(
                    "HAVEN_COVE_MODEL_DIR",
                    "/opt/haven/models/tone-models",
                )
            ),
            repo_id=os.getenv("HAVEN_COVE_REPO_ID", COVE_REPO_ID),
            revision=os.getenv("HAVEN_COVE_REPO_REVISION", COVE_REPO_REVISION),
            gpt_path=os.getenv("HAVEN_COVE_GPT_PATH", COVE_GPT_PATH),
            sovits_path=os.getenv("HAVEN_COVE_SOVITS_PATH", COVE_SOVITS_PATH),
            reference_path=os.getenv(
                "HAVEN_COVE_REFERENCE_PATH", COVE_REFERENCE_PATH
            ),
            prompt_text=os.getenv("HAVEN_COVE_PROMPT_TEXT", COVE_PROMPT_TEXT),
        )


@dataclass(frozen=True)
class CoveAssets:
    """Resolved files required by GPT-SoVITS inference."""

    model_dir: Path
    gpt: Path
    sovits: Path
    reference_audio: Path
    hubert: Path
    bert: Path

    def missing(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in (
                self.gpt,
                self.sovits,
                self.reference_audio,
                self.hubert,
                self.bert,
            )
            if not path.exists()
        )


def resolve_cove_assets(config: CoveConfig) -> CoveAssets:
    root = config.model_dir
    return CoveAssets(
        model_dir=root,
        gpt=root / config.gpt_path,
        sovits=root / config.sovits_path,
        reference_audio=root / config.reference_path,
        hubert=root / "chinese-hubert-base",
        bert=root / "chinese-roberta-wwm-ext-large",
    )


def download_cove_assets(
    config: CoveConfig | None = None,
    downloader: Callable[..., Any] | None = None,
) -> CoveAssets:
    """Download only the pinned Cove assets, unless they are already baked in."""

    selected = config or CoveConfig.from_environment()
    assets = resolve_cove_assets(selected)
    if not assets.missing():
        return assets

    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - image dependency failure
            raise RuntimeError("huggingface_hub is required to download Cove assets") from exc
        downloader = snapshot_download

    selected.model_dir.mkdir(parents=True, exist_ok=True)
    downloader(
        repo_id=selected.repo_id,
        revision=selected.revision,
        allow_patterns=list(COVE_ASSET_PATTERNS),
        local_dir=str(selected.model_dir),
    )

    missing = resolve_cove_assets(selected).missing()
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Cove model assets after download: {joined}")
    return resolve_cove_assets(selected)


def validate_text(text: Any) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = text.strip()
    if not normalized:
        raise ValueError("text cannot be empty")
    if len(normalized) > MAX_TEXT_LENGTH:
        raise ValueError(f"text cannot exceed {MAX_TEXT_LENGTH} characters")
    return normalized


def build_inference_params(
    text: str,
    assets: CoveAssets,
    config: CoveConfig,
    *,
    speed: float = 1.0,
    seed: int = 233333,
) -> dict[str, Any]:
    if not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    if not -1 <= seed <= 2_147_483_647:
        raise ValueError("seed must be between -1 and 2147483647")

    return {
        "text": validate_text(text),
        "text_lang": "en",
        "ref_audio_path": str(assets.reference_audio),
        "prompt_text": config.prompt_text,
        "prompt_lang": "en",
        "top_k": 5,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "cut4",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": speed,
        "fragment_interval": 0.07,
        "seed": seed,
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }


def _apply_torch_load_compat(torch_module: Any) -> None:
    """Allow the pinned, trusted GPT-SoVITS checkpoints on newer PyTorch."""

    original_load = torch_module.load
    if getattr(original_load, "_haven_weights_compat", False):
        return

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    compatible_load._haven_weights_compat = True  # type: ignore[attr-defined]
    torch_module.load = compatible_load


def _apply_gpt_sovits_import_compat() -> None:
    """Apply the final typing fix used by the previously working Cove Space."""

    spec = importlib.util.find_spec("gpt_sovits_python")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("The pinned gpt_sovits_python wheel is not installed")

    package_root = Path(next(iter(spec.submodule_search_locations)))
    cleaner = package_root / "text" / "cleaner.py"
    if not cleaner.exists() or "from importlib import import_module" not in cleaner.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("The installed GPT-SoVITS wheel lacks the Cove English patch")

    for filename in (
        "patched_mha_with_cache.py",
        "patched_mha_with_cache_onnx.py",
    ):
        target = package_root / "AR" / "modules" / filename
        if not target.exists():
            continue
        source = target.read_text(encoding="utf-8")
        imports: list[str] = []
        if "from torch import Tensor" not in source:
            imports.append("from torch import Tensor")
        if "from typing import Optional, Tuple" not in source:
            imports.append("from typing import Optional, Tuple")
        if imports:
            target.write_text("\n".join(imports) + "\n" + source, encoding="utf-8")


def _write_wav(sample_rate: int, audio_data: Any) -> Path:
    import numpy as np
    import scipy.io.wavfile as wavfile

    audio = np.asarray(audio_data)
    if audio.dtype.kind == "f":
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)

    output_dir = Path(tempfile.mkdtemp(prefix="haven-cove-"))
    output_path = output_dir / "cove.wav"
    wavfile.write(str(output_path), int(sample_rate), audio)
    return output_path


class CoveSynthesizer:
    """Load the fixed Cove voice once, then produce WAV files safely."""

    def __init__(self, config: CoveConfig | None = None) -> None:
        self.config = config or CoveConfig.from_environment()
        self.assets: CoveAssets | None = None
        self.pipeline: Any = None
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        if self.pipeline is not None:
            return

        nltk_data = Path("/opt/haven/nltk_data")
        if nltk_data.exists():
            os.environ.setdefault("NLTK_DATA", str(nltk_data))

        import torch

        _apply_torch_load_compat(torch)
        _apply_gpt_sovits_import_compat()
        from gpt_sovits_python import TTS, TTS_Config

        self.assets = download_cove_assets(self.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_config = {
            "default": {
                "device": device,
                "is_half": device == "cuda",
                "t2s_weights_path": str(self.assets.gpt),
                "vits_weights_path": str(self.assets.sovits),
                "cnhuhbert_base_path": str(self.assets.hubert),
                "bert_base_path": str(self.assets.bert),
            }
        }
        self.pipeline = TTS(TTS_Config(model_config))

    def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        seed: int = 233333,
    ) -> Path:
        if self.pipeline is None:
            self.load()
        if self.assets is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("Cove assets were not loaded")

        params = build_inference_params(
            text,
            self.assets,
            self.config,
            speed=speed,
            seed=seed,
        )
        with self._inference_lock:
            sample_rate, audio_data = next(self.pipeline.run(params))
        return _write_wav(sample_rate, audio_data)

