"""Modal deployment entrypoint for the shared Haven provider.

Deploy with:

    modal deploy modal_app.py

The endpoint accepts a POST JSON body such as {"text": "Hello", "voice": "haven"}
and returns the same response shape as the RunPod handler.
"""

from __future__ import annotations

import os
from typing import Any

from haven.config import load_settings
from haven.model_download import download_pretrained_models, preload_nltk_resources
from providers.modal import ModalProvider


def _preload_resources() -> None:
    """Run once during Modal image construction, before serving requests."""

    settings = load_settings()
    download_pretrained_models(settings.model_dir, settings.pretrained_model_urls)
    preload_nltk_resources()


def _build_app() -> Any:
    import modal

    settings = load_settings()
    model_dir = "/opt/haven/models"
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "git")
        .pip_install(*settings.modal_pip_packages)
        .add_local_python_source("haven", "providers", copy=True)
        .env({"HAVEN_MODEL_DIR": model_dir})
        .run_function(
            _preload_resources,
            env={
                "HAVEN_MODEL_DIR": model_dir,
                "HAVEN_PRETRAINED_MODEL_URLS": os.getenv(
                    "HAVEN_PRETRAINED_MODEL_URLS", ""
                ),
            },
        )
    )

    app = modal.App(name=settings.modal_app_name)

    @app.function(image=image, gpu=settings.modal_gpu, timeout=settings.modal_timeout)
    @modal.fastapi_endpoint(method="POST", docs=True)
    def synthesize(payload: dict[str, Any]) -> dict[str, Any]:
        # dict annotation makes Modal parse the POST body as JSON.
        return ModalProvider().handle_http(payload)

    return app


try:
    import modal as _modal  # noqa: F401
except ImportError:  # Local tests do not need the Modal SDK installed.
    app = None
else:
    app = _build_app()
