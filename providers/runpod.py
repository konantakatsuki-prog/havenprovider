"""RunPod serverless adapter preserving the original job contract."""

from __future__ import annotations

from typing import Any

from haven.engine import SharedTTSEngine


class RunPodProvider:
    def __init__(self, engine: SharedTTSEngine | None = None) -> None:
        self.engine = engine or SharedTTSEngine()

    def handle_job(self, job: dict[str, Any]) -> dict[str, Any]:
        # Keep the original job["input"] contract unchanged.
        return self.engine.synthesize(job["input"])
