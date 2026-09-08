"""Planned third/self-hosted provider wrapper."""

from __future__ import annotations

from haven.engine import SharedTTSEngine

from .local import LocalProvider


class SelfHostedProvider(LocalProvider):
    """Use the same engine contract behind a future self-hosted transport."""

    def __init__(self, engine: SharedTTSEngine | None = None, endpoint: str | None = None) -> None:
        super().__init__(engine)
        self.endpoint = endpoint
