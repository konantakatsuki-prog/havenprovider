"""Common provider adapter types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from haven.engine import SharedTTSEngine


Payload = Mapping[str, Any]


class Provider(Protocol):
    def handle(self, payload: Payload) -> dict[str, Any]:
        ...


class EngineProvider:
    """Small adapter that keeps transport concerns out of the TTS engine."""

    def __init__(self, engine: SharedTTSEngine | None = None) -> None:
        self.engine = engine or SharedTTSEngine()

    def handle(self, payload: Payload) -> dict[str, Any]:
        return self.engine.synthesize(payload)
