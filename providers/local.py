"""Direct local provider wrapper."""

from __future__ import annotations

from haven.engine import SharedTTSEngine

from .base import EngineProvider, Payload


class LocalProvider(EngineProvider):
    """Run the shared engine in the current Python process."""

    def handle_local(self, payload: Payload) -> dict[str, object]:
        return self.handle(payload)


def handle_local(payload: Payload, engine: SharedTTSEngine | None = None) -> dict[str, object]:
    return LocalProvider(engine).handle_local(payload)
