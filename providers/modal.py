"""Modal provider wrapper; deployment wiring lives in modal_app.py."""

from __future__ import annotations

from haven.engine import SharedTTSEngine

from .base import EngineProvider, Payload


class ModalProvider(EngineProvider):
    """Adapt a JSON HTTP body to the shared engine's stable response."""

    def handle_http(self, payload: Payload) -> dict[str, object]:
        return self.handle(payload)


def handle_modal(payload: Payload, engine: SharedTTSEngine | None = None) -> dict[str, object]:
    return ModalProvider(engine).handle_http(payload)
