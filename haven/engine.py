"""Provider-independent request normalization and TTS engine boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SpeechRequest:
    """The stable request shape shared by local, Modal, and other providers."""

    text: Any
    voice: Any

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SpeechRequest":
        if not isinstance(payload, Mapping):
            raise TypeError("TTS payload must be a JSON object")
        return cls(
            text=payload.get("text", ""),
            voice=payload.get("voice", "haven"),
        )


class TTSBackend(Protocol):
    """Backend contract; a GPT-SoVITS adapter can implement this later."""

    def synthesize(self, request: SpeechRequest) -> dict[str, Any]:
        ...


class EchoTTSBackend:
    """Compatibility backend matching the downloaded repository's behavior."""

    def synthesize(self, request: SpeechRequest) -> dict[str, Any]:
        return {
            "ok": True,
            "text": request.text,
            "voice": request.voice,
            "message": "Haven provider is alive",
        }


class SharedTTSEngine:
    """Shared engine facade used by every provider wrapper."""

    def __init__(self, backend: TTSBackend | None = None) -> None:
        self._backend = backend or EchoTTSBackend()

    def synthesize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = SpeechRequest.from_payload(payload)
        return self._backend.synthesize(request)
