"""High-level assistant: conversation state + backend + media sources."""

from __future__ import annotations

from typing import Iterator

from .backends import VLMBackend, create_backend
from .config import AppConfig
from .conversation import Conversation
from .media import MediaItem, Webcam


class VisionAssistant:
    def __init__(self, config: AppConfig | None = None, backend: VLMBackend | None = None):
        self.config = config or AppConfig()
        self.backend = backend or create_backend(self.config.model)
        self.conversation = Conversation(self.config.chat)
        self._webcam: Webcam | None = None

    # ------------------------------------------------------------------ webcam
    @property
    def webcam(self) -> Webcam:
        """Lazily started camera reader (so the app works without a camera)."""
        if self._webcam is None:
            self._webcam = Webcam(self.config.media)
        if not self._webcam.running:
            self._webcam.start()
        return self._webcam

    @property
    def webcam_running(self) -> bool:
        return self._webcam is not None and self._webcam.running

    # ------------------------------------------------------------------ chat
    def chat(self, text: str, media: list[MediaItem] | None = None, **gen) -> Iterator[str]:
        """Stream a reply to ``text`` (+ optional media) and record it in the conversation."""
        self.conversation.add_user(text, media)
        parts: list[str] = []
        try:
            for chunk in self.backend.stream(self.conversation.to_messages(), **gen):
                parts.append(chunk)
                yield chunk
        except BaseException:
            # Keep history consistent: drop the unanswered question (incl. Ctrl+C mid-reply).
            if parts:
                self.conversation.add_assistant("".join(parts).strip() + " [interrupted]")
            else:
                self.conversation.pop_last_user()
            raise
        self.conversation.add_assistant("".join(parts).strip())

    def ask(self, text: str, media: list[MediaItem] | None = None, **gen) -> str:
        return "".join(self.chat(text, media, **gen))

    def oneshot(self, text: str, media: list[MediaItem] | None = None, **gen) -> Iterator[str]:
        """Stateless query (not added to history) - used for live monitoring."""
        conv = Conversation(self.config.chat)
        conv.add_user(text, media)
        yield from self.backend.stream(conv.to_messages(), **gen)

    def reset(self) -> None:
        self.conversation.reset()

    def close(self) -> None:
        if self._webcam is not None:
            self._webcam.stop()
        self.backend.close()
