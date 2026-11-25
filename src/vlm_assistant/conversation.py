"""Backend-agnostic conversation state.

Messages are produced in a neutral format that each backend converts to its own wire format::

    {"role": "user", "content": [
        {"type": "image", "image": PIL.Image},
        {"type": "video", "video": VideoItem},
        {"type": "text",  "text": "..."},
    ]}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ChatConfig
from .media import ImageItem, MediaItem, VideoItem


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str
    media: list[MediaItem] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class Conversation:
    def __init__(self, cfg: ChatConfig | None = None):
        self.cfg = cfg or ChatConfig()
        self.system_prompt = self.cfg.system_prompt
        self.turns: list[Turn] = []

    # ------------------------------------------------------------------ mutation
    def add_user(self, text: str, media: list[MediaItem] | None = None) -> Turn:
        turn = Turn("user", text, list(media or []))
        self.turns.append(turn)
        return turn

    def add_assistant(self, text: str) -> Turn:
        turn = Turn("assistant", text)
        self.turns.append(turn)
        return turn

    def pop_last_user(self) -> None:
        """Undo a user turn whose generation failed."""
        if self.turns and self.turns[-1].role == "user":
            self.turns.pop()

    def reset(self) -> None:
        self.turns.clear()

    # ------------------------------------------------------------------ export
    def _window(self) -> list[Turn]:
        """Last ``max_history_turns`` exchanges, always starting on a user turn."""
        max_msgs = max(1, self.cfg.max_history_turns) * 2
        window = self.turns[-max_msgs:]
        while window and window[0].role != "user":
            window = window[1:]
        return window

    def to_messages(self) -> list[dict]:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})

        window = self._window()
        user_idx = [i for i, t in enumerate(window) if t.role == "user"]
        keep_media = set(user_idx[-self.cfg.keep_media_turns:]) if self.cfg.keep_media_turns > 0 else set()

        for i, turn in enumerate(window):
            content: list[dict] = []
            for item in turn.media:
                if i in keep_media:
                    if isinstance(item, ImageItem):
                        content.append({"type": "image", "image": item.image})
                    elif isinstance(item, VideoItem):
                        content.append({"type": "video", "video": item})
                else:
                    content.append({"type": "text", "text": f"[earlier {item.describe()} - no longer shown]"})
            if turn.text:
                content.append({"type": "text", "text": turn.text})
            if not content:
                content.append({"type": "text", "text": ""})
            messages.append({"role": turn.role, "content": content})
        return messages

    def transcript(self) -> list[dict]:
        """JSON-serialisable transcript (media replaced by descriptions)."""
        return [
            {
                "role": t.role,
                "text": t.text,
                "media": [m.describe() for m in t.media],
                "timestamp": t.timestamp,
            }
            for t in self.turns
        ]

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"system_prompt": self.system_prompt, "turns": self.transcript()}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
