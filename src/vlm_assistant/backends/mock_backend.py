"""A model-free backend: describes what it received. For tests, CI and UI development."""

from __future__ import annotations

import time
from typing import Iterator

from ..media import VideoItem
from .base import GenerationStats, VLMBackend, _Timer


class MockBackend(VLMBackend):
    name = "mock"
    delay = 0.0  # seconds per chunk, to visualise streaming

    def stream(self, messages: list[dict], **gen_overrides) -> Iterator[str]:
        self.last_stats = stats = GenerationStats()
        timer = _Timer(stats)
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        seen = []
        question = ""
        if last_user:
            for part in last_user["content"]:
                if part["type"] == "image":
                    w, h = part["image"].size
                    seen.append(f"an image ({w}x{h})")
                elif part["type"] == "video":
                    v: VideoItem = part["video"]
                    seen.append(f"a video ({len(v.frames)} frames, {v.duration:.1f}s)")
                elif part["type"] == "text":
                    question = part["text"]
        n_turns = sum(1 for m in messages if m["role"] == "user")
        reply = (
            f"[mock] I received {', '.join(seen) if seen else 'no media'} "
            f"and the question {question!r} (user turn {n_turns})."
        )
        stats.prompt_tokens = sum(len(str(m["content"])) // 4 for m in messages)
        words = reply.split(" ")
        for i, word in enumerate(words):
            if self.delay:
                time.sleep(self.delay)
            timer.tick()
            yield word + (" " if i < len(words) - 1 else "")
        stats.completion_tokens = len(words)
        timer.done()
