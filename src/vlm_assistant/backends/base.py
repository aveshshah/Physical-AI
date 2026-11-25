from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from ..config import ModelConfig


@dataclass
class GenerationStats:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    time_to_first_token: float | None = None
    total_time: float | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or not self.total_time or self.time_to_first_token is None:
            return None
        decode_time = self.total_time - self.time_to_first_token
        return self.completion_tokens / decode_time if decode_time > 0 else None

    def summary(self) -> str:
        parts = []
        if self.prompt_tokens is not None:
            parts.append(f"prompt {self.prompt_tokens} tok")
        if self.completion_tokens is not None:
            parts.append(f"output {self.completion_tokens} tok")
        if self.time_to_first_token is not None:
            parts.append(f"TTFT {self.time_to_first_token:.2f}s")
        tps = self.tokens_per_second
        if tps:
            parts.append(f"{tps:.1f} tok/s")
        if self.total_time is not None:
            parts.append(f"total {self.total_time:.2f}s")
        return ", ".join(parts)


class VLMBackend(ABC):
    """A vision-language model that turns neutral messages into streamed text."""

    name = "base"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.last_stats = GenerationStats()

    @abstractmethod
    def stream(self, messages: list[dict], **gen_overrides) -> Iterator[str]:
        """Yield text chunks of the assistant reply for ``messages``."""

    def generate(self, messages: list[dict], **gen_overrides) -> str:
        return "".join(self.stream(messages, **gen_overrides))

    def gen_params(self, **overrides) -> dict:
        params = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "repetition_penalty": self.cfg.repetition_penalty,
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        return params

    def close(self) -> None:  # pragma: no cover - optional
        pass


class _Timer:
    """Helper that records TTFT / total time around a streaming loop."""

    def __init__(self, stats: GenerationStats):
        self.stats = stats
        self.t0 = time.perf_counter()
        self._first = False

    def tick(self) -> None:
        if not self._first:
            self._first = True
            self.stats.time_to_first_token = time.perf_counter() - self.t0

    def done(self) -> None:
        self.stats.total_time = time.perf_counter() - self.t0
