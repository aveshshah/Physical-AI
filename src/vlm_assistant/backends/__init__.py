from __future__ import annotations

from ..config import ModelConfig
from .base import GenerationStats, VLMBackend


def create_backend(cfg: ModelConfig) -> VLMBackend:
    """Instantiate the backend named in ``cfg.backend`` (heavy imports are lazy)."""
    if cfg.backend == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(cfg)
    if cfg.backend == "openai":
        from .openai_backend import OpenAICompatBackend

        return OpenAICompatBackend(cfg)
    if cfg.backend == "mock":
        from .mock_backend import MockBackend

        return MockBackend(cfg)
    raise ValueError(f"Unknown backend {cfg.backend!r} (expected transformers, openai or mock)")


__all__ = ["create_backend", "VLMBackend", "GenerationStats"]
