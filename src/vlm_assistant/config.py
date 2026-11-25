"""Configuration dataclasses, loadable from YAML and overridable from the CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise visual assistant running locally on the user's machine. "
    "You can see images, webcam snapshots, short webcam clips and recorded videos that the user "
    "shares. Describe only what is actually visible; if something is unclear or out of frame, say so "
    "instead of guessing. When a video or clip is provided, reason about what changes over time and "
    "refer to timestamps when useful."
)


@dataclass
class ModelConfig:
    # "transformers" (local GPU, Hugging Face), "openai" (any OpenAI-compatible server such as
    # llama.cpp / vLLM / Ollama / LM Studio) or "mock" (no model, for tests and UI work).
    backend: str = "transformers"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"

    # --- transformers backend ---
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32 | auto
    quantization: str | None = None  # None | "4bit" | "8bit" (needs bitsandbytes)
    attn_implementation: str | None = None  # e.g. "flash_attention_2" or "sdpa"
    device_map: str = "auto"

    # --- openai-compatible backend ---
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "EMPTY"
    request_timeout: float = 600.0

    # --- generation (Qwen's recommended defaults for the Instruct VL models) ---
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0


@dataclass
class MediaConfig:
    # Images are downscaled so the longest side is at most this many pixels. Qwen3-VL uses
    # 32x32-pixel visual tokens, so 1280 px ~= 1600 tokens for a 16:9 image.
    image_max_side: int = 1280
    # Video / clip frames are smaller: many frames add up quickly.
    video_frame_max_side: int = 640
    video_fps: float = 2.0
    video_max_frames: int = 32

    # Webcam: an int device index, or a string (file path, RTSP/HTTP stream URL, GStreamer pipeline).
    webcam_source: int | str = 0
    webcam_width: int = 1280
    webcam_height: int = 720
    # Rolling buffer of recent frames used for "what just happened?" questions.
    webcam_buffer_seconds: float = 10.0
    webcam_buffer_fps: float = 2.0


@dataclass
class ChatConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Number of past user/assistant exchanges kept in context.
    max_history_turns: int = 12
    # Media is only re-sent for the most recent N user turns; older images/videos are replaced
    # by a short text placeholder to keep the context (and VRAM) bounded.
    keep_media_turns: int = 2


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)

    # ---------------------------------------------------------------- I/O
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppConfig":
        cfg = cls()
        _update_dataclass(cfg, data or {})
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> "AppConfig":
        if path is None:
            return cls()
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(obj: Any, data: dict[str, Any], prefix: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key: {prefix}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section {prefix}{key} must be a mapping")
            _update_dataclass(current, value, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, value)
