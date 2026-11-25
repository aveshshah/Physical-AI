"""Local, GPU-accelerated vision-language assistant built on Qwen3-VL."""

from .assistant import VisionAssistant
from .config import AppConfig, ChatConfig, MediaConfig, ModelConfig
from .media import FrameBuffer, ImageItem, VideoItem, Webcam, load_image, load_media, sample_video

__version__ = "0.1.0"

__all__ = [
    "VisionAssistant",
    "AppConfig",
    "ChatConfig",
    "MediaConfig",
    "ModelConfig",
    "FrameBuffer",
    "ImageItem",
    "VideoItem",
    "Webcam",
    "load_image",
    "load_media",
    "sample_video",
]
