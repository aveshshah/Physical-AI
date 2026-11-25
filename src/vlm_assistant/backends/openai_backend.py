"""Any OpenAI-compatible chat server: llama.cpp ``llama-server``, vLLM, Ollama, LM Studio...

This is the path for the robot: run a quantised GGUF of Qwen3-VL with llama.cpp on the robot's
own computer (e.g. a Jetson), or on a desktop GPU on the same network, and point this client at it.

Videos are sent as a sequence of time-stamped frames (images), which every server supports.
"""

from __future__ import annotations

import base64
import json
from typing import Iterator

from ..config import ModelConfig
from ..media import VideoItem, pil_to_jpeg_bytes
from .base import GenerationStats, VLMBackend, _Timer


def _data_url(img) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(pil_to_jpeg_bytes(img, quality=88)).decode("ascii")


def to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        parts: list[dict] = []
        for part in msg["content"]:
            kind = part["type"]
            if kind == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif kind == "image":
                parts.append({"type": "image_url", "image_url": {"url": _data_url(part["image"])}})
            elif kind == "video":
                video: VideoItem = part["video"]
                parts.append(
                    {
                        "type": "text",
                        "text": f"The following {len(video.frames)} images are consecutive frames of a "
                        f"{video.duration:.1f}-second video ({video.source}), in chronological order.",
                    }
                )
                for img, t in zip(video.frames, video.timestamps):
                    parts.append({"type": "text", "text": f"<{t:.1f} seconds>"})
                    parts.append({"type": "image_url", "image_url": {"url": _data_url(img)}})
            else:
                raise ValueError(f"Unsupported content type: {kind}")
        # Plain-string content for text-only system/assistant turns: widest server compatibility.
        if all(p["type"] == "text" for p in parts):
            out.append({"role": msg["role"], "content": "\n".join(p["text"] for p in parts)})
        else:
            out.append({"role": msg["role"], "content": parts})
    return out


class OpenAICompatBackend(VLMBackend):
    name = "openai"

    def __init__(self, cfg: ModelConfig, session=None):
        super().__init__(cfg)
        import requests

        self.session = session or requests.Session()
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"

    def _payload(self, messages: list[dict], params: dict) -> dict:
        payload = {
            "model": self.cfg.model_id,
            "messages": to_openai_messages(messages),
            "max_tokens": params["max_new_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Non-standard but accepted by llama.cpp, vLLM and Ollama's OpenAI endpoint.
        payload["top_k"] = params["top_k"]
        if params["repetition_penalty"] and params["repetition_penalty"] != 1.0:
            payload["repetition_penalty"] = params["repetition_penalty"]
        return payload

    def stream(self, messages: list[dict], **gen_overrides) -> Iterator[str]:
        params = self.gen_params(**gen_overrides)
        self.last_stats = stats = GenerationStats()
        timer = _Timer(stats)
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        with self.session.post(
            self.url,
            json=self._payload(messages, params),
            headers=headers,
            stream=True,
            timeout=self.cfg.request_timeout,
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"Server returned {resp.status_code}: {resp.text[:500]}")
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if usage:
                    stats.prompt_tokens = usage.get("prompt_tokens")
                    stats.completion_tokens = usage.get("completion_tokens")
                for choice in event.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        timer.tick()
                        yield delta
        timer.done()
