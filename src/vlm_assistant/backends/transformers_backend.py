"""Local GPU inference with Hugging Face transformers (Qwen3-VL)."""

from __future__ import annotations

import logging
import threading
from typing import Iterator

from ..config import ModelConfig
from ..media import VideoItem
from .base import GenerationStats, VLMBackend, _Timer

log = logging.getLogger(__name__)

# Qwen3-VL: 16-px ViT patches merged 2x2 -> one visual token per 32x32 pixels.
QWEN3_PATCH_SIZE = 16
# Upper bound on pixels per video (all frames together). ~24k visual tokens.
VIDEO_TOTAL_PIXELS = 24_000 * 32 * 32


def to_qwen_messages(messages: list[dict]) -> list[dict]:
    """Neutral messages -> the dict format expected by ``qwen_vl_utils.process_vision_info``."""
    out = []
    for msg in messages:
        content = []
        for part in msg["content"]:
            kind = part["type"]
            if kind == "text":
                content.append({"type": "text", "text": part["text"]})
            elif kind == "image":
                content.append({"type": "image", "image": part["image"]})
            elif kind == "video":
                video: VideoItem = part["video"]
                content.append(
                    {
                        "type": "video",
                        "video": list(video.frames),
                        # qwen_vl_utils builds timestamps as frame_index / sample_fps.
                        "sample_fps": float(video.fps),
                        "total_pixels": VIDEO_TOTAL_PIXELS,
                    }
                )
            else:
                raise ValueError(f"Unsupported content type: {kind}")
        out.append({"role": msg["role"], "content": content})
    return out


class TransformersBackend(VLMBackend):
    name = "transformers"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:  # other VLM families / older naming
            from transformers import AutoModelForImageTextToText as ModelCls

        self.torch = torch
        if not torch.cuda.is_available():
            log.warning("CUDA is not available - Qwen3-VL-8B on CPU will be extremely slow.")

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }[cfg.dtype]
        kwargs: dict = {"dtype": dtype, "device_map": cfg.device_map}
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        if cfg.quantization:
            from transformers import BitsAndBytesConfig

            if cfg.quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if dtype == "auto" else dtype,
                )
            elif cfg.quantization == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"Unknown quantization {cfg.quantization!r} (use 4bit or 8bit)")

        log.info("Loading %s (%s)...", cfg.model_id, ", ".join(f"{k}={v}" for k, v in kwargs.items()))
        self.model = ModelCls.from_pretrained(cfg.model_id, **kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self._lock = threading.Lock()  # one generation at a time on the GPU

    # ------------------------------------------------------------------ inputs
    def _prepare(self, messages: list[dict]):
        from qwen_vl_utils import process_vision_info

        qmsgs = to_qwen_messages(messages)
        text = self.processor.apply_chat_template(qmsgs, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            qmsgs,
            image_patch_size=QWEN3_PATCH_SIZE,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadata = None
        if videos is not None:
            videos, video_metadata = map(list, zip(*videos))
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            do_resize=False,  # already resized by qwen_vl_utils
            **video_kwargs,
        )
        return inputs.to(self.model.device)

    # ------------------------------------------------------------------ generation
    def stream(self, messages: list[dict], **gen_overrides) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        params = self.gen_params(**gen_overrides)
        temperature = params.pop("temperature")
        gen_kwargs = dict(
            max_new_tokens=params["max_new_tokens"],
            repetition_penalty=params["repetition_penalty"],
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=params["top_p"], top_k=params["top_k"])
        else:
            gen_kwargs.update(do_sample=False)

        self.last_stats = stats = GenerationStats()
        timer = _Timer(stats)

        with self._lock:
            inputs = self._prepare(messages)
            stats.prompt_tokens = int(inputs["input_ids"].shape[-1])
            streamer = TextIteratorStreamer(
                self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600.0
            )
            result: dict = {}

            def _run():
                try:
                    with self.torch.inference_mode():
                        out = self.model.generate(**inputs, streamer=streamer, **gen_kwargs)
                    result["n"] = int(out.shape[-1] - inputs["input_ids"].shape[-1])
                except Exception as exc:  # surface errors in the consumer thread
                    result["error"] = exc
                    streamer.end()

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            try:
                for chunk in streamer:
                    if chunk:
                        timer.tick()
                        yield chunk
            finally:
                worker.join()
                timer.done()
                stats.completion_tokens = result.get("n")
            if "error" in result:
                raise result["error"]

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
