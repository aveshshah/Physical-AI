"""Offline smoke test of the transformers backend using a tiny *random-weight* Qwen3-VL.

It builds a minimal tokenizer, processor and model locally (no downloads), saves them to a temp
directory and runs the real ``TransformersBackend`` on an image and on a video. The output text is
gibberish (random weights) - the point is to verify that the preprocessing, chat template, vision
token placement, video metadata and streaming generation all fit together with the installed
transformers / qwen-vl-utils versions.

    python scripts/tiny_qwen3vl_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for c in message['content'] %}"
    "{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif c['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}"
    "{% endfor %}{% endif %}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
           "<|image_pad|>", "<|video_pad|>"]


def build_tiny_model(out: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    words = [chr(c) for c in range(32, 127)] + [f"w{i}" for i in range(300)]
    vocab = {tok: i for i, tok in enumerate(SPECIAL + words + ["<unk>"])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="<unk>", eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL[1:],
    )

    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    tid = tokenizer.convert_tokens_to_ids
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
            max_position_embeddings=8192,
        ),
        vision_config=dict(
            depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64,
            patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=256,
            deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=tid("<|image_pad|>"),
        video_token_id=tid("<|video_pad|>"),
        vision_start_token_id=tid("<|vision_start|>"),
        vision_end_token_id=tid("<|vision_end|>"),
    )
    model = Qwen3VLForConditionalGeneration(config)
    model.generation_config.eos_token_id = tid("<|im_end|>")
    model.generation_config.pad_token_id = tid("<|endoftext|>")
    model.save_pretrained(out)
    processor.save_pretrained(out)


def main() -> None:
    from vlm_assistant.backends.transformers_backend import TransformersBackend
    from vlm_assistant.config import ChatConfig, MediaConfig, ModelConfig
    from vlm_assistant.conversation import Conversation
    from vlm_assistant.media import ImageItem, VideoItem

    with tempfile.TemporaryDirectory() as tmp:
        build_tiny_model(Path(tmp))
        backend = TransformersBackend(
            ModelConfig(model_id=tmp, dtype="float32", device_map="cpu", max_new_tokens=12, temperature=0.0)
        )
        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
        frames = [Image.fromarray(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)) for _ in range(6)]
        video = VideoItem(frames=frames, timestamps=[i * 0.5 for i in range(6)], source="synthetic", fps=2.0)

        conv = Conversation(ChatConfig(system_prompt="sys"))
        conv.add_user("what is this?", [ImageItem(img, "synthetic")])
        out = "".join(backend.stream(conv.to_messages()))
        print("image  ->", repr(out), "|", backend.last_stats.summary())
        assert backend.last_stats.completion_tokens and backend.last_stats.completion_tokens > 0

        conv.add_assistant(out)
        conv.add_user("and this clip?", [video])
        out = "".join(backend.stream(conv.to_messages()))
        print("video  ->", repr(out), "|", backend.last_stats.summary())
        assert backend.last_stats.completion_tokens and backend.last_stats.completion_tokens > 0

        _ = MediaConfig  # imported for completeness
        print("OK: transformers backend pipeline works end to end")


if __name__ == "__main__":
    main()
