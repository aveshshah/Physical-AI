"""Report GPU / library status and recommend a config for Qwen3-VL-8B.

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform


def version(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__})"


def main() -> None:
    print(f"python        {platform.python_version()} on {platform.system()} {platform.machine()}")
    for mod in ["torch", "torchvision", "transformers", "accelerate", "qwen_vl_utils", "bitsandbytes",
                "flash_attn", "cv2", "gradio"]:
        print(f"{mod:<13} {version(mod)}")

    try:
        import torch
    except ImportError:
        print("\nPyTorch is not installed -> use the 'openai' backend or install torch with CUDA.")
        return
    if not torch.cuda.is_available():
        print("\nNo CUDA GPU visible to PyTorch. Options: install a CUDA build of torch, or use the")
        print("'openai' backend against llama.cpp / Ollama (configs/edge_llamacpp.yaml).")
        return

    print(f"\nCUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s):")
    best = 0.0
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024**3
        best = max(best, gb)
        print(f"  [{i}] {p.name}  {gb:.1f} GB  compute {p.major}.{p.minor}  bf16={torch.cuda.is_bf16_supported()}")

    print("\nRecommendation for Qwen3-VL-8B-Instruct:")
    if best >= 20:
        print("  configs/default.yaml  (bf16, ~17.5 GB weights + activations)")
    elif best >= 8:
        print("  configs/low_vram.yaml (4-bit NF4, ~7 GB) - pip install -e '.[quant]'")
    else:
        print("  < 8 GB VRAM: use Qwen/Qwen3-VL-4B-Instruct or -2B-Instruct with low_vram.yaml,")
        print("  or a GGUF via llama.cpp (configs/edge_llamacpp.yaml).")


if __name__ == "__main__":
    main()
