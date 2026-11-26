"""QLoRA fine-tune of Qwen3-VL as the robot's behaviour + local planner (VLM venv).

    .venv/bin/python scripts/finetune_planner.py                      # train on runs/planner_data
    .venv/bin/python scripts/finetune_planner.py --eval-only --adapter runs/planner_lora
    .venv/bin/python scripts/finetune_planner.py --eval-only            # the base model, same prompt

Data: sim/gen_planner_data.py (camera image with the route painted on the road -> JSON answer).
The base model is loaded in 4-bit NF4; LoRA adapters go on the language model's attention and MLP
projections (the vision encoder stays frozen). Only the answer tokens are trained on.
Evaluation: action accuracy (and per class), offset error when the action is right, blocker-box IoU,
answer latency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from talk2drive.vlm_planner import FT_PROMPT, parse_plan_ft  # noqa: E402


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answer_text(target: dict) -> str:
    """Blocker first, then the decision: the model locates the object before choosing a side."""
    d = {"blocker": target.get("blocker"), "action": target["action"], "offset_m": target["offset_m"]}
    return json.dumps(d, separators=(", ", ": "))


DECISION_WEIGHT = 5.0  # loss weight on the action / offset tokens vs the box digits


def decision_mask(processor, ids, n_prompt: int):
    """1 for answer tokens from '"action"' onwards (the decision), 0 elsewhere."""
    import torch

    w = torch.zeros_like(ids, dtype=torch.float32)
    text = ""
    for k in range(n_prompt, ids.shape[1]):
        text += processor.tokenizer.decode(ids[0, k:k + 1])
        if '"action"' in text:
            w[0, k - 2:] = 1.0  # include the key's own tokens
            break
    return w


def messages_for(image, with_answer: str | None = None) -> list[dict]:
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": FT_PROMPT}]}]
    if with_answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": with_answer}]})
    return msgs


def encode(processor, image, answer: str | None):
    """Model inputs; with an answer, labels masked to the answer tokens only."""
    from qwen_vl_utils import process_vision_info

    prompt_msgs = messages_for(image)
    images, _ = process_vision_info(prompt_msgs, image_patch_size=16)
    prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    if answer is None:
        return processor(text=[prompt_text], images=images, return_tensors="pt", do_resize=False)
    full_text = prompt_text + answer + "<|im_end|>\n"
    enc = processor(text=[full_text], images=images, return_tensors="pt", do_resize=False)
    n_prompt = processor(text=[prompt_text], images=images, return_tensors="pt", do_resize=False)["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    enc["labels"] = labels
    enc["n_prompt"] = n_prompt
    return enc


def load_model(model_id: str, adapter: str | None, train: bool):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelCls
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = ModelCls.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0", quantization_config=q)
    processor = AutoProcessor.from_pretrained(model_id)
    if train:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                         target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    elif adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    else:
        model.eval()
    return model, processor


def evaluate(model, processor, records: list[dict], root: Path, limit: int | None = None) -> dict:
    import torch
    from PIL import Image

    recs = records[:limit] if limit else records
    rows = []
    for r in recs:
        img = Image.open(root / r["image"]).convert("RGB")
        if max(img.size) > 960:  # same size as training and the run-time server (media.image_max_side)
            img = img.resize((960, round(img.size[1] * 960 / img.size[0])))
        enc = encode(processor, img, None).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=80, do_sample=False)
        ms = (time.perf_counter() - t0) * 1e3
        text = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        pred = parse_plan_ft(text)
        tgt = r.get("target")
        correct_set = r.get("correct") or [tgt["action"]]
        row = {"image": r["image"], "truth": correct_set, "pred": pred["action"] if pred else "unparsed", "ms": ms, "raw": text}
        row["ok"] = row["pred"] in correct_set
        if tgt and pred and row["ok"] and tgt["action"].startswith("bypass"):
            row["offset_err"] = abs(pred.get("offset_m", 0.0) - tgt["offset_m"])
        if tgt and pred and tgt.get("blocker") and pred.get("blocker") and pred["blocker"].get("bbox_2d"):
            row["iou"] = iou(tgt["blocker"]["bbox_2d"], pred["blocker"]["bbox_2d"])
        rows.append(row)
    classes = {}
    for row in rows:
        for c in row["truth"][:1]:
            classes.setdefault(c, []).append(row["ok"])
    off = [r["offset_err"] for r in rows if "offset_err" in r]
    ious = [r["iou"] for r in rows if "iou" in r]
    return {"n": len(rows), "accuracy": sum(r["ok"] for r in rows) / max(1, len(rows)),
            "by_class": {k: round(sum(v) / len(v), 3) for k, v in sorted(classes.items())},
            "offset_abs_err_m": (sum(off) / len(off)) if off else None,
            "blocker_iou": (sum(ious) / len(ious)) if ious else None,
            "mean_ms": sum(r["ms"] for r in rows) / max(1, len(rows)), "rows": rows}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="runs/planner_data")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default="runs/planner_lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--eval-n", type=int, default=200, help="validation samples scored by generation after training")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--testset", default="runs/planner_testset", help="hand-made 36-image set (labels.json) also scored")
    args = ap.parse_args()

    import torch
    from PIL import Image

    root = REPO / args.data
    out = REPO / args.out
    train = load_records(root / "train.jsonl")
    val = load_records(root / "val.jsonl")
    print(f"[ft] {len(train)} train, {len(val)} val samples", flush=True)

    if args.eval_only:
        model, processor = load_model(args.model, args.adapter, train=False)
    else:
        model, processor = load_model(args.model, None, train=True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
        steps = int(len(train) * args.epochs)
        n_opt = math.ceil(steps / args.accum)
        warm = max(1, n_opt // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / max(1, n_opt)))))
        model.train()
        rng = random.Random(0)
        order = []
        while len(order) < steps:
            idx = list(range(len(train)))
            rng.shuffle(idx)
            order += idx
        order = order[:steps]
        t0, run_loss = time.time(), []
        for k, i in enumerate(order):
            r = train[i]
            img = Image.open(root / r["image"]).convert("RGB")
            enc = encode(processor, img, answer_text(r["target"]))
            n_prompt = enc.pop("n_prompt")
            enc = enc.to(model.device)
            labels = enc.pop("labels")
            wts = 1.0 + (DECISION_WEIGHT - 1.0) * decision_mask(processor, labels.clamp(min=0), n_prompt).to(model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits[:, :-1].float()
            tgt = labels[:, 1:]
            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                                   ignore_index=-100, reduction="none").view_as(tgt)
            w = wts[:, 1:] * (tgt != -100)
            loss = (ce * w).sum() / w.sum() / args.accum
            loss.backward()
            run_loss.append(loss.item() * args.accum)
            if (k + 1) % args.accum == 0 or k == steps - 1:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if (k + 1) % 100 == 0:
                el = time.time() - t0
                print(f"[ft] {k + 1}/{steps}  loss {sum(run_loss[-100:]) / len(run_loss[-100:]):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {el / (k + 1):.2f} s/sample, ~{(steps - k - 1) * el / (k + 1) / 60:.0f} min left",
                      flush=True)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        (out / "train_info.json").write_text(json.dumps({"model": args.model, "samples": steps, "epochs": args.epochs,
                                                          "lr": args.lr, "accum": args.accum, "prompt": FT_PROMPT,
                                                          "minutes": (time.time() - t0) / 60}, indent=1))
        print(f"[ft] adapter saved to {out}", flush=True)
        model.eval()

    res = {"val": evaluate(model, processor, val, root, args.eval_n)}
    ts = REPO / args.testset / "labels.json"
    if ts.exists():
        res["testset"] = evaluate(model, processor, json.loads(ts.read_text()), REPO / args.testset)
    tag = "lora" if (args.adapter or not args.eval_only) else "base"
    dest = (out if not args.eval_only or args.adapter else REPO / "runs") / f"eval_{tag}.json"
    dest.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"[ft] {k}: " + json.dumps({kk: vv for kk, vv in v.items() if kk != "rows"}), flush=True)


if __name__ == "__main__":
    main()
