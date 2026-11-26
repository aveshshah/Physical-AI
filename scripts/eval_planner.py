"""Score the VLM planner's decisions on the labelled test set (sim/m3_planner_testset.py). VLM venv.

    .venv/bin/python scripts/eval_planner.py --url http://127.0.0.1:8766
    .venv/bin/python scripts/eval_planner.py --prompt-file my_prompt.txt
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from talk2drive.vlm_planner import PLANNER_PROMPT, parse_plan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="runs/planner_testset")
    ap.add_argument("--url", default="http://127.0.0.1:8766")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else PLANNER_PROMPT.format(width=1.2)
    root = REPO / args.set
    labels = json.loads((root / "labels.json").read_text())
    model = json.loads(urllib.request.urlopen(args.url + "/health", timeout=5).read())["model"]
    rows, by_dist, by_kind = [], defaultdict(list), defaultdict(list)
    for lb in labels:
        b = base64.b64encode((root / lb["image"]).read_bytes()).decode()
        req = urllib.request.Request(args.url + "/ground", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"image_png_b64": b, "query": "", "prompt": prompt}).encode())
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        plan = parse_plan(r["raw"])
        act = plan["action"] if plan else "unparsed"
        ok = act in lb["correct"]
        rows.append(dict(image=lb["image"], kind=lb["kind"], distance=lb["distance_m"], answer=act, correct=ok, ms=r["ms"],
                         raw=r["raw"]))
        by_dist[lb["distance_m"]].append(ok)
        by_kind[lb["kind"]].append(ok)
        print(f"{'ok ' if ok else 'BAD'} {lb['image']:32s} -> {act:13s} ({r['ms']:.0f} ms)", flush=True)
    acc = sum(r["correct"] for r in rows) / len(rows)
    summary = {"model": model, "accuracy": acc, "n": len(rows),
               "by_distance": {f"{int(d)} m": sum(v) / len(v) for d, v in sorted(by_dist.items())},
               "by_situation": {k: sum(v) / len(v) for k, v in by_kind.items()},
               "mean_ms": sum(r["ms"] for r in rows) / len(rows)}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
