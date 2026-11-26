"""Visual grounding with Qwen3-VL: "where is <thing> in this image?" -> bounding box.

Qwen3-VL answers grounding prompts with boxes in coordinates normalised to 0..1000 (independent of
the image size the processor resized to). ``parse_boxes`` accepts the JSON it usually writes, code
fences included, and plain ``[x1, y1, x2, y2]`` lists.
"""

from __future__ import annotations

import json
import re

GROUND_PROMPT = (
    "Locate {query} in the image. Output JSON only, a list with one entry per matching object: "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "<short name>"}}]. '
    "If there is no such object, output []."
)


def parse_boxes(text: str) -> list[dict]:
    """[{bbox_2d: [x1, y1, x2, y2] (0..1000), label}] from a model answer; [] if none."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        lists = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", t)
        return [{"bbox_2d": [float(v) for v in b], "label": ""} for b in lists]
    if isinstance(data, dict):
        data = [data]
    out = []
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and isinstance(d.get("bbox_2d"), (list, tuple)) and len(d["bbox_2d"]) == 4:
            out.append({"bbox_2d": [float(v) for v in d["bbox_2d"]], "label": str(d.get("label", ""))})
        elif isinstance(d, (list, tuple)) and len(d) == 4 and all(isinstance(v, (int, float)) for v in d):
            out.append({"bbox_2d": [float(v) for v in d], "label": ""})
    return out


def to_pixels(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    """0..1000 normalised box -> pixel box (x1, y1, x2, y2), clipped to the image."""
    x1, y1, x2, y2 = box
    sx, sy = width / 1000.0, height / 1000.0
    x1, x2 = sorted((x1 * sx, x2 * sx))
    y1, y2 = sorted((y1 * sy, y2 * sy))
    return max(0.0, x1), max(0.0, y1), min(width - 1.0, x2), min(height - 1.0, y2)
