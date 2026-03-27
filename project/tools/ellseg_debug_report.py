from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


SORT_MODES = {
    "pupil_minus_iris_dist": {"key": "pupil_minus_iris_dist", "reverse": True},
    "iris_edge_margin_min": {"key": "iris_edge_margin_min", "reverse": False},
    "pupil_component_count": {"key": "pupil_component_count", "reverse": True},
}


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"records.jsonl not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            records.append(json.loads(s))
    return records


def _metric_value(rec: Dict[str, Any], key: str) -> float:
    v = rec.get(key)
    if v is None:
        return -1e9 if key != "iris_edge_margin_min" else 1e9
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1e9 if key != "iris_edge_margin_min" else 1e9


def _load_frame_image(base_dir: Path, sub_dir: str, frame_idx: int) -> np.ndarray:
    p = base_dir / sub_dir / f"frame_{frame_idx:06d}.jpg"
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    return img


def _annotate(img: np.ndarray, lines: List[str]) -> np.ndarray:
    canvas = img.copy()
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        y += 22
    return canvas


def _resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    w = int(round(img.shape[1] * float(h) / float(max(1, img.shape[0]))))
    return cv2.resize(img, (max(1, w), h), interpolation=cv2.INTER_AREA)


def build_report(stability_dir: Path, sort_by: str, top_k: int) -> Tuple[List[Dict[str, Any]], Path]:
    records = _read_records(stability_dir / "records.jsonl")
    spec = SORT_MODES[sort_by]
    key = str(spec["key"])
    reverse = bool(spec["reverse"])

    sorted_records = sorted(records, key=lambda r: _metric_value(r, key), reverse=reverse)
    picks = sorted_records[: int(max(1, top_k))]

    out_dir = stability_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[np.ndarray] = []
    for rec in picks:
        frame_idx = int(rec.get("frame_idx", -1))
        roi_vis = _load_frame_image(stability_dir, "roi_vis", frame_idx)
        mask_color = _load_frame_image(stability_dir, "mask_color", frame_idx)
        mask_debug = _load_frame_image(stability_dir, "mask_component_debug", frame_idx)
        target_h = max(roi_vis.shape[0], mask_color.shape[0], mask_debug.shape[0], 200)
        roi_vis = _resize_to_h(roi_vis, target_h)
        mask_color = _resize_to_h(mask_color, target_h)
        mask_debug = _resize_to_h(mask_debug, target_h)

        info = np.zeros((target_h, 560, 3), dtype=np.uint8)
        info = _annotate(
            info,
            [
                f"frame={frame_idx} valid={bool(rec.get('valid', False))}",
                f"{sort_by}={rec.get(key)}",
                f"dist={rec.get('pupil_minus_iris_dist')} comp(p/i)={rec.get('pupil_component_count')}/{rec.get('iris_component_count')}",
                f"area(p/i)={rec.get('pupil_mask_area')}/{rec.get('iris_mask_area')}",
                f"margin_min(p/i)={rec.get('pupil_edge_margin_min')}/{rec.get('iris_edge_margin_min')}",
                f"mask_ok={rec.get('mask_ok')} reason={rec.get('mask_reject_reason')}",
                f"geometry_ok={rec.get('geometry_ok')} reason={rec.get('geometry_reject_reason')}",
            ],
        )
        row = np.hstack([roi_vis, mask_color, mask_debug, info])
        rows.append(row)

    if rows:
        report_img = np.vstack(rows)
    else:
        report_img = np.zeros((200, 800, 3), dtype=np.uint8)
        cv2.putText(report_img, "No records found", (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    out_img = out_dir / f"top_{top_k}_{sort_by}.jpg"
    cv2.imwrite(str(out_img), report_img)
    out_json = out_dir / f"top_{top_k}_{sort_by}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(picks, f, ensure_ascii=False, indent=2)
    return picks, out_img


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bad-frame report from ellseg_stability output directory.")
    parser.add_argument("stability_dir", type=str, help="Path to ellseg_stability_xxx directory")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="pupil_minus_iris_dist",
        choices=list(SORT_MODES.keys()),
        help="Sort metric",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-K frames to export")
    args = parser.parse_args()

    picks, out_img = build_report(Path(args.stability_dir), sort_by=args.sort_by, top_k=args.top_k)
    print(f"[INFO] selected={len(picks)}")
    print(f"[INFO] report_image={out_img}")


if __name__ == "__main__":
    main()
