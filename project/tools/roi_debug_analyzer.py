"""
ROI debug jsonl analyzer.

按固定格式输出每一行 ROI 状态，便于和 crop 文件名、行号对应排查。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _fmt_list(value: Any) -> str:
    if isinstance(value, list):
        return str(value)
    if value is None:
        return "[]"
    if isinstance(value, tuple):
        return str(list(value))
    return str(value)


def _build_entry_block(line_no: int, rec: Dict[str, Any]) -> str:
    mode = str(rec["current_mode"])

    roi = _fmt_list(rec["final_roi"])
    union_bbox = _fmt_list(rec["union_mask_bbox"])

    geometry_confidence = rec["geometry_confidence"]
    cg_2 = _fmt_float(geometry_confidence, 2)

    face_stable = _fmt_bool(rec["face_bbox_is_stable"])
    shift = _fmt_float(rec["face_bbox_corner_max_shift"], 4)
    face_bbox = _fmt_list(rec["face_bbox"])

    track_from_previous_bbox_applied = bool(rec["track_from_previous_bbox_applied"])

    # Output as TSV row (tab-separated), no frame_id.
    return "\t".join(
        [
            f"L{line_no}",
            str(mode),
            str(cg_2),
            str(face_stable),
            str(shift),
            _fmt_bool(track_from_previous_bbox_applied),
            face_bbox,
            roi,
            union_bbox,
        ]
    )


def analyze_roi_debug_jsonl(
    jsonl_path: str,
    *,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> List[str]:
    """
    读取 ROI debug jsonl 并返回格式化输出块列表。

    Args:
        jsonl_path: jsonl 文件路径
        start_line: 起始行（含，1-based）
        end_line: 结束行（含，1-based），None 表示到文件末尾
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    blocks: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            if idx < start_line:
                continue
            if end_line is not None and idx > end_line:
                break
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            blocks.append(_build_entry_block(idx, rec))
    return blocks


def _iter_all_blocks(paths: Iterable[str], start_line: int, end_line: Optional[int]) -> Iterable[str]:
    for p in paths:
        yield f"\n===== {p} ====="
        yield "L\tmode\tcg\tface_stable\tshift\ttrack_from_previous_bbox_applied\tface_bbox\troi\tunion_bbox"
        for block in analyze_roi_debug_jsonl(p, start_line=start_line, end_line=end_line):
            yield block


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze roi_debug_*.jsonl and print compact formatted report.")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="One or more jsonl paths, e.g. project/debug/roi/roi_debug_left.jsonl",
    )
    parser.add_argument("--start-line", type=int, default=1, help="Start line (1-based, inclusive)")
    parser.add_argument("--end-line", type=int, default=None, help="End line (1-based, inclusive)")
    args = parser.parse_args()

    for line in _iter_all_blocks(args.paths, args.start_line, args.end_line):
        print(line)


if __name__ == "__main__":
    main()

