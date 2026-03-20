"""
Compatibility points schema helpers (主链兼容层，非 EllSeg 原生输出)。

本模块集中放置与“兼容点格式/顺序/计数”相关的常量与校验，避免污染主链 extractor。

重要约束：
- 不改变 compatibility export 语义：
  - pupil = 1 (EllSeg pupil_center)
  - iris = 4 (iris ellipse cardinal points)
  - optional landmarks = empty lists
- 不引入旧 Face Landmarker 的索引语义（468/473 等属于 LEGACY_FACE_LANDMARK_INDICES，不在此模块）
"""

from __future__ import annotations

from typing import Any, Dict, List

from project.data.data_models import LEGACY_COMPAT_POINT_SCHEMA

# Compatibility schema 键顺序与点数（保持对外语义不变）
CANONICAL_ORDER = tuple(LEGACY_COMPAT_POINT_SCHEMA.keys())
POINT_COUNTS = LEGACY_COMPAT_POINT_SCHEMA


def validate_fitting_dict(result: Dict[str, Dict[str, List[Any]]]) -> bool:
    """检查 result 是否包含左右眼且各类型点数正确（compatibility schema 完整性）。"""
    if not result or "left" not in result or "right" not in result:
        return False
    for side in ("left", "right"):
        eye = result[side]
        for ft, count in POINT_COUNTS.items():
            if ft not in eye or len(eye[ft]) != count:
                return False
    return True


def get_landmark_indices() -> Dict[str, Dict[str, List[int]]]:
    """
    返回 compatibility schema 的“顺序索引映射”，用于可视化等场景按类型上色。

    说明：
    - 这是当前 compatibility export 的顺序（按 CANONICAL_ORDER 串行展开）
    - 不是历史 Face Landmarker 的 468/473 等索引
    """
    out: Dict[str, Dict[str, List[int]]] = {"left": {}, "right": {}}
    idx = 0
    for side in ("left", "right"):
        for ft in CANONICAL_ORDER:
            n = POINT_COUNTS[ft]
            out[side][ft] = list(range(idx, idx + n))
            idx += n
    return out


__all__ = [
    "CANONICAL_ORDER",
    "POINT_COUNTS",
    "validate_fitting_dict",
    "get_landmark_indices",
]

