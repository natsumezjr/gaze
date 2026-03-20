from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# EllSeg 官方权重名称
ELLSEG_MODEL_FILENAME: str = "all.git_ok"

# 权重搜索目录
ELLSEG_MODEL_SEARCH_DIRS: List[str] = [
    PROJECT_ROOT / "weights",
    PROJECT_ROOT / "EllSeg/weights",
]

# EllSeg 仓库位置
ELLSEG_REPO_ROOT: Optional[str] = None
ELLSEG_REPO_SEARCH_DIRS: List[str] = [
    PROJECT_ROOT / "EllSeg",
]

# backend
ELLSEG_BACKEND: str = "official"

# YuNet
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

YUNET_MODEL_SEARCH_DIRS = [
    PROJECT_ROOT / "weights",
    # 默认将项目内的配置资源目录纳入搜索范围（避免把模型硬耦合到某个运行目录）
    PROJECT_ROOT / "project" / "config",
]