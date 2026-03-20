"""
Lightweight model/repo locator utility.

职责：提供稳定、可复用的“模型文件定位 / 仓库根目录定位”逻辑。
该模块刻意保持轻量，供 face detector / EllSeg 等组件通过 config 指定文件名与搜索目录后复用。

注意：本次仅补充职责说明，不改变任何查找逻辑与行为。
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ModelLocateError(FileNotFoundError):
    filename: str
    searched_dirs: List[str]
    hint: Optional[str] = None


def _normalize_dirs(dirs: Sequence[PathLike]) -> List[Path]:
    return [Path(d).expanduser().resolve() for d in dirs]


def find_model_file(model_filename: str, search_dirs: Sequence[PathLike]) -> Path:
    dirs = _normalize_dirs(search_dirs)

    for d in dirs:
        cand = d / model_filename
        if cand.exists():
            return cand.resolve()

    raise ModelLocateError(
        filename=model_filename,
        searched_dirs=[str(d) for d in dirs],
        hint="model file not found",
    )


def find_repo_root(
    explicit_root: Optional[PathLike],
    search_dirs: Sequence[PathLike],
) -> Path:

    if explicit_root:
        root = Path(explicit_root).resolve()
        if root.exists():
            return root

    dirs = _normalize_dirs(search_dirs)

    for d in dirs:
        if d.exists():
            return d

    raise ModelLocateError(
        filename="__repo_root__",
        searched_dirs=[str(d) for d in dirs],
    )