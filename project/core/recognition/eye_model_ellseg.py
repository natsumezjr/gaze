"""
EllSeg 眼部 ROI 模型（主链）：适配推理 + 原生几何输出。

----------------------------
EllSeg 官方调用约定（以 README + evaluate_ellseg.py 为准）
----------------------------
1. 官方无单独 API 文档，以仓库 README 与 evaluate_ellseg.py 为准。
2. 标准推理入口语义：
   - README 推荐 evaluate_ellseg.py 作为快速推理入口。
   - evaluate_ellseg_on_image(frame, model) 返回：seg_map, latent, pupil_ellipse, iris_ellipse。
3. 输入约定：
   - 输入为灰度图；
   - tensor shape [1, 1, H, W]；
   - 官方流程默认 preprocess_frame(frame, (240, 320), align_width)。
4. 输出约定：
   - seg_map 为官方原生输出，不自行伪造；
   - 椭圆来源：a) network ellipse（中心来自 segmentation，轴/角来自回归头），
     b) ellipse-from-output（对 seg_map 做 ElliFit/RANSAC）。
5. seg_map 标签约定（与 evaluate_ellseg.py 一致）：
   - seg_map == 2 : pupil
   - seg_map == 1 : iris
   - 其他值为非 pupil/非 iris 区域。
6. README 的 --save_maps 仍为 "coming soon"，不假设官方标准落盘接口；内存中 seg_map 即官方标准输出。

本文件：保持 EllSeg 原生 seg_map / pupil_ellipse / iris_ellipse 等关键几何输出，只向上游提供模型原生结果所需的数据。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

from project.config.logging_config import setup_logging
from project.config.model_config import (
    ELLSEG_BACKEND,
    ELLSEG_MODEL_FILENAME,
    ELLSEG_MODEL_SEARCH_DIRS,
    ELLSEG_REPO_ROOT,
    ELLSEG_REPO_SEARCH_DIRS,
)
from project.data.data_models import Ellipse2D, Point2D

from project.data.model_locator import ModelLocateError, find_model_file, find_repo_root

logger = setup_logging(__name__, logging.DEBUG)


def ellipse_to_cardinal_points(ellipse: Ellipse2D) -> Tuple[Point2D, Point2D, Point2D, Point2D]:
    """
    从 fitted iris ellipse 推导 left/right/top/bottom 四个 cardinal 点。
    用于 compatibility export，非 EllSeg 原生 landmarks。
    """
    import math

    a = ellipse.major_axis * 0.5
    b = ellipse.minor_axis * 0.5
    angle_rad = math.radians(ellipse.angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    # 局部坐标: left=(-a,0), right=(a,0), top=(0,-b), bottom=(0,b)
    pts_local = [(-a, 0), (a, 0), (0, -b), (0, b)]
    out = []
    for lx, ly in pts_local:
        gx = ellipse.cx + lx * c - ly * s
        gy = ellipse.cy + lx * s + ly * c
        out.append(Point2D(x=gx, y=gy))
    return out[0], out[1], out[2], out[3]  # left, right, top, bottom


@dataclass
class EllSegResult:
    """
    EllSeg 原生输出结构。
    - valid: 是否有效
    - segmentation_mask: 分割 mask（可选，HxW）
    - pupil_center: 瞳孔中心（椭圆拟合）
    - pupil_ellipse: 瞳孔椭圆
    - iris_ellipse: 虹膜椭圆
    - pupil_confidence: 瞳孔置信度
    - iris_confidence: 虹膜置信度
    """

    valid: bool = False
    segmentation_mask: Optional[np.ndarray] = None
    pupil_center: Optional[Point2D] = None
    pupil_ellipse: Optional[Ellipse2D] = None
    iris_ellipse: Optional[Ellipse2D] = None
    pupil_confidence: Optional[float] = None
    iris_confidence: Optional[float] = None
    raw_output: Optional[dict] = field(default_factory=dict)


def _fit_ellipse_from_mask(mask: np.ndarray, min_points: int = 5) -> Tuple[Optional[Ellipse2D], Optional[Point2D], float]:
    """
    从二值 mask 拟合椭圆，返回 (Ellipse2D, center, confidence)。
    """
    binary = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < min_points:
        return None, None, 0.0
    try:
        ellipse = cv2.fitEllipse(largest)
        (cx, cy), (ma, mi), angle = ellipse
        major = max(ma, mi)
        minor = min(ma, mi)
        conf = min(1.0, float(cv2.contourArea(largest)) / max(1.0, float(mask.size) * 0.01))
        el = Ellipse2D(
            cx=float(cx),
            cy=float(cy),
            major_axis=float(major),
            minor_axis=float(minor),
            angle_deg=float(angle),
            confidence=conf,
        )
        center = Point2D(x=el.cx, y=el.cy)
        return el, center, conf
    except Exception as e:
        logger.debug("ellipse fit failed: %s", e)
        return None, None, 0.0


class EllSegAdapter:
    """
    EllSeg 适配器：分割 mask -> 椭圆拟合 -> EllSegResult。
    """

    def __init__(self) -> None:
        self._backend = ELLSEG_BACKEND
        logger.info("EllSegAdapter backend from config: %s", self._backend)

        if self._backend == "legacy_onnx":
            # legacy ONNX：仍然输出 EllSegResult（mask + fitted ellipses）
            self._seg = PupilSegmentationModel()  # type: ignore[name-defined]
            self._official_model = None
            self._device = None
            return

        if self._backend != "official":
            raise ValueError(f"Unknown ELLSEG_BACKEND: {self._backend}")

        # official：定位权重文件 + 定位 EllSeg 仓库根目录 + 加载模型
        logger.info("model filename from config: %s", ELLSEG_MODEL_FILENAME)
        try:
            weights_path = find_model_file(ELLSEG_MODEL_FILENAME, ELLSEG_MODEL_SEARCH_DIRS)
        except ModelLocateError as e:
            raise FileNotFoundError(f"EllSeg weights not found.\n{e}") from None
        logger.info("resolved model path: %s", str(weights_path))

        try:
            repo_root = find_repo_root(ELLSEG_REPO_ROOT, ELLSEG_REPO_SEARCH_DIRS)
        except ModelLocateError as e:
            raise FileNotFoundError(
                "EllSeg repo root not found.\n"
                f"{e}\n"
                f"Expected repo directory name like: {ELLSEG_REPO_SEARCH_DIRS}"
            ) from None
        logger.info("resolved EllSeg repo root: %s", str(repo_root))

        try:
            self._official_model, self._device = self._load_official_ellseg(repo_root, weights_path)
            self._seg = None
            logger.info("model load success: %s", str(weights_path))
        except Exception as e:
            raise RuntimeError(f"EllSeg model load error (file found but load failed): {weights_path}\n{e}") from e

    @staticmethod
    def _load_official_ellseg(repo_root: Path, weights_path: Path):
        """
        对齐 EllSeg 官方“权重加载方式”的项目内封装。
        """
        import sys

        try:
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "No module named 'torch'. EllSeg official backend requires PyTorch.\n"
                "Fix options:\n"
                "- Install torch in the SAME environment you run this project.\n"
                "- Or set ELLSEG_BACKEND='legacy_onnx' in project/config/model_config.py to use the temporary ONNX backend."
            ) from e

        # EllSeg 官方仓库较老，部分代码仍使用 np.int/np.bool 等旧别名。
        # 这里做最小兼容 shim，避免要求整个项目降级到旧 numpy（Python 3.12 下也不可行）。
        import numpy as np

        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]
        if not hasattr(np, "bool"):
            np.bool = bool  # type: ignore[attr-defined]
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]

        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from modelSummary import model_dict  # type: ignore

        net_dict = torch.load(str(weights_path), map_location="cpu")
        model = model_dict["ritnet_v3"]
        model.load_state_dict(net_dict["state_dict"], strict=True)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, device

    def infer(self, eye_patch_rgb: np.ndarray) -> Optional[EllSegResult]:
        """
        Args:
            eye_patch_rgb: HxWx3 RGB uint8 (ROI crop)

        Returns:
            EllSegResult 或 None（失败时）
        """
        try:
            H, W = eye_patch_rgb.shape[:2]
            if H <= 0 or W <= 0:
                return None

            if self._backend == "legacy_onnx":
                pupil_mask, iris_mask, score = self._seg.infer(eye_patch_rgb)  # type: ignore[union-attr]
                if pupil_mask is None or iris_mask is None:
                    logger.warning("EllSegAdapter(legacy_onnx): mask none")
                    return None

                # 合并 mask 用于 segmentation_mask
                seg_mask = np.zeros((H, W), dtype=np.float32)
                seg_mask = np.maximum(seg_mask, pupil_mask)
                seg_mask = np.maximum(seg_mask, iris_mask)

                # pupil ellipse
                pupil_el, pupil_center, pupil_conf = _fit_ellipse_from_mask(pupil_mask)
                if pupil_el is None or pupil_center is None:
                    logger.warning("EllSegAdapter(legacy_onnx): pupil ellipse invalid")
                    return None
                logger.debug(
                    "ellseg native output ready: pupil ellipse valid cx=%.1f cy=%.1f conf=%.3f",
                    pupil_center.x,
                    pupil_center.y,
                    pupil_conf,
                )

                # iris ellipse
                iris_el, _, iris_conf = _fit_ellipse_from_mask(iris_mask)
                if iris_el is None:
                    logger.warning("EllSegAdapter(legacy_onnx): iris ellipse invalid")
                    return None
                logger.debug(
                    "ellseg native output ready: iris ellipse valid cx=%.1f cy=%.1f conf=%.3f",
                    iris_el.cx,
                    iris_el.cy,
                    iris_conf,
                )

                logger.debug("segmentation mask ready H=%d W=%d", H, W)
                return EllSegResult(
                    valid=True,
                    segmentation_mask=seg_mask,
                    pupil_center=pupil_center,
                    pupil_ellipse=pupil_el,
                    iris_ellipse=iris_el,
                    pupil_confidence=pupil_conf,
                    iris_confidence=iris_conf,
                    raw_output={"score": score, "backend": "legacy_onnx"},
                )

            # official EllSeg
            seg_map, pupil_el, iris_el = self._infer_official(eye_patch_rgb)
            if seg_map is None or pupil_el is None or iris_el is None:
                return None
            logger.debug("ellseg native output ready")
            logger.debug("segmentation mask ready H=%d W=%d", H, W)
            return EllSegResult(
                valid=True,
                segmentation_mask=seg_map,
                pupil_center=Point2D(x=pupil_el.cx, y=pupil_el.cy),
                pupil_ellipse=pupil_el,
                iris_ellipse=iris_el,
                pupil_confidence=pupil_el.confidence,
                iris_confidence=iris_el.confidence,
                raw_output={"backend": "official"},
            )
        except Exception as e:
            logger.warning("EllSegAdapter.infer 失败: %s", e)
            return None

    def _infer_official(self, eye_patch_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Ellipse2D], Optional[Ellipse2D]]:
        """
        基于 EllSeg 官方代码的最小推理路径（项目内封装）。
        """
        import math
        import torch

        if self._official_model is None or self._device is None:
            return None, None, None

        H, W = eye_patch_rgb.shape[:2]
        if H <= 0 or W <= 0:
            return None, None, None

        gray = cv2.cvtColor(eye_patch_rgb, cv2.COLOR_RGB2GRAY)
        in_w, in_h = 320, 240  # EllSeg 官方脚本默认尺寸
        resized = cv2.resize(gray, (in_w, in_h), interpolation=cv2.INTER_LANCZOS4)
        mu = float(resized.mean())
        sigma = float(resized.std()) if float(resized.std()) > 1e-6 else 1.0
        norm = (resized.astype(np.float32) - mu) / sigma
        inp = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(torch.float32).to(self._device)

        with torch.no_grad():
            model = self._official_model
            x4, x3, x2, x1, x = model.enc(inp)  # type: ignore[attr-defined]
            el_out = model.elReg(x, 0)  # type: ignore[attr-defined]
            seg_out = model.dec(x4, x3, x2, x1, x)  # type: ignore[attr-defined]

        # imports from official repo (already on sys.path in _load_official_ellseg)
        from helperfunctions import my_ellipse  # type: ignore
        from loss import get_seg2ptLoss  # type: ignore
        from utils import get_predictions  # type: ignore

        seg_out_cpu = seg_out.detach().cpu()
        el_out_cpu = el_out.squeeze().detach().cpu()

        seg_map_240x320 = get_predictions(seg_out_cpu).squeeze().numpy().astype(np.uint8)

        # Get EllSeg proposed ellipse predictions (evaluate_ellseg.py, ellseg_ellipses=1)
        _, norm_pupil_center = get_seg2ptLoss(seg_out_cpu[:, 2, ...], torch.zeros(2,), temperature=4)
        _, norm_iris_center = get_seg2ptLoss(-seg_out_cpu[:, 0, ...], torch.zeros(2,), temperature=4)
        norm_pupil_center = norm_pupil_center.squeeze().cpu()
        norm_iris_center = norm_iris_center.squeeze().cpu()

        norm_pupil_ellipse = torch.cat([norm_pupil_center, el_out_cpu[7:10]])
        norm_iris_ellipse = torch.cat([norm_iris_center, el_out_cpu[2:5]])

        # Transformation H: normalized [-1,1] -> pixel coords (W,H)
        Hm = np.array(
            [[in_w / 2.0, 0.0, in_w / 2.0], [0.0, in_h / 2.0, in_h / 2.0], [0.0, 0.0, 1.0]]
        )

        pup_param = my_ellipse(norm_pupil_ellipse.numpy()).transform(Hm)[0][:-1]  # [cx,cy,a,b,theta(rad)]
        iri_param = my_ellipse(norm_iris_ellipse.numpy()).transform(Hm)[0][:-1]

        if np.any(np.asarray(pup_param) == -1) or np.any(np.asarray(iri_param) == -1):
            logger.warning("EllSegAdapter(official): ellipse invalid from network")
            return None, None, None

        # Scale back to ROI size
        sx = W / float(in_w)
        sy = H / float(in_h)

        def _to_ellipse2d(param: np.ndarray) -> Ellipse2D:
            cx, cy, a, b, theta = [float(x) for x in param.tolist()]
            # a/b are semi-axes (in helperfunctions.my_ellipse); convert to full axis lengths
            major = 2.0 * max(a * sx, b * sy)
            minor = 2.0 * min(a * sx, b * sy)
            ang_deg = float(theta * 180.0 / math.pi)
            return Ellipse2D(
                cx=cx * sx,
                cy=cy * sy,
                major_axis=major,
                minor_axis=minor,
                angle_deg=ang_deg,
                confidence=1.0,
            )

        pupil_el = _to_ellipse2d(np.asarray(pup_param))
        iris_el = _to_ellipse2d(np.asarray(iri_param))

        seg_map = cv2.resize(seg_map_240x320, (W, H), interpolation=cv2.INTER_NEAREST)
        return seg_map, pupil_el, iris_el


class EyeRoiModel:
    """
    眼部 ROI 模型：仅输出 EllSeg 原生几何（通过 EllSegAdapter）。
    """

    def __init__(self) -> None:
        self._adapter = EllSegAdapter()

    def infer(self, eye_patch_rgb: np.ndarray) -> Optional[EllSegResult]:
        # Keep EllSegResult as adapter-native output.
        return self._adapter.infer(eye_patch_rgb)


__all__ = [
    "EllSegResult",
    "ellipse_to_cardinal_points",
    "_fit_ellipse_from_mask",
    "EllSegAdapter",
    "EyeRoiModel",
]

