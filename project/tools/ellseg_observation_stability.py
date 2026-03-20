from __future__ import annotations

"""
Quantify EllSeg observation stability (pupil/iris geometry jitter).

Goal:
- Under a single gaze point (user keeps still), measure across frames:
  - pupil center frame-to-frame jitter (std of cx/cy)
  - iris center frame-to-frame jitter (std of cx/cy)
  - pupil-iris relative position stability (std of dx/dy, std of distance)
  - ellipse axes/angle jitter (std of major/minor/angle)

This tool is intentionally standalone:
- It uses EyeRoiModel + EllSeg native outputs only.
- It does NOT change any ROI strategies in the main pipeline.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from project.core.recognition.eye_model_ellseg import EyeRoiModel, EllSegResult
from project.core.recognition.runtime_visualization import (
    ellseg_segmentation_mask_to_bgr,
)


def clamp_roi(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, min(int(w), img_w - x))
    h = max(1, min(int(h), img_h - y))
    return x, y, w, h


def fit_circular_std_deg(angles_deg: np.ndarray) -> float:
    """
    Circular std for ellipse angle (0..180 can wrap).
    OpenCV fitEllipse returns angle in degrees in [0,180).
    We map to unit circle at 2x angle to make 0/180 consistent.
    """
    if angles_deg.size == 0:
        return 0.0
    ang = np.deg2rad(angles_deg.astype(np.float64))
    # Use doubled angle so that theta and theta+90 (180 span) are handled more consistently.
    z = np.exp(1j * 2.0 * ang)
    R = np.abs(z.mean())
    if R <= 1e-12:
        return float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))
    # Equivalent circular std estimate on the circle.
    return float(np.sqrt(-2.0 * np.log(R)))


def mean_std(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "rmse": 0.0}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    rmse = float(np.sqrt(np.mean((x - mean) ** 2)))
    return {"n": int(x.size), "mean": mean, "std": std, "rmse": rmse}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


class EllSegStabilityRunner:
    def __init__(
        self,
        device_index: int,
        side: str,
        split_half: bool,
        autofocus: bool,
        backend_dshow: bool,
        num_frames: int,
        interval_ms: float,
        output_root: str,
        roi: Optional[Tuple[int, int, int, int]],
        roi_zoom: int,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be left/right")
        self.side = side
        self.split_half = split_half
        self.num_frames = int(num_frames)
        self.interval_ms = float(interval_ms)
        self.roi_zoom = int(max(1, roi_zoom))
        self.roi = roi
        self.output_root = output_root

        backend = cv2.CAP_DSHOW if backend_dshow else None
        if backend is not None:
            self.cap = cv2.VideoCapture(device_index, backend)
            if not self.cap.isOpened():
                print("[WARN] CAP_DSHOW failed to open camera. Falling back to default backend...")
                self.cap = cv2.VideoCapture(device_index)
        else:
            self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index={device_index}")

        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        except Exception:
            pass

        self.eye_model = EyeRoiModel()

    def _split_left_right(self, frame: np.ndarray) -> np.ndarray:
        if not self.split_half:
            if self.side != "left":
                raise ValueError("split_half must be enabled for side=right")
            return frame
        h, w = frame.shape[:2]
        half = w // 2
        if self.side == "left":
            return frame[:, :half]
        return frame[:, half:]

    def _select_roi_interactively(self, eye_img: np.ndarray) -> Tuple[int, int, int, int]:
        win = f"Select ROI ({self.side}) - drag, press ENTER/SPACE"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(win, eye_img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win)
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise RuntimeError("ROI selection invalid (w/h <= 0)")
        img_h, img_w = eye_img.shape[:2]
        x, y, w, h = clamp_roi(x, y, w, h, img_w=img_w, img_h=img_h)
        return int(x), int(y), int(w), int(h)

    def _draw_roi_and_geometry(self, eye_img: np.ndarray, roi: Tuple[int, int, int, int], result: Optional[EllSegResult]) -> np.ndarray:
        x, y, w, h = roi
        vis = eye_img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if result is None or not result.valid:
            return vis

        # Draw ellipses on full eye half.
        if result.iris_ellipse is not None:
            el = result.iris_ellipse
            cx = int(round(el.cx + x))
            cy = int(round(el.cy + y))
            ax1 = int(round((el.major_axis * 0.5)))
            ax2 = int(round((el.minor_axis * 0.5)))
            cv2.ellipse(vis, (cx, cy), (max(1, ax1), max(1, ax2)), float(el.angle_deg), 0, 360, (0, 200, 255), 1)

        if result.pupil_ellipse is not None:
            el = result.pupil_ellipse
            cx = int(round(el.cx + x))
            cy = int(round(el.cy + y))
            ax1 = int(round((el.major_axis * 0.5)))
            ax2 = int(round((el.minor_axis * 0.5)))
            cv2.ellipse(vis, (cx, cy), (max(1, ax1), max(1, ax2)), float(el.angle_deg), 0, 360, (0, 255, 0), 1)

        if result.pupil_center is not None:
            pc = result.pupil_center
            cv2.circle(vis, (int(round(pc.x + x)), int(round(pc.y + y))), 3, (0, 255, 0), -1)

        return vis

    def run(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.output_root, f"ellseg_stability_{self.side}_{ts}")
        mask_dir = os.path.join(out_dir, "mask_color")
        vis_dir = os.path.join(out_dir, "roi_vis")
        ensure_dir(out_dir)
        ensure_dir(mask_dir)
        ensure_dir(vis_dir)

        records_path = os.path.join(out_dir, "records.jsonl")
        summary_path = os.path.join(out_dir, "summary.json")
        config_path = os.path.join(out_dir, "config.json")

        win_eye = f"EllSeg Stability Viewer ({self.side})"
        cv2.namedWindow(win_eye, cv2.WINDOW_NORMAL)

        # Grab first frame for ROI selection.
        while True:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                eye_img = self._split_left_right(frame)
                break
            time.sleep(0.01)

        eye_h, eye_w = eye_img.shape[:2]
        if self.roi is None:
            self.roi = self._select_roi_interactively(eye_img)
        else:
            x, y, w, h = self.roi
            self.roi = clamp_roi(x, y, w, h, img_w=eye_w, img_h=eye_h)

        roi = self.roi

        # Save config.
        config = {
            "side": self.side,
            "split_half": self.split_half,
            "device_index": None,
            "num_frames": self.num_frames,
            "interval_ms": self.interval_ms,
            "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
            "roi_zoom": self.roi_zoom,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Start capture: side={self.side}, roi={roi}, frames={self.num_frames}")

        valid_records = []
        n_total = 0
        n_valid = 0
        interrupted = False

        with open(records_path, "w", encoding="utf-8") as rec_f:
            try:
                while n_total < self.num_frames:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        time.sleep(0.01)
                        continue
                    n_total += 1

                    eye_img = self._split_left_right(frame)
                    x, y, w, h = roi
                    crop_bgr = eye_img[y : y + h, x : x + w]
                    if crop_bgr is None or crop_bgr.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    result = self.eye_model.infer(crop_rgb)

                    # Prepare visualization (saved every valid frame; mask requires native seg_map).
                    vis_full = self._draw_roi_and_geometry(eye_img, roi, result)
                    # Save ROI vis (zoomed)
                    vis_zoom = cv2.resize(
                        vis_full[y : y + h, x : x + w],
                        (max(1, w * self.roi_zoom), max(1, h * self.roi_zoom)),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    sharp = float(cv2.Laplacian(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                    cv2.putText(
                        vis_zoom,
                        f"frame={n_total} sharp={sharp:.1f}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Save per-frame overlay and mask (only if segmentation_mask exists).
                    out_vis_path = os.path.join(vis_dir, f"frame_{n_total:06d}.jpg")
                    cv2.imwrite(out_vis_path, vis_zoom)

                    record: Dict[str, Any] = {
                        "frame_idx": n_total,
                        "valid": bool(
                            result is not None
                            and result.valid
                            and result.iris_ellipse is not None
                            and result.pupil_ellipse is not None
                            and result.pupil_center is not None
                        ),
                        "sharp_laplacian_var": sharp,
                        "pupil_center_roi": None,
                        "pupil_major": None,
                        "pupil_minor": None,
                        "pupil_angle_deg": None,
                        "iris_center_roi": None,
                        "iris_major": None,
                        "iris_minor": None,
                        "iris_angle_deg": None,
                        "pupil_minus_iris_dx": None,
                        "pupil_minus_iris_dy": None,
                        "pupil_minus_iris_dist": None,
                        "mask_exists": bool(result is not None and result.segmentation_mask is not None),
                        "roi_xywh": [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])],
                        "crop_size_hw": [int(h), int(w)],
                    }

                    if record["valid"]:
                        assert result is not None
                        assert result.pupil_center is not None
                        assert result.iris_ellipse is not None
                        assert result.pupil_ellipse is not None

                        pc = result.pupil_center
                        ic = result.iris_ellipse

                        record["pupil_center_roi"] = [float(pc.x), float(pc.y)]
                        record["pupil_major"] = float(result.pupil_ellipse.major_axis)
                        record["pupil_minor"] = float(result.pupil_ellipse.minor_axis)
                        record["pupil_angle_deg"] = float(result.pupil_ellipse.angle_deg)

                        record["iris_center_roi"] = [float(ic.cx), float(ic.cy)]
                        record["iris_major"] = float(ic.major_axis)
                        record["iris_minor"] = float(ic.minor_axis)
                        record["iris_angle_deg"] = float(ic.angle_deg)

                        dx = float(pc.x - ic.cx)
                        dy = float(pc.y - ic.cy)
                        dist = float(np.hypot(dx, dy))
                        record["pupil_minus_iris_dx"] = dx
                        record["pupil_minus_iris_dy"] = dy
                        record["pupil_minus_iris_dist"] = dist

                        # Save mask_color if available.
                        mask_img = ellseg_segmentation_mask_to_bgr(result.segmentation_mask, dst_h=h, dst_w=w)
                        if mask_img is not None:
                            mask_path = os.path.join(mask_dir, f"frame_{n_total:06d}.jpg")
                            cv2.imwrite(mask_path, mask_img)

                        valid_records.append(record)
                        n_valid += 1

                    rec_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Live viewer + early quit.
                    cv2.imshow(win_eye, vis_zoom)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        break

                    if self.interval_ms > 0:
                        time.sleep(self.interval_ms / 1000.0)
            except KeyboardInterrupt:
                interrupted = True
                print("\n[INFO] Ctrl+C received, finalizing current capture...")

        # Final statistics on valid frames only.
        def arr(key: str) -> np.ndarray:
            vals = [r.get(key) for r in valid_records]
            return np.array([v for v in vals if v is not None], dtype=np.float64)

        pupil_cx = np.array([r["pupil_center_roi"][0] for r in valid_records], dtype=np.float64) if valid_records else np.array([], dtype=np.float64)
        pupil_cy = np.array([r["pupil_center_roi"][1] for r in valid_records], dtype=np.float64) if valid_records else np.array([], dtype=np.float64)
        iris_cx = np.array([r["iris_center_roi"][0] for r in valid_records], dtype=np.float64) if valid_records else np.array([], dtype=np.float64)
        iris_cy = np.array([r["iris_center_roi"][1] for r in valid_records], dtype=np.float64) if valid_records else np.array([], dtype=np.float64)

        dx = arr("pupil_minus_iris_dx")
        dy = arr("pupil_minus_iris_dy")
        dist = arr("pupil_minus_iris_dist")

        pupil_major = arr("pupil_major")
        pupil_minor = arr("pupil_minor")
        pupil_angle = arr("pupil_angle_deg")
        iris_major = arr("iris_major")
        iris_minor = arr("iris_minor")
        iris_angle = arr("iris_angle_deg")

        summary = {
            "n_total_frames": int(n_total),
            "n_valid_frames": int(n_valid),
            "interrupted_by_ctrl_c": bool(interrupted),
            "pupil_center_cx": mean_std(pupil_cx),
            "pupil_center_cy": mean_std(pupil_cy),
            "iris_center_cx": mean_std(iris_cx),
            "iris_center_cy": mean_std(iris_cy),
            "pupil_minus_iris_dx": mean_std(dx),
            "pupil_minus_iris_dy": mean_std(dy),
            "pupil_minus_iris_dist": mean_std(dist),
            "pupil_major": mean_std(pupil_major),
            "pupil_minor": mean_std(pupil_minor),
            "pupil_angle_circular_std_deg": float(fit_circular_std_deg(pupil_angle)),
            "iris_major": mean_std(iris_major),
            "iris_minor": mean_std(iris_minor),
            "iris_angle_circular_std_deg": float(fit_circular_std_deg(iris_angle)),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("[RESULT] EllSeg observation stability summary")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        self.cap.release()
        cv2.destroyAllWindows()


def parse_roi(s: str) -> Tuple[int, int, int, int]:
    # "x,y,w,h"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("roi must be 'x,y,w,h'")
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify EllSeg pupil/iris observation jitter across frames.")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--split-half", action="store_true", help="Split 3840x1080 side-by-side into left/right halves.")
    parser.add_argument("--no-split-half", dest="split_half", action="store_false")
    parser.set_defaults(split_half=True)

    parser.add_argument("--roi-xywh", type=str, default=None, help="Fixed ROI 'x,y,w,h' (skip interactive selection).")
    parser.add_argument("--roi-zoom", type=int, default=4, help="Zoom factor for saved ROI vis images.")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--interval-ms", type=float, default=0.0)
    parser.add_argument("--autofocus", action="store_true", help="Enable autofocus at start.")
    parser.add_argument("--no-autofocus", dest="autofocus", action="store_false")
    parser.set_defaults(autofocus=False)
    parser.add_argument("--backend-dshow", action="store_true", help="Use cv2.CAP_DSHOW first (Windows).")
    parser.add_argument("--output-root", type=str, default=None, help="Output root dir.")

    args = parser.parse_args()

    if args.output_root is None:
        # project/debug/... (works from repo root)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project/
        args.output_root = os.path.join(base_dir, "debug", "ellseg_stability")

    roi = parse_roi(args.roi_xywh) if args.roi_xywh else None

    runner = EllSegStabilityRunner(
        device_index=args.device_index,
        side=args.side,
        split_half=args.split_half,
        autofocus=args.autofocus,
        backend_dshow=args.backend_dshow,
        num_frames=args.num_frames,
        interval_ms=args.interval_ms,
        output_root=args.output_root,
        roi=roi,
        roi_zoom=args.roi_zoom,
    )
    runner.run()


if __name__ == "__main__":
    main()

