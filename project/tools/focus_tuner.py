from __future__ import annotations

"""
Focus tuning + manual ROI zoom viewer.

What it shows (real-time):
- Left original (half frame or dedicated camera)
- Right original (half frame or dedicated camera)
- Left ROI zoom (3-5x, manual selection)
- Right ROI zoom (3-5x, manual selection)
- A simple sharpness score on ROI (Laplacian variance)

Controls:
- l: select ROI on Left window
- r: select ROI on Right window
- [ / ]: decrease / increase focus (if CAP_PROP_FOCUS is supported)
- a: toggle autofocus (CAP_PROP_AUTOFOCUS)
- +/- zoom: change zoom factor (default 4)
- q / ESC: quit
"""

import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np


def laplacian_sharpness(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clamp_roi(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, min(int(w), img_w - x))
    h = max(1, min(int(h), img_h - y))
    return x, y, w, h


def set_focus(cap: cv2.VideoCapture, value: float) -> None:
    # Some backends/cameras ignore this silently.
    try:
        cap.set(cv2.CAP_PROP_FOCUS, float(value))
    except Exception:
        pass


def set_autofocus(cap: cv2.VideoCapture, enabled: bool) -> None:
    try:
        # 1/0 semantics depend on the driver; we try both.
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if enabled else 0)
    except Exception:
        pass


class FocusTuner:
    def __init__(
        self,
        device_index: int,
        split_half: bool,
        zoom: int,
        autofocus: bool,
        focus_step: float,
        backend: Optional[int],
    ) -> None:
        self.device_index = device_index
        self.split_half = split_half
        self.zoom = int(max(1, zoom))
        self.focus_step = float(focus_step)
        self.autofocus_enabled = bool(autofocus)

        self.cap = cv2.VideoCapture(device_index, backend) if backend is not None else cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index={device_index}")

        # Apply initial autofocus / focus.
        set_autofocus(self.cap, self.autofocus_enabled)

        self.last_frame_time = 0.0

        self.left_roi: Optional[Tuple[int, int, int, int]] = None
        self.right_roi: Optional[Tuple[int, int, int, int]] = None

        # Populated after first frame.
        self.frame_h: int = 0
        self.frame_w: int = 0

    def _split_left_right(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        if not self.split_half:
            raise ValueError("split_half is False but _split_left_right was called")
        half = w // 2
        left = frame[:, :half]
        right = frame[:, half:]
        return left, right

    def _roi_zoom_view(self, img: np.ndarray, roi: Optional[Tuple[int, int, int, int]], zoom: int) -> np.ndarray:
        if img is None or img.size == 0:
            return img
        h, w = img.shape[:2]
        canvas = img.copy()
        if roi is None:
            cv2.putText(canvas, "ROI not set", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            return canvas

        x, y, rw, rh = clamp_roi(roi[0], roi[1], roi[2], roi[3], w, h)
        roi_img = img[y : y + rh, x : x + rw].copy()
        zoomed = cv2.resize(
            roi_img,
            (max(1, rw * zoom), max(1, rh * zoom)),
            interpolation=cv2.INTER_CUBIC,
        )

        sharp = laplacian_sharpness(roi_img)
        cv2.putText(
            zoomed,
            f"sharp={sharp:.1f} (lap var)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Also draw a small marker box on original for feedback.
        cv2.rectangle(canvas, (x, y), (x + rw, y + rh), (0, 255, 255), 1)
        return zoomed

    def run(self) -> None:
        win_left = "Left"
        win_right = "Right"
        win_left_zoom = "Left ROI Zoom"
        win_right_zoom = "Right ROI Zoom"

        cv2.namedWindow(win_left, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_right, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_left_zoom, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_right_zoom, cv2.WINDOW_NORMAL)

        print("Keys: l/r select ROI, [/] focus, a autofocus, +/- zoom, q or ESC quit")

        # Try read once for display sizes.
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            self.frame_h, self.frame_w = frame.shape[:2]

            left, right = self._split_left_right(frame)

            # Current focus value to overlay.
            focus_val = self.cap.get(cv2.CAP_PROP_FOCUS)
            af_val = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)

            # Show original halves.
            cv2.putText(
                left,
                f"focus={focus_val:.2f} af={int(af_val)} zoom={self.zoom}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                right,
                f"focus={focus_val:.2f} af={int(af_val)} zoom={self.zoom}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            left_zoom = self._roi_zoom_view(left, self.left_roi, self.zoom)
            right_zoom = self._roi_zoom_view(right, self.right_roi, self.zoom)

            cv2.imshow(win_left, left)
            cv2.imshow(win_right, right)
            cv2.imshow(win_left_zoom, left_zoom)
            cv2.imshow(win_right_zoom, right_zoom)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("l"):
                # ROI selection on the left image.
                roi = cv2.selectROI(win_left, left, fromCenter=False, showCrosshair=True)
                x, y, w, h = roi
                if w > 0 and h > 0:
                    self.left_roi = (int(x), int(y), int(w), int(h))
                continue
            if key == ord("r"):
                roi = cv2.selectROI(win_right, right, fromCenter=False, showCrosshair=True)
                x, y, w, h = roi
                if w > 0 and h > 0:
                    self.right_roi = (int(x), int(y), int(w), int(h))
                continue

            if key == ord("a"):
                self.autofocus_enabled = not self.autofocus_enabled
                set_autofocus(self.cap, self.autofocus_enabled)
                continue

            if key in (ord("]"), ord("+")):
                focus_val = self.cap.get(cv2.CAP_PROP_FOCUS)
                set_focus(self.cap, focus_val + self.focus_step)
                continue
            if key in (ord("["), ord("-")):
                focus_val = self.cap.get(cv2.CAP_PROP_FOCUS)
                set_focus(self.cap, focus_val - self.focus_step)
                continue

            # Zoom control with keys '+'/'-' is already used for focus +/- via above.
            # Provide separate zoom keys: '=' and '_' for zoom factor.
            if key == ord("="):
                self.zoom = min(8, self.zoom + 1)
                continue
            if key == ord("_"):
                self.zoom = max(1, self.zoom - 1)
                continue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=1, help="Camera index for cv2.VideoCapture")
    parser.add_argument("--split-half", action="store_true", help="Split side-by-side frame into left/right halves")
    parser.add_argument("--no-split-half", dest="split_half", action="store_false")
    parser.set_defaults(split_half=True)

    parser.add_argument("--zoom", type=int, default=4, help="ROI zoom factor (3-5 recommended)")
    parser.add_argument("--autofocus", action="store_true", help="Enable autofocus at start")
    parser.add_argument("--no-autofocus", dest="autofocus", action="store_false")
    parser.set_defaults(autofocus=False)
    parser.add_argument("--focus-step", type=float, default=1.0, help="Focus step for [/] keys")

    parser.add_argument("--backend-dshow", action="store_true", help="Use CAP_DSHOW backend (Windows)")
    args = parser.parse_args()

    backend = cv2.CAP_DSHOW if args.backend_dshow else None

    tuner = FocusTuner(
        device_index=args.device_index,
        split_half=args.split_half,
        zoom=args.zoom,
        autofocus=args.autofocus,
        focus_step=args.focus_step,
        backend=backend,
    )
    tuner.run()

    tuner.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

