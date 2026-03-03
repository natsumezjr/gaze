"""检查标定图像分辨率：列出每张图的 (宽, 高) 及左/右半幅尺寸，与 camera_params 中的 image_size 对照。

用法（在项目根目录 D:\\my_projects\\gaze 下）:
  poetry run python -m project.config.check_calibration_resolution
  或
  python -m project.config.check_calibration_resolution
"""
import json
import os
import sys

import cv2


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_dir = os.path.join(base_dir, "calibration")
    params_path = os.path.join(base_dir, "camera_params.json")

    # 1) 从 camera_params 读标称 image_size
    json_image_size = None
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "left" in data and "image_size" in data["left"]:
            json_image_size = tuple(data["left"]["image_size"])
            print(f"camera_params.json 中 left.image_size: {json_image_size[0]} x {json_image_size[1]}")
        if "stereo" in data and "image_size" in data["stereo"]:
            st = tuple(data["stereo"]["image_size"])
            print(f"camera_params.json 中 stereo.image_size: {st[0]} x {st[1]}")
    else:
        print(f"未找到 {params_path}")
    print()

    # 2) 收集要检查的图片路径
    paths = []
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        used = data.get("stereo", {}).get("used_pair_paths", [])
        for p in used:
            if os.path.exists(p):
                paths.append(p)
    if not paths:
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            paths.extend(
                os.path.join(calibration_dir, f)
                for f in sorted(os.listdir(calibration_dir))
                if f.lower().endswith(ext[1:])
            )
        paths = [p for p in paths if os.path.exists(p)]

    if not paths:
        print(f"未找到任何标定图片（已查 used_pair_paths 与目录 {calibration_dir}）")
        return 1

    print(f"共 {len(paths)} 张图片，分辨率如下：")
    print("-" * 60)
    seen = set()
    first_roi_size = None
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [无法读取] {os.path.basename(path)}")
            continue
        h, w = img.shape[:2]
        half_w = w // 2
        roi_size = (half_w, h)  # 左/右半幅 (宽, 高)
        if first_roi_size is None:
            first_roi_size = roi_size
        key = (w, h)
        if key not in seen:
            seen.add(key)
        print(f"  {w} x {h}  (左/右半幅: {roi_size[0]} x {roi_size[1]})  {os.path.basename(path)}")

    print("-" * 60)
    if first_roi_size:
        print(f"标定逻辑下首张有效图的 image_size 将为: {first_roi_size[0]} x {first_roi_size[1]}")
    if json_image_size:
        if first_roi_size and tuple(first_roi_size) != json_image_size:
            print(f"与 JSON 中 left.image_size {json_image_size} 不一致（若仅部分图分辨率不同则可能来自首张成功角点图）")
        else:
            print(f"与 JSON 中 left.image_size 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
