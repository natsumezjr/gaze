import cv2
import numpy as np
import json
import base64
import threading
import time
from datetime import datetime
from pathlib import Path
from recognition.config.settings import DATA_PATH

def encode_image_to_base64(bgr_image):
    """
    将BGR格式（OpenCV默认）的图像编码为base64字符串
    """
    _, buffer = cv2.imencode('.jpg', bgr_image)
    return base64.b64encode(buffer).decode('utf-8')

def clean_old_entries(data_dict, max_age_sec=10):
    now = time.time()
    return {
        k: v for k, v in data_dict.items()
        if (now - datetime.fromisoformat(v["timestamp"]).timestamp()) < max_age_sec
    }

def convert_to_json(bgr_image, frame_id):
    """
    将BGR格式（OpenCV默认）的图像和帧信息转为JSON结构
    """
    h, w = bgr_image.shape[:2]
    return {
        "frame_id": frame_id,
        "timestamp": datetime.now().isoformat(),
        "camera": {
            "resolution": [w, h]
        },
        "rgb_data": encode_image_to_base64(bgr_image),  # 实际为BGR格式的base64字符串
        "depth_data": None,  # 无深度数据
        "eye_centers": {"left": None, "right": None},
        "pupil_centers": {"left": None, "right": None},
        "iris_boundaries": {"left": [], "right": []}
    }

def auto_clean_thread(data_dict, lock, max_age_sec=10, interval=1):
    while True:
        with lock:
            keys_to_del = [k for k, v in data_dict.items()
                           if (time.time() - datetime.fromisoformat(v["timestamp"]).timestamp()) > max_age_sec]
            for k in keys_to_del:
                del data_dict[k]
        time.sleep(interval)

def main():
    cap = cv2.VideoCapture(0)
    frame_id = 0
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    output_path = f"{DATA_PATH}/rgbd_input.json"
    data_dict = {}
    lock = threading.Lock()

    # 启动自动清理线程
    threading.Thread(target=auto_clean_thread, args=(data_dict, lock), daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            data = convert_to_json(frame, frame_id)
            frame_key = f"frame_{frame_id}"
            with lock:
                data_dict[frame_key] = data
                # 写入前再清理一遍，保证数据不过期
                data_dict = clean_old_entries(data_dict, max_age_sec=10)
                with open(output_path, "w") as f:
                    json.dump(data_dict, f, indent=2)

            print(f"[✔] Frame {frame_id} written, 当前保留 {len(data_dict)} 帧")
            frame_id += 1

            cv2.imshow("RGB", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
