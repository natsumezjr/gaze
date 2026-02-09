"""RGB-D 摄像头测试脚本
测试 _open_rgb_d 函数能否正常打开 RGB-D 摄像头
显示实时画面，按 Ctrl+C 或 q 退出
支持 1080P 分辨率设置（双目相机 3840x1080 或单路 1920x1080）
"""
import sys
import platform
import cv2

# 确保项目根目录在路径中
if __name__ == "__main__":
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# 1080P 分辨率候选（双目相机通常输出左右并排 3840x1080，单路为 1920x1080）
RESOLUTION_PRESETS = [
    (3840, 1080),   # 双目并排 1080P
    (1920, 1080),   # 单路 1080P
    (2560, 720),    # 部分双目相机
    (1280, 720),    # 720P 备用
]


def test_open_rgb_d(camera_index: int = 1, width: int = 1920, height: int = 1080):
    """打开 RGB-D 摄像头并设置分辨率"""
    print(f"正在尝试打开 RGB-D 摄像头 (index={camera_index})...")
    try:
        # Windows 下使用 DirectShow 可获得更好的分辨率控制
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("[失败] 摄像头打开失败，可能设备未连接或已被占用")
            return None

        # 先设置分辨率，再读取（必须在首次 read 之前设置）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 读取一帧以确认实际生效的分辨率
        ret, _ = cap.read()
        if ret:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[成功] 摄像头已打开 | 请求: {width}x{height} | 实际: {actual_w}x{actual_h}")
            if (actual_w, actual_h) != (width, height):
                print(f"       (相机可能不支持 {width}x{height}，已使用最接近的格式)")
        else:
            print("[成功] 摄像头已打开")

        return cap
    except Exception as e:
        print(f"[异常] 打开摄像头失败: {e}")
        return None


def main():
    print("=" * 50)
    print("RGB-D 摄像头测试 (1080P) - 按 Ctrl+C 或 q 退出")
    print("=" * 50)

    cap = None
    for w, h in RESOLUTION_PRESETS:
        cap = test_open_rgb_d(camera_index=1, width=w, height=h)
        if cap is not None:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w >= 1280 or actual_h >= 720:  # 至少接近 720P
                print(f"\n使用分辨率: {actual_w}x{actual_h}")
                break
        if cap is not None:
            cap.release()
            cap = None

    if cap is None:
        # 回退：不设分辨率，用默认
        print("\n尝试使用默认分辨率...")
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("测试结果: 失败")
            return 1
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[成功] 摄像头已打开 | 默认分辨率: {actual_w}x{actual_h}")

    window_name = "RGB-D Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 按画面比例固定初始窗口大小 (3840:1080 ≈ 32:9)
    init_h = 400
    init_w = int(init_h * actual_w / actual_h) if actual_h else 1422
    cv2.resizeWindow(window_name, init_w, init_h)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[警告] 读取帧失败")
                break
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头资源已释放")
    return 0


if __name__ == "__main__":
    exit(main())
