import numpy as np
import mediapipe as mp
import cv2
from typing import List, Dict
from project.recg_fit_data.data_manager import FITTING_TYPE
from project.recg_fit_data.data_manager import KeyCoordinates, EYE_TYPE
from project.recognition.config.settings import *




def calculate_visibility(landmark: List[float], image: np.ndarray, landmark_index: int) -> float:
    """计算关键点的可见性分数"""
    return 0.6

# 复用单例 FaceMesh 实例与短期容错
_FACE_MESH = None
_FACE_DET = None
_LAST_LANDMARKS: List[List[float]] | None = None
_MISS_COUNT = 0
_MISS_GRACE = 5  # 丢失容错帧数


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        mp_face_mesh = mp.solutions.face_mesh
        _FACE_MESH = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5
        )
    return _FACE_MESH

def _get_face_detector():
    """缓存的人脸检测器（比FaceMesh快，用于兜底找ROI）"""
    global _FACE_DET
    if _FACE_DET is None:
        mp_fd = mp.solutions.face_detection
        _FACE_DET = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.3)
    return _FACE_DET


def _compute_roi_from_last(landmarks: List[List[float]], image_shape: tuple[int, int], scale: float = 1.6):
    """由上一次关键点估计ROI，并做扩大与边界裁剪"""
    if not landmarks:
        return None
    h, w = image_shape[:2]
    pts = np.array(landmarks, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    x0 = max(0.0, float(xs.min()))
    y0 = max(0.0, float(ys.min()))
    x1 = min(float(w - 1), float(xs.max()))
    y1 = min(float(h - 1), float(ys.max()))
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale
    rx0 = int(max(0.0, cx - bw * 0.5))
    ry0 = int(max(0.0, cy - bh * 0.5))
    rx1 = int(min(w, cx + bw * 0.5))
    ry1 = int(min(h, cy + bh * 0.5))
    if rx1 - rx0 < 20 or ry1 - ry0 < 20:
        return None
    return rx0, ry0, rx1, ry1


def extract_landmarks(bgr_image: np.ndarray) -> List[List[float]]:
    """
    输入：一张包含人脸的BGR格式numpy数组图像
    输出：478个精确的2D/3D关键点，格式为[[x, y, z, visibility], ...]
    """
    global _LAST_LANDMARKS, _MISS_COUNT
    face_mesh = _get_face_mesh()
    
    # ROI优先，再回退全图
    used_roi = False
    roi = _compute_roi_from_last(_LAST_LANDMARKS, bgr_image.shape) if _LAST_LANDMARKS is not None else None
    if roi is not None:
        x0, y0, x1, y1 = roi
        roi_img = bgr_image[y0:y1, x0:x1]
        rgb_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_roi)
        used_roi = True
    else:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks and roi is not None:
        # 回退全图再试
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        used_roi = False
    
    # 仍失败 → 兜底：快速人脸检测获取ROI再跑一次FaceMesh
    if not results.multi_face_landmarks:
        fd = _get_face_detector()
        rgb_full = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        fd_res = fd.process(rgb_full)
        if fd_res and fd_res.detections:
            h, w = bgr_image.shape[:2]
            # 选取最居中的人脸
            cx_img, cy_img = w * 0.5, h * 0.5
            def det_score_center(d):
                bbox = d.location_data.relative_bounding_box
                bx = (bbox.xmin + bbox.width * 0.5) * w
                by = (bbox.ymin + bbox.height * 0.5) * h
                # 距离中心越近越好，同时考虑置信度
                dist = (bx - cx_img) ** 2 + (by - cy_img) ** 2
                score = d.score[0] if d.score else 0.0
                return (-score, dist)
            best = sorted(fd_res.detections, key=det_score_center)[0]
            bbox = best.location_data.relative_bounding_box
            x0b = max(0, int(bbox.xmin * w))
            y0b = max(0, int(bbox.ymin * h))
            x1b = min(w, int((bbox.xmin + bbox.width) * w))
            y1b = min(h, int((bbox.ymin + bbox.height) * h))
            # 扩大一些
            bw = x1b - x0b
            bh = y1b - y0b
            x0e = max(0, int(x0b - bw * 0.3))
            y0e = max(0, int(y0b - bh * 0.3))
            x1e = min(w, int(x1b + bw * 0.3))
            y1e = min(h, int(y1b + bh * 0.3))
            if x1e - x0e >= 20 and y1e - y0e >= 20:
                roi_img2 = bgr_image[y0e:y1e, x0e:x1e]
                rgb_roi2 = cv2.cvtColor(roi_img2, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_roi2)
                if results.multi_face_landmarks:
                    used_roi = True
                    roi = (x0e, y0e, x1e, y1e)

    if not results.multi_face_landmarks:
        # 使用短期容错：短暂丢失时复用上一帧有效关键点
        if _LAST_LANDMARKS is not None and _MISS_COUNT < _MISS_GRACE:
            _MISS_COUNT += 1
            return _LAST_LANDMARKS
        return []
    
    face_landmarks = results.multi_face_landmarks[0]
    height, width = bgr_image.shape[:2]
    
    landmarks = []
    for i, landmark in enumerate(face_landmarks.landmark):
        if used_roi and roi is not None:
            x = landmark.x * (roi[2] - roi[0]) + roi[0]
            y = landmark.y * (roi[3] - roi[1]) + roi[1]
        else:
            x = landmark.x * width
            y = landmark.y * height
        z = landmark.z
        visibility = calculate_visibility([x, y, z], bgr_image, i)
        landmarks.append([x, y, z, visibility])
    
    # 记录最近一次成功结果并重置丢失计数
    _LAST_LANDMARKS = landmarks
    _MISS_COUNT = 0
    return landmarks

def get_landmark_indices() -> Dict[str, Dict[str, List[int]]]:
    """返回MediaPipe Face Mesh 拟合所需的关键点索引映射"""
    return {
        "left":{
            "pupil": [468],
            "iris": [469, 470, 471, 472],
            "inner_canthus": [133],
            "upper_eyelid": [157, 158, 159, 160, 173],
            "lower_eyelid": [145, 153, 154, 155, 161],
            "outer_canthus": [246]
        },
        "right":{
            "pupil": [473],
            "iris": [474, 475, 476, 477],
            "inner_canthus": [362],
            "upper_eyelid": [384, 385, 386, 387, 398],
            "lower_eyelid": [374, 380, 381, 382, 390],
            "outer_canthus": [466]
        }
    }
    

def get_fitting_landmarks(landmarks: List[List[float]]) -> KeyCoordinates:
    """
    获取所有眼部关键点，整合为fitting模块所需的格式
    
    :param landmarks: 关键点列表，每个点包含[x, y, z, visibility]
    :return: 整合后的眼部关键点字典
    """
    result = {}
    for eye in EYE_TYPE:
        result[eye] = {}
        for fitting_type in FITTING_TYPE:
            result[eye][fitting_type] = []
            
            
    if not landmarks or len(landmarks) != 478:
        return result
    
    try:
        indices = get_landmark_indices()
        
        # 填充数据
        for eye in EYE_TYPE:
            for fitting_type in FITTING_TYPE:
                    result[eye][fitting_type] = [np.array(landmarks[i]) for i in indices[eye][fitting_type] if i < len(landmarks)]
        
        return result
        
    except Exception as e:
        print(f"整合眼部关键点失败: {e}")
        return {
            'left': {'pupil': [], 'iris': [], 'inner_canthus': [], 'upper_eyelid': [], 'lower_eyelid': [], 'outer_canthus': []},
            'right': {'pupil': [], 'iris': [], 'inner_canthus': [], 'upper_eyelid': [], 'lower_eyelid': [], 'outer_canthus': []}
        }

def validate_landmarks(landmarks: List[List[float]]) -> bool:
    """验证关键点数据的有效性"""
    if not landmarks or len(landmarks) != 478:
        return False
    
    try:
        landmarks_array = np.array(landmarks)
        if not np.all(np.isfinite(landmarks_array)):
            return False
        
        x_coords = landmarks_array[:, 0]
        y_coords = landmarks_array[:, 1]
        z_coords = landmarks_array[:, 2]
        
        if np.any(x_coords < -1000) or np.any(x_coords > 10000) or \
           np.any(y_coords < -1000) or np.any(y_coords > 10000) or \
           np.any(z_coords < -10) or np.any(z_coords > 10):
            return False
        
        return True
    except Exception:
        return False
    

__all__ = ['extract_landmarks', 'get_fitting_landmarks', 'validate_landmarks']

if __name__ == "__main__":
    print(" Landmark Extractor 模块测试")
    print("=" * 50)
    
    def initialize_camera():
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        return None
    
    def draw_landmarks_on_image(image, landmarks):
        if not landmarks:
            return image
        
        try:
            fitting_data = get_fitting_landmarks(landmarks)
            
            # 定义不同关键点的颜色
            colors = {
                'pupil': {'left': (255, 0, 0), 'right': (0, 0, 255)},      # 瞳孔：左眼蓝色，右眼红色
                'iris': (0, 255, 0),                                          # 虹膜：绿色
                'inner_canthus': (255, 255, 0),                               # 内眼角：青色
                'upper_eyelid': (0, 255, 255),                                # 上眼睑：黄色
                'lower_eyelid': (0, 165, 255),                                # 下眼睑：橙色
                'outer_canthus': (255, 0, 255)                                # 外眼角：紫色
            }
            
            # 绘制瞳孔中心点
            for eye in EYE_TYPE:
                pupil = fitting_data[eye]['pupil']
                if pupil and len(pupil) > 0:
                    pupil_point = pupil[0]  # 瞳孔是列表的第一个元素
                    x, y = int(pupil_point[0]), int(pupil_point[1])
                    color = colors['pupil'][eye]
                    cv2.circle(image, (x, y), 4, color, -1)
                
                # 绘制其他关键点
                for part, points in fitting_data[eye].items():
                    if part != 'pupil' and points:
                        color = colors[part]
                        for point in points:
                            x, y = int(point[0]), int(point[1])
                            cv2.circle(image, (x, y), 2, color, -1)
                    
        except Exception as e:
            print(f"绘制关键点失败: {e}")
        
        return image
    
    def run_real_time_test():
        print("🚀 启动实时测试...")
        
        cap = initialize_camera()
        if cap is None:
            print("❌ 无法初始化摄像头")
            return
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 提取关键点
                landmarks = extract_landmarks(frame)
                
                if landmarks:
                    # 获取拟合数据
                    fitting_data = get_fitting_landmarks(landmarks)
                    
                    # 显示统计信息
                    print(f"帧 {frame_count}: ✅ 检测到 {len(landmarks)} 个关键点")
                    for eye in EYE_TYPE:
                        pupil = fitting_data[eye]['pupil']
                        if pupil and len(pupil) > 0:
                            pupil_point = pupil[0]  # 瞳孔是列表的第一个元素
                            print(f"   {eye}眼瞳孔: ({float(pupil_point[0]):.1f}, {float(pupil_point[1]):.1f})")
                    
                    # 绘制关键点
                    frame = draw_landmarks_on_image(frame, landmarks)
                else:
                    print(f"帧 {frame_count}: ❌ 未检测到人脸")
                
                # 显示图像
                cv2.imshow('Landmark Extractor Test', frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\n👋 用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ 测试完成")
    
    try:
        run_real_time_test()
    except Exception as e:
        print(f"❌ 测试错误: {e}")