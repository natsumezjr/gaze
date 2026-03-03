"""关键点可视化器 - 在图像上绘制识别到的关键点"""
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from project.data.data_models import KeyCoordinates, Point3DWithVisibility, FITTING_TYPE, EYE_TYPE, Landmark

# 配置日志
from project.config.logging_config import setup_logging 
logger = setup_logging(__name__)



class KeypointVisualizer:
    """关键点可视化器 - 在图像上绘制按眼睛和类型分组的关键点"""
    
    def __init__(self, camera_params: Dict):
        """
        初始化可视化器
        
        Args:
            camera_params: 相机参数字典，包含内参用于坐标转换
        """
        self.camera_params = camera_params
        self._extract_camera_intrinsics()
        
        # 关键点类型颜色配置 (BGR格式)
        self.color_map = {
            "pupil": (0, 0, 255),           # 红色
            "iris": (255, 0, 0),            # 蓝色
            "inner_canthus": (0, 255, 0),  # 绿色
            "outer_canthus": (0, 255, 255), # 黄色
            "upper_eyelid": (255, 0, 255),  # 紫色
            "lower_eyelid": (255, 255, 0),  # 青色
        }
        
        # 绘制参数
        self.point_radius = 3
        self.line_thickness = 1
        self.show_labels = False  # 默认不显示标签（性能考虑）
    
    def _extract_camera_intrinsics(self):
        """提取相机内参"""
        intrinsic_params = self.camera_params.get("intrinsic_params", {})
        self.fx = intrinsic_params.get("fx", 925.0)
        self.fy = intrinsic_params.get("fy", 925.0)
        self.cx = intrinsic_params.get("cx", 640.0)
        self.cy = intrinsic_params.get("cy", 360.0)
        
        logger.debug(f"相机内参: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
    
    def _3d_to_2d(self, point_3d: Point3DWithVisibility) -> Optional[Tuple[int, int]]:
        """
        将3D坐标（米单位）转换为2D像素坐标
        
        Args:
            point_3d: 3D坐标点（带可见性）
            
        Returns:
            (x, y) 像素坐标，如果转换失败返回None
        """
        try:
            x_3d, y_3d, z_3d = point_3d.x, point_3d.y, point_3d.z
            
            # 检查深度值有效性
            if not np.isfinite(z_3d) or z_3d <= 0:
                return None
            
            # 检查可见性
            if point_3d.visibility < 0.3:  # 可见性阈值
                return None
            
            # 反向投影公式：x_pixel = (X_3d * fx / Z_3d) + cx
            x_pixel = (x_3d * self.fx / z_3d) + self.cx
            y_pixel = (y_3d * self.fy / z_3d) + self.cy
            
            # 转换为整数
            x_pixel = int(round(x_pixel))
            y_pixel = int(round(y_pixel))
            
            return (x_pixel, y_pixel)
            
        except Exception as e:
            logger.warning(f"3D到2D坐标转换失败: {e}")
            return None
    
    def _get_color_for_type(self, fitting_type: str) -> Tuple[int, int, int]:
        """
        根据关键点类型获取颜色
        
        Args:
            fitting_type: 关键点类型（pupil, iris等）
            
        Returns:
            (B, G, R) 颜色元组
        """
        return self.color_map.get(fitting_type, (255, 255, 255))  # 默认白色
    
    def _draw_point_group(self, 
                         frame: np.ndarray, 
                         points: List[Point3DWithVisibility], 
                         color: Tuple[int, int, int],
                         fitting_type: str) -> None:
        """
        绘制一组关键点
        
        Args:
            frame: BGR图像
            points: 关键点列表
            color: 颜色 (B, G, R)
            fitting_type: 关键点类型（用于判断是否绘制连接线）
        """
        if not points:
            return
        
        # 转换所有点到2D坐标
        points_2d = []
        for point_3d in points:
            point_2d = self._3d_to_2d(point_3d)
            if point_2d is not None:
                points_2d.append(point_2d)
        
        if not points_2d:
            return
        
        # 绘制点
        for x, y in points_2d:
            # 检查坐标是否在图像范围内
            h, w = frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), self.point_radius, color, -1)  # 实心圆
        
        # 对于某些类型，绘制连接线（如iris边界）
        if fitting_type == "iris" and len(points_2d) > 2:
            # 绘制iris边界连接线
            pts = np.array(points_2d, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=self.line_thickness)
        elif fitting_type in ["upper_eyelid", "lower_eyelid"] and len(points_2d) > 1:
            # 绘制眼睑连接线
            pts = np.array(points_2d, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=self.line_thickness)
    
    def draw_keypoints(self, 
                      frame: np.ndarray, 
                      key_coordinates: KeyCoordinates) -> np.ndarray:
        """
        在图像上绘制关键点
        
        Args:
            frame: BGR图像 (numpy array, shape: H×W×3)
            key_coordinates: 关键点坐标数据
            
        Returns:
            visualized_frame: 绘制了关键点的图像
        """
        # 复制图像（避免修改原始图像）
        visualized_frame = frame.copy()
        
        if key_coordinates is None:
            return visualized_frame
        
        try:
            # 遍历左右眼
            for eye in EYE_TYPE:
                # 获取该眼的数据
                eye_data = key_coordinates.left_eye if eye == "left" else key_coordinates.right_eye
                
                # 遍历每种关键点类型
                for fitting_type in FITTING_TYPE:
                    # 获取该类型的关键点
                    points = eye_data.get(fitting_type, [])
                    
                    if not points:
                        continue
                    
                    # 获取颜色
                    color = self._get_color_for_type(fitting_type)
                    
                    # 绘制关键点组
                    self._draw_point_group(visualized_frame, points, color, fitting_type)
            
            logger.debug("关键点绘制完成")
            
        except Exception as e:
            logger.error(f"绘制关键点时出错: {e}")
        
        return visualized_frame

    def draw_2d_landmarks(self, frame: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
        """
        在图像上绘制 2D 关键点（像素坐标），用于稳定显示，避免因深度 nan 导致 3D 投影乱跳。
        """
        out = frame.copy()
        if not landmarks:
            return out
        h, w = out.shape[:2]
        for lm in landmarks:
            x, y = int(round(lm.x)), int(round(lm.y))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(out, (x, y), self.point_radius, (0, 255, 0), -1)
        return out

    def draw_2d_fitting_landmarks(self, frame: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
        """
        仅绘制拟合用关键点（与 get_landmark_indices 一致），按类型区分颜色，与之前 3D 绘制逻辑一致。
        """
        from project.core.recognition.landmark_extractor import get_landmark_indices
        out = frame.copy()
        if not landmarks:
            return out
        h, w = out.shape[:2]
        indices = get_landmark_indices()
        for eye in EYE_TYPE:
            for fitting_type in FITTING_TYPE:
                color = self._get_color_for_type(fitting_type)
                for idx in indices.get(eye, {}).get(fitting_type, []):
                    if idx >= len(landmarks):
                        continue
                    lm = landmarks[idx]
                    x, y = int(round(lm.x)), int(round(lm.y))
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(out, (x, y), self.point_radius, color, -1)
                # 可选：iris/眼睑连线（与 _draw_point_group 一致）
                pts = []
                for idx in indices.get(eye, {}).get(fitting_type, []):
                    if idx < len(landmarks):
                        x, y = int(round(landmarks[idx].x)), int(round(landmarks[idx].y))
                        if 0 <= x < w and 0 <= y < h:
                            pts.append([x, y])
                if fitting_type == "iris" and len(pts) > 2:
                    cv2.polylines(out, [np.array(pts, dtype=np.int32)], True, color, self.line_thickness)
                elif fitting_type in ("upper_eyelid", "lower_eyelid") and len(pts) > 1:
                    cv2.polylines(out, [np.array(pts, dtype=np.int32)], False, color, self.line_thickness)
        return out

    def set_drawing_params(self, 
                          point_radius: Optional[int] = None,
                          line_thickness: Optional[int] = None,
                          show_labels: Optional[bool] = None):
        """
        设置绘制参数
        
        Args:
            point_radius: 点半径
            line_thickness: 线粗细
            show_labels: 是否显示标签
        """
        if point_radius is not None:
            self.point_radius = point_radius
        if line_thickness is not None:
            self.line_thickness = line_thickness
        if show_labels is not None:
            self.show_labels = show_labels

