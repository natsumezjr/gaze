from project.data.data_manager import RecgFitDataManager
from project.core.fitting.fitting_strategy import fit_all_eyes
from project.config.screen_config import SCREEN_CONFIG
from project.config.logging_config import setup_logging
from project.data.data_models import Point3D, Point2D
from project.core.fitting.kappa_calibrator import KAPPA_STORAGE, apply_kappa
from project.managers import CALLBACK_MANAGER
from project.events.event_types import GAZE_POINT_UPDATE
import numpy as np
import logging
from typing import Optional, Dict

logger = setup_logging(__name__, logging.DEBUG)


class FittingManager:
    def __init__(self, recg_fit_data_manager: RecgFitDataManager, frame_id: Optional[int] = None):
        self.data_manager = recg_fit_data_manager
        self.frame_id = frame_id
        self.kapa_storage = KAPPA_STORAGE
        self.intersections = {
            "left": None,
            "right": None
        }
        self.gaze = {
            "left": Point3D(0, 0, 0),
            "right": Point3D(0, 0, 0)
        }

    
    def run(self):
        try:
            # 执行拟合
            logger.debug("开始执行拟合...")
            fitting_results = fit_all_eyes(self.data_manager)
            logger.debug(f"拟合完成，结果数量: {len(fitting_results)}")
            logger.debug(f"拟合结果键: {list(fitting_results.keys())}")
            
            self.log_fitting_results(fitting_results)
            
            eyeball_centers: Dict[str, Point3D] = {}
            gaze_directions: Dict[str, np.ndarray] = {}
            for eye, result in fitting_results.items():
                logger.debug(f"处理 {eye} 眼的拟合结果...")
                parameters = result.parameters
                eyeball_center = parameters.center
                
                pupil_center = self.data_manager.get_coordinate_point(eye, "pupil")[0]
                pupil_center = Point3D(pupil_center.x, pupil_center.y, pupil_center.z)
                eyeball_center_pt = Point3D.from_ndarray(eyeball_center)
                eyeball_centers[eye] = eyeball_center_pt

                # 计算原始视线方向（从 eyeball 到 pupil）
                optical_axis = (pupil_center - eyeball_center_pt).to_ndarray()
                self.log_gaze_direction(eye, optical_axis)
                # 如果 kappa valid，应用 kappa 校正
                kappa = self.kapa_storage.get_kappa(eye)
                if kappa and self.kapa_storage.is_kappa_valid():
                    corrected_direction = apply_kappa(optical_axis, kappa)
                    distance = np.linalg.norm(optical_axis)
                    if distance > 0:
                        corrected_gaze_point = eyeball_center_pt + Point3D.from_ndarray(corrected_direction * distance)
                    else:
                        corrected_gaze_point = pupil_center
                    self.gaze[eye] = corrected_gaze_point
                else:
                    self.gaze[eye] = pupil_center
                
                gaze_vec = (self.gaze[eye] - eyeball_center_pt).to_ndarray()
                norm = np.linalg.norm(gaze_vec)
                gaze_directions[eye] = (gaze_vec / norm) if norm > 1e-9 else gaze_vec
                
                pixel_intersection = SCREEN_CONFIG.calculate_gaze_intersection(self.gaze[eye], eyeball_center_pt)
                self.intersections[eye] = pixel_intersection
                self.log_intersection_and_region(eye, pixel_intersection)
                if pixel_intersection:
                    CALLBACK_MANAGER.emit(GAZE_POINT_UPDATE, point=pixel_intersection, color="#0000FF")
                
        except Exception as e:
            logger.error(f"拟合过程中发生异常: {e}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            
            
    def log_gaze_direction(self, eye: str, optical_axis: np.ndarray) -> None:
        """记录视线向量与方向描述（屏幕坐标系 x 右为正、y 下为正）。"""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        oa_norm = np.linalg.norm(optical_axis)
        optical_axis_unit = (optical_axis / oa_norm) if oa_norm > 1e-9 else optical_axis
        dx, dy, dz = optical_axis_unit[0], optical_axis_unit[1], optical_axis_unit[2]
        h_dir = "偏右" if dx > 0.05 else ("偏左" if dx < -0.05 else "水平居中")
        v_dir = "偏下" if dy > 0.05 else ("偏上" if dy < -0.05 else "垂直居中")
        dir_desc = f"{v_dir}{h_dir}" if (v_dir != "垂直居中" or h_dir != "水平居中") else "正前方"
        logger.debug(
            f"{eye} 视线向量(未归一化)=%.3f 单位向量=(%.4f,%.4f,%.4f) 方向={dir_desc}",
            oa_norm, dx, dy, dz,
        )

    def log_intersection_and_region(self, eye: str, pixel_intersection: Optional[Point2D]) -> None:
        """记录视线焦点像素坐标及屏幕方位（左上、左中、左下、中上、正中、中下、右上、右中、右下）。"""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        screen_region = "未知"
        if pixel_intersection:
            rw, rh = SCREEN_CONFIG.resolution_px[0], SCREEN_CONFIG.resolution_px[1]
            if rw > 0 and rh > 0:
                px, py = pixel_intersection.x, pixel_intersection.y
                h_third, v_third = rw / 3.0, rh / 3.0
                h_pos = "左" if px < h_third else ("右" if px >= 2 * h_third else "中")
                v_pos = "上" if py < v_third else ("下" if py >= 2 * v_third else "中")
                screen_region = v_pos + h_pos
        logger.debug(f"{eye} 视线焦点: {pixel_intersection} 屏幕方位: {screen_region}")

    def log_fitting_results(self, fitting_results: dict) -> None:
        logger.debug(f"==============================================")
        logger.debug(f"拟合结果:")
        for eye, result in fitting_results.items():
            logger.debug(f"  {eye.upper()} 眼: {result}")
        logger.debug(f"==============================================")
        
        

        
        