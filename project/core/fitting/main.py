from project.data.data_manager import RecgFitDataManager
from project.core.fitting.fitting_strategy import fit_all_eyes
from project.config.screen_config import SCREEN_CONFIG
from project.config.logging_config import setup_logging 
from project.data.data_models import Point3D
from project.core.fitting.kappa_calibrator import KAPPA_STORAGE, apply_kappa
import numpy as np
logger = setup_logging(__name__)


class FittingManager:
    def __init__(self, recg_fit_data_manager: RecgFitDataManager):
        self.data_manager = recg_fit_data_manager
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
            
            for eye, result in fitting_results.items():
                logger.debug(f"处理 {eye} 眼的拟合结果...")
                parameters = result.parameters
                eyeball_center = parameters.center
                
                pupil_center = self.data_manager.get_coordinate_point(eye, "pupil")[0]
                pupil_center = Point3D(pupil_center.x, pupil_center.y, pupil_center.z)
                eyeball_center = Point3D.from_ndarray(eyeball_center)

                # 计算原始视线方向（从 eyeball 到 pupil）
                optical_axis = (pupil_center - eyeball_center).to_ndarray()
                
                # 如果 kappa valid，应用 kappa 校正
                kappa = self.kapa_storage.get_kappa(eye)
                if kappa and self.kapa_storage.is_kappa_valid():
                    corrected_direction = apply_kappa(optical_axis, kappa)
                    # 将校正后的方向转换为点（从 eyeball 沿校正方向延伸）
                    distance = np.linalg.norm(optical_axis)
                    if distance > 0:
                        corrected_gaze_point = eyeball_center + Point3D.from_ndarray(corrected_direction * distance)
                    else:
                        corrected_gaze_point = pupil_center
                    self.gaze[eye] = corrected_gaze_point
                else:
                    # kappa 未 valid，使用原始视线
                    self.gaze[eye] = pupil_center
                
                pixel_intersection = SCREEN_CONFIG.calculate_gaze_intersection(self.gaze[eye], eyeball_center)
                self.intersections[eye] = pixel_intersection
                
                logger.info(f"-----------------------------------------------")
                logger.info(f"{eye} 视线焦点: {pixel_intersection}")
                logger.info(f"-----------------------------------------------")
                
                # 在 UI 上显示实现点（蓝色）
                try:
                    from project.client.kappa.ui import EyeCalibrationApp
                    ui = EyeCalibrationApp.get_instance()
                    if ui:
                        ui.add_gaze_point(pixel_intersection, color="#0000FF")  # 蓝色
                except Exception as e:
                    logger.debug(f"显示实现点失败: {e}")
                
        except Exception as e:
            logger.error(f"拟合过程中发生异常: {e}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            
            
        
        
        
    
    
    def log_fitting_results(self, fitting_results: dict):
        logger.info(f"==============================================")
        logger.info(f"拟合结果:")
        for eye, result in fitting_results.items():
            logger.info(f"  {eye.upper()} 眼: {result}")
        logger.info(f"==============================================")
        
        

        
        