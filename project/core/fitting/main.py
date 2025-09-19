from project.data.data_manager import RecgFitDataManager
from project.core.fitting.fitting_strategy import fit_all_eyes, FittingResult
from project.managers import SEMAPHORE_MANAGER, SemaphoreManager, FRAME_ID_MANAGER
from project.config.screen_config import SCREEN_CONFIG
from project.config.logging_config import setup_logging, get_logger
from project.data.data_models import Point3D, Vector3D
from project.core.fitting.kappa_calibrator import KAPPA_STORAGE
from project.data.data_models import EYE_TYPE
setup_logging()
logger = get_logger(__name__)

class FittingManager:
    def __init__(self, recg_fit_data_manager: RecgFitDataManager, semaphore_manager: SemaphoreManager = SEMAPHORE_MANAGER):
        self.semaphore_manager = semaphore_manager
        self.frame_id_manager = FRAME_ID_MANAGER
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
                
                if not self.kapa_storage.is_kappa_valid():
                    self.kapa_storage.append_center(eye, "eyeball", eyeball_center)
                    self.kapa_storage.append_center(eye, "pupil", pupil_center)
                
                pupil_center = Point3D(pupil_center.x, pupil_center.y, pupil_center.z)
                eyeball_center = Point3D.from_ndarray(eyeball_center)

                self.gaze[eye] = Point3D(pupil_center.x, pupil_center.y, pupil_center.z)
                pixel_intersection = SCREEN_CONFIG.calculate_gaze_intersection(pupil_center, eyeball_center)
                self.intersections[eye] = pixel_intersection
                
                logger.info(f"-----------------------------------------------")
                logger.info(f"{eye} 视线焦点: {pixel_intersection}")
                logger.info(f"-----------------------------------------------")
                
            """ from project.main import kappa_calibrate_event, EYE_CALIBRATION_APP  
            from project.client.kappa.ui import CalibrationState  
            if not self.kapa_storage.is_pixel_valid():
                kappa_calibrate_event.set()
                
                if EYE_CALIBRATION_APP.state == CalibrationState.CALIBRATING:
                    EYE_CALIBRATION_APP.submit_data(self.intersections["left"], self.intersections["right"])
                    
                if EYE_CALIBRATION_APP.state == CalibrationState.COMPLETED:
                    calibration_result = EYE_CALIBRATION_APP.get_calibration_result()
                    logger.info(f"校准结果: {calibration_result}")
                    self.kapa_storage.set_pixel(calibration_result["left_eye"].points, calibration_result["right_eye"].points)
                    
            if self.kapa_storage.is_pixel_valid() and not self.kapa_storage.is_kappa_valid():
                from project.core.fitting.kappa_calibrator import build_samples_from_arrays, estimate_kappa, apply_kappa
                for eye in EYE_TYPE:
                    eyeball_centers = self.kapa_storage.get_center(eye, "eyeball")
                    pupil_centers = self.kapa_storage.get_center(eye, "pupil")
                    target_pixels = self.kapa_storage.get_pixel(eye)
                    target_points = []
                    for i, target_pixel in enumerate(target_pixels):
                        target_points.append(SCREEN_CONFIG.pixel_to_3d_intersection(target_pixel, eyeball_centers[i]))
                    samples = build_samples_from_arrays(eyeball_centers, pupil_centers, target_points)
                    kappa = estimate_kappa(samples)
                    self.kapa_storage.set_kappa(eye, kappa)
                    
            if self.kapa_storage.is_kappa_valid():
                for eye in EYE_TYPE:
                    gaze = apply_kappa(self.gaze[eye].to_ndarray(), self.kapa_storage.get_kappa(eye))
                    self.gaze[eye] = Vector3D.from_ndarray(gaze)
            return self.gaze """
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
        
        

        
        