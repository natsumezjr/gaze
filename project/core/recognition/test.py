from project.data.data_manager import CameraDataManager
from project.core.recognition.detector import FaceDetector
from project.data.data_models import BGRImage, DepthMap
import logging
import time
import numpy as np
import cv2

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_camera_initialization():
    """测试摄像头初始化"""
    logger.info("=== 测试摄像头初始化 ===")
    
    rgb_d = False
    camera_data_manager = CameraDataManager(rgb_d=rgb_d)
    
    if camera_data_manager.initialize_camera():
        logger.info("✓ 摄像头初始化成功")
        
        # 获取摄像头参数
        camera_params = camera_data_manager.get_camera_params()
        logger.info(f"摄像头参数: {camera_params}")
        
        # 测试读取一帧
        cap = camera_data_manager.get_cap()
        ret, frame = cap.read()
        
        if ret:
            logger.info(f"✓ 成功读取帧，尺寸: {frame.shape}")
            logger.info(f"帧数据类型: {frame.dtype}")
            logger.info(f"帧数据范围: {frame.min()} - {frame.max()}")
            
            # 显示帧信息
            h, w = frame.shape[:2]
            logger.info(f"图像尺寸: {w}x{h}")
            
            return True, camera_data_manager, frame
        else:
            logger.error("✗ 无法读取摄像头帧")
            return False, None, None
    else:
        logger.error("✗ 摄像头初始化失败")
        return False, None, None

def test_face_detection_with_debug(rgb_image, depth_map):
    """带调试信息的人脸检测测试"""
    logger.info("=== 测试人脸检测 ===")
    
    try:
        # 创建人脸检测器
        camera_params = {
            "intrinsic_params": {
                "fx": 500.0,
                "fy": 500.0,
                "cx": 320.0,
                "cy": 240.0
            },
            "depth_scale": 0.001
        }
        
        face_detector = FaceDetector(camera_params, rgb_d=False)
        logger.info("✓ FaceDetector创建成功")
        
        # 测试图像转换
        logger.info(f"输入图像类型: {type(rgb_image)}")
        logger.info(f"输入图像尺寸: {rgb_image.shape if hasattr(rgb_image, 'shape') else 'N/A'}")
        
        # 创建BGRImage对象
        bgr_image = BGRImage(data=rgb_image)
        logger.info(f"✓ BGRImage创建成功: {bgr_image}")
        
        # 创建DepthMap对象
        depth_map_obj = DepthMap(data=depth_map, unit="meter")
        logger.info(f"✓ DepthMap创建成功: {depth_map_obj}")
        
        # 执行人脸检测
        logger.info("开始人脸检测...")
        detected = face_detector.detect_face(bgr_image, depth_map_obj)
        
        if detected:
            logger.info("✓ 人脸检测成功")
            
            # 获取检测信息
            detection_info = face_detector.get_detection_info()
            logger.info(f"检测信息: {detection_info}")
            
            # 获取关键点
            landmarks = face_detector.get_landmarks()
            logger.info(f"关键点数量: {len(landmarks)}")
            
            if landmarks:
                logger.info(f"前5个关键点: {landmarks[:5]}")
            
            # 更新拟合数据
            key_coordinates = face_detector.update_fitting_data()
            logger.info(f"拟合数据: {key_coordinates}")
            
        else:
            logger.warning("✗ 人脸检测失败")
            
            # 添加详细的调试信息
            logger.info("=== 详细调试信息 ===")
            
            # 检查BGRImage对象
            logger.info(f"BGRImage对象: {bgr_image}")
            logger.info(f"BGRImage数据形状: {bgr_image.data.shape}")
            logger.info(f"BGRImage数据类型: {bgr_image.data.dtype}")
            
            # 检查DepthMap对象
            logger.info(f"DepthMap对象: {depth_map_obj}")
            logger.info(f"DepthMap数据形状: {depth_map_obj.data.shape}")
            logger.info(f"DepthMap数据类型: {depth_map_obj.data.dtype}")
            
            # 尝试直接使用MediaPipe测试
            logger.info("尝试直接使用MediaPipe测试...")
            test_mediapipe_directly(rgb_image)
            
        return detected
        
    except Exception as e:
        logger.error(f"人脸检测测试失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False

def test_mediapipe_directly(rgb_image):
    """直接测试 MediaPipe FaceLandmarker 人脸检测"""
    try:
        from project.core.recognition.landmark_extractor import _get_face_landmarker
        import mediapipe as mp

        logger.info("初始化 MediaPipe FaceLandmarker...")
        landmarker = _get_face_landmarker()

        rgb_converted = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        logger.info(f"RGB转换后图像尺寸: {rgb_converted.shape}")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_converted)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            logger.info(f"✓ MediaPipe 检测到 {len(result.face_landmarks)} 个人脸")

            face_landmarks = result.face_landmarks[0]
            logger.debug(f"关键点数量: {len(face_landmarks)} (期望: 478个)")

            for i in range(min(5, len(face_landmarks))):
                landmark = face_landmarks[i]
                logger.info(f"关键点 {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
        else:
            logger.warning("✗ MediaPipe 未检测到人脸")

            debug_filename = "debug_frame.jpg"
            cv2.imwrite(debug_filename, rgb_image)
            logger.info(f"调试图像已保存为: {debug_filename}")

    except Exception as e:
        logger.error(f"MediaPipe 直接测试失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")

def test_with_saved_image():
    """使用保存的图像进行测试"""
    logger.info("=== 使用保存的图像测试 ===")
    
    try:
        # 尝试加载调试图像
        debug_image = cv2.imread("debug_frame.jpg")
        if debug_image is not None:
            logger.info(f"✓ 成功加载调试图像: {debug_image.shape}")
            
            # 创建深度图
            h, w = debug_image.shape[:2]
            depth_map = np.ones((h, w), dtype=np.float32) * 0.6
            
            # 测试人脸检测
            return test_face_detection_with_debug(debug_image, depth_map)
        else:
            logger.warning("未找到调试图像文件")
            return False
            
    except Exception as e:
        logger.error(f"使用保存图像测试失败: {e}")
        return False

def test_recg_fit_data_manager():
    """测试RecgFitDataManager的详细数据"""
    logger.info("=== 测试RecgFitDataManager数据 ===")
    
    try:
        # 1. 初始化摄像头
        logger.info("1. 初始化摄像头...")
        camera_manager = CameraDataManager(rgb_d=False)
        
        if not camera_manager.initialize_camera():
            logger.error("摄像头初始化失败")
            return
        
        # 2. 读取一帧图像
        cap = camera_manager.get_cap()
        ret, frame = cap.read()
        
        if not ret:
            logger.error("无法读取摄像头数据")
            return
        
        # 3. 创建深度图
        h, w = frame.shape[:2]
        depth_scale = camera_manager.get_depth_scale()
        depth_map = np.ones((h, w), dtype=np.float32) * (0.6 / depth_scale)
        
        # 4. 创建数据对象
        bgr_image = BGRImage(data=frame)
        depth_map_obj = DepthMap(data=depth_map, unit="meter")
        
        # 5. 创建人脸检测器
        camera_params = camera_manager.get_camera_params()
        face_detector = FaceDetector(camera_params, rgb_d=False)
        
        # 6. 执行人脸检测
        detected = face_detector.detect_face(bgr_image, depth_map_obj)
        
        if not detected:
            logger.error("人脸检测失败")
            return
        
        # 7. 获取拟合数据
        key_coordinates = face_detector.update_fitting_data()
        
        # 7.4. 调试原始坐标数据
        logger.info("7.4. 原始坐标数据调试:")
        for eye in ["left", "right"]:
            for fitting_type in ["pupil", "iris", "inner_canthus", "upper_eyelid", "lower_eyelid", "outer_canthus"]:
                points = key_coordinates.get_points(eye, fitting_type)
                if points:
                    logger.info(f"  {eye}眼{fitting_type} (转换前):")
                    for i, point in enumerate(points[:2]):  # 只显示前2个点
                        logger.info(f"    [{i}] x={point.x:.6f}m, y={point.y:.6f}m, z={point.z:.6f}m")
        
        # 7.5. 调试深度图信息
        logger.info("7.5. 深度图调试信息:")
        logger.info(f"  深度图尺寸: {depth_map_obj.height}x{depth_map_obj.width}")
        logger.info(f"  深度图单位: {depth_map_obj.unit}")
        logger.info(f"  深度图范围: {np.min(depth_map_obj.data):.6f}m - {np.max(depth_map_obj.data):.6f}m")
        logger.info(f"  深度图平均值: {np.mean(depth_map_obj.data):.6f}m")
        
        # 7.6. 调试相机参数
        logger.info("7.6. 相机参数:")
        logger.info(f"  fx: {camera_params['intrinsic_params']['fx']}")
        logger.info(f"  fy: {camera_params['intrinsic_params']['fy']}")
        logger.info(f"  cx: {camera_params['intrinsic_params']['cx']}")
        logger.info(f"  cy: {camera_params['intrinsic_params']['cy']}")
        
        # 8. 创建调试版本的RecgFitDataManager
        from project.data.data_manager import RecgFitDataManager
        debug_data_manager = RecgFitDataManager(
            key_coordinates=key_coordinates,
            debug_log=True,  # 启用调试日志
        )
        
        # 9. 输出数据摘要
        logger.info("2. 数据摘要:")
        summary = debug_data_manager.get_data_summary()
        for eye in ["left", "right"]:
            logger.info(f"  {eye} eye:")
            for fitting_type in ["pupil", "iris", "inner_canthus", "upper_eyelid", "lower_eyelid", "outer_canthus"]:
                count = summary[eye][fitting_type]
                logger.info(f"    {fitting_type}: {count} points")
        
        # 10. 输出详细坐标数据
        logger.info("3. 详细坐标数据:")
        for eye in ["left", "right"]:
            for fitting_type in ["pupil", "iris", "inner_canthus", "upper_eyelid", "lower_eyelid", "outer_canthus"]:
                points = debug_data_manager.get_coordinate_point(eye, fitting_type)
                if points:
                    logger.info(f"  {eye}眼{fitting_type} ({len(points)}个点):")
                    for i, point in enumerate(points):
                        logger.info(f"    [{i}] x={point.x:.2f}mm, y={point.y:.2f}mm, z={point.z:.2f}mm, visibility={point.visibility:.3f}")
                else:
                    logger.info(f"  {eye}眼{fitting_type}: 无数据")
        
        # 11. 输出RecgFitDataManager的字符串表示
        logger.info("4. RecgFitDataManager字符串表示:")
        logger.info(str(debug_data_manager))
        
        
        return True
        
    except Exception as e:
        logger.error(f"RecgFitDataManager测试失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False
    

def main():
    """主测试函数"""
    logger.info("开始人脸检测系统测试...")
    
    # 测试1: 摄像头初始化
    success, camera_manager, frame = test_camera_initialization()
    
    if not success:
        logger.error("摄像头初始化失败，无法继续测试")
        return
    
    try:
        # 测试2: 人脸检测
        h, w = frame.shape[:2]
        depth_scale = camera_manager.get_depth_scale()
        depth_map = np.ones((h, w), dtype=np.float32) * (0.6 / depth_scale)
        
        detection_success = test_face_detection_with_debug(frame, depth_map)
        
        if not detection_success:
            logger.info("尝试使用保存的图像进行测试...")
            test_with_saved_image()
        
        # 测试3: RecgFitDataManager数据调试
        if detection_success:
            logger.info("=== 测试RecgFitDataManager数据 ===")
            test_recg_fit_data_manager()
        
        # 测试4: 连续检测（如果第一次成功）
        if detection_success:
            logger.info("=== 测试连续检测 ===")
            test_continuous_detection(camera_manager)
            
    except KeyboardInterrupt:
        logger.info("用户中断测试")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
    finally:
        # 清理资源
        if camera_manager:
            camera_manager.release_camera()
            logger.info("摄像头资源已释放")

def test_continuous_detection(camera_manager):
    """测试连续检测"""
    logger.info("开始连续检测测试（按Ctrl+C停止）...")
    
    try:
        camera_params = {
            "intrinsic_params": {
                "fx": 500.0,
                "fy": 500.0,
                "cx": 320.0,
                "cy": 240.0
            },
            "depth_scale": 0.001
        }
        
        face_detector = FaceDetector(camera_params, rgb_d=False)
        cap = camera_manager.get_cap()
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("无法读取摄像头数据")
                time.sleep(0.1)
                continue
            
            # 创建深度图
            h, w = frame.shape[:2]
            depth_scale = camera_manager.get_depth_scale()
            depth_map = np.ones((h, w), dtype=np.float32) * (0.6 / depth_scale)
            
            # 创建数据对象
            bgr_image = BGRImage(data=frame)
            depth_map_obj = DepthMap(data=depth_map, unit="meter")
            
            # 检测人脸
            detected = face_detector.detect_face(bgr_image, depth_map_obj)
            
            frame_count += 1
            if detected:
                detection_count += 1
                logger.info(f"帧 {frame_count}: ✓ 检测到人脸")
            else:
                logger.info(f"帧 {frame_count}: ✗ 未检测到人脸")
            
            # 每10帧显示一次统计
            if frame_count % 10 == 0:
                success_rate = (detection_count / frame_count) * 100
                logger.info(f"检测统计: {detection_count}/{frame_count} ({success_rate:.1f}%)")
            
            time.sleep(0.1)  # 控制帧率
            
    except KeyboardInterrupt:
        logger.info("连续检测测试被用户中断")
    except Exception as e:
        logger.error(f"连续检测测试失败: {e}")
        
if __name__ == "__main__":
    main()