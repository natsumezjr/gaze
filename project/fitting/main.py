
import logging
import numpy as np
from project.fitting.core.fitting_strategy import fit_all_eyes
from project.recg_fit_data.data_manager import RECG_FIT_DATA_MANAGER, FITTING_TYPE, EYE_TYPE

# 导入专业版kappa校准器相关功能
from project.fitting.core.kappa_calibrator_pro import (
    estimate_kappa as estimate_kappa_pro,
    apply_kappa,
    evaluate_fit,
    incremental_update,
    build_samples_from_arrays
)
from project.fitting.core.gaze_estimator import compute_theoretical_gaze

def calculate_distances_to_eyeball(eye: str, fitting_result, data_manager):
    """
    计算各点到眼球中心和表面的距离
    
    Args:
        eye: 眼睛类型 ("left" 或 "right")
        fitting_result: 拟合结果
        data_manager: 数据管理器
    """
    if not fitting_result or not fitting_result.converged:
        logging.warning(f"{eye}眼拟合未收敛，无法计算距离")
        return
    
    center = fitting_result.parameters.center
    radius = fitting_result.parameters.radius
    
    logging.info(f"\n{'='*80}")
    logging.info(f"{eye.upper()}眼距离分析报告")
    logging.info(f"{'='*80}")
    logging.info(f"眼球中心: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] mm")
    logging.info(f"眼球半径: {radius:.2f} mm")
    logging.info(f"拟合置信度: {fitting_result.confidence:.3f}")
    logging.info(f"最终残差: {fitting_result.final_residual_norm:.3f}")
    logging.info(f"{'='*80}")
    
    # 统计信息
    total_points = 0
    points_in_sphere = 0
    points_outside_sphere = 0
    
    # 按类型分组显示
    for fitting_type in FITTING_TYPE:
        points = data_manager.get_coordinate_point(eye, fitting_type)
        if not points:
            continue
            
        logging.info(f"\n【{fitting_type.upper()}】点分析 ({len(points)}个点):")
        logging.info("-" * 60)
        
        type_in_sphere = 0
        type_outside_sphere = 0
        
        for i, point in enumerate(points):
            if len(point) >= 3:
                # 提取3D坐标
                point_3d = point[:3]
                
                # 计算到球心的距离
                distance_to_center = np.linalg.norm(point_3d - center)
                
                # 计算到球面的距离（正值表示在球外，负值表示在球内）
                distance_to_surface = distance_to_center - radius
                
                # 获取可见性（如果存在）
                visibility = point[3] if len(point) >= 4 else "N/A"
                
                # 判断位置
                if distance_to_surface <= 0:
                    position = "球内"
                    type_in_sphere += 1
                    points_in_sphere += 1
                else:
                    position = "球外"
                    type_outside_sphere += 1
                    points_outside_sphere += 1
                
                total_points += 1
                
                # 简化输出格式
                logging.info(f"  点{i+1}: 距离球心={distance_to_center:.5f}mm, "
                           f"距离球面={distance_to_surface:+.5f}mm ({position}), "
                           f"可见性={visibility}")
        
        # 该类型统计
        if len(points) > 0:
            logging.info(f"  统计: {type_in_sphere}个球内, {type_outside_sphere}个球外")
    
    # 总体统计
    logging.info(f"\n{'='*80}")
    logging.info(f"{eye.upper()}眼总体统计:")
    logging.info(f"总点数: {total_points}")
    logging.info(f"球内点数: {points_in_sphere} ({points_in_sphere/total_points*100:.5f}%)")
    logging.info(f"球外点数: {points_outside_sphere} ({points_outside_sphere/total_points*100:.5f}%)")
    logging.info(f"{'='*80}")

def demo_kappa_calibration_pro(use_real_camera=False, camera_config=None):
    """演示专业版Kappa校准功能"""
    print("\n" + "="*80)
    print("专业版KAPPA校准演示")
    print("="*80)
    
    logging.info("\n" + "="*80)
    logging.info("专业版KAPPA校准演示")
    logging.info("="*80)
    
    try:
        # 创建模拟校准数据
        np.random.seed(42)
        N_samples = 10
        
        # 模拟眼球中心和瞳孔中心
        eyes = np.random.randn(N_samples, 3) * 0.1  # 眼球中心，小偏移
        pupils = np.random.randn(N_samples, 3) * 0.05  # 瞳孔中心，更小偏移
        
        # 模拟相机内参矩阵
        K = np.array([[1000, 0, 480],
                      [0, 1000, 270],
                      [0, 0, 1]])
        
        # 模拟目标像素坐标
        target_pixels = np.random.randint(0, 960, (N_samples, 2))
        
        print(f"创建了{N_samples}个模拟校准样本")
        print(f"眼球中心范围: {np.min(eyes, axis=0)} 到 {np.max(eyes, axis=0)}")
        print(f"瞳孔中心范围: {np.min(pupils, axis=0)} 到 {np.max(pupils, axis=0)}")
        
        # 使用专业版kappa校准器
        print("使用专业版kappa校准器...")
        logging.info("使用专业版kappa校准器...")
        samples_pro = build_samples_from_arrays(
            eyes, pupils, 
            target_pixels=target_pixels, 
            K=K
        )
        
        kappa_pro, info_pro = estimate_kappa_pro(samples_pro, lock_roll=True)
        print(f"专业版校准结果 - Kappa: {np.degrees(kappa_pro)}°")
        print(f"校准信息: {info_pro}")
        logging.info(f"专业版校准结果 - Kappa: {np.degrees(kappa_pro)}°")
        logging.info(f"校准信息: {info_pro}")
        
        # 应用kappa补偿
        print("\n应用kappa补偿...")
        logging.info("\n应用kappa补偿...")
        theoretical_gaze = compute_theoretical_gaze(eyes[0], pupils[0])
        compensated_gaze = apply_kappa(theoretical_gaze.reshape(1, 3), kappa_pro)[0]
        
        print(f"理论视线: {theoretical_gaze}")
        print(f"补偿后视线: {compensated_gaze}")
        logging.info(f"理论视线: {theoretical_gaze}")
        logging.info(f"补偿后视线: {compensated_gaze}")
        
        # 评估拟合质量
        print("\n评估拟合质量...")
        logging.info("\n评估拟合质量...")
        fit_quality = evaluate_fit(samples_pro, kappa_pro, K, return_pixel_err=True)
        print(f"拟合质量: {fit_quality}")
        logging.info(f"拟合质量: {fit_quality}")
        
        # 演示增量更新
        print("\n演示增量kappa更新...")
        logging.info("\n演示增量kappa更新...")
        kappa_prev = kappa_pro.copy()
        v = theoretical_gaze
        d = compensated_gaze
        
        for i in range(3):
            kappa_new = incremental_update(kappa_prev, v, d, beta=0.1)
            angle_change = np.degrees(np.linalg.norm(kappa_new - kappa_prev))
            print(f"第{i+1}次更新: Kappa变化 {angle_change:.4f}°")
            logging.info(f"第{i+1}次更新: Kappa变化 {angle_change:.4f}°")
            kappa_prev = kappa_new.copy()
        
        print("Kappa校准演示成功完成！")
        return kappa_pro, info_pro, fit_quality
        
    except Exception as e:
        print(f"Kappa校准演示失败: {e}")
        logging.error(f"Kappa校准演示失败: {e}")
        import traceback
        error_trace = traceback.format_exc()
        print(f"详细错误信息: {error_trace}")
        logging.error(f"详细错误信息: {error_trace}")
        return None, None, None

def main():
    """
    眼动追踪数据初步测评
    """
    #key_coordinates_log(RECG_FIT_DATA_MANAGER)
    
    print("="*80)
    print("程序开始执行")
    print("="*80)
    
    logging.info("-" * 50)
    logging.info("开始拟合...")
    
    try:
        results = fit_all_eyes(params_only=False)
        
        data_manager = RECG_FIT_DATA_MANAGER
        # 关闭debug日志避免刷屏
        data_manager._debug_log = False
        
        for key, result in results.items():
            logging.info(f"拟合结果: {key}：\n{result}")
            
            # 计算各点到眼球中心和表面的距离
            calculate_distances_to_eyeball(key, result, data_manager)
            
    except Exception as e:
        logging.warning(f"眼球拟合过程出现错误: {e}")
        print(f"眼球拟合过程出现错误: {e}")
        logging.info("继续运行kappa校准演示...")
    
    # 运行kappa校准演示（独立运行，不受拟合结果影响）
    print("\n" + "="*80)
    print("开始运行KAPPA校准演示")
    print("="*80)
    
    logging.info("\n" + "="*80)
    logging.info("开始运行KAPPA校准演示")
    logging.info("="*80)
    
    # 摄像头配置选项
    print("\n摄像头配置选项:")
    print("1. 使用模拟数据（默认）")
    print("2. 使用真实摄像头")
    print("3. 自定义摄像头配置")
    
    try:
        choice = input("\n请选择 (1/2/3，直接回车使用默认): ").strip()
        
        if choice == "2":
            # 真实摄像头模式
            print("切换到真实摄像头模式...")
            use_real_camera = True
            camera_config = None
            
        elif choice == "3":
            # 自定义配置
            print("自定义摄像头配置:")
            focal_length = float(input("焦距（像素，默认1000）: ") or "1000")
            principal_x = float(input("主点X坐标（像素，默认960）: ") or "960")
            principal_y = float(input("主点Y坐标（像素，默认540）: ") or "540")
            
            camera_config = {
                'focal_length': focal_length,
                'principal_point': (principal_x, principal_y),
                'resolution': (1920, 1080)
            }
            use_real_camera = False  # 仍使用模拟数据，但用自定义配置
            print(f"使用自定义配置: 焦距={focal_length}, 主点=({principal_x}, {principal_y})")
            
        else:
            # 默认模拟数据模式
            print("使用默认模拟数据模式...")
            use_real_camera = False
            camera_config = None
        
        print("调用demo_kappa_calibration_pro函数...")
        kappa_pro, info_pro, fit_quality = demo_kappa_calibration_pro(
            use_real_camera=use_real_camera,
            camera_config=camera_config
        )
        
        if kappa_pro is not None:
            print("Kappa校准演示完成！")
            print(f"最终Kappa角: {np.degrees(kappa_pro)}°")
            logging.info("Kappa校准演示完成！")
        else:
            print("Kappa校准演示失败")
            
    except Exception as e:
        print(f"Kappa校准演示失败: {e}")
        logging.error(f"Kappa校准演示失败: {e}")
        import traceback
        error_trace = traceback.format_exc()
        print(f"详细错误信息: {error_trace}")
        logging.error(f"详细错误信息: {error_trace}")
    
    print("\n程序执行完成！")
    logging.info("\n程序执行完成！")

if __name__ == "__main__":
    # 创建空的RecgFitDataManager实例用于测试
    main()