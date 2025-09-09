
import logging
import numpy as np
from project.fitting.core.fitting_strategy import fit_all_eyes
from project.recg_fit_data.data_manager import RECG_FIT_DATA_MANAGER, FITTING_TYPE, EYE_TYPE

from project.fitting.app.state import SESSION_MANAGER, FittingResultLite

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


def main():
    """
    眼球拟合主程序 - 专注于眼球形状拟合
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

            # 写入会话管理器，保存最近一次拟合结果（用于标定/前端交互）
            try:
                lite = FittingResultLite(
                    center=result.parameters.center,
                    radius=float(result.parameters.radius),
                    confidence=float(result.confidence),
                    converged=bool(result.converged),
                    final_residual_norm=float(result.final_residual_norm),
                    strategy_used=str(result.strategy_used)
                )
                SESSION_MANAGER.set_last_fitting(key, lite)
            except Exception as _e:
                logging.warning(f"保存最近拟合结果失败: {_e}")
            
    except Exception as e:
        logging.warning(f"眼球拟合过程出现错误: {e}")
        print(f"眼球拟合过程出现错误: {e}")
        logging.info("继续运行kappa校准演示...")
    
    
    print("\n程序执行完成！")
    logging.info("\n程序执行完成！")

if __name__ == "__main__":
    # 创建空的RecgFitDataManager实例用于测试
    main()