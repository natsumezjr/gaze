import numpy as np
import time
import logging
from typing import Dict, List, Union
from project.fitting.core.eyeballs_fitter import ransac_sphere_eyeball,ransac_sphere_eyeball_restart
from project.fitting.utils.center_fitter import center_fitter

def main(key_coordinates: dict):
    """
    眼动追踪数据初步测评
    """
    
    # 简洁输出关键点信息
    logging.info("=" * 50)
    logging.info("眼动追踪数据测评")
    logging.info("=" * 50)
    
    # 瞳孔中心
    if 'pupil_center' in key_coordinates:
        pupil = key_coordinates['pupil_center']
        if pupil['left'] is not None and pupil['right'] is not None:
            left = pupil['left']
            right = pupil['right']
            # 只显示前3维坐标，忽略visibility
            left_3d = left[:3] if len(left) >= 3 else left
            right_3d = right[:3] if len(right) >= 3 else right
            logging.info(f"瞳孔中心: 左({left_3d[0]:.3f}, {left_3d[1]:.3f}, {left_3d[2]:.3f}) | 右({right_3d[0]:.3f}, {right_3d[1]:.3f}, {right_3d[2]:.3f})")
        else:
            logging.info("瞳孔中心: 无效")
    
    # 虹膜边界
    if 'iris_boundaries' in key_coordinates:
        iris = key_coordinates['iris_boundaries']
        if iris['left'] is not None and iris['right'] is not None:
            left_count = len(iris['left'])
            right_count = len(iris['right'])
            logging.info(f"虹膜边界: 左眼{left_count}个点 | 右眼{right_count}个点")
        else:
            logging.info("虹膜边界: 无效")
            
    # 眼睛轮廓
    if 'eyes_contours' in key_coordinates:
        contour = key_coordinates['eyes_contours']
        if contour['left'] is not None and contour['right'] is not None:
            left_count = len(contour['left'])
            right_count = len(contour['right'])
            logging.info(f"眼睛轮廓: 左眼{left_count}个点 | 右眼{right_count}个点")
        else:
            logging.info("眼睛轮廓: 无效")
    
    logging.info("-" * 50)
    logging.info("开始RANSAC球体拟合...")
    
    # 执行拟合
    try:
        result = ransac_sphere_eyeball(key_coordinates, center_fitter)
        logging.info("拟合完成!")
        
        # 输出拟合结果
        logging.info("-" * 50)
        logging.info("拟合结果:")
        
        # 左眼结果
        left_shape, left_center = result["left"]
        if left_center is not None and len(left_center) == 3:
            logging.info(f"左眼中心: ({left_center[0]:.3f}, {left_center[1]:.3f}, {left_center[2]:.3f})")
            if left_shape and hasattr(left_shape, 'axes'):
                axes = left_shape.axes
                # 转换为毫米显示
                radius_mm = axes[0] * 1000
                logging.info(f"左眼半径: {radius_mm:.2f}mm")
                
                # 分析左眼关键点的分布情况
                logging.info("-" * 30)
                logging.info("左眼关键点分布分析:")
                _analyze_eye_points(key_coordinates, "left", left_center, radius_mm/1000)
        else:
            logging.info("左眼中心: 拟合失败")
        
        # 右眼结果
        right_shape, right_center = result["right"]
        if right_center is not None and len(right_center) == 3:
            logging.info(f"右眼中心: ({right_center[0]:.3f}, {right_center[1]:.3f}, {right_center[2]:.3f})")
            if right_shape and hasattr(right_shape, 'axes'):
                axes = right_shape.axes
                # 转换为毫米显示
                radius_mm = axes[0] * 1000
                logging.info(f"右眼半径: {radius_mm:.2f}mm")
                
                # 分析右眼关键点的分布情况
                logging.info("-" * 30)
                logging.info("右眼关键点分布分析:")
                _analyze_eye_points(key_coordinates, "right", right_center, radius_mm/1000)
        else:
            logging.info("右眼中心: 拟合失败")
            
    except Exception as e:
        logging.info(f"拟合过程中出现错误: {e}")
    
    logging.info("=" * 50)
    
    # 适当睡眠防止刷屏
    time.sleep(0.5)


def _analyze_eye_points(key_coordinates: dict, eye_side: str, eye_center: np.ndarray, eye_radius: float):
    """
    分析指定眼睛的关键点分布情况
    
    Args:
        key_coordinates: 关键点坐标字典
        eye_side: 眼睛侧别 ("left" 或 "right")
        eye_center: 眼球中心坐标 (numpy数组)
        eye_radius: 眼球半径 (米)
    """
    import numpy as np
    
    # 定义各结构类型的规范阈值（毫米）
    # 注意：这些点应该在眼球外部，且距离在阈值范围内
    # 参考_int_center_fitter.md中的解剖结构偏移范围
    thresholds = {
        "pupil_center": 0.3,      # 瞳孔中心：偏移范围0.1-0.3mm，取上限0.3mm
        "iris_boundary": 1.2,     # 虹膜边界：偏移范围0.5-1.2mm，取上限1.2mm
        "eye_contour": 2.0        # 眼睛轮廓：偏移范围1.0-2.0mm，取上限2.0mm
    }
    
    total_points = 0
    valid_points = 0
    
    # 分析瞳孔中心点
    if 'pupil_center' in key_coordinates:
        pupil = key_coordinates['pupil_center']
        if eye_side in pupil and pupil[eye_side] is not None:
            point = np.array(pupil[eye_side])
            # 只使用前3维坐标进行计算，忽略visibility
            point_3d = point[:3] if len(point) >= 3 else point
            distance = np.linalg.norm(point_3d - eye_center)
             
            # 计算到眼球表面的距离
            surface_distance = abs(distance - eye_radius)
            surface_distance_mm = surface_distance * 1000
            threshold_mm = thresholds["pupil_center"]
             
            # 判断位置
            if distance < eye_radius:
                position = "眼球内部"
            else:
                position = "眼球外部"
             
            # 判断是否符合规范
            # 只有在眼球外部且距离在阈值范围内的点才是正确的
            is_valid = (distance >= eye_radius) and (surface_distance_mm <= threshold_mm)
            status = "✓ 符合规范" if is_valid else "✗ 超出规范"
             
            logging.info(f"瞳孔中心点: {position}, 距离眼球表面: {surface_distance_mm:.2f}mm, {status}")
            total_points += 1
            if is_valid:
                valid_points += 1
    
    # 分析虹膜边界点
    if 'iris_boundaries' in key_coordinates:
        iris = key_coordinates['iris_boundaries']
        if eye_side in iris and iris[eye_side] is not None:
            iris_points = iris[eye_side]
            if isinstance(iris_points, list) and len(iris_points) > 0:
                logging.info(f"虹膜边界点 ({len(iris_points)}个):")
                for i, point_data in enumerate(iris_points):
                    if point_data is not None:
                        point = np.array(point_data)
                        # 只使用前3维坐标进行计算，忽略visibility
                        point_3d = point[:3] if len(point) >= 3 else point
                        distance = np.linalg.norm(point_3d - eye_center)
                         
                        # 计算到眼球表面的距离
                        surface_distance = abs(distance - eye_radius)
                        surface_distance_mm = surface_distance * 1000
                        threshold_mm = thresholds["iris_boundary"]
                         
                        # 判断位置
                        if distance < eye_radius:
                            position = "眼球内部"
                        else:
                            position = "眼球外部"
                         
                        # 判断是否符合规范
                        # 只有在眼球外部且距离在阈值范围内的点才是正确的
                        is_valid = (distance >= eye_radius) and (surface_distance_mm <= threshold_mm)
                        status = "✓ 符合规范" if is_valid else "✗ 超出规范"
                         
                        logging.info(f"  点{i+1}: {position}, 距离眼球表面: {surface_distance_mm:.2f}mm, {status}")
                        total_points += 1
                        if is_valid:
                            valid_points += 1
    
    # 分析眼睛轮廓点
    if 'eyes_contours' in key_coordinates:
        contour = key_coordinates['eyes_contours']
        if eye_side in contour and contour[eye_side] is not None:
            contour_points = contour[eye_side]
            if isinstance(contour_points, list) and len(contour_points) > 0:
                logging.info(f"眼睛轮廓点 ({len(contour_points)}个):")
                for i, point_data in enumerate(contour_points):
                    if point_data is not None:
                        point = np.array(point_data)
                        # 只使用前3维坐标进行计算，忽略visibility
                        point_3d = point[:3] if len(point) >= 3 else point
                        distance = np.linalg.norm(point_3d - eye_center)
                         
                        # 计算到眼球表面的距离
                        surface_distance = abs(distance - eye_radius)
                        surface_distance_mm = surface_distance * 1000
                        threshold_mm = thresholds["eye_contour"]
                         
                        # 判断位置
                        if distance < eye_radius:
                            position = "眼球内部"
                        else:
                            position = "眼球外部"
                         
                        # 判断是否符合规范
                        # 只有在眼球外部且距离在阈值范围内的点才是正确的
                        is_valid = (distance >= eye_radius) and (surface_distance_mm <= threshold_mm)
                        status = "✓ 符合规范" if is_valid else "✗ 超出规范"
                         
                        logging.info(f"  点{i+1}: {position}, 距离眼球表面: {surface_distance_mm:.2f}mm, {status}")
                        total_points += 1
                        if is_valid:
                            valid_points += 1
    
    # 输出统计信息
    if total_points > 0:
        valid_ratio = (valid_points / total_points) * 100
        logging.info(f"总结: {eye_side}眼共{total_points}个关键点, {valid_points}个符合规范, 符合率: {valid_ratio:.1f}%")
        
        # 质量评估
        if valid_ratio >= 90:
            quality = "优秀"
        elif valid_ratio >= 80:
            quality = "良好"
        elif valid_ratio >= 70:
            quality = "一般"
        else:
            quality = "较差"
        
        logging.info(f"数据质量评估: {quality}")
    else:
        logging.info(f"总结: {eye_side}眼没有可分析的关键点")

if __name__ == "__main__":
    main({})