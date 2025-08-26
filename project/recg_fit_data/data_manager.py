FITTING_TYPE = ["pupil", "iris", "inner_canthus", "upper_eyelid", "lower_eyelid", "outer_canthus"]
EYE_TYPE = ["left", "right"]

from typing import Dict, List, Union
import numpy as np
import logging

# 定义坐标点类型
CoordinatePoint = np.ndarray  # shape: (4,), dtype: float32 [x, y, z, visibility]

KeyCoordinates = Dict[str, Dict[str, Union[List[CoordinatePoint], None]]]

# 单眼关键点坐标类型（给CenterFitter使用）
SingleEyeKeyCoordinates = Dict[str, Union[List[CoordinatePoint], None]]


class RecgFitDataManager:
    
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RecgFitDataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, key_coordinates: KeyCoordinates = None, debug_log: bool = False, convert2mm: bool = True):
        try:
            self._key_coordinates = key_coordinates or {
                'left': {'pupil': None, 'iris': [], 'inner_canthus': [], 'upper_eyelid': [], 'lower_eyelid': [], 'outer_canthus': []},
                'right': {'pupil': None, 'iris': [], 'inner_canthus': [], 'upper_eyelid': [], 'lower_eyelid': [], 'outer_canthus': []}
            }
            self._debug_log = debug_log
            self._convert2mm = convert2mm
            self._convert_coords_2mm()
        except Exception as e:
            logging.error(f"RecgFitDataManager初始化失败: {e}")
            raise
        
        
    def _convert_coords_2mm(self) -> None:
        if self._convert2mm:
            for eye in EYE_TYPE:
                for fitting_type in FITTING_TYPE:
                    coordinate_point = self._key_coordinates[eye][fitting_type]
                    if isinstance(coordinate_point, np.ndarray) and coordinate_point.dtype.kind in 'fc':
                        # 检查是否已经是毫米单位（坐标值大于10）
                        if len(coordinate_point) >= 3:
                            # 检查前3维坐标是否已经大于10（毫米单位）
                            if np.any(np.abs(coordinate_point[:3]) > 10.0):
                                # 已经是毫米单位，跳过转换
                                continue
                        
                        # 确保是数值类型的numpy数组
                        # 只转换坐标部分（前3维），保持visibility（第4维）不变
                        if len(coordinate_point) >= 4:
                            # 复制数组，避免修改原始数据
                            converted_point = coordinate_point.copy()
                            # 只转换前3维坐标（x, y, z）
                            converted_point[:3] = coordinate_point[:3] * 1000
                            # 保持第4维visibility不变
                            converted_point[3] = coordinate_point[3]
                            self._key_coordinates[eye][fitting_type] = converted_point
                        else:
                            # 如果只有3维，全部转换
                            self._key_coordinates[eye][fitting_type] = coordinate_point * 1000
                    elif isinstance(coordinate_point, list):
                        # 如果是列表，处理列表中的每个元素
                        converted_list = []
                        for point in coordinate_point:
                            if isinstance(point, np.ndarray) and point.dtype.kind in 'fc':
                                # 检查是否已经是毫米单位
                                if len(point) >= 3 and np.any(np.abs(point[:3]) > 10.0):
                                    # 已经是毫米单位，直接添加
                                    converted_list.append(point)
                                    continue
                                
                                if len(point) >= 4:
                                    # 复制数组，避免修改原始数据
                                    converted_point = point.copy()
                                    # 只转换前3维坐标（x, y, z）
                                    converted_point[:3] = point[:3] * 1000
                                    # 保持第4维visibility不变
                                    converted_point[3] = point[3]
                                    converted_list.append(converted_point)
                                else:
                                    # 如果只有3维，全部转换
                                    converted_list.append(point * 1000)
                            else:
                                converted_list.append(point)
                        self._key_coordinates[eye][fitting_type] = converted_list
    # ==================== Getter方法 ====================
        
    def get_key_coordinates(self) -> KeyCoordinates:
        return self._key_coordinates
    
    def get_eye_coordinates(self, eye: str) -> SingleEyeKeyCoordinates:
        if self._check_eye_type(eye):
            return self._key_coordinates[eye]
        else:
            raise ValueError(f"无效的眼睛类型: {eye}")
    
    def get_coordinate_point(self, eye: str, fitting_type: str) -> List[CoordinatePoint]:
        """
        获取指定眼睛和类型的坐标点，统一返回列表格式
        
        Args:
            eye: 眼睛类型 ("left" 或 "right")
            fitting_type: 拟合类型
            
        Returns:
            List[CoordinatePoint]: 坐标点列表，如果没有数据则返回空列表
        """
        self._log_debug()
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")
        
        if fitting_type not in FITTING_TYPE:
            raise ValueError(f"无效的拟合类型: {fitting_type}")
        
        coordinate_point = self._key_coordinates[eye][fitting_type]
        
        # 确保返回列表格式
        if coordinate_point is None:
            return []
        elif isinstance(coordinate_point, list):
            return coordinate_point
        elif isinstance(coordinate_point, np.ndarray):
            # 如果是单个坐标点，包装成列表
            return [coordinate_point]
        else:
            logging.warning(f"未知的坐标点类型: {type(coordinate_point)}")
            return []
    
    def get_point_count(self, eye: str) -> int:
        """
        获取指定眼睛的坐标点数量
        """
        total_count = 0
        for fitting_type in FITTING_TYPE:
            points = self.get_coordinate_point(eye, fitting_type)
            if points:
                total_count += len(points)
        return total_count
    
    def get_mean_visibility(self, eye: str) -> float:
        """
        获取指定眼睛的置信度
        """
        visibility_values = []
        for fitting_type in FITTING_TYPE:
            points = self.get_coordinate_point(eye, fitting_type)
            if points:
                for point in points:
                    if isinstance(point, np.ndarray) and len(point) >= 4:
                        visibility_values.append(float(point[3]))
        
        if visibility_values:
            return np.mean(visibility_values)
        else:
            return 0.0
        
    
    
    
    # ==================== Setter方法 ====================
    
    def set_key_coordinates(self, key_coordinates: KeyCoordinates) -> None:
        """
        设置完整的关键点坐标数据
        
        Args:
            key_coordinates: 完整的关键点坐标字典
        """
        self._validate_key_coordinates(key_coordinates)
        self._key_coordinates = key_coordinates
        # 转换坐标到毫米
        self._convert_coords_2mm()
        if self._debug_log:
            logging.info("RecgFitDataManager: 已更新完整的关键点坐标数据")
        self._log_debug()
    
    def set_eye_coordinates(self, eye: str, eye_coordinates: SingleEyeKeyCoordinates) -> None:
        """
        设置指定眼睛的关键点坐标数据
        
        Args:
            eye: 眼睛类型 ("left" 或 "right")
            eye_coordinates: 单眼关键点坐标字典
        """
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")
        
        self._validate_eye_coordinates(eye_coordinates)
        self._key_coordinates[eye] = eye_coordinates
        # 转换坐标到毫米
        self._convert_coords_2mm()
        if self._debug_log:
            logging.info(f"RecgFitDataManager: 已更新{eye}眼的关键点坐标数据")
        self._log_debug()
    
    def set_coordinate_point(self, eye: str, fitting_type: str, coordinate_point: List[CoordinatePoint]) -> None:
        """
        设置指定眼睛和类型的关键点坐标
        
        Args:
            eye: 眼睛类型 ("left" 或 "right")
            fitting_type: 拟合类型 (pupil, iris, inner_canthus, upper_eyelid, lower_eyelid, outer_canthus)
            coordinate_point: 坐标点或坐标点列表
        """
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")
        
        if fitting_type not in FITTING_TYPE:
            raise ValueError(f"无效的拟合类型: {fitting_type}")
        
        self._validate_coordinate_point(coordinate_point)
        self._key_coordinates[eye][fitting_type] = coordinate_point
        # 转换坐标到毫米
        self._convert_coords_2mm()
        if self._debug_log:
            logging.info(f"RecgFitDataManager: 已更新{eye}眼{fitting_type}的坐标点数据")
        self._log_debug()
    
    
    
    def clear_eye_data(self, eye: str) -> None:
        """
        清空指定眼睛的所有数据
        
        Args:
            eye: 眼睛类型 ("left" 或 "right")
        """
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")
        
        self._key_coordinates[eye] = {
            'pupil': None, 
            'iris': [], 
            'inner_canthus': [], 
            'upper_eyelid': [], 
            'lower_eyelid': [], 
            'outer_canthus': []
        }
        if self._debug_log:
            logging.info(f"RecgFitDataManager: 已清空{eye}眼的所有数据")
    
    def clear_all_data(self) -> None:
        """
        清空所有眼睛的数据
        """
        for eye in EYE_TYPE:
            self.clear_eye_data(eye)
        if self._debug_log:
            logging.info("RecgFitDataManager: 已清空所有数据")
    
    # ==================== 验证方法 ====================
    
    def _validate_key_coordinates(self, key_coordinates: KeyCoordinates) -> None:
        """
        验证关键点坐标数据的有效性
        
        Args:
            key_coordinates: 关键点坐标字典
        """
        if not isinstance(key_coordinates, dict):
            raise ValueError("key_coordinates必须是字典类型")
        
        for eye in EYE_TYPE:
            if eye not in key_coordinates:
                raise ValueError(f"缺少{eye}眼的数据")
            
            eye_data = key_coordinates[eye]
            if not isinstance(eye_data, dict):
                raise ValueError(f"{eye}眼数据必须是字典类型")
            
            for fitting_type in FITTING_TYPE:
                if fitting_type not in eye_data:
                    raise ValueError(f"{eye}眼缺少{fitting_type}类型的数据")
    
    def _validate_eye_coordinates(self, eye_coordinates: SingleEyeKeyCoordinates) -> None:
        """
        验证单眼关键点坐标数据的有效性
        
        Args:
            eye_coordinates: 单眼关键点坐标字典
        """
        if not isinstance(eye_coordinates, dict):
            raise ValueError("eye_coordinates必须是字典类型")
        
        for fitting_type in FITTING_TYPE:
            if fitting_type not in eye_coordinates:
                raise ValueError(f"缺少{fitting_type}类型的数据")
    
    def _validate_coordinate_point(self, coordinate_point: Union[CoordinatePoint, List[CoordinatePoint], None]) -> None:
        """
        验证坐标点数据的有效性
        
        Args:
            coordinate_point: 坐标点或坐标点列表
        """
        if coordinate_point is None:
            return
        
        if isinstance(coordinate_point, np.ndarray):
            if coordinate_point.shape != (4,) or coordinate_point.dtype != np.float32:
                raise ValueError("单个坐标点必须是shape为(4,)的float32数组")
        elif isinstance(coordinate_point, list):
            for point in coordinate_point:
                if not isinstance(point, np.ndarray) or point.shape != (4,) or point.dtype != np.float32:
                    raise ValueError("坐标点列表中的每个元素必须是shape为(4,)的float32数组")
        else:
            raise ValueError("坐标点必须是numpy数组类型或numpy数组列表")
        
    def _check_eye_type(self, eye: str) -> bool:
        if eye in EYE_TYPE:
            return True
        else:
            return False
        
    def _log_debug(self) -> None:
        def _show_coordinate_point(eye:str, fitting_type:str, coordinate_point: CoordinatePoint) -> None:
            try:
                if isinstance(coordinate_point, np.ndarray) and len(coordinate_point) >= 4:
                    logging.debug(f"{eye}眼{fitting_type}: ({coordinate_point[0]:.2f}mm, {coordinate_point[1]:.2f}mm, {coordinate_point[2]:.2f}mm, visibility:{coordinate_point[3]:.2f})")
                else:
                    logging.debug(f"{eye}眼{fitting_type}: 坐标点格式错误: {coordinate_point}")
            except Exception as e:
                logging.debug(f"{eye}眼{fitting_type}: 显示坐标点时出错: {e}")

                
        if self._debug_log:
            for eye in EYE_TYPE:
                for fitting_type in FITTING_TYPE:
                    coordinate_point = self._key_coordinates[eye][fitting_type]
                    if coordinate_point is None:
                        logging.debug(f"{eye}眼{fitting_type}: 无数据")
                    elif isinstance(coordinate_point, list):
                        if coordinate_point:
                            for point in coordinate_point:
                                _show_coordinate_point(eye, fitting_type, point)
                        else:
                            logging.debug(f"{eye}眼{fitting_type}: 空列表")
                    elif isinstance(coordinate_point, np.ndarray):
                        _show_coordinate_point(eye, fitting_type, coordinate_point)
                    else:
                        logging.debug(f"{eye}眼{fitting_type}: 未知类型 {type(coordinate_point)}")
                        
RECG_FIT_DATA_MANAGER = RecgFitDataManager()
        
    
    











