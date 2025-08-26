import cv2
import numpy as np
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Tuple


class CameraDataManager:
    """
    数据管理器单例类
    负责管理BGR图像和深度图数据
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化（只在第一次创建时执行）"""
        if not hasattr(self, '_initialized'):
            self._data_dict = {}  # 私有成员变量：{frame_id: frame_data}
            self._resolution = None  # 分辨率 (width, height)
            self._initialized = True
    
    def add_frame(self, frame_id: int, bgr_image: np.ndarray, 
                  depth_map: Optional[np.ndarray] = None) -> None:
        """
        添加一帧数据
        
        :param frame_id: 帧ID
        :param bgr_image: BGR格式的图像数据 (H, W, 3)
        :param depth_map: 深度图数据 (H, W)，可选
        """
        # 验证图像格式
        if not self._validate_bgr_image(bgr_image):
            raise ValueError("BGR图像格式无效")
        
        # 验证深度图格式（如果提供）
        if depth_map is not None and not self._validate_depth_map(depth_map):
            raise ValueError("深度图格式无效")
        
        # 自动设置分辨率（使用第一帧的分辨率）
        if self._resolution is None:
            h, w = bgr_image.shape[:2]
            self._resolution = (w, h)
        
        # 构建帧数据
        frame_data = {
            "bgr_image": bgr_image,
            "depth_map": depth_map,
            "timestamp": datetime.now().isoformat()
        }
        
        # 存储数据
        self._data_dict[frame_id] = frame_data
    
    def get_image(self, frame_id: int) -> Optional[np.ndarray]:
        """
        获取指定帧的BGR图像
        
        :param frame_id: 帧ID
        :return: BGR图像数组，如果不存在则返回None
        """
        frame_data = self._data_dict.get(frame_id)
        return frame_data["bgr_image"] if frame_data else None
    
    def get_depth(self, frame_id: int) -> Optional[np.ndarray]:
        """
        获取指定帧的深度图
        
        :param frame_id: 帧ID
        :return: 深度图数组，如果不存在则返回None
        """
        frame_data = self._data_dict.get(frame_id)
        return frame_data["depth_map"] if frame_data else None
    
    def get_frame_data(self, frame_id: int) -> Optional[Dict]:
        """
        获取指定帧的完整数据
        
        :param frame_id: 帧ID
        :return: 帧数据字典，如果不存在则返回None
        """
        return self._data_dict.get(frame_id)
    
    def get_resolution(self) -> Optional[Tuple[int, int]]:
        """
        获取分辨率
        
        :return: (width, height) 元组，如果未设置则返回None
        """
        return self._resolution
    
    def get_frame_count(self) -> int:
        """
        获取当前帧数量
        
        :return: 帧数量
        """
        return len(self._data_dict)
    
    def clear_all(self) -> None:
        """清空所有数据"""
        self._data_dict.clear()
        self._resolution = None
    
    def remove_frame(self, frame_id: int) -> bool:
        """
        删除指定帧
        
        :param frame_id: 帧ID
        :return: 是否成功删除
        """
        if frame_id in self._data_dict:
            del self._data_dict[frame_id]
            return True
        return False
    
    def _validate_bgr_image(self, image: np.ndarray) -> bool:
        """
        验证BGR图像格式
        
        :param image: 待验证的图像
        :return: 是否有效
        """
        if not isinstance(image, np.ndarray):
            return False
        
        if len(image.shape) != 3:
            return False
        
        if image.shape[2] != 3:
            return False
        
        if image.dtype != np.uint8:
            return False
        
        return True
    
    def _validate_depth_map(self, depth_map: np.ndarray) -> bool:
        """
        验证深度图格式
        
        :param depth_map: 待验证的深度图
        :return: 是否有效
        """
        if not isinstance(depth_map, np.ndarray):
            return False
        
        if len(depth_map.shape) != 2:
            return False
        
        if depth_map.dtype != np.float32:
            return False
        
        return True


# 全局单例实例
data_manager = CameraDataManager()


# 便捷函数接口
def add_frame(frame_id: int, bgr_image: np.ndarray, 
              depth_map: Optional[np.ndarray] = None) -> None:
    """添加帧数据的便捷函数"""
    data_manager.add_frame(frame_id, bgr_image, depth_map)


def get_image(frame_id: int) -> Optional[np.ndarray]:
    """获取图像数据的便捷函数"""
    return data_manager.get_image(frame_id)


def get_depth(frame_id: int) -> Optional[np.ndarray]:
    """获取深度图数据的便捷函数"""
    return data_manager.get_depth(frame_id)


def get_resolution() -> Optional[Tuple[int, int]]:
    """获取分辨率的便捷函数"""
    return data_manager.get_resolution()


def get_frame_count() -> int:
    """获取帧数量的便捷函数"""
    return data_manager.get_frame_count()

# 接口测试
def main():
    """示例：使用RecgFitDataManager类并进行严格的数据检查"""
    cap = cv2.VideoCapture(0)
    frame_id = 0
    
    # 输入要打印的帧数
    try:
        print_frames = int(input("请输入要打印详细信息的帧数: "))
    except ValueError:
        print_frames = 2
        print(f"输入无效，使用默认值: {print_frames}")
    
    # 存储检查结果
    check_results = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头数据")
                break
            
            # 模拟深度图
            h, w = frame.shape[:2]
            depth_map = np.random.rand(h, w).astype(np.float32) * 2.0
            
            # 添加帧数据
            data_manager.add_frame(frame_id, frame, depth_map)
            
            # 严格检查指定数量的帧
            if frame_id < print_frames:
                print(f"\n{'='*50}")
                print(f"=== Frame {frame_id} 严格数据检查 ===")
                print(f"{'='*50}")
                
                # 1. 基础数据检查
                print("\n1. 基础数据检查:")
                print(f"   - 分辨率: {data_manager.get_resolution()}")
                print(f"   - 当前帧数量: {data_manager.get_frame_count()}")
                print(f"   - 帧ID存在性: {frame_id in data_manager._data_dict}")
                
                # 2. get_image接口严格检查
                print("\n2. get_image() 接口检查:")
                retrieved_image = get_image(frame_id)
                
                # 类型检查
                is_correct_type = isinstance(retrieved_image, np.ndarray)
                print(f"   ✅ 类型检查: {type(retrieved_image)} {'✓' if is_correct_type else '✗'}")
                
                if retrieved_image is not None:
                    # 维度检查
                    is_correct_dims = len(retrieved_image.shape) == 3
                    print(f"   ✅ 维度检查: {len(retrieved_image.shape)}D {'✓' if is_correct_dims else '✗'}")
                    
                    # 通道数检查
                    is_correct_channels = retrieved_image.shape[2] == 3
                    print(f"   ✅ 通道数检查: {retrieved_image.shape[2]} {'✓' if is_correct_channels else '✗'}")
                    
                    # 数据类型检查
                    is_correct_dtype = retrieved_image.dtype == np.uint8
                    print(f"   ✅ 数据类型检查: {retrieved_image.dtype} {'✓' if is_correct_dtype else '✗'}")
                    
                    # 数据范围检查
                    min_val, max_val = retrieved_image.min(), retrieved_image.max()
                    is_correct_range = 0 <= min_val <= max_val <= 255
                    print(f"   ✅ 数据范围检查: [{min_val}, {max_val}] {'✓' if is_correct_range else '✗'}")
                    
                    # 形状一致性检查
                    original_shape = frame.shape
                    retrieved_shape = retrieved_image.shape
                    is_shape_consistent = original_shape == retrieved_shape
                    print(f"   ✅ 形状一致性: {original_shape} vs {retrieved_shape} {'✓' if is_shape_consistent else '✗'}")
                    
                    # 数据一致性检查
                    is_data_consistent = np.array_equal(frame, retrieved_image)
                    print(f"   ✅ 数据一致性: {'✓' if is_data_consistent else '✗'}")
                
                # 3. get_depth接口严格检查
                print("\n3. get_depth() 接口检查:")
                retrieved_depth = get_depth(frame_id)
                
                # 类型检查
                is_correct_type = isinstance(retrieved_depth, np.ndarray)
                print(f"   ✅ 类型检查: {type(retrieved_depth)} {'✓' if is_correct_type else '✗'}")
                
                if retrieved_depth is not None:
                    # 维度检查
                    is_correct_dims = len(retrieved_depth.shape) == 2
                    print(f"   ✅ 维度检查: {len(retrieved_depth.shape)}D {'✓' if is_correct_dims else '✗'}")
                    
                    # 数据类型检查
                    is_correct_dtype = retrieved_depth.dtype == np.float32
                    print(f"   ✅ 数据类型检查: {retrieved_depth.dtype} {'✓' if is_correct_dtype else '✗'}")
                    
                    # 数据范围检查（深度图应该是正数）
                    min_val, max_val = retrieved_depth.min(), retrieved_depth.max()
                    is_correct_range = min_val >= 0
                    print(f"   ✅ 数据范围检查: [{min_val:.3f}, {max_val:.3f}] {'✓' if is_correct_range else '✗'}")
                    
                    # 形状一致性检查
                    original_shape = depth_map.shape
                    retrieved_shape = retrieved_depth.shape
                    is_shape_consistent = original_shape == retrieved_shape
                    print(f"   ✅ 形状一致性: {original_shape} vs {retrieved_shape} {'✓' if is_shape_consistent else '✗'}")
                    
                    # 数据一致性检查
                    is_data_consistent = np.array_equal(depth_map, retrieved_depth)
                    print(f"   ✅ 数据一致性: {'✓' if is_data_consistent else '✗'}")
                
                # 4. 接口规范检查
                print("\n4. 接口规范检查:")
                print("   📋 接口要求:")
                print("      - get_image(): np.ndarray(shape=(H,W,3), dtype=np.uint8)")
                print("      - get_depth(): np.ndarray(shape=(H,W), dtype=np.float32)")
                
                # 5. 性能检查
                print("\n5. 性能检查:")
                import time
                start_time = time.time()
                _ = get_image(frame_id)
                image_time = time.time() - start_time
                print(f"   ⏱️  get_image() 耗时: {image_time*1000:.2f}ms")
                
                start_time = time.time()
                _ = get_depth(frame_id)
                depth_time = time.time() - start_time
                print(f"   ⏱️  get_depth() 耗时: {depth_time*1000:.2f}ms")
                
                # 记录检查结果
                check_results.append({
                    'frame_id': frame_id,
                    'image_checks': {
                        'type': is_correct_type,
                        'dims': is_correct_dims if retrieved_image is not None else False,
                        'channels': is_correct_channels if retrieved_image is not None else False,
                        'dtype': is_correct_dtype if retrieved_image is not None else False,
                        'range': is_correct_range if retrieved_image is not None else False,
                        'shape_consistent': is_shape_consistent if retrieved_image is not None else False,
                        'data_consistent': is_data_consistent if retrieved_image is not None else False
                    },
                    'depth_checks': {
                        'type': isinstance(retrieved_depth, np.ndarray),
                        'dims': len(retrieved_depth.shape) == 2 if retrieved_depth is not None else False,
                        'dtype': retrieved_depth.dtype == np.float32 if retrieved_depth is not None else False,
                        'range': retrieved_depth.min() >= 0 if retrieved_depth is not None else False,
                        'shape_consistent': retrieved_depth.shape == depth_map.shape if retrieved_depth is not None else False,
                        'data_consistent': np.array_equal(depth_map, retrieved_depth) if retrieved_depth is not None else False
                    }
                })
                
            else:
                print(f"[✔] Frame {frame_id} added, 当前保留 {data_manager.get_frame_count()} 帧")
            
            # 显示图像并检查ESC键 - 确保这行在每次循环都执行
            cv2.imshow("RGB", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                print("\n检测到ESC键，正在退出...")
                break
            elif key == ord('q'):  # 也可以按q退出
                print("\n检测到q键，正在退出...")
                break
            
            frame_id += 1
    
    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，正在退出...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 6. 总结报告
        print(f"\n{'='*50}")
        print("=== 严格检查总结报告 ===")
        print(f"{'='*50}")
        
        if check_results:
            total_frames = len(check_results)
            image_pass_count = sum(1 for r in check_results if all(r['image_checks'].values()))
            depth_pass_count = sum(1 for r in check_results if all(r['depth_checks'].values()))
            
            print(f"\n📊 检查统计:")
            print(f"   - 检查帧数: {total_frames}")
            print(f"   - 图像接口完全通过: {image_pass_count}/{total_frames}")
            print(f"   - 深度图接口完全通过: {depth_pass_count}/{total_frames}")
            
            print(f"\n✅ 接口一致性:")
            print(f"   - get_image() 符合规范: {'✓' if image_pass_count == total_frames else '✗'}")
            print(f"   - get_depth() 符合规范: {'✓' if depth_pass_count == total_frames else '✗'}")
            
            if image_pass_count == total_frames and depth_pass_count == total_frames:
                print(f"\n🎉 所有检查通过！接口完全符合规范！")
            else:
                print(f"\n⚠️  存在检查失败项，请查看详细报告")


if __name__ == "__main__":
    main()