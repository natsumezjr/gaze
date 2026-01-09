"""动态效果模块 - 实现脉冲和发光动画效果"""
import math
import time
from typing import Optional, Tuple, Callable
from project.data.data_models import Point2D
from project.config.logging_config import setup_logging, get_logger
from project.client.kappa.ui_config import ANIMATION_CONFIG

setup_logging()
logger = get_logger(__name__)

class AnimationEffect:
    """动态效果管理器"""
    
    def __init__(self, canvas, detection_range_px: Optional[int] = None):
        """
        初始化动态效果
        
        Args:
            canvas: Tkinter Canvas对象
            detection_range_px: 检测范围（像素），如果为None则使用配置值
        """
        self.canvas = canvas
        self.detection_range = detection_range_px or ANIMATION_CONFIG["detection_range_px"]
        
        # 动画状态
        self.is_animating = False
        self.animation_id = None
        self.start_time = None
        
        # 当前目标点
        self.target_point: Optional[Point2D] = None
        
        # 当前粗略视线位置
        self.rough_gaze: Optional[Point2D] = None
        
        # 更新回调函数
        self.update_callback: Optional[Callable[[float, float], None]] = None
        
        # 原始圆圈大小和颜色（用于动画）
        self.original_size = None
        self.original_colors = None
        
        logger.info(f"动态效果管理器已初始化，检测范围: {self.detection_range}px")
    
    def set_target_point(self, point: Point2D):
        """设置目标标定点"""
        self.target_point = point
        logger.debug(f"目标点已设置: {point}")
    
    def update_rough_gaze(self, gaze_point: Point2D):
        """更新粗略视线位置"""
        self.rough_gaze = gaze_point
        self._check_and_animate()
    
    def _check_and_animate(self):
        """检查距离并触发动画"""
        if not self.target_point or not self.rough_gaze:
            return
        
        # 计算距离
        distance = math.sqrt(
            (self.target_point.x - self.rough_gaze.x) ** 2 +
            (self.target_point.y - self.rough_gaze.y) ** 2
        )
        
        # 如果在检测范围内，启动动画
        if distance <= self.detection_range:
            if not self.is_animating:
                self._start_animation()
        else:
            if self.is_animating:
                self._stop_animation()
    
    def _start_animation(self):
        """启动动画"""
        self.is_animating = True
        self.start_time = time.time()
        self._animate()
        logger.debug("动态效果已启动")
    
    def _stop_animation(self):
        """停止动画"""
        self.is_animating = False
        if self.animation_id:
            self.canvas.after_cancel(self.animation_id)
            self.animation_id = None
        logger.debug("动态效果已停止")
    
    def _animate(self):
        """执行动画循环"""
        if not self.is_animating:
            return
        
        current_time = time.time()
        elapsed = (current_time - self.start_time) * 1000  # 转换为毫秒
        
        # 脉冲效果：计算缩放比例
        pulse_cycle = elapsed % ANIMATION_CONFIG["pulse_duration"]
        pulse_progress = pulse_cycle / ANIMATION_CONFIG["pulse_duration"]
        
        # 使用正弦波实现平滑的脉冲效果
        pulse_scale = ANIMATION_CONFIG["pulse_min_scale"] + \
                     (ANIMATION_CONFIG["pulse_max_scale"] - ANIMATION_CONFIG["pulse_min_scale"]) * \
                     (math.sin(pulse_progress * 2 * math.pi) + 1) / 2
        
        # 发光效果：计算强度
        glow_intensity = ANIMATION_CONFIG["glow_intensity_min"] + \
                        (ANIMATION_CONFIG["glow_intensity_max"] - ANIMATION_CONFIG["glow_intensity_min"]) * \
                        (math.sin(pulse_progress * 2 * math.pi) + 1) / 2
        
        # 更新Canvas中的元素（通过回调）
        if self.update_callback:
            try:
                self.update_callback(pulse_scale, glow_intensity)
            except Exception as e:
                logger.error(f"动画更新回调执行失败: {e}", exc_info=True)
        
        # 继续下一帧
        self.animation_id = self.canvas.after(16, self._animate)  # 约60fps
    
    def set_update_callback(self, callback: Callable[[float, float], None]):
        """设置更新回调（用于更新Canvas元素）"""
        self.update_callback = callback
    
    def cleanup(self):
        """清理资源"""
        self._stop_animation()
        self.target_point = None
        self.rough_gaze = None
        self.update_callback = None
