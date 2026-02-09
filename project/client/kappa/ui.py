import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from project.data.data_models import Point2D
from project.config.logging_config import setup_logging 
from project.managers import CALLBACK_MANAGER
from project.events.event_types import KapaCallbackEventTypes, CALIBRATION_START_REQUEST, ROUGH_GAZE_UPDATE, CALIBRATION_POINT_SUBMIT, CALIBRATION_COMPLETE
from project.client.kappa.ui_config import (
    BackgroundColor, BACKGROUND_COLOR_MAP, TEXT_COLOR_MAP,
    CALIBRATION_POINT_COLORS, CALIBRATION_POINTS, ANIMATION_CONFIG,
    SCREEN_MARGIN_RATIO, CIRCLE_SIZE_RATIO
)
from project.client.kappa.keyboard_handler import KeyboardHandler
from project.client.kappa.animation_effect import AnimationEffect
from project.client.kappa.dotted_surface import DottedSurface
from project.client.kappa.gooey_text import GooeyText
from project.data.data_models import CalibrationRequest, CalibrationResponse
logger = setup_logging(__name__)


# 配置类
@dataclass
class CalibrationConfig:
    """校准配置"""
    points: List[Tuple[int, int]]  # 校准点坐标 (百分比)
    colors: List[Tuple[str, str]]  # 颜色配置 (背景色, 前景色)
    circle_size: int = 50
    animation_duration: float = 1.0
    countdown_duration: int = 3

@dataclass
class CalibrationResult:
    points: List[Point2D]
    accuracy: float
    error_message: Optional[str]
    calibration_time: Optional[datetime]
    
    def __str__(self):
        return f"目标点: {self.points}\n准确度: {self.accuracy}\n错误信息: {self.error_message}\n校准时间: {self.calibration_time}"

class CalibrationState(Enum):
    """校准状态"""
    IDLE = "idle"
    RUNNING = "running"
    COUNTDOWN = "countdown"
    CALIBRATING = "calibrating"
    WAITING_FOR_USER_CONFIRM = "waiting_for_user_confirm"  # 等待用户按键确认
    COMPLETED = "completed"

class EyeCalibrationApp:
    """眼动校准应用 - UI主动发送点位，支持多背景色系统（单例模式）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self, callback_manager=None):
        # 单例模式：避免重复初始化
        if hasattr(self, '_initialized'):
            return
        
        self.callback_manager = callback_manager or CALLBACK_MANAGER
        self.root = tk.Tk()
        self.root.title("眼动校准系统")
        
        # 使用 Tkinter 跨平台方法获取屏幕分辨率
        self.root.update_idletasks()  # 确保窗口已初始化
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        
        # 背景色配置（从ui_config导入）
        self.background_colors = [BackgroundColor.BLACK, BackgroundColor.GRAY, BackgroundColor.WHITE]
        self.background_color_map = BACKGROUND_COLOR_MAP
        self.text_color_map = TEXT_COLOR_MAP
        self.current_background_index = 0
        self.current_point_index = 0
        self.total_points = 27  # 3组 × 9点
        
        # 状态变量
        self.state = CalibrationState.IDLE
        self.is_calibrating = False
        
        # 当前标定请求数据（存储 frame_id 和 eye_type）
        self.current_calibration_request: Optional[CalibrationRequest] = None
        
        # 配置
        self.config = CalibrationConfig(
            points=CALIBRATION_POINTS,
            colors=CALIBRATION_POINT_COLORS
        )
        
        # 校准圆圈
        self.calibration_circle = None
        self.inner_circle = None
        self.outer_circle = None  # 用于动画更新
        self.original_circle_size = None  # 保存原始大小
        
        # 进度显示
        self.progress_frame = None
        self.progress_bar = None
        self.progress_label = None
        self.progress_text = None
        
        # 倒计时显示
        self.countdown_label = None
        
        # 提示信息
        self.instruction_label = None
        

        # 动态点阵背景
        self.dotted_background = None
        
        # GooeyText 文字效果
        self.gooey_text = None

        # 实现点显示相关（需要在 setup_ui 之前初始化，因为 setup_ui 会调用 _redraw_gaze_points）
        self.gaze_points_canvas = None  # 用于显示实现点的 Canvas
        self.gaze_points: List[Tuple[Point2D, str]] = []  # 存储实现点 (point, color)
        self.gaze_point_color = "#0000FF"  # 默认蓝色
        self.gaze_point_size = 8  # 实现点大小（半径），增大以便更容易看到

        
        # 注册回调（接收后端启动请求和粗略视线位置）
        self._register_callbacks()
        
        # 界面元素
        self.setup_ui()
        
        # 初始化键盘处理器
        self.keyboard_handler = KeyboardHandler(
            self.root,
            on_confirm=self._confirm_and_submit_point
        )
        
        # 动态效果将在创建校准圆圈时初始化
        self.animation_effect = None
        
        self._initialized = True
        
        logger.info(f"屏幕分辨率: {self.screen_width}x{self.screen_height}")
        logger.info(f"校准点数量: {len(self.config.points)} × 3组 = {self.total_points}个点")
    
    def _register_callbacks(self):
        """注册回调函数"""
        self.callback_manager.register(
            CALIBRATION_START_REQUEST,
            self._on_calibration_start_request
        )
        self.callback_manager.register(
            ROUGH_GAZE_UPDATE,
            self._on_rough_gaze_update
        )
        logger.info("回调函数已注册")
    
    def _on_rough_gaze_update(self, calibration_request: CalibrationRequest):
        """接收粗略视线位置更新（包含 CalibrationRequest）"""
        # 存储当前的 calibration_request（包含 frame_id 和 eye_type）
        self.current_calibration_request = calibration_request
        
        if self.animation_effect:
            # 在主线程中更新粗略视线位置
            self.root.after(0, lambda: self.animation_effect.update_rough_gaze(calibration_request.intersection))
    
    def _on_calibration_start_request(self, frame_id: int = None):
        """收到校准启动请求（从后端回调）"""
        logger.info(f"收到校准启动请求，frame_id: {frame_id}")
        if self.state == CalibrationState.IDLE:
            # 在主线程中启动校准
            self.root.after(0, self.start_calibration)
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建全屏 Canvas 用于显示实现点（背景层）
        # 使用父窗口的背景色，实现视觉上的"透明"效果
        parent_bg = self.root.cget('bg')
        self.gaze_points_canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            bg=parent_bg,  # 使用父窗口背景色
            highlightthickness=0
        )
        self.gaze_points_canvas.place(x=0, y=0)
        # 注意：Canvas 的层级由创建顺序决定，先创建的在下层
        # 由于 Canvas 在其他 UI 元素之前创建，它自然在最底层
        # 禁用鼠标事件，让点击穿透到下层
        self.gaze_points_canvas.bind("<Button-1>", lambda e: None)
        self.gaze_points_canvas.bind("<Button-2>", lambda e: None)
        self.gaze_points_canvas.bind("<Button-3>", lambda e: None)
        
        # 初始背景色（黑色）
        self._set_background_color(BackgroundColor.BLACK)
        
        # 创建动态点阵背景（先创建，确保在最底层）
        self.dotted_background = DottedSurface(
            self.root,
            theme='dark',
            bg_color='#000000'
        )
        self.dotted_background.place(x=0, y=0, relwidth=1, relheight=1)
        
        # 绑定回车键开始校准
        self.root.bind('<Return>', lambda event: self.start_calibration())
        self.root.focus_set()  # 确保窗口可以接收键盘事件
        
        # 创建 GooeyText 文字效果
        self.gooey_text = GooeyText(
            self.root,
            texts=["EyeCalibration"],
            morph_time=1.0,
            cooldown_time=0.25,
            font_size=60,
            text_color="#FFFFFF",
            bg_color="transparent"
        )
        self.gooey_text.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # 进度指示器（初始隐藏）
        self.progress_frame = tk.Frame(self.root, bg=self.background_color_map[BackgroundColor.BLACK])
        self.progress_label = tk.Label(
            self.progress_frame,
            text="校准进度",
            font=("Arial", 12, "bold"),
            fg=self.text_color_map[BackgroundColor.BLACK],
            bg=self.background_color_map[BackgroundColor.BLACK]
        )
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            length=min(400, self.screen_width * 0.3),
            mode='determinate'
        )
        self.progress_bar.pack(pady=5)
        
        self.progress_text = tk.Label(
            self.progress_frame,
            text=f"0 / {self.total_points}",
            font=("Arial", 10),
            fg=self.text_color_map[BackgroundColor.BLACK],
            bg=self.background_color_map[BackgroundColor.BLACK]
        )
        self.progress_text.pack(pady=5)
        
        # 倒计时显示（初始隐藏）
        self.countdown_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 48, "bold"),
            fg=self.text_color_map[BackgroundColor.BLACK],
            bg=self.background_color_map[BackgroundColor.BLACK]
        )
        
        # 提示信息（初始隐藏）
        self.instruction_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 16),
            fg="#FFFF00",  # 黄色提示文字
            bg=self.background_color_map[BackgroundColor.BLACK]
        )
    
    def _set_background_color(self, bg_color: BackgroundColor):
        """设置背景色"""
        color_hex = self.background_color_map[bg_color]
        text_color = self.text_color_map[bg_color]
        
        self.root.configure(bg=color_hex)
        
        # 更新 Canvas 的背景色以匹配父窗口
        if self.gaze_points_canvas:
            self.gaze_points_canvas.configure(bg=color_hex)
        
        # 更新所有UI元素的背景色和文字颜色
        self._update_ui_colors(color_hex, text_color)
        
        logger.info(f"背景色切换为: {bg_color.value}")
    
    def _update_ui_colors(self, bg_color: str, text_color: str):
        """更新所有UI元素的颜色"""
        # 更新已存在的UI元素
        if hasattr(self, 'progress_frame') and self.progress_frame:
            self.progress_frame.config(bg=bg_color)
        if hasattr(self, 'progress_label') and self.progress_label:
            self.progress_label.config(bg=bg_color, fg=text_color)
        if hasattr(self, 'progress_text') and self.progress_text:
            self.progress_text.config(bg=bg_color, fg=text_color)
        if hasattr(self, 'countdown_label') and self.countdown_label:
            self.countdown_label.config(bg=bg_color, fg=text_color)
        if hasattr(self, 'instruction_label') and self.instruction_label:
            self.instruction_label.config(bg=bg_color)
        
        # 重新绘制实现点（背景切换时需要重新绘制）
        self._redraw_gaze_points()
    
    def add_gaze_point(self, point: Point2D, color: Optional[str] = None):
        """
        添加实现点（实际注视点）到 UI 上显示
        
        Args:
            point: 实现点坐标 (Point2D)
            color: 点的颜色（十六进制字符串，如 "#0000FF"），默认为蓝色
        """
        if color is None:
            color = self.gaze_point_color
        
        # 存储点
        self.gaze_points.append((point, color))
        
        # 在主线程中绘制点
        self.root.after(0, lambda: self._draw_gaze_point(point, color))
    
    def _draw_gaze_point(self, point: Point2D, color: str):
        """在 Canvas 上绘制单个实现点"""
        if not self.gaze_points_canvas:
            return
        
        try:
            x, y = int(point.x), int(point.y)
            size = self.gaze_point_size
            
            # 绘制圆形点
            self.gaze_points_canvas.create_oval(
                x - size, y - size,
                x + size, y + size,
                fill=color,
                outline=color,
                width=1,
                tags="gaze_point"
            )
        except Exception as e:
            logger.error(f"绘制实现点失败: {e}")
    
    def _redraw_gaze_points(self):
        """重新绘制所有实现点（用于背景切换等情况）"""
        if not self.gaze_points_canvas:
            return
        
        # 检查 gaze_points 是否已初始化
        if not hasattr(self, 'gaze_points'):
            return
        
        # 清除所有现有的实现点
        self.gaze_points_canvas.delete("gaze_point")
        
        # 重新绘制所有点
        for point, color in self.gaze_points:
            self._draw_gaze_point(point, color)
    
    def clear_gaze_points(self):
        """清除所有实现点"""
        self.gaze_points.clear()
        if self.gaze_points_canvas:
            self.gaze_points_canvas.delete("gaze_point")
    
    def set_gaze_point_color(self, color: str):
        """
        设置实现点的默认颜色
        
        Args:
            color: 颜色（十六进制字符串，如 "#0000FF" 表示蓝色）
        """
        self.gaze_point_color = color
    
    def set_gaze_point_size(self, size: int):
        """
        设置实现点的大小（半径）
        
        Args:
            size: 点的半径（像素）
        """
        self.gaze_point_size = size
        # 重新绘制所有点以应用新大小
        self._redraw_gaze_points()
    
    def _update_instruction(self, message: str):
        """更新提示信息"""
        if self.instruction_label:
            self.instruction_label.config(text=message)
            if not self.instruction_label.winfo_viewable():
                self.instruction_label.pack(expand=True, pady=20)
            self.root.update()
    
    def _hide_instruction(self):
        """隐藏提示信息"""
        if self.instruction_label:
            self.instruction_label.pack_forget()
    
    def update_status(self, message: str):
        """更新状态显示"""
        # 状态显示已移除，此方法保留用于兼容性
        self.root.update()
    
    def start_calibration(self):
        """开始校准流程 - UI主动显示第一个点"""
        if self.is_calibrating:
            return
        
        self.is_calibrating = True
        self.state = CalibrationState.COUNTDOWN
        self.current_background_index = 0
        self.current_point_index = 0
        
        # 隐藏并停止动态背景
        if self.dotted_background:
            self.dotted_background.stop_animation()
            self.dotted_background.place_forget()
        
        # 隐藏并停止 GooeyText 文字效果
        if self.gooey_text:
            self.gooey_text.stop_animation()
            self.gooey_text.place_forget()
        
        # 隐藏所有文字元素（进度指示器、倒计时、提示信息等）
        if self.progress_frame:
            self.progress_frame.pack_forget()
        if self.countdown_label:
            self.countdown_label.pack_forget()
        if self.instruction_label:
            self.instruction_label.pack_forget()
        
        # 设置第一个背景色（黑色）
        self._set_background_color(self.background_colors[0])
        
        # 跳过倒计时，直接开始校准序列
        self.start_calibration_sequence()
    
    def start_countdown(self):
        """开始倒计时"""
        self.state = CalibrationState.COUNTDOWN
        self.update_status("准备开始校准...")
        
        # 显示倒计时
        self.countdown_label.pack(expand=True)
        self.countdown_animation(3)
    
    def countdown_animation(self, count: int):
        """倒计时动画"""
        if count > 0:
            self.countdown_label.config(text=str(count))
            self.root.after(1000, lambda: self.countdown_animation(count - 1))
        else:
            # 倒计时结束，隐藏倒计时标签
            self.countdown_label.pack_forget()
            # 开始校准序列 - 主动显示第一个点
            self.start_calibration_sequence()
    
    def start_calibration_sequence(self):
        """开始校准序列 - 主动显示第一个点"""
        self.state = CalibrationState.CALIBRATING
        self._show_next_point()
    
    def _show_next_point(self):
        """显示下一个校准点"""
        if self.current_point_index >= self.total_points:
            self._finish_calibration()
            return
        
        # 计算当前是第几组的第几个点
        point_in_group = self.current_point_index % 9
        
        # 如果开始新的一组（point_in_group == 0 且不是第一个点），切换背景色
        if point_in_group == 0 and self.current_point_index > 0:
            self.current_background_index += 1
            if self.current_background_index < len(self.background_colors):
                self._set_background_color(self.background_colors[self.current_background_index])
        
        # 显示当前点
        point_percent = self.config.points[point_in_group]
        self._display_calibration_point(point_percent)
        
        # 不更新进度显示（隐藏所有文字）
    
    def _display_calibration_point(self, point_percent: Tuple[int, int]):
        """显示校准点"""
        # 计算屏幕坐标
        x = int((point_percent[0] / 100) * self.screen_width)
        y = int((point_percent[1] / 100) * self.screen_height)
        
        # 创建校准圆圈
        self.create_calibration_circle(point_percent, x, y)
        
        # 设置状态为等待用户确认
        self.state = CalibrationState.WAITING_FOR_USER_CONFIRM
        
        # 不显示任何文字提示，只保留校准点
    
    def create_calibration_circle(self, point: Tuple[int, int], x: int = None, y: int = None):
        """创建校准圆圈（集成动态效果）"""
        if self.calibration_circle:
            if self.animation_effect:
                self.animation_effect.cleanup()
            self.calibration_circle.destroy()
        
        # 如果没有提供坐标，计算屏幕坐标
        if x is None or y is None:
            x = int((point[0] / 100) * self.screen_width)
            y = int((point[1] / 100) * self.screen_height)
        
        # 根据屏幕分辨率调整圆圈大小（使用相对值：屏幕宽度的2%）
        circle_size = self.screen_width * CIRCLE_SIZE_RATIO
        # 添加最小/最大限制，防止极端分辨率下过小或过大
        circle_size = max(30, min(80, circle_size))
        self.original_circle_size = circle_size
        
        # 获取当前背景色
        current_bg = self.background_colors[self.current_background_index]
        bg_color_hex = self.background_color_map[current_bg]
        
        # 计算Canvas放置位置，确保不会超出屏幕范围
        canvas_width = circle_size * 2
        canvas_height = circle_size * 2
        canvas_x = max(0, min(x - circle_size, self.screen_width - canvas_width))
        canvas_y = max(0, min(y - circle_size, self.screen_height - canvas_height))
        
        self.calibration_circle = tk.Canvas(
            self.root,
            width=canvas_width,  # 扩大Canvas以容纳动画效果
            height=canvas_height,
            bg=bg_color_hex,
            highlightthickness=0
        )
        self.calibration_circle.place(x=canvas_x, y=canvas_y)
        
        # 获取当前点的颜色配置
        point_index = self.current_point_index % 9
        color = self.config.colors[point_index]
        
        # 计算中心点（在Canvas中，相对于Canvas的实际位置）
        # 如果Canvas位置被调整（边界检查），需要相应调整中心点
        center_x = x - canvas_x
        center_y = y - canvas_y
        
        # 绘制外圈（无边框）
        margin = circle_size // 10
        self.outer_circle = self.calibration_circle.create_oval(
            center_x - circle_size + margin, center_y - circle_size + margin,
            center_x + circle_size - margin, center_y + circle_size - margin,
            fill=color[0],
            outline="",
            width=0,
            tags="outer_circle"
        )
        
        # 绘制内圈
        inner_margin = circle_size // 5
        self.inner_circle = self.calibration_circle.create_oval(
            center_x - circle_size + inner_margin, center_y - circle_size + inner_margin,
            center_x + circle_size - inner_margin, center_y + circle_size - inner_margin,
            fill=color[1],
            outline="",
            tags="inner_circle"
        )
        
        # 初始化动态效果
        self.animation_effect = AnimationEffect(
            self.calibration_circle,
            detection_range_px=ANIMATION_CONFIG["detection_range_px"]
        )
        
        # 设置目标点
        target_point = Point2D(x, y)
        self.animation_effect.set_target_point(target_point)
        
        # 设置更新回调（更新圆圈大小和颜色）
        self.animation_effect.set_update_callback(
            lambda scale, intensity: self._update_circle_animation(scale, intensity, center_x, center_y, color, current_bg)
        )
    
    def _update_circle_animation(self, pulse_scale: float, glow_intensity: float, 
                                 center_x: int, center_y: int, color: Tuple[str, str], 
                                 current_bg: BackgroundColor):
        """更新圆圈动画效果"""
        if not self.calibration_circle or not self.original_circle_size:
            return
        
        try:
            # 计算新的圆圈大小
            new_size = int(self.original_circle_size * pulse_scale)
            margin = new_size // 10
            inner_margin = new_size // 5
            
            # 更新外圈大小和颜色（发光效果）
            # 混合原始颜色和发光颜色
            from project.client.kappa.ui_config import ANIMATION_CONFIG
            glow_color = ANIMATION_CONFIG["glow_color"]
            
            # 更新外圈大小（无边框）
            self.calibration_circle.coords(
                self.outer_circle,
                center_x - new_size + margin, center_y - new_size + margin,
                center_x + new_size - margin, center_y + new_size - margin
            )
            self.calibration_circle.itemconfig(
                self.outer_circle,
                outline="",
                width=0
            )
            
            # 更新内圈大小
            self.calibration_circle.coords(
                self.inner_circle,
                center_x - new_size + inner_margin, center_y - new_size + inner_margin,
                center_x + new_size - inner_margin, center_y + new_size - inner_margin
            )
        except Exception as e:
            logger.error(f"更新动画效果失败: {e}", exc_info=True)
    
    def _confirm_and_submit_point(self):
        """确认并提交当前点位数据"""
        if self.state != CalibrationState.WAITING_FOR_USER_CONFIRM:
            return
        
        # 检查是否有 calibration_request（包含 frame_id 和 eye_type）
        if self.current_calibration_request is None:
            logger.warning("没有 calibration_request，无法提交标定点")
            return
        
        # 获取当前目标点位（Point2D）
        point_percent = self.config.points[self.current_point_index % 9]
        target_x = int((point_percent[0] / 100) * self.screen_width)
        target_y = int((point_percent[1] / 100) * self.screen_height)
        target_pixel = Point2D(target_x, target_y)
        
        # 获取当前背景色
        current_bg = self.background_colors[self.current_background_index].value
        
        # 创建 CalibrationResponse
        calibration_response = CalibrationResponse(
            frame_id=self.current_calibration_request.frame_id,
            eye_type=self.current_calibration_request.eye_type,
            target_pixel=target_pixel,
            background_color=current_bg
        )
        
        logger.info(f"提交校准点 {self.current_point_index + 1}/{self.total_points}: frame_id={calibration_response.frame_id}, eye={calibration_response.eye_type}, target_pixel={target_pixel}, 背景色: {current_bg}")
        
        # 通过回调发送 CalibrationResponse 给后端
        self.callback_manager.emit(
            CALIBRATION_POINT_SUBMIT,
            calibration_response=calibration_response
        )
        
        # 隐藏当前校准圆圈和清理动画
        if self.calibration_circle:
            if self.animation_effect:
                self.animation_effect.cleanup()
                self.animation_effect = None
            self.calibration_circle.destroy()
            self.calibration_circle = None
        
        # 继续下一个点
        self.current_point_index += 1
        if self.current_point_index < self.total_points:
            # 短暂延迟后显示下一个点
            self.root.after(200, self._show_next_point)
        else:
            self._finish_calibration()
    
    def update_progress(self, current: int, total: int):
        """更新进度显示（已禁用，不显示任何文字）"""
        # 不显示任何进度信息，只保留校准点
        pass
    
    def _finish_calibration(self):
        """完成校准"""
        self.state = CalibrationState.COMPLETED
        self.is_calibrating = False
        
        # 隐藏校准圆圈和清理动画
        if self.calibration_circle:
            if self.animation_effect:
                self.animation_effect.cleanup()
                self.animation_effect = None
            self.calibration_circle.destroy()
            self.calibration_circle = None
        
        # 隐藏提示信息
        self._hide_instruction()
        
        # 通过回调通知后端校准完成
        self.callback_manager.emit(
            CALIBRATION_COMPLETE,
            calibration_result={"total_points": self.total_points}
        )
        
        # 显示完成消息
        self._show_completion_message()
    
    def _show_completion_message(self):
        """显示完成消息"""
        # 隐藏进度指示器
        if self.progress_frame:
            self.progress_frame.pack_forget()
        
        # 显示完成消息
        completion_frame = tk.Frame(self.root, bg=self.background_color_map[self.background_colors[self.current_background_index]])
        completion_frame.pack(expand=True)
        
        # 完成图标
        icon_label = tk.Label(
            completion_frame,
            text="✓",
            font=("Arial", 48),
            fg="green",
            bg=self.background_color_map[self.background_colors[self.current_background_index]]
        )
        icon_label.pack(pady=20)
        
        # 完成文本
        text_label = tk.Label(
            completion_frame,
            text="校准完成",
            font=("Arial", 24, "bold"),
            fg=self.text_color_map[self.background_colors[self.current_background_index]],
            bg=self.background_color_map[self.background_colors[self.current_background_index]]
        )
        text_label.pack(pady=10)
        
        # 数据统计
        stats_label = tk.Label(
            completion_frame,
            text=f"总共收集 {self.total_points} 个校准点数据\n"
                 f"黑色背景: 9个点\n"
                 f"灰色背景: 9个点\n"
                 f"白色背景: 9个点\n"
                 f"屏幕分辨率: {self.screen_width}x{self.screen_height}",
            font=("Arial", 12),
            fg=self.text_color_map[self.background_colors[self.current_background_index]],
            bg=self.background_color_map[self.background_colors[self.current_background_index]]
        )
        stats_label.pack(pady=5)
        
        # 按钮框架
        button_frame = tk.Frame(completion_frame, bg=self.background_color_map[self.background_colors[self.current_background_index]])
        button_frame.pack(pady=20)
        
        # 重新校准按钮
        restart_button = tk.Button(
            button_frame,
            text="重新校准",
            font=("Arial", 14),
            bg="#f0f0f0",
            fg="black",
            relief="solid",
            borderwidth=1,
            padx=20,
            pady=10,
            command=self.reset_calibration
        )
        restart_button.pack(side=tk.LEFT, padx=10)
    
    def get_calibration_result(self) -> dict:
        """获取校准结果 - 返回包含左右眼结果的字典"""
        # 由于现在只发送点位数据，不收集左右眼数据
        # 返回基本信息
        return {
            "total_points": self.total_points,
            "background_colors": [bg.value for bg in self.background_colors],
            "points_per_background": 9
        }
    
    def reset_calibration(self):
        """重置校准"""
        # 清理所有状态
        self.state = CalibrationState.IDLE
        self.is_calibrating = False
        self.current_point_index = 0
        self.current_background_index = 0
        
        # 隐藏所有UI元素和清理动画
        if self.calibration_circle:
            if self.animation_effect:
                self.animation_effect.cleanup()
                self.animation_effect = None
            self.calibration_circle.destroy()
            self.calibration_circle = None
        
        # 重新显示开始界面
        # 注意：这里会销毁窗口并重新初始化，背景会在 setup_ui() 中重新创建
        self.root.destroy()
        self.__init__(callback_manager=self.callback_manager)
        self.root.mainloop()
    
    def run(self):
        """运行应用"""
        if self.state == CalibrationState.IDLE:
            self.state = CalibrationState.RUNNING
            self.root.mainloop()
    
    def close(self):
        """关闭应用"""
        try:
            if self.root:
                try:
                    if self.root.winfo_exists():
                        self.root.after(0, self.root.destroy)
                        logger.info("UI 已关闭")
                except Exception as e:
                    logger.error(f"UI winfo_exists/destroy 失败: {e}", exc_info=True)
                    try:
                        self.root.destroy()
                    except Exception as e2:
                        logger.error(f"UI destroy 失败: {e2}", exc_info=True)
        except Exception as e:
            logger.error(f"关闭 UI 时出错: {e}", exc_info=True)

# 使用示例
def example_usage():
    """使用示例"""
    app = EyeCalibrationApp()
    
    # 启动应用
    app.run()

# 主函数
if __name__ == "__main__":
    example_usage()
