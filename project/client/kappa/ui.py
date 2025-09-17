import tkinter as tk
from tkinter import ttk, messagebox
import json
import time
import math
import threading
import queue
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from project.data.data_models import Point2D
from project.config.screen_config import _CURRENT_RES_W, _CURRENT_RES_H
from project.config.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

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

# 校准点配置 - 6点六边形布局
CALIBRATION_POINTS = [
    (33, 20),   # 左上
    (67, 20),   # 右上
    (10, 50),   # 左中
    (90, 50),   # 右中
    (33, 80),   # 左下
    (67, 80)    # 右下
]

CALIBRATION_COLORS = [
    ("#DC3545", "#C82333"),  # 深红色系
    ("#20C997", "#1E7E34"),  # 深绿色系
    ("#0D6EFD", "#0A58CA"),  # 深蓝色系
    ("#198754", "#155724"),  # 深薄荷绿系
    ("#FD7E14", "#E8590C"),  # 深橙色系
    ("#6610F2", "#520DC2")   # 深紫色系
]

class CalibrationState(Enum):
    """校准状态"""
    IDLE = "idle"
    RUNNING = "running"
    COUNTDOWN = "countdown"
    CALIBRATING = "calibrating"
    WAITING_FOR_DATA = "waiting_for_data"
    COMPLETED = "completed"

class EyeCalibrationApp:
    """眼动校准应用 - 真正阻塞式数据收集版本"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("眼动校准系统")
        
        # 使用当前屏幕分辨率
        self.screen_width = _CURRENT_RES_W if _CURRENT_RES_W > 0 else 1920
        self.screen_height = _CURRENT_RES_H if _CURRENT_RES_H > 0 else 1080
        
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        self.root.configure(bg="white")
        
        # 状态变量
        self.state = CalibrationState.IDLE
        self.current_point_index = 0
        self.collected_left_eye_points = []  # 存储收集到的左眼Point2D数据
        self.collected_right_eye_points = []  # 存储收集到的右眼Point2D数据
        self.is_calibrating = False
        
        # 阻塞机制
        self.left_eye_queue = queue.Queue()  # 左眼数据队列
        self.right_eye_queue = queue.Queue()  # 右眼数据队列
        self.current_target_pixel = None  # 当前目标像素坐标
        self.waiting_for_data = False  # 是否正在等待数据
        self.left_eye_data_received = False  # 左眼数据是否已接收
        self.right_eye_data_received = False  # 右眼数据是否已接收
        
        # 配置
        self.config = CalibrationConfig(
            points=CALIBRATION_POINTS,
            colors=CALIBRATION_COLORS
        )
        
        # 校准圆圈
        self.calibration_circle = None
        self.inner_circle = None
        
        # 进度显示
        self.progress_frame = None
        self.progress_bar = None
        self.progress_label = None
        
        # 倒计时显示
        self.countdown_label = None
        
        # 等待数据提示
        self.waiting_label = None
        
        # 界面元素
        self.setup_ui()
        
        logger.info(f"屏幕分辨率: {self.screen_width}x{self.screen_height}")
    
    def setup_ui(self):
        """设置用户界面"""
        # 主标题
        self.title_label = tk.Label(
            self.root,
            text="眼动校准系统",
            font=("Arial", 24, "bold"),
            fg="black",
            bg="white"
        )
        self.title_label.pack(pady=50)
        
        # 开始按钮
        self.start_button = tk.Button(
            self.root,
            text="开始校准",
            font=("Arial", 16),
            bg="#f0f0f0",
            fg="black",
            relief="solid",
            borderwidth=1,
            padx=30,
            pady=15,
            command=self.start_calibration
        )
        self.start_button.pack(pady=20)
        
        # 状态显示
        self.status_label = tk.Label(
            self.root,
            text=f"准备就绪 - 屏幕分辨率: {self.screen_width}x{self.screen_height}",
            font=("Arial", 12),
            fg="gray",
            bg="white"
        )
        self.status_label.pack(pady=10)
        
        # 进度指示器（初始隐藏）
        self.progress_frame = tk.Frame(self.root, bg="white")
        self.progress_label = tk.Label(
            self.progress_frame,
            text="校准进度",
            font=("Arial", 12, "bold"),
            fg="black",
            bg="white"
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
            text="0 / 6",
            font=("Arial", 10),
            fg="gray",
            bg="white"
        )
        self.progress_text.pack(pady=5)
        
        # 倒计时显示（初始隐藏）
        self.countdown_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 48, "bold"),
            fg="black",
            bg="white"
        )
        
        # 等待数据提示（初始隐藏）
        self.waiting_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 16),
            fg="blue",
            bg="white"
        )
    
    def update_status(self, message: str):
        """更新状态显示"""
        self.status_label.config(text=message)
        self.root.update()
    
    def start_calibration(self):
        """开始校准流程"""
        if self.is_calibrating:
            return
        
        self.is_calibrating = True
        self.state = CalibrationState.COUNTDOWN
        self.current_point_index = 0
        self.collected_left_eye_points = []
        self.collected_right_eye_points = []
        
        # 隐藏开始按钮和标题
        self.start_button.pack_forget()
        self.title_label.pack_forget()
        
        # 显示进度指示器
        if self.progress_frame is None:
            logger.error("progress_frame 是 None，无法显示进度指示器")
            return
        self.progress_frame.pack(pady=20)
        
        # 开始倒计时
        self.start_countdown()
    
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
            # 开始校准序列
            self.start_calibration_sequence()
    
    def start_calibration_sequence(self):
        """开始校准序列"""
        self.state = CalibrationState.CALIBRATING
        self.move_to_point(0)
    
    def move_to_point(self, point_index: int):
        """移动到指定校准点"""
        if point_index >= len(self.config.points):
            self.finish_calibration()
            return
        
        self.current_point_index = point_index
        point = self.config.points[point_index]
        color = self.config.colors[point_index]
        
        # 更新进度
        self.update_progress(point_index + 1, len(self.config.points))
        
        # 创建校准圆圈
        self.create_calibration_circle(point, color)
        
        # 开始校准
        self.start_calibration_at_point(point_index)
    
    def create_calibration_circle(self, point: Tuple[int, int], color: Tuple[str, str]):
        """创建校准圆圈"""
        if self.calibration_circle:
            self.calibration_circle.destroy()
        
        # 计算屏幕坐标 - 使用实际屏幕分辨率
        x = int((point[0] / 100) * self.screen_width)
        y = int((point[1] / 100) * self.screen_height)
        
        # 根据屏幕分辨率调整圆圈大小
        circle_size = max(60, min(100, self.screen_width // 20))
        
        self.calibration_circle = tk.Canvas(
            self.root,
            width=circle_size,
            height=circle_size,
            bg="white",
            highlightthickness=0
        )
        self.calibration_circle.place(x=x - circle_size//2, y=y - circle_size//2)
        
        # 绘制外圈
        margin = circle_size // 10
        self.calibration_circle.create_oval(
            margin, margin, circle_size - margin, circle_size - margin,
            fill=color[0],
            outline="black",
            width=2
        )
        
        # 绘制内圈
        inner_margin = circle_size // 5
        self.inner_circle = self.calibration_circle.create_oval(
            inner_margin, inner_margin, 
            circle_size - inner_margin, circle_size - inner_margin,
            fill=color[1],
            outline=""
        )
    
    def start_calibration_at_point(self, point_index: int):
        """在指定点开始校准 - 真正阻塞式等待数据"""
        point = self.config.points[point_index]
        
        # 计算目标像素坐标 - 使用实际屏幕分辨率
        target_x = int((point[0] / 100) * self.screen_width)
        target_y = int((point[1] / 100) * self.screen_height)
        
        # 创建目标Point2D对象
        self.current_target_pixel = Point2D(target_x, target_y)
        
        # 显示等待数据提示
        self.show_waiting_for_data(self.current_target_pixel)
        
        # 启动内圈收缩动画
        self.trigger_inner_circle_animation()
        
        # 设置状态为等待数据
        self.state = CalibrationState.WAITING_FOR_DATA
        self.waiting_for_data = True
        
        # 在后台线程中阻塞等待双眼数据
        def wait_for_data():
            try:
                # 等待左右眼数据
                left_eye_data = self.left_eye_queue.get(timeout=30)  # 30秒超时
                right_eye_data = self.right_eye_queue.get(timeout=30)  # 30秒超时
                
                # 在主线程中处理数据
                self.root.after(0, lambda: self.process_received_data(left_eye_data, right_eye_data))
                
            except queue.Empty:
                # 超时处理
                self.root.after(0, lambda: self.handle_data_timeout())
        
        # 启动等待线程
        threading.Thread(target=wait_for_data, daemon=True).start()
    
    def show_waiting_for_data(self, target_pixel: Point2D):
        """显示等待数据提示"""
        self.waiting_label.config(
            text=f"等待双眼数据传入...\n目标坐标: Point2D({target_pixel.x}, {target_pixel.y})\n\n请调用 app.submit_data(left_eye_data, right_eye_data) 传入双眼数据"
        )
        self.waiting_label.pack(expand=True)
        self.update_status(f"校准点 {self.current_point_index + 1} - 等待双眼数据传入...")
    
    def hide_waiting_for_data(self):
        """隐藏等待数据提示"""
        self.waiting_label.pack_forget()
    
    def submit_data(self, left_eye_data: Point2D, right_eye_data: Point2D):
        """外部调用此方法提交双眼数据 - 这是唯一的接口"""
        if not self.waiting_for_data:
            logger.warning("警告: 当前不在等待数据状态")
            return False
        
        # 将左右眼数据分别放入对应队列
        self.left_eye_queue.put(left_eye_data)
        self.right_eye_queue.put(right_eye_data)
        return True
    
    def process_received_data(self, left_eye_data: Point2D, right_eye_data: Point2D):
        """处理接收到的双眼数据"""
        if not self.waiting_for_data:
            return
        
        # 隐藏等待提示
        self.hide_waiting_for_data()
        
        # 存储双眼数据
        self.collected_left_eye_points.append(left_eye_data)
        self.collected_right_eye_points.append(right_eye_data)
        
        logger.info(f"接收到校准点 {self.current_point_index + 1} 数据:")
        logger.info(f"  左眼: Point2D({left_eye_data.x}, {left_eye_data.y})")
        logger.info(f"  右眼: Point2D({right_eye_data.x}, {right_eye_data.y})")
        
        # 重置等待状态
        self.waiting_for_data = False
        
        # 继续到下一个点
        self.continue_to_next_point()
    
    def handle_data_timeout(self):
        """处理数据超时"""
        if not self.waiting_for_data:
            return
        
        self.hide_waiting_for_data()
        self.waiting_for_data = False
        
        # 显示超时消息
        messagebox.showerror("超时", f"校准点 {self.current_point_index + 1} 等待数据超时")
        
        # 重置校准
        self.reset_calibration()
    
    def trigger_inner_circle_animation(self):
        """触发内圈收缩动画"""
        if self.inner_circle:
            # 简单的收缩动画
            for i in range(10):
                self.root.after(i * 200, lambda i=i: self.animate_inner_circle(i))
    
    def animate_inner_circle(self, step: int):
        """内圈收缩动画"""
        if self.inner_circle and self.calibration_circle:
            # 计算收缩比例
            scale = 1.0 - (step * 0.1)
            if scale < 0.25:
                scale = 0.25
            
            # 更新内圈大小
            circle_size = self.calibration_circle.winfo_width()
            center = circle_size // 2
            radius = (circle_size // 5) * scale
            
            self.calibration_circle.coords(
                self.inner_circle,
                center - radius, center - radius,
                center + radius, center + radius
            )
    
    def continue_to_next_point(self):
        """继续到下一个校准点"""
        if self.current_point_index + 1 < len(self.config.points):
            self.move_to_point(self.current_point_index + 1)
        else:
            self.finish_calibration()
    
    def update_progress(self, current: int, total: int):
        """更新进度显示"""
        self.progress_bar.config(value=(current / total) * 100)
        self.progress_text.config(text=f"{current} / {total}")
        self.update_status(f"校准点 {current}/{total} - 屏幕分辨率: {self.screen_width}x{self.screen_height}")
    
    def finish_calibration(self):
        """完成校准"""
        self.state = CalibrationState.COMPLETED
        self.is_calibrating = False
        
        # 隐藏校准圆圈
        if self.calibration_circle:
            self.calibration_circle.destroy()
        
        # 显示完成消息
        self.show_completion_message()
    
    def show_completion_message(self):
        """显示完成消息"""
        # 隐藏进度指示器
        self.progress_frame.pack_forget()
        
        # 显示完成消息
        completion_frame = tk.Frame(self.root, bg="white")
        completion_frame.pack(expand=True)
        
        # 完成图标
        icon_label = tk.Label(
            completion_frame,
            text="✓",
            font=("Arial", 48),
            fg="green",
            bg="white"
        )
        icon_label.pack(pady=20)
        
        # 完成文本
        text_label = tk.Label(
            completion_frame,
            text="校准完成",
            font=("Arial", 24, "bold"),
            fg="black",
            bg="white"
        )
        text_label.pack(pady=10)
        
        # 数据统计
        stats_label = tk.Label(
            completion_frame,
            text=f"左眼收集 {len(self.collected_left_eye_points)} 个校准点数据\n右眼收集 {len(self.collected_right_eye_points)} 个校准点数据\n屏幕分辨率: {self.screen_width}x{self.screen_height}",
            font=("Arial", 12),
            fg="gray",
            bg="white"
        )
        stats_label.pack(pady=5)
        
        # 显示收集的数据
        data_text = tk.Text(
            completion_frame,
            height=12,
            width=80,
            font=("Courier", 10),
            wrap=tk.WORD
        )
        data_text.pack(pady=10, padx=20)
        
        # 插入数据
        data_text.insert(tk.END, "收集的双眼校准数据:\n\n")
        data_text.insert(tk.END, "左眼数据:\n")
        for i, point in enumerate(self.collected_left_eye_points):
            data_text.insert(tk.END, f"  点 {i+1}: Point2D({point.x}, {point.y})\n")
        
        data_text.insert(tk.END, "\n右眼数据:\n")
        for i, point in enumerate(self.collected_right_eye_points):
            data_text.insert(tk.END, f"  点 {i+1}: Point2D({point.x}, {point.y})\n")
        
        data_text.config(state=tk.DISABLED)
        
        # 按钮框架
        button_frame = tk.Frame(completion_frame, bg="white")
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
        
        # 获取结果按钮
        get_result_button = tk.Button(
            button_frame,
            text="获取校准结果",
            font=("Arial", 14),
            bg="#28a745",
            fg="white",
            relief="solid",
            borderwidth=1,
            padx=20,
            pady=10,
            command=self.get_calibration_result
        )
        get_result_button.pack(side=tk.LEFT, padx=10)
    
    def get_calibration_result(self) -> dict:
        """获取校准结果 - 返回包含左右眼结果的字典"""
        if len(self.collected_left_eye_points) != len(self.config.points) or len(self.collected_right_eye_points) != len(self.config.points):
            error_msg = f"校准点数量不匹配: 期望 {len(self.config.points)}, 左眼实际 {len(self.collected_left_eye_points)}, 右眼实际 {len(self.collected_right_eye_points)}"
            return {
                "left_eye": CalibrationResult(
                    points=[],
                    accuracy=0.0,
                    error_message=error_msg,
                    calibration_time=None
                ),
                "right_eye": CalibrationResult(
                    points=[],
                    accuracy=0.0,
                    error_message=error_msg,
                    calibration_time=None
                )
            }
        
        # 计算左右眼准确度
        left_accuracy = self.calculate_accuracy(self.collected_left_eye_points)
        right_accuracy = self.calculate_accuracy(self.collected_right_eye_points)
        
        left_result = CalibrationResult(
            points=self.collected_left_eye_points.copy(),
            accuracy=left_accuracy,
            error_message=None,
            calibration_time=datetime.now()
        )
        
        right_result = CalibrationResult(
            points=self.collected_right_eye_points.copy(),
            accuracy=right_accuracy,
            error_message=None,
            calibration_time=datetime.now()
        )
        
        # 显示结果
        messagebox.showinfo(
            "校准结果", 
            f"双眼校准完成！\n"
            f"左眼收集点数: {len(left_result.points)}, 准确度: {left_result.accuracy:.2f}%\n"
            f"右眼收集点数: {len(right_result.points)}, 准确度: {right_result.accuracy:.2f}%\n"
            f"校准时间: {left_result.calibration_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return {
            "left_eye": left_result,
            "right_eye": right_result
        }
    
    def calculate_accuracy(self, collected_points: List[Point2D]) -> float:
        """计算校准准确度"""
        if len(collected_points) != len(self.config.points):
            return 0.0
        
        total_error = 0.0
        for i, (collected_point, expected_percentage) in enumerate(zip(collected_points, self.config.points)):
            # 计算期望的像素坐标
            expected_x = int((expected_percentage[0] / 100) * self.screen_width)
            expected_y = int((expected_percentage[1] / 100) * self.screen_height)
            
            # 计算误差
            error = math.sqrt((collected_point.x - expected_x)**2 + (collected_point.y - expected_y)**2)
            total_error += error
        
        # 计算平均误差
        avg_error = total_error / len(collected_points)
        
        # 转换为准确度百分比
        max_error = math.sqrt(self.screen_width**2 + self.screen_height**2)
        accuracy = max(0.0, (1.0 - avg_error / max_error) * 100)
        
        return accuracy
    
    def reset_calibration(self):
        """重置校准"""
        # 清理所有状态
        self.state = CalibrationState.IDLE
        self.is_calibrating = False
        self.current_point_index = 0
        self.collected_left_eye_points = []
        self.collected_right_eye_points = []
        self.waiting_for_data = False
        self.left_eye_data_received = False
        self.right_eye_data_received = False
        
        # 清空队列
        while not self.left_eye_queue.empty():
            try:
                self.left_eye_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.right_eye_queue.empty():
            try:
                self.right_eye_queue.get_nowait()
            except queue.Empty:
                break
        
        # 重新显示开始界面
        self.root.destroy()
        self.__init__()
        self.root.mainloop()
    
    def run(self):
        """运行应用"""
        if self.state == CalibrationState.IDLE:
            self.state = CalibrationState.RUNNING
            self.root.mainloop()

# 使用示例
def example_usage():
    """使用示例"""
    app = EyeCalibrationApp()
    
    # 启动应用
    app.run()

# 主函数
if __name__ == "__main__":
    example_usage()