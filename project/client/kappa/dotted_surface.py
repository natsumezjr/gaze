"""动态点阵表面组件 - 使用 Tkinter Canvas 实现 3D 点阵动画效果"""
import tkinter as tk
from typing import Optional, Tuple, List
import math


class DottedSurface(tk.Canvas):
    """动态点阵表面组件
    
    使用 Tkinter Canvas 和 2D 投影模拟 Three.js 的 3D 点阵效果。
    支持亮色/暗色主题，响应窗口大小变化。
    """
    
    # 配置参数（与原始 React 组件保持一致）
    SEPARATION = 150  # 点之间的间距
    AMOUNTX = 40      # X 方向点数
    AMOUNTY = 60      # Y 方向点数
    POINT_SIZE = 8    # 点的显示大小
    ANIMATION_SPEED = 0.1  # 动画速度
    WAVE_AMPLITUDE = 50    # 波动幅度
    
    # 3D 相机参数（模拟原始 Three.js 设置）
    CAMERA_X = 0
    CAMERA_Y = 355
    CAMERA_Z = 1220
    FOV = 60  # 视野角度（度）
    # 垂直偏移调整（向下移动背景）
    VERTICAL_OFFSET = 50  # 向下偏移像素数
    
    def __init__(
        self,
        parent: tk.Widget,
        theme: str = 'dark',
        bg_color: Optional[str] = None,
        *args,
        **kwargs
    ):
        """初始化动态点阵表面
        
        Args:
            parent: 父组件
            theme: 主题 ('dark' 或 'light')
            bg_color: 背景颜色（如果为 None，则根据主题自动设置）
            *args, **kwargs: 传递给 Canvas 的其他参数
        """
        # 设置背景色
        if bg_color is None:
            bg_color = '#000000' if theme == 'dark' else '#FFFFFF'
        
        # 初始化 Canvas
        kwargs.setdefault('bg', bg_color)
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, *args, **kwargs)
        
        self.theme = theme
        self.bg_color = bg_color
        
        # 点的颜色（根据主题设置）
        if theme == 'dark':
            self.point_color = '#C8C8C8'  # 浅灰色
        else:
            self.point_color = '#000000'  # 黑色
        
        # 动画状态
        self.count = 0.0
        self.animation_id: Optional[int] = None
        self.is_running = False
        
        # 点阵数据
        self.points_3d: List[Tuple[float, float, float]] = []
        self.point_items: List[int] = []  # Canvas 对象 ID
        
        # 窗口尺寸
        self.width = 0
        self.height = 0
        
        # 初始化点阵
        self._initialize_points()
        
        # 绑定窗口大小变化事件
        self.bind('<Configure>', self._on_resize)
        
        # 启动动画
        self.start_animation()
    
    def _initialize_points(self):
        """初始化点阵的 3D 坐标"""
        self.points_3d = []
        
        for ix in range(self.AMOUNTX):
            for iy in range(self.AMOUNTY):
                x = ix * self.SEPARATION - (self.AMOUNTX * self.SEPARATION) / 2
                y = 0  # 初始 Y 坐标为 0，将在动画中更新
                z = iy * self.SEPARATION - (self.AMOUNTY * self.SEPARATION) / 2
                
                self.points_3d.append((x, y, z))
    
    def _project_3d_to_2d(
        self,
        x_3d: float,
        y_3d: float,
        z_3d: float
    ) -> Tuple[float, float]:
        """将 3D 坐标投影到 2D Canvas 坐标
        
        使用透视投影算法模拟 Three.js 的相机效果。
        
        Args:
            x_3d: 3D X 坐标
            y_3d: 3D Y 坐标
            z_3d: 3D Z 坐标
            
        Returns:
            (x_2d, y_2d): 2D Canvas 坐标
        """
        if self.width == 0 or self.height == 0:
            return (0, 0)
        
        # 计算相对于相机的坐标
        rel_x = x_3d - self.CAMERA_X
        rel_y = y_3d - self.CAMERA_Y
        rel_z = z_3d - self.CAMERA_Z
        
        # 透视投影
        # 使用简单的透视投影公式
        distance = abs(rel_z)
        if distance == 0:
            distance = 0.001  # 避免除零
        
        # 计算投影比例（模拟 FOV）
        fov_rad = math.radians(self.FOV)
        scale = self.CAMERA_Z / distance
        
        # 垂直压缩比例（让背景更矮）
        vertical_scale = 0.6  # 垂直方向压缩到60%
        
        # 投影到屏幕坐标
        x_2d = rel_x * scale + self.width / 2
        y_2d = rel_y * scale * vertical_scale + self.height / 2 + self.VERTICAL_OFFSET
        
        # 上下镜像翻转
        y_2d = self.height - y_2d
        
        return (x_2d, y_2d)
    
    def _update_animation(self):
        """更新动画帧"""
        if not self.is_running:
            return
        
        # 更新计数器
        self.count += self.ANIMATION_SPEED
        
        # 清除所有点
        for item_id in self.point_items:
            self.delete(item_id)
        self.point_items.clear()
        
        # 更新并绘制所有点
        point_index = 0
        for ix in range(self.AMOUNTX):
            for iy in range(self.AMOUNTY):
                # 获取原始 3D 坐标
                x_3d, _, z_3d = self.points_3d[point_index]
                
                # 计算动画后的 Y 坐标（使用正弦波）
                y_3d = (
                    math.sin((ix + self.count) * 0.3) * self.WAVE_AMPLITUDE +
                    math.sin((iy + self.count) * 0.5) * self.WAVE_AMPLITUDE
                )
                
                # 投影到 2D
                x_2d, y_2d = self._project_3d_to_2d(x_3d, y_3d, z_3d)
                
                # 只绘制在屏幕范围内的点
                if 0 <= x_2d <= self.width and 0 <= y_2d <= self.height:
                    # 根据 Z 距离调整点的大小（距离越远越小）
                    z_distance = abs(z_3d - self.CAMERA_Z)
                    size_factor = max(0.3, min(1.0, self.CAMERA_Z / (z_distance + 100)))
                    point_size = self.POINT_SIZE * size_factor
                    
                    # 根据 Z 距离调整透明度（距离越远越透明）
                    opacity = max(0.3, min(0.8, self.CAMERA_Z / (z_distance + 500)))
                    
                    # 绘制点（使用椭圆模拟圆形）
                    item_id = self.create_oval(
                        x_2d - point_size / 2,
                        y_2d - point_size / 2,
                        x_2d + point_size / 2,
                        y_2d + point_size / 2,
                        fill=self.point_color,
                        outline='',
                        width=0
                    )
                    self.point_items.append(item_id)
                
                point_index += 1
        
        # 调度下一帧
        self.animation_id = self.after(16, self._update_animation)  # 约 60fps
    
    def _on_resize(self, event: tk.Event):
        """处理窗口大小变化事件"""
        if event.widget == self:
            new_width = event.width
            new_height = event.height
            
            if new_width != self.width or new_height != self.height:
                self.width = new_width
                self.height = new_height
                # 重新配置 Canvas 大小
                self.config(width=new_width, height=new_height)
    
    def start_animation(self):
        """启动动画"""
        if not self.is_running:
            self.is_running = True
            # 获取当前窗口大小
            self.update_idletasks()
            self.width = self.winfo_width()
            self.height = self.winfo_height()
            
            # 如果窗口大小为 1（默认值），使用父窗口大小
            if self.width <= 1 or self.height <= 1:
                if self.master:
                    self.width = self.master.winfo_width()
                    self.height = self.master.winfo_height()
            
            # 如果还是无效，使用屏幕大小
            if self.width <= 1 or self.height <= 1:
                self.width = self.winfo_screenwidth()
                self.height = self.winfo_screenheight()
            
            self._update_animation()
    
    def stop_animation(self):
        """停止动画"""
        self.is_running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
    
    def set_theme(self, theme: str):
        """设置主题
        
        Args:
            theme: 'dark' 或 'light'
        """
        self.theme = theme
        
        if theme == 'dark':
            self.point_color = '#C8C8C8'
            self.bg_color = '#000000'
        else:
            self.point_color = '#000000'
            self.bg_color = '#FFFFFF'
        
        self.config(bg=self.bg_color)
    
    def destroy(self):
        """清理资源"""
        self.stop_animation()
        super().destroy()
