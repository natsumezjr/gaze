"""GooeyText 文字变形效果组件 - 使用 Tkinter 实现文字模糊变形动画"""
import tkinter as tk
from typing import List, Optional
import math
import time


class GooeyText(tk.Canvas):
    """GooeyText 文字变形效果组件
    
    实现类似 React GooeyText 的文字变形动画效果，支持模糊和透明度渐变。
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        texts: List[str],
        morph_time: float = 1.0,
        cooldown_time: float = 0.25,
        font_size: int = 60,
        text_color: str = "#FFFFFF",
        bg_color: str = "transparent",
        *args,
        **kwargs
    ):
        """初始化 GooeyText 组件
        
        Args:
            parent: 父组件
            texts: 文字列表
            morph_time: 变形时间（秒）
            cooldown_time: 冷却时间（秒）
            font_size: 字体大小
            text_color: 文字颜色
            bg_color: 背景颜色（transparent 表示透明）
            *args, **kwargs: 传递给 Canvas 的其他参数
        """
        # 设置 Canvas 参数
        kwargs.setdefault('highlightthickness', 0)
        kwargs.setdefault('borderwidth', 0)
        if bg_color == "transparent":
            kwargs.setdefault('bg', parent.cget('bg'))
        else:
            kwargs.setdefault('bg', bg_color)
        
        super().__init__(parent, *args, **kwargs)
        
        self.texts = texts
        self.morph_time = morph_time
        self.cooldown_time = cooldown_time
        self.font_size = font_size
        self.text_color = text_color
        
        # 动画状态
        self.text_index = len(texts) - 1
        self.morph = 0.0
        self.cooldown = cooldown_time
        self.last_time = time.time()
        self.animation_id: Optional[int] = None
        self.is_running = False
        
        # 文字对象 ID
        self.text1_id: Optional[int] = None
        self.text2_id: Optional[int] = None
        
        # 文字中心位置（用于避免重影）
        self.center_x = 0
        self.center_y = 0
        
        # 字体
        self.font = ("Arial", font_size, "bold")
        
        # 计算 Canvas 大小（基于文字大小）
        self.update_idletasks()
        test_text = max(texts, key=len) if texts else "EyeCalibration"
        self.config(
            width=len(test_text) * font_size * 0.6,
            height=font_size * 1.5
        )
        
        # 启动动画
        self.start_animation()
    
    def _set_morph(self, fraction: float):
        """设置变形状态
        
        Args:
            fraction: 变形进度 (0.0 到 1.0)
        """
        # 计算模糊值（模拟 CSS blur）
        # blur = min(8 / fraction - 8, 100) if fraction > 0 else 100
        # 简化：使用透明度模拟模糊效果
        blur_factor = min(8 / (fraction + 0.01) - 8, 100) if fraction > 0 else 100
        
        # 计算透明度
        opacity1 = max(0, min(100, int((1 - fraction) ** 0.4 * 100)))
        opacity2 = max(0, min(100, int(fraction ** 0.4 * 100)))
        
        # 更新 text2（新文字）
        if self.text2_id:
            # 计算颜色（带透明度）
            color2 = self._apply_opacity(self.text_color, opacity2 / 100.0)
            self.itemconfig(self.text2_id, fill=color2)
            # 基于原始中心位置计算新位置，避免累积偏移
            offset = blur_factor * 0.05  # 减小偏移量
            self.coords(self.text2_id, self.center_x + offset, self.center_y)
        
        # 更新 text1（旧文字）
        if self.text1_id:
            color1 = self._apply_opacity(self.text_color, opacity1 / 100.0)
            self.itemconfig(self.text1_id, fill=color1)
            # 基于原始中心位置计算新位置，避免累积偏移
            offset = blur_factor * 0.05  # 减小偏移量
            self.coords(self.text1_id, self.center_x - offset, self.center_y)
    
    def _apply_opacity(self, color: str, opacity: float) -> str:
        """应用透明度到颜色（简化实现）
        
        Args:
            color: 十六进制颜色（如 "#FFFFFF"）
            opacity: 透明度 (0.0 到 1.0)
            
        Returns:
            带透明度的颜色字符串（Tkinter 不支持真正的 alpha，这里返回原色）
        """
        # Tkinter Canvas 不支持真正的 alpha 通道
        # 这里通过调整颜色亮度来模拟透明度
        if opacity < 0.3:
            return "#888888"  # 很暗
        elif opacity < 0.6:
            return "#CCCCCC"  # 中等
        else:
            return color  # 原色
    
    def _do_cooldown(self):
        """执行冷却阶段"""
        self.morph = 0.0
        
        # 重置文字状态
        if self.text1_id:
            self.itemconfig(self.text1_id, fill=self._apply_opacity(self.text_color, 0.0))
        if self.text2_id:
            self.itemconfig(self.text2_id, fill=self.text_color)
    
    def _do_morph(self):
        """执行变形阶段"""
        self.morph -= self.cooldown
        self.cooldown = 0.0
        
        fraction = self.morph / self.morph_time
        
        if fraction > 1.0:
            self.cooldown = self.cooldown_time
            fraction = 1.0
        
        self._set_morph(fraction)
    
    def _animate(self):
        """动画循环"""
        if not self.is_running:
            return
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        should_increment_index = self.cooldown > 0
        self.cooldown -= dt
        
        if self.cooldown <= 0:
            if should_increment_index:
                # 切换到下一个文字
                self.text_index = (self.text_index + 1) % len(self.texts)
                
                # 更新文字内容
                text1_content = self.texts[self.text_index % len(self.texts)]
                text2_content = self.texts[(self.text_index + 1) % len(self.texts)]
                
                # 清除旧文字
                if self.text1_id:
                    self.delete(self.text1_id)
                if self.text2_id:
                    self.delete(self.text2_id)
                
                # 创建新文字
                width = int(self.winfo_width() or self.cget('width') or 800)
                height = int(self.winfo_height() or self.cget('height') or 150)
                self.center_x = width / 2
                self.center_y = height / 2
                
                self.text1_id = self.create_text(
                    self.center_x, self.center_y,
                    text=text1_content,
                    font=self.font,
                    fill=self._apply_opacity(self.text_color, 0.0),
                    anchor=tk.CENTER
                )
                
                self.text2_id = self.create_text(
                    self.center_x, self.center_y,
                    text=text2_content,
                    font=self.font,
                    fill=self.text_color,
                    anchor=tk.CENTER
                )
            
            self._do_morph()
        else:
            self._do_cooldown()
        
        # 继续下一帧
        self.animation_id = self.after(16, self._animate)  # 约 60fps
    
    def start_animation(self):
        """启动动画"""
        if not self.is_running:
            self.is_running = True
            self.last_time = time.time()
            
            # 初始化文字
            if self.texts:
                width = self.winfo_width()
                height = self.winfo_height()
                if width <= 1:
                    width = int(self.cget('width') or 800)
                if height <= 1:
                    height = int(self.cget('height') or 150)
                
                # 确保是整数
                width = int(width)
                height = int(height)
                
                self.center_x = width / 2
                self.center_y = height / 2
                
                text1_content = self.texts[self.text_index % len(self.texts)]
                text2_content = self.texts[(self.text_index + 1) % len(self.texts)]
                
                self.text1_id = self.create_text(
                    self.center_x, self.center_y,
                    text=text1_content,
                    font=self.font,
                    fill=self._apply_opacity(self.text_color, 0.0),
                    anchor=tk.CENTER
                )
                
                self.text2_id = self.create_text(
                    self.center_x, self.center_y,
                    text=text2_content,
                    font=self.font,
                    fill=self.text_color,
                    anchor=tk.CENTER
                )
            
            self._animate()
    
    def stop_animation(self):
        """停止动画"""
        self.is_running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
    
    def destroy(self):
        """清理资源"""
        self.stop_animation()
        super().destroy()
