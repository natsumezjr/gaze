"""键盘事件处理模块 - 可独立配置和修改"""
from typing import Callable, Optional, Dict
from project.config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

class KeyboardHandler:
    """键盘事件处理器 - 可独立配置和修改"""
    
    def __init__(self, root, on_confirm: Optional[Callable] = None):
        """
        初始化键盘处理器
        
        Args:
            root: Tkinter根窗口
            on_confirm: 确认按键的回调函数
        """
        self.root = root
        self.on_confirm = on_confirm
        
        # 可配置的按键映射
        self.key_map = {
            'space': self._handle_confirm,
            'Return': self._handle_confirm,
            'Enter': self._handle_confirm,
        }
        
        # 绑定键盘事件
        self.root.bind('<KeyPress>', self._on_key_press)
        self.root.focus_set()
        
        logger.info("键盘处理器已初始化")
    
    def _on_key_press(self, event):
        """键盘事件处理"""
        key = event.keysym
        if key in self.key_map:
            self.key_map[key]()
    
    def _handle_confirm(self):
        """处理确认按键"""
        if self.on_confirm:
            try:
                self.on_confirm()
            except Exception as e:
                logger.error(f"确认回调执行失败: {e}", exc_info=True)
    
    def set_confirm_callback(self, callback: Callable):
        """设置确认回调函数"""
        self.on_confirm = callback
        logger.debug("确认回调已更新")
    
    def update_key_map(self, key: str, handler: Callable):
        """更新按键映射"""
        self.key_map[key] = handler
        logger.debug(f"按键映射已更新: {key}")
    
    def unbind(self):
        """解绑键盘事件"""
        self.root.unbind('<KeyPress>')
        logger.debug("键盘事件已解绑")
