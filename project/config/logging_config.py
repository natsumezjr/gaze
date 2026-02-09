# project/config/logging_config.py
import copy
import logging
import os
from datetime import datetime

# ANSI 颜色码（终端彩色输出）
_COLORS = {
    "RESET": "\033[0m",
    "GRAY": "\033[90m",
    "RED": "\033[31m",
    "YELLOW": "\033[33m",
    "GREEN": "\033[32m",
    "CYAN": "\033[36m",
    "MAGENTA": "\033[35m",
}

# 各级别对应颜色
_LEVEL_COLORS = {
    logging.DEBUG: _COLORS["CYAN"],
    logging.INFO: _COLORS["GREEN"],
    logging.WARNING: _COLORS["YELLOW"],
    logging.ERROR: _COLORS["RED"],
    logging.CRITICAL: _COLORS["RED"],
}


class ColoredFormatter(logging.Formatter):
    """控制台彩色日志格式化器（不修改原 record，避免污染文件日志）"""

    def format(self, record):
        record = copy.copy(record)
        color = _LEVEL_COLORS.get(record.levelno, _COLORS["RESET"])
        record.levelname = f"{color}{record.levelname}{_COLORS['RESET']}"
        record.name = f"{_COLORS['GRAY']}{record.name}{_COLORS['RESET']}"
        return super().format(record)


def setup_logging(name: str, level=logging.INFO, log_to_file=True):
    """
    设置日志配置
    
    Args:
        level: 日志级别
        log_to_file: 是否保存到文件
    """
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 获取根logger
    root_logger = logging.getLogger(name)
    root_logger.setLevel(level)
    
    # 清除现有的handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台输出（彩色）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(console_handler)
    
    # 文件输出
    if log_to_file:
        # 按日期命名的日志文件
        log_filename = f"gaze_{datetime.now().strftime('%Y%m%d')}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 错误日志单独文件
        error_log_filename = f"gaze_error_{datetime.now().strftime('%Y%m%d')}.log"
        error_log_filepath = os.path.join(log_dir, error_log_filename)
        
        error_handler = logging.FileHandler(error_log_filepath, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
    
    # 抑制TensorFlow警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    return root_logger