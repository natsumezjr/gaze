# project/config/logging_config.py
import logging
import os
from datetime import datetime

def setup_logging(level=logging.INFO, log_to_file=True):
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
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
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

def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger
    
    Args:
        name: logger名称
        
    Returns:
        logging.Logger: logger对象
    """
    return logging.getLogger(name)