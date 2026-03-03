# 眼动追踪系统主入口
# 系统启动、模块协调、全局状态管理
from project.runtime import run_application, request_system_stop
from project.config.logging_config import setup_logging

logger = setup_logging(__name__)

# 向后兼容：其他脚本可能直接 from project.main import request_system_stop
__all__ = ["main", "request_system_stop"]


def main():
    """主函数 - 委托给运行时模块执行"""
    run_application(rgb_d=True)


if __name__ == "__main__":
    main()
