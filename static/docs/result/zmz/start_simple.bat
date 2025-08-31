@echo off
chcp 65001 >nul
echo ========================================
echo   眼动追踪 Web API 服务 - 简化启动
echo ========================================
echo.

echo 正在启动Web API服务...
echo 服务地址: http://localhost:5000
echo 测试页面: http://localhost:5000/static/test_frontend.html
echo.
echo 按 Ctrl+C 停止服务
echo.

python -m project.fitting.app.run_web_api

pause
