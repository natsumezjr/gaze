@echo off
echo 正在检查Git安装...

REM 检查Git是否可用
git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Git已安装
    echo 正在获取远程仓库信息...
    git fetch origin
    echo 正在检查本地分支状态...
    git status
    echo 正在拉取最新更改...
    git pull origin main
    echo 同步完成！
) else (
    echo 错误: Git未找到
    echo 请按照以下步骤操作:
    echo 1. 访问 https://git-scm.com/download/win
    echo 2. 下载并安装Git for Windows
    echo 3. 重启命令提示符
    echo 4. 重新运行此脚本
    echo.
    echo 或者使用GitHub Desktop:
    echo 1. 访问 https://desktop.github.com/
    echo 2. 下载并安装GitHub Desktop
    echo 3. 克隆仓库: https://github.com/natsumezjr/gaze
)

pause 