@echo off
echo 正在设置Git环境变量...

:: 设置Git路径
set "GIT_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd"

:: 添加到用户PATH环境变量
setx PATH "%PATH%;%GIT_PATH%"

echo Git路径已添加到系统PATH中
echo Git版本信息：
"%GIT_PATH%\git.exe" --version

echo.
echo 设置完成！请重新打开命令提示符或PowerShell以使更改生效。
pause 