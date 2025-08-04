# GitHub仓库同步脚本
# 使用方法：.\sync_repository.ps1

Write-Host "正在检查Git安装..." -ForegroundColor Green

# 检查Git是否可用
try {
    $gitVersion = git --version 2>$null
    if ($gitVersion) {
        Write-Host "Git已安装: $gitVersion" -ForegroundColor Green
        
        Write-Host "正在获取远程仓库信息..." -ForegroundColor Yellow
        git fetch origin
        
        Write-Host "正在检查本地分支状态..." -ForegroundColor Yellow
        git status
        
        Write-Host "正在拉取最新更改..." -ForegroundColor Yellow
        git pull origin main
        
        Write-Host "同步完成！" -ForegroundColor Green
    } else {
        throw "Git未找到"
    }
} catch {
    Write-Host "错误: $_" -ForegroundColor Red
    Write-Host "请按照以下步骤操作:" -ForegroundColor Yellow
    Write-Host "1. 访问 https://git-scm.com/download/win" -ForegroundColor Cyan
    Write-Host "2. 下载并安装Git for Windows" -ForegroundColor Cyan
    Write-Host "3. 重启PowerShell" -ForegroundColor Cyan
    Write-Host "4. 重新运行此脚本" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "或者使用GitHub Desktop:" -ForegroundColor Yellow
    Write-Host "1. 访问 https://desktop.github.com/" -ForegroundColor Cyan
    Write-Host "2. 下载并安装GitHub Desktop" -ForegroundColor Cyan
    Write-Host "3. 克隆仓库: https://github.com/natsumezjr/gaze" -ForegroundColor Cyan
}

Write-Host "按任意键继续..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 