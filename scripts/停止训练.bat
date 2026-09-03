@echo off
chcp 65001 >nul
rem STOP 文件必须写在项目根目录（train.py 按 PROJECT_ROOT\STOP 检测）
echo stop> "%~dp0..\STOP"
echo.
echo  已请求停止训练（STOP 文件已创建）。
echo  训练会在当前这轮跑完后自动保存模型并退出，稍等几秒到几十秒。
echo.
pause
