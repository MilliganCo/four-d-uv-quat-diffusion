@echo off
chcp 65001 >nul
cd /d E:\4D-UV-Quat-Diffusion
echo Запуск GUI мониторинга...
echo Откройте браузер: http://localhost:8501
venv4d\Scripts\streamlit.exe run ui/app.py -- --run_dir runs/cifar10
pause


