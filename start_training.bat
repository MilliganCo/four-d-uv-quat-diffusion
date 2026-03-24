@echo off
chcp 65001 >nul
cd /d E:\4D-UV-Quat-Diffusion
echo Запуск обучения...
venv4d\Scripts\python.exe train.py --config configs/cifar10.yaml
pause


