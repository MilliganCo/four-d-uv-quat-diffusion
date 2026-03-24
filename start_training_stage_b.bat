@echo off
REM Запуск обучения Stage B с uv_kernel=3, 64x64 и текстовыми промптами
REM Использование: start_training_stage_b.bat

echo ========================================
echo ОБУЧЕНИЕ STAGE B
echo ========================================
echo Конфигурация: uv_kernel=3, 64x64, текстовые промпты
echo ========================================
echo.

cd /d E:\4D-UV-Quat-Diffusion
venv4d\Scripts\python.exe train_conditional.py --config configs/cifar10_stage_b_64x64.yaml

pause

