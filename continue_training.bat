@echo off
REM Скрипт для продолжения обучения модели
REM Использование: continue_training.bat

echo ========================================
echo ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ МОДЕЛИ
echo ========================================
echo.
echo Модель будет продолжать обучение с последнего чекпоинта
echo Чекпоинты сохраняются каждые 20,000 шагов
echo.
echo Для остановки нажмите Ctrl+C
echo ========================================
echo.

cd /d E:\4D-UV-Quat-Diffusion
venv4d\Scripts\python.exe train.py --config configs/cifar10.yaml

pause

