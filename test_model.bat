@echo off
REM Быстрая проверка модели
REM Использование: test_model.bat [путь_к_чекпоинту] [количество_образцов] [DDIM_шагов]

setlocal enabledelayedexpansion

set "CKPT=%~1"
set "N=%~2"
set "STEPS=%~3"

if "%CKPT%"=="" set "CKPT=runs\cifar10\ckpt_last.pt"
if "%N%"=="" set "N=16"
if "%STEPS%"=="" set "STEPS=50"

echo ========================================
echo Генерация образцов из модели
echo ========================================
echo Чекпоинт: %CKPT%
echo Количество образцов: %N%
echo DDIM шагов: %STEPS%
echo ========================================

venv4d\Scripts\python.exe quick_test.py --ckpt %CKPT% --n %N% --steps %STEPS%

echo.
echo ========================================
echo Генерация завершена!
echo Результаты сохранены в папке: samples\
echo ========================================
pause

