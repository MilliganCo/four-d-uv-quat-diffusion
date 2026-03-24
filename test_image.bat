@echo off
REM Простой скрипт для тестирования своего изображения
REM Использование: test_image.bat путь_к_изображению [добавить_шум]

setlocal enabledelayedexpansion

set "IMAGE=%~1"
set "ADD_NOISE=%~2"

if "%IMAGE%"=="" (
    echo Ошибка: Укажите путь к изображению
    echo Использование: test_image.bat путь_к_изображению [--add_noise]
    pause
    exit /b 1
)

echo ========================================
echo Тестирование своего изображения
echo ========================================
echo Изображение: %IMAGE%
if "%ADD_NOISE%"=="--add_noise" (
    echo Режим: С добавлением шума (демонстрация денойзинга)
) else (
    echo Режим: Обычная обработка
)
echo ========================================
echo.

if "%ADD_NOISE%"=="--add_noise" (
    venv4d\Scripts\python.exe test_my_image_simple.py --image "%IMAGE%" --add_noise
) else (
    venv4d\Scripts\python.exe test_my_image_simple.py --image "%IMAGE%"
)

echo.
echo ========================================
echo Готово! Проверьте папку samples\
echo ========================================
pause

