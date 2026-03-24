# test_model.ps1 - PowerShell скрипт для быстрой проверки модели
param(
    [string]$Ckpt = "runs/cifar10/ckpt_last.pt",
    [int]$N = 16,
    [int]$Steps = 50
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Генерация образцов из модели" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Чекпоинт: $Ckpt" -ForegroundColor Yellow
Write-Host "Количество образцов: $N" -ForegroundColor Yellow
Write-Host "DDIM шагов: $Steps" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

& venv4d\Scripts\python.exe quick_test.py --ckpt $Ckpt --n $N --steps $Steps

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Генерация завершена!" -ForegroundColor Green
Write-Host "Результаты сохранены в папке: samples\" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

