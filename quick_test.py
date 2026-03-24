# quick_test.py - быстрая проверка модели с визуализацией
from __future__ import annotations

import os, argparse, yaml
import torch
from torchvision.utils import save_image, make_grid
from models.unet4d import UNet4D
from diffusion.schedules import Diffusion
from PIL import Image


@torch.no_grad()
def main(ckpt_path: str, n: int = 16, steps: int = 50):
    """Быстрая проверка модели с несколькими образцами"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Устройство: {device}")
    
    # Загружаем конфигурацию из чекпоинта или используем дефолтную
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'cfg' in ckpt:
        cfg = ckpt['cfg']
        print(f"Конфигурация загружена из чекпоинта")
    else:
        # Используем дефолтную конфигурацию
        with open('configs/cifar10.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        print("Использую конфигурацию из configs/cifar10.yaml")
    
    s = cfg['stride_s']
    X = cfg['image_size'] // s
    Y = cfg['image_size'] // s
    
    print(f"Параметры: X={X}, Y={Y}, s={s}")
    print(f"Размер батча: {n}, DDIM шагов: {steps}")
    
    # Создаем модель
    model = UNet4D(
        in_channels=cfg['channels'],
        base_channels=cfg['base_channels'],
        channel_mults=tuple(cfg['channel_mults']),
        num_res_blocks=cfg['num_res_blocks'],
        uv_kernel=cfg['uv_kernel']
    ).to(device)
    
    # Загружаем веса
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print("Загружены EMA веса")
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        print("Загружены обычные веса")
    else:
        raise ValueError("В чекпоинте нет весов модели!")
    
    model.eval()
    
    # Создаем диффузию
    diff = Diffusion(T=cfg['num_timesteps'], schedule=cfg['beta_schedule'], device=device)
    
    # Генерируем образцы
    print(f"\nГенерация {n} образцов...")
    imgs = diff.ddim_sample(
        model,
        x_shape=(n, cfg['channels'], X, Y),
        steps=steps,
        s=s,
        shift_avg=cfg.get('shift_averaging', True),
        device=device
    )
    
    # Денормализация
    imgs = imgs.clamp(-1, 1)
    imgs = (imgs + 1) / 2
    
    # Сохраняем сетку
    nrow = int(n ** 0.5)
    grid = make_grid(imgs, nrow=nrow)
    
    # Создаем директорию для результатов
    out_dir = 'samples'
    os.makedirs(out_dir, exist_ok=True)
    
    # Определяем имя файла из чекпоинта
    ckpt_name = os.path.basename(ckpt_path).replace('.pt', '')
    out_path = os.path.join(out_dir, f'{ckpt_name}_samples.png')
    
    save_image(grid, out_path)
    print(f"\n[OK] Образцы сохранены: {out_path}")
    
    # Сохраняем также отдельные изображения
    individual_dir = os.path.join(out_dir, f'{ckpt_name}_individual')
    os.makedirs(individual_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        img_path = os.path.join(individual_dir, f'sample_{i:03d}.png')
        save_image(img, img_path)
    print(f"[OK] Отдельные изображения сохранены в: {individual_dir}")
    
    # Показываем статистику
    print(f"\n[Статистика]")
    print(f"   - Размер изображений: {imgs.shape[2]}x{imgs.shape[3]}")
    print(f"   - Количество каналов: {imgs.shape[1]}")
    print(f"   - Количество образцов: {imgs.shape[0]}")
    print(f"   - Диапазон значений: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"   - Среднее значение: {imgs.mean():.3f}")
    print(f"   - Стандартное отклонение: {imgs.std():.3f}")
    
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Быстрая проверка обученной модели')
    ap.add_argument('--ckpt', type=str, default='runs/cifar10/ckpt_last.pt',
                    help='Путь к чекпоинту (по умолчанию: последний чекпоинт)')
    ap.add_argument('--n', type=int, default=16,
                    help='Количество образцов для генерации (по умолчанию: 16)')
    ap.add_argument('--steps', type=int, default=50,
                    help='Количество DDIM шагов (по умолчанию: 50)')
    args = ap.parse_args()
    
    main(args.ckpt, args.n, args.steps)

