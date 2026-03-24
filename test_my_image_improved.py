# test_my_image_improved.py - Улучшенный денойзинг с итеративным процессом
from __future__ import annotations

import os, argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from models.unet4d import UNet4D
from diffusion.schedules import Diffusion
from ops.fold_unfold_4d import image_to_4d, field4d_to_image


def iterative_denoise(model, x_noisy, diff, noise_level, steps=10, device='cuda'):
    """
    Итеративный денойзинг - постепенное удаление шума за несколько шагов
    Это более эффективно, чем одношаговое восстановление
    """
    x = x_noisy.clone()
    B = x.shape[0]
    
    # Создаем последовательность временных шагов от noise_level до 0
    ts = torch.linspace(noise_level, 0, steps + 1, device=device).long()
    
    print(f"  Итеративный денойзинг: {steps} шагов")
    print(f"  Временные шаги: {ts.tolist()}")
    
    for i, t in enumerate(ts[:-1]):
        t_batch = t.expand(B)
        t_prev = ts[i + 1]
        
        # Предсказываем шум
        eps_pred = model(x, t_batch.float())
        
        # Параметры для текущего и предыдущего шага
        alpha_bar_t = diff.alphas_cumprod[t]
        alpha_bar_prev = diff.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        
        # Оценка x0
        x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
        
        # DDIM шаг (без случайного шума для детерминированности)
        if t_prev >= 0:
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev = torch.sqrt(1 - alpha_bar_prev)
            
            # Обновление x
            x = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * eps_pred
        else:
            x = x0_pred
        
        if (i + 1) % max(1, steps // 5) == 0:
            print(f"    Шаг {i+1}/{steps}: t={t.item()} -> t={t_prev.item()}")
    
    return x


@torch.no_grad()
def main(image_path: str, ckpt_path: str, output_path: str = None, 
         noise_level: int = None, denoise_steps: int = 20):
    """Улучшенное тестирование с итеративным денойзингом"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("УЛУЧШЕННЫЙ ДЕНОЙЗИНГ С ИТЕРАТИВНЫМ ПРОЦЕССОМ")
    print("=" * 70)
    print(f"\nУстройство: {device}")
    print(f"Изображение: {image_path}")
    print(f"Чекпоинт: {ckpt_path}")
    print(f"Шагов денойзинга: {denoise_steps}\n")
    
    # Загрузка модели
    print("[1] Загрузка модели...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if 'cfg' in ckpt:
        cfg = ckpt['cfg']
    else:
        import yaml
        with open('configs/cifar10.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
    
    s = cfg['stride_s']
    
    model = UNet4D(
        in_channels=cfg['channels'],
        base_channels=cfg['base_channels'],
        channel_mults=tuple(cfg['channel_mults']),
        num_res_blocks=cfg['num_res_blocks'],
        uv_kernel=cfg['uv_kernel']
    ).to(device)
    
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print("  Загружены EMA веса")
    else:
        model.load_state_dict(ckpt['model'])
        print("  Загружены обычные веса")
    
    if 'step' in ckpt:
        print(f"  Шаг обучения: {ckpt['step']:,}")
    
    model.eval()
    print("  ✓ Модель загружена\n")
    
    # Загрузка изображения
    print("[2] Загрузка изображения...")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    print(f"  Оригинальный размер: {original_size[0]}x{original_size[1]}")
    
    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"  Размер после преобразования: {img_tensor.shape[2]}x{img_tensor.shape[3]}\n")
    
    # Преобразование в 4D
    print("[3] Преобразование в 4D поле...")
    x4 = image_to_4d(img_tensor, s=s, du=0, dv=0)
    print(f"  4D поле: {x4.shape}\n")
    
    # Добавление шума
    print("[4] Добавление шума...")
    diff = Diffusion(T=cfg['num_timesteps'], schedule=cfg['beta_schedule'], device=device)
    
    if noise_level is None:
        # Используем более низкий уровень шума для лучшего восстановления
        noise_level = cfg['num_timesteps'] // 3  # 33% вместо 50%
    
    print(f"  Уровень шума: {noise_level} / {cfg['num_timesteps']} ({noise_level/cfg['num_timesteps']*100:.1f}%)")
    
    noise = torch.randn_like(x4)
    t = torch.tensor([noise_level], device=device)
    x_noisy = diff.q_sample(x4, t, noise)
    print("  ✓ Шум добавлен\n")
    
    # Итеративный денойзинг
    print("[5] Итеративный денойзинг...")
    x_denoised = iterative_denoise(model, x_noisy, diff, noise_level, steps=denoise_steps, device=device)
    print("  ✓ Денойзинг завершен\n")
    
    # Обратное преобразование
    print("[6] Преобразование обратно в изображение...")
    imgs = []
    for du in range(s):
        for dv in range(s):
            shifted = x_denoised.roll((du, dv), dims=(-2, -1))
            imgs.append(field4d_to_image(shifted))
    result = torch.stack(imgs, dim=0).mean(dim=0)
    
    result = result.clamp(-1, 1)
    result = (result + 1) / 2
    print("  ✓ Преобразование завершено\n")
    
    # Сохранение
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f'samples/tested_{base_name}_improved.png'
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Сравнение: оригинал, зашумленное, восстановленное
    orig_norm = img_tensor.clamp(-1, 1)
    orig_norm = (orig_norm + 1) / 2
    
    noisy_img = field4d_to_image(x_noisy[0:1])
    noisy_img = noisy_img.clamp(-1, 1)
    noisy_img = (noisy_img + 1) / 2
    
    comparison = torch.cat([orig_norm, noisy_img, result], dim=0)
    save_image(comparison, output_path, nrow=3)
    
    print("=" * 70)
    print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ")
    print("=" * 70)
    print(f"Файл: {output_path}")
    print("  - Слева: Оригинал")
    print("  - В центре: С шумом")
    print("  - Справа: Восстановленное (итеративный денойзинг)")
    print()
    
    # Также сохраняем только результат
    result_path = output_path.replace('.png', '_result_only.png')
    save_image(result, result_path)
    print(f"Только результат: {result_path}\n")
    
    # Статистика
    mse = ((result - orig_norm) ** 2).mean()
    psnr = -10 * torch.log10(mse) if mse > 0 else float('inf')
    print(f"Статистика восстановления:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Улучшенный денойзинг с итеративным процессом')
    ap.add_argument('--image', type=str, required=True, help='Путь к изображению')
    ap.add_argument('--ckpt', type=str, default='runs/cifar10/ckpt_last.pt', help='Путь к чекпоинту')
    ap.add_argument('--out', type=str, default=None, help='Путь для сохранения')
    ap.add_argument('--noise_level', type=int, default=None, help='Уровень шума (0-T), по умолчанию T/3')
    ap.add_argument('--steps', type=int, default=20, help='Количество шагов итеративного денойзинга')
    args = ap.parse_args()
    
    main(args.image, args.ckpt, args.out, args.noise_level, args.steps)

