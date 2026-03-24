# test_my_image_detailed.py - Детальное тестирование с максимальным выводом
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


def print_section(title: str):
    """Красивый вывод секции"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_info(label: str, value, unit: str = ""):
    """Форматированный вывод информации"""
    print(f"  {label:.<50} {value} {unit}")


@torch.no_grad()
def main(image_path: str, ckpt_path: str, output_path: str = None, noise_level: int = None):
    """Детальное тестирование с максимальным выводом информации"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print_section("НАЧАЛО ТЕСТИРОВАНИЯ")
    print_info("Устройство", device)
    print_info("Изображение", image_path)
    print_info("Чекпоинт", ckpt_path)
    
    # ========== ЗАГРУЗКА МОДЕЛИ ==========
    print_section("ШАГ 1: ЗАГРУЗКА МОДЕЛИ")
    print("  Загрузка чекпоинта...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if 'cfg' in ckpt:
        cfg = ckpt['cfg']
        print_info("Конфигурация", "загружена из чекпоинта")
    else:
        import yaml
        with open('configs/cifar10.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        print_info("Конфигурация", "загружена из configs/cifar10.yaml")
    
    s = cfg['stride_s']
    print_info("Параметр stride_s", s)
    print_info("Размер изображения (обучение)", f"{cfg['image_size']}x{cfg['image_size']}")
    print_info("Количество каналов", cfg['channels'])
    print_info("Количество временных шагов", cfg['num_timesteps'])
    print_info("Расписание бета", cfg['beta_schedule'])
    
    # Информация о модели
    print("\n  Создание модели...")
    model = UNet4D(
        in_channels=cfg['channels'],
        base_channels=cfg['base_channels'],
        channel_mults=tuple(cfg['channel_mults']),
        num_res_blocks=cfg['num_res_blocks'],
        uv_kernel=cfg['uv_kernel']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print_info("Базовые каналы", cfg['base_channels'])
    print_info("Множители каналов", str(cfg['channel_mults']))
    print_info("Количество Res блоков", cfg['num_res_blocks'])
    print_info("UV kernel размер", cfg['uv_kernel'])
    print_info("Всего параметров", f"{total_params:,}")
    print_info("Обучаемых параметров", f"{trainable_params:,}")
    print_info("Размер модели (MB)", f"{total_params * 4 / 1024 / 1024:.2f}")
    
    # Загрузка весов
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print_info("Веса", "EMA (Exponential Moving Average)")
    else:
        model.load_state_dict(ckpt['model'])
        print_info("Веса", "обычные")
    
    if 'step' in ckpt:
        print_info("Шаг обучения чекпоинта", f"{ckpt['step']:,}")
    
    model.eval()
    print("  ✓ Модель готова к работе")
    
    # ========== ЗАГРУЗКА ИЗОБРАЖЕНИЯ ==========
    print_section("ШАГ 2: ЗАГРУЗКА И ПРЕОБРАЗОВАНИЕ ИЗОБРАЖЕНИЯ")
    
    print(f"  Загрузка: {image_path}")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    print_info("Оригинальный размер", f"{original_size[0]}x{original_size[1]}")
    
    # Анализ изображения
    img_array = np.array(img)
    print_info("Режим изображения", img.mode)
    print_info("Диапазон значений", f"[{img_array.min()}, {img_array.max()}]")
    print_info("Среднее значение", f"{img_array.mean():.2f}")
    print_info("Стандартное отклонение", f"{img_array.std():.2f}")
    
    # Преобразование
    print("\n  Преобразование в тензор...")
    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    print_info("Размер после resize", f"{img_tensor.shape[2]}x{img_tensor.shape[3]}")
    print_info("Форма тензора", str(list(img_tensor.shape)))
    print_info("Диапазон после нормализации", f"[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    print_info("Среднее значение", f"{img_tensor.mean():.3f}")
    print_info("Стандартное отклонение", f"{img_tensor.std():.3f}")
    
    # ========== ПРЕОБРАЗОВАНИЕ В 4D ПОЛЕ ==========
    print_section("ШАГ 3: ПРЕОБРАЗОВАНИЕ В 4D ПОЛЕ")
    
    print("  Преобразование изображения в 4D представление...")
    print(f"  Параметр s (stride): {s}")
    print(f"  Это означает разбиение на блоки {s}x{s}")
    
    du, dv = 0, 0
    x4 = image_to_4d(img_tensor, s=s, du=du, dv=dv)
    
    print_info("Форма 4D поля", str(list(x4.shape)))
    print_info("Batch размер", x4.shape[0])
    print_info("Каналы", x4.shape[1])
    print_info("X размерность", x4.shape[2])
    print_info("Y размерность", x4.shape[3])
    print_info("U размерность", x4.shape[4])
    print_info("V размерность", x4.shape[5])
    print_info("Всего 'точек' в 4D", f"{x4.shape[2] * x4.shape[3]:,}")
    print_info("Вариантов на точку", f"{x4.shape[4] * x4.shape[5]}")
    print_info("Всего элементов", f"{x4.numel():,}")
    print_info("Память (MB)", f"{x4.numel() * 4 / 1024 / 1024:.2f}")
    
    # Статистика 4D поля
    print("\n  Статистика 4D поля:")
    print_info("Минимальное значение", f"{x4.min():.4f}")
    print_info("Максимальное значение", f"{x4.max():.4f}")
    print_info("Среднее значение", f"{x4.mean():.4f}")
    print_info("Стандартное отклонение", f"{x4.std():.4f}")
    
    # ========== ДОБАВЛЕНИЕ ШУМА ==========
    print_section("ШАГ 4: ДОБАВЛЕНИЕ ШУМА И ДЕНОЙЗИНГ")
    
    diff = Diffusion(T=cfg['num_timesteps'], schedule=cfg['beta_schedule'], device=device)
    
    if noise_level is None:
        noise_level = cfg['num_timesteps'] // 2
    
    print_info("Уровень шума (t)", f"{noise_level} / {cfg['num_timesteps']}")
    print_info("Процент зашумления", f"{noise_level / cfg['num_timesteps'] * 100:.1f}%")
    
    # Параметры диффузии для этого уровня
    alpha_bar_t = diff.alphas_cumprod[noise_level]
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
    
    print("\n  Параметры диффузии:")
    print_info("alpha_bar_t", f"{alpha_bar_t:.6f}")
    print_info("sqrt(alpha_bar_t)", f"{sqrt_alpha_bar:.6f}")
    print_info("sqrt(1 - alpha_bar_t)", f"{sqrt_one_minus:.6f}")
    
    # Генерируем шум
    print("\n  Генерация случайного шума...")
    noise = torch.randn_like(x4)
    print_info("Форма шума", str(list(noise.shape)))
    print_info("Среднее шума", f"{noise.mean():.4f}")
    print_info("Стандартное отклонение шума", f"{noise.std():.4f}")
    
    # Добавляем шум
    print("\n  Добавление шума к изображению...")
    t = torch.tensor([noise_level], device=device)
    x_noisy = diff.q_sample(x4, t, noise)
    
    print_info("Форма зашумленного", str(list(x_noisy.shape)))
    print_info("Минимальное значение", f"{x_noisy.min():.4f}")
    print_info("Максимальное значение", f"{x_noisy.max():.4f}")
    print_info("Среднее значение", f"{x_noisy.mean():.4f}")
    print_info("Стандартное отклонение", f"{x_noisy.std():.4f}")
    
    # Вычисляем SNR (Signal-to-Noise Ratio)
    signal_power = (x4 ** 2).mean()
    noise_power = ((x_noisy - x4) ** 2).mean()
    snr_db = 10 * torch.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    print_info("SNR (dB)", f"{snr_db:.2f}")
    
    # ========== ОБРАБОТКА МОДЕЛЬЮ ==========
    print_section("ШАГ 5: ОБРАБОТКА МОДЕЛЬЮ")
    
    print("  Модель анализирует зашумленное изображение...")
    print("  Предсказывает, какой шум был добавлен...")
    
    eps_pred = model(x_noisy, t.float())
    
    print_info("Форма предсказания шума", str(list(eps_pred.shape)))
    print_info("Минимальное значение", f"{eps_pred.min():.4f}")
    print_info("Максимальное значение", f"{eps_pred.max():.4f}")
    print_info("Среднее значение", f"{eps_pred.mean():.4f}")
    print_info("Стандартное отклонение", f"{eps_pred.std():.4f}")
    
    # Сравнение предсказанного и реального шума
    actual_noise = x_noisy - x4
    mse_noise = ((eps_pred - actual_noise) ** 2).mean()
    print_info("MSE между предсказанным и реальным шумом", f"{mse_noise:.6f}")
    
    # ========== ВОССТАНОВЛЕНИЕ ==========
    print_section("ШАГ 6: ВОССТАНОВЛЕНИЕ ИЗОБРАЖЕНИЯ")
    
    print("  Удаление предсказанного шума...")
    x0_pred = (x_noisy - sqrt_one_minus * eps_pred) / sqrt_alpha_bar
    
    print_info("Форма восстановленного", str(list(x0_pred.shape)))
    print_info("Минимальное значение", f"{x0_pred.min():.4f}")
    print_info("Максимальное значение", f"{x0_pred.max():.4f}")
    print_info("Среднее значение", f"{x0_pred.mean():.4f}")
    print_info("Стандартное отклонение", f"{x0_pred.std():.4f}")
    
    # Ошибка восстановления
    mse_reconstruction = ((x0_pred - x4) ** 2).mean()
    psnr = -10 * torch.log10(mse_reconstruction) if mse_reconstruction > 0 else float('inf')
    print_info("MSE восстановления", f"{mse_reconstruction:.6f}")
    print_info("PSNR (dB)", f"{psnr:.2f}")
    
    # ========== ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ ==========
    print_section("ШАГ 7: ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ В 2D")
    
    print("  Преобразование 4D поля обратно в изображение...")
    print("  Использование shift averaging для лучшего качества...")
    
    imgs = []
    for du in range(s):
        for dv in range(s):
            shifted = x0_pred.roll((du, dv), dims=(-2, -1))
            img_2d = field4d_to_image(shifted)
            imgs.append(img_2d)
    
    result = torch.stack(imgs, dim=0).mean(dim=0)
    print_info("Количество вариантов для усреднения", len(imgs))
    print_info("Форма результата", str(list(result.shape)))
    
    # Денормализация
    result = result.clamp(-1, 1)
    result = (result + 1) / 2
    
    print_info("Диапазон после денормализации", f"[{result.min():.4f}, {result.max():.4f}]")
    print_info("Среднее значение", f"{result.mean():.4f}")
    print_info("Стандартное отклонение", f"{result.std():.4f}")
    
    # ========== СОХРАНЕНИЕ ==========
    print_section("ШАГ 8: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f'samples/tested_{base_name}_detailed.png'
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Подготовка изображений для сравнения
    orig_norm = img_tensor.clamp(-1, 1)
    orig_norm = (orig_norm + 1) / 2
    
    noisy_img = field4d_to_image(x_noisy[0:1])
    noisy_img = noisy_img.clamp(-1, 1)
    noisy_img = (noisy_img + 1) / 2
    
    comparison = torch.cat([orig_norm, noisy_img, result], dim=0)
    save_image(comparison, output_path, nrow=3)
    
    print_info("Сравнение сохранено", output_path)
    print("    - Слева: Оригинал (32x32)")
    print("    - В центре: С шумом")
    print("    - Справа: Восстановленное")
    
    # Отдельно результат
    result_path = output_path.replace('.png', '_result_only.png')
    save_image(result, result_path)
    print_info("Только результат", result_path)
    
    # Статистика финального результата
    print("\n  Финальная статистика:")
    print_info("Размер результата", f"{result.shape[2]}x{result.shape[3]}")
    print_info("Количество каналов", result.shape[1])
    print_info("Диапазон значений", f"[{result.min():.4f}, {result.max():.4f}]")
    print_info("Среднее значение", f"{result.mean():.4f}")
    print_info("Стандартное отклонение", f"{result.std():.4f}")
    
    print_section("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("  Все результаты сохранены в папке samples/")
    print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Детальное тестирование изображения с максимальным выводом')
    ap.add_argument('--image', type=str, required=True, help='Путь к изображению')
    ap.add_argument('--ckpt', type=str, default='runs/cifar10/ckpt_last.pt', help='Путь к чекпоинту')
    ap.add_argument('--out', type=str, default=None, help='Путь для сохранения')
    ap.add_argument('--noise_level', type=int, default=None, help='Уровень шума (0-T), по умолчанию T/2')
    args = ap.parse_args()
    
    main(args.image, args.ckpt, args.out, args.noise_level)

