# test_my_image.py - Тестирование своего изображения
from __future__ import annotations

import os, argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from models.unet4d import UNet4D
from diffusion.schedules import Diffusion
from ops.fold_unfold_4d import image_to_4d, field4d_to_image


@torch.no_grad()
def main(image_path: str, ckpt_path: str, output_path: str = None):
    """
    Простой скрипт для тестирования своего изображения
    
    Что происходит:
    1. Загружаем ваше изображение
    2. Преобразуем его в формат, который понимает модель (4D поле)
    3. Добавляем шум (как в процессе обучения)
    4. Модель пытается убрать шум (денойзинг)
    5. Преобразуем обратно в обычное изображение
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Устройство: {device}")
    
    # Загружаем чекпоинт
    print(f"Загрузка модели из: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if 'cfg' in ckpt:
        cfg = ckpt['cfg']
    else:
        import yaml
        with open('configs/cifar10.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
    
    s = cfg['stride_s']  # обычно 2
    print(f"Параметр s (stride): {s}")
    
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
    else:
        model.load_state_dict(ckpt['model'])
        print("Загружены обычные веса")
    
    model.eval()
    
    # Загружаем ваше изображение
    print(f"\nЗагрузка изображения: {image_path}")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    print(f"Оригинальный размер: {original_size[0]}x{original_size[1]}")
    
    # Преобразуем в тензор и нормализуем
    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),  # Изменяем размер до 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 32, 32]
    print(f"Размер после преобразования: {img_tensor.shape}")
    
    # ШАГ 1: Преобразуем изображение в 4D поле
    print("\n[ШАГ 1] Преобразование изображения в 4D поле...")
    print("  Что происходит: Ваше 2D изображение (ширина x высота) превращается")
    print("  в 4D поле (X x Y x U x V), где U и V - это 'сдвиги' или 'варианты'")
    print("  изображения. Это как создать несколько копий изображения со сдвигами.")
    
    du, dv = 0, 0  # Можно попробовать разные сдвиги
    x4 = image_to_4d(img_tensor, s=s, du=du, dv=dv)
    print(f"  4D поле: {x4.shape} (Batch, Channels, X, Y, U, V)")
    
    # ШАГ 2: Добавляем шум (симулируем процесс диффузии)
    print("\n[ШАГ 2] Добавление шума...")
    print("  Что происходит: Добавляем случайный шум к вашему изображению.")
    print("  Это имитирует процесс, когда изображение постепенно 'зашумляется'.")
    
    diff = Diffusion(T=cfg['num_timesteps'], schedule=cfg['beta_schedule'], device=device)
    
    # Выбираем уровень шума (0 = без шума, T-1 = максимальный шум)
    noise_level = cfg['num_timesteps'] // 2  # Средний уровень шума
    print(f"  Уровень шума: {noise_level} из {cfg['num_timesteps']}")
    
    noise = torch.randn_like(x4)
    t = torch.tensor([noise_level], device=device)
    x_noisy = diff.q_sample(x4, t, noise)
    print(f"  Зашумленное изображение готово")
    
    # ШАГ 3: Модель пытается убрать шум
    print("\n[ШАГ 3] Денойзинг (удаление шума)...")
    print("  Что происходит: Модель анализирует зашумленное изображение и")
    print("  пытается предсказать, какой шум был добавлен, чтобы его убрать.")
    print("  Это как 'умный фильтр', который знает, как выглядели оригинальные изображения.")
    
    eps_pred = model(x_noisy, t.float())
    print(f"  Модель предсказала шум")
    
    # ШАГ 4: Восстанавливаем изображение
    print("\n[ШАГ 4] Восстановление изображения...")
    print("  Что происходит: Убираем предсказанный шум из зашумленного изображения.")
    print("  Получаем 'очищенную' версию вашего изображения.")
    
    alpha_bar_t = diff.alphas_cumprod[noise_level]
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
    
    # Оценка оригинального изображения
    x0_pred = (x_noisy - sqrt_one_minus * eps_pred) / sqrt_alpha_bar
    
    # ШАГ 5: Преобразуем обратно в обычное изображение
    print("\n[ШАГ 5] Преобразование обратно в 2D изображение...")
    print("  Что происходит: 4D поле (X x Y x U x V) преобразуется обратно")
    print("  в обычное 2D изображение (ширина x высота).")
    print("  Если использовались сдвиги, они усредняются для лучшего качества.")
    
    # Используем shift averaging для лучшего качества
    from ops.fold_unfold_4d import field4d_to_image
    imgs = []
    for du in range(s):
        for dv in range(s):
            shifted = x0_pred.roll((du, dv), dims=(-2, -1))
            imgs.append(field4d_to_image(shifted))
    result = torch.stack(imgs, dim=0).mean(dim=0)
    
    # Денормализация
    result = result.clamp(-1, 1)
    result = (result + 1) / 2  # [0, 1]
    
    print(f"  Результат готов: {result.shape}")
    
    # Сохраняем результаты
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f'samples/tested_{base_name}.png'
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Сохраняем сравнение: оригинал, зашумленное, восстановленное
    comparison = torch.cat([
        img_tensor,  # Оригинал (нормализованный)
        x_noisy[0, :, :, :, 0, 0].unsqueeze(0),  # Зашумленное (первый срез)
        result  # Восстановленное
    ], dim=0)
    
    # Денормализуем все для сохранения
    comparison = comparison.clamp(-1, 1)
    comparison = (comparison + 1) / 2
    
    save_image(comparison, output_path, nrow=3)
    print(f"\n[ГОТОВО] Результаты сохранены: {output_path}")
    print("  Слева: Ваше оригинальное изображение (изменено до 32x32)")
    print("  В центре: Зашумленная версия")
    print("  Справа: Восстановленное моделью изображение")
    
    # Также сохраняем только результат
    result_path = output_path.replace('.png', '_result.png')
    save_image(result, result_path)
    print(f"  Только результат: {result_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Тестирование своего изображения')
    ap.add_argument('--image', type=str, required=True, help='Путь к вашему изображению')
    ap.add_argument('--ckpt', type=str, default='runs/cifar10/ckpt_last.pt', help='Путь к чекпоинту')
    ap.add_argument('--out', type=str, default=None, help='Путь для сохранения результата')
    args = ap.parse_args()
    
    main(args.image, args.ckpt, args.out)

