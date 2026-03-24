# test_my_image_simple.py - Простое объяснение и тестирование своего изображения
"""
ЧТО ПРОИСХОДИТ С ВАШИМ ИЗОБРАЖЕНИЕМ:

1. ЗАГРУЗКА И ПОДГОТОВКА
   - Ваше изображение 64x64 загружается
   - Автоматически уменьшается до 32x32 (размер, на котором обучена модель)
   - Преобразуется в формат, который понимает модель

2. ПРЕОБРАЗОВАНИЕ В 4D ПОЛЕ
   - Обычное изображение: 2D (ширина x высота)
   - Модель работает с 4D полем: (X x Y x U x V)
   - Это как создать "сетку" из вашего изображения:
     * X, Y - основные координаты (как обычные пиксели, но в 2 раза меньше)
     * U, V - "сдвиги" или "варианты" изображения (обычно 2x2 = 4 варианта)
   - Представьте: ваше изображение разбивается на блоки 2x2, и каждый блок
     становится отдельной "точкой" в 4D пространстве

3. ДОБАВЛЕНИЕ ШУМА (опционально, для демонстрации)
   - К изображению добавляется случайный шум
   - Это имитирует процесс, когда изображение "портится"
   - Модель обучена восстанавливать такие "испорченные" изображения

4. ОБРАБОТКА МОДЕЛЬЮ
   - Модель анализирует изображение (или зашумленное изображение)
   - Она пытается "понять", что на изображении
   - Используя знания, полученные при обучении на CIFAR-10, модель
     улучшает изображение или убирает шум

5. ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ
   - 4D поле преобразуется обратно в обычное 2D изображение
   - Если использовались сдвиги (U, V), они усредняются для лучшего качества
   - Получается финальное изображение 32x32

ВАЖНО:
- Модель обучена на CIFAR-10 (32x32, 10 классов объектов)
- Ваше изображение будет уменьшено до 32x32
- Модель лучше всего работает с объектами, похожими на CIFAR-10:
  (автомобили, самолеты, кошки, собаки, птицы, олени, лошади, корабли, грузовики)
"""

from __future__ import annotations

import os, argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from models.unet4d import UNet4D
from diffusion.schedules import Diffusion
from ops.fold_unfold_4d import image_to_4d, field4d_to_image


def explain_process():
    """Объяснение процесса простыми словами"""
    print("=" * 60)
    print("ЧТО ПРОИСХОДИТ С ВАШИМ ИЗОБРАЖЕНИЕМ:")
    print("=" * 60)
    print()
    print("1. ЗАГРУЗКА:")
    print("   - Ваше изображение загружается")
    print("   - Автоматически изменяется размер до 32x32")
    print("   - (Модель обучена на изображениях 32x32)")
    print()
    print("2. ПРЕОБРАЗОВАНИЕ В 4D ПОЛЕ:")
    print("   - Обычное изображение: 2D (ширина x высота)")
    print("   - Модель работает с 4D: (X x Y x U x V)")
    print("   - Ваше изображение разбивается на блоки 2x2")
    print("   - Каждый блок становится 'точкой' в 4D пространстве")
    print("   - Это как создать 4 варианта вашего изображения со сдвигами")
    print()
    print("3. ОБРАБОТКА МОДЕЛЬЮ:")
    print("   - Модель анализирует изображение")
    print("   - Использует знания, полученные при обучении")
    print("   - Может улучшить качество или убрать шум")
    print()
    print("4. ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ:")
    print("   - 4D поле превращается обратно в обычное изображение")
    print("   - Разные варианты усредняются для лучшего качества")
    print("   - Получается финальное изображение 32x32")
    print()
    print("=" * 60)
    print()


@torch.no_grad()
def main(image_path: str, ckpt_path: str, output_path: str = None, add_noise: bool = False):
    """Тестирование своего изображения"""
    
    explain_process()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Устройство: {device}\n")
    
    # Загружаем модель
    print(f"[1/5] Загрузка модели из: {ckpt_path}")
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
    else:
        model.load_state_dict(ckpt['model'])
    model.eval()
    print("   Модель загружена ✓\n")
    
    # Загружаем изображение
    print(f"[2/5] Загрузка изображения: {image_path}")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    print(f"   Оригинальный размер: {original_size[0]}x{original_size[1]}")
    
    # Преобразуем в тензор
    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"   Размер после преобразования: {img_tensor.shape[2]}x{img_tensor.shape[3]}\n")
    
    # Преобразуем в 4D поле
    print(f"[3/5] Преобразование в 4D поле...")
    x4 = image_to_4d(img_tensor, s=s, du=0, dv=0)
    print(f"   4D поле: {x4.shape} (Batch, Channels, X, Y, U, V)")
    print(f"   Это означает: {x4.shape[2]}x{x4.shape[3]} основных точек,")
    print(f"   каждая с {x4.shape[4]}x{x4.shape[5]} вариантами\n")
    
    # Обработка моделью
    print(f"[4/5] Обработка моделью...")
    
    if add_noise:
        # Добавляем шум для демонстрации денойзинга
        diff = Diffusion(T=cfg['num_timesteps'], schedule=cfg['beta_schedule'], device=device)
        noise_level = cfg['num_timesteps'] // 2
        noise = torch.randn_like(x4)
        t = torch.tensor([noise_level], device=device)
        x_noisy = diff.q_sample(x4, t, noise)
        
        print(f"   Добавлен шум (уровень {noise_level}/{cfg['num_timesteps']})")
        eps_pred = model(x_noisy, t.float())
        
        # Восстанавливаем
        alpha_bar_t = diff.alphas_cumprod[noise_level]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
        x4_result = (x_noisy - sqrt_one_minus * eps_pred) / sqrt_alpha_bar
    else:
        # Просто пропускаем через модель (без шума)
        # Используем небольшой уровень шума для "улучшения"
        t = torch.tensor([0], device=device)
        eps_pred = model(x4, t.float())
        # Небольшая коррекция
        x4_result = x4 - 0.1 * eps_pred
    
    print("   Обработка завершена ✓\n")
    
    # Обратное преобразование
    print(f"[5/5] Преобразование обратно в изображение...")
    
    # Используем shift averaging
    imgs = []
    for du in range(s):
        for dv in range(s):
            shifted = x4_result.roll((du, dv), dims=(-2, -1))
            imgs.append(field4d_to_image(shifted))
    result = torch.stack(imgs, dim=0).mean(dim=0)
    
    # Денормализация
    result = result.clamp(-1, 1)
    result = (result + 1) / 2
    
    print(f"   Результат готов: {result.shape[2]}x{result.shape[3]}\n")
    
    # Сохраняем
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f'samples/tested_{base_name}.png'
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Сравнение
    if add_noise:
        # Оригинал, зашумленное, восстановленное
        orig_norm = img_tensor.clamp(-1, 1)
        orig_norm = (orig_norm + 1) / 2
        
        noisy_img = field4d_to_image(x_noisy[0:1])
        noisy_img = noisy_img.clamp(-1, 1)
        noisy_img = (noisy_img + 1) / 2
        
        comparison = torch.cat([orig_norm, noisy_img, result], dim=0)
        save_image(comparison, output_path, nrow=3)
        print(f"[ГОТОВО] Результаты сохранены: {output_path}")
        print("   Слева: Оригинал (32x32)")
        print("   В центре: С шумом")
        print("   Справа: Восстановленное")
    else:
        # Оригинал и результат
        orig_norm = img_tensor.clamp(-1, 1)
        orig_norm = (orig_norm + 1) / 2
        
        comparison = torch.cat([orig_norm, result], dim=0)
        save_image(comparison, output_path, nrow=2)
        print(f"[ГОТОВО] Результаты сохранены: {output_path}")
        print("   Слева: Оригинал (32x32)")
        print("   Справа: Обработанное моделью")
    
    # Отдельно результат
    result_path = output_path.replace('.png', '_result_only.png')
    save_image(result, result_path)
    print(f"   Только результат: {result_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Тестирование своего изображения с объяснением процесса',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python test_my_image_simple.py --image my_photo.jpg
  python test_my_image_simple.py --image my_photo.jpg --add_noise
  python test_my_image_simple.py --image my_photo.jpg --ckpt runs/cifar10/ckpt_60000.pt
        """
    )
    ap.add_argument('--image', type=str, required=True, help='Путь к вашему изображению')
    ap.add_argument('--ckpt', type=str, default='runs/cifar10/ckpt_last.pt', help='Путь к чекпоинту')
    ap.add_argument('--out', type=str, default=None, help='Путь для сохранения результата')
    ap.add_argument('--add_noise', action='store_true', help='Добавить шум для демонстрации денойзинга')
    args = ap.parse_args()
    
    main(args.image, args.ckpt, args.out, args.add_noise)

