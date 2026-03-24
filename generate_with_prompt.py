# generate_with_prompt.py - Генерация изображений с текстовыми промптами
from __future__ import annotations

import os, argparse, yaml
import torch
from torchvision.utils import save_image, make_grid
from models.unet4d import UNet4D
from models.text_encoder import get_text_encoder, CIFAR10_CLASSES, CIFAR10_CLASS_DESCRIPTIONS
from diffusion.schedules import Diffusion


@torch.no_grad()
def main(cfg_path: str, ckpt_path: str, out_dir: str, prompt: str = None, 
         class_label: int = None, n: int = 4, steps: int = 100):
    """
    Генерация изображений с текстовыми промптами
    
    Args:
        cfg_path: Путь к конфигурации
        ckpt_path: Путь к чекпоинту
        out_dir: Директория для сохранения
        prompt: Текстовый промпт (например, "cat", "a cat sitting")
        class_label: Метка класса (0-9) вместо промпта
        n: Количество образцов
        steps: Количество DDIM шагов
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Устройство: {device}")
    
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    use_text = cfg.get('use_text_conditioning', False)
    if not use_text:
        print("ВНИМАНИЕ: Модель не поддерживает условную генерацию!")
        print("Используется безусловная генерация.")
    
    s = cfg['stride_s']
    X = cfg['image_size'] // s
    Y = cfg['image_size'] // s
    
    print(f"Разрешение: {cfg['image_size']}x{cfg['image_size']}")
    print(f"4D размеры: X={X}, Y={Y}, U={s}, V={s}")
    
    # Создание модели
    text_dim = cfg.get('text_dim', 128) if use_text else 0
    model = UNet4D(
        in_channels=cfg['channels'],
        base_channels=cfg['base_channels'],
        channel_mults=tuple(cfg['channel_mults']),
        num_res_blocks=cfg['num_res_blocks'],
        uv_kernel=cfg['uv_kernel'],
        text_dim=text_dim
    ).to(device)
    
    # Загрузка весов модели
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print("Загружены EMA веса модели")
    else:
        model.load_state_dict(ckpt['model'])
        print("Загружены обычные веса модели")
    model.eval()
    
    # Текстовый энкодер
    text_encoder = None
    text_emb = None
    
    if use_text:
        encoder_type = cfg.get('text_encoder', 'class_label')
        text_encoder = get_text_encoder(encoder_type, text_dim).to(device)
        
        if 'text_encoder' in ckpt:
            text_encoder.load_state_dict(ckpt['text_encoder'])
            print("Текстовый энкодер загружен")
        elif 'ema_text' in ckpt:
            text_encoder.load_state_dict(ckpt['ema_text'])
            print("Загружены EMA веса текстового энкодера")
        else:
            print("ВНИМАНИЕ: Текстовый энкодер не найден в чекпоинте!")
            print("Используется случайная инициализация.")
        
        text_encoder.eval()
        
        # Определяем метку класса
        if class_label is not None:
            label = class_label
            class_name = CIFAR10_CLASSES[label]
            print(f"Используется класс: {class_name} (метка {label})")
        elif prompt:
            # Пытаемся найти класс по промпту
            prompt_lower = prompt.lower()
            label = None
            for i, cls in enumerate(CIFAR10_CLASSES):
                if cls in prompt_lower:
                    label = i
                    class_name = cls
                    print(f"Найден класс по промпту: {class_name} (метка {label})")
                    break
            
            if label is None:
                print(f"Класс не найден в промпте '{prompt}'")
                print(f"Доступные классы: {', '.join(CIFAR10_CLASSES)}")
                print("Используется случайный класс.")
                label = torch.randint(0, 10, (1,)).item()
                class_name = CIFAR10_CLASSES[label]
        else:
            # Случайный класс
            label = torch.randint(0, 10, (1,)).item()
            class_name = CIFAR10_CLASSES[label]
            print(f"Случайный класс: {class_name} (метка {label})")
        
        # Кодируем метку класса
        labels = torch.tensor([label] * n, device=device)
        text_emb = text_encoder(labels)
        print(f"Текстовые эмбеддинги созданы: {text_emb.shape}")
    
    # Создание диффузии
    diff = Diffusion(T=cfg['num_timesteps'], schedule=cfg['beta_schedule'], device=device)
    
    # Генерация
    print(f"\nГенерация {n} образцов...")
    print(f"DDIM шагов: {steps}")
    if use_text and text_emb is not None:
        print(f"Класс: {class_name}")
    
    imgs = diff.ddim_sample(
        model,
        x_shape=(n, cfg['channels'], X, Y),
        steps=steps,
        s=s,
        shift_avg=cfg.get('shift_averaging', True),
        device=device,
        text_emb=text_emb
    )
    
    # Денормализация
    imgs = imgs.clamp(-1, 1)
    imgs = (imgs + 1) / 2
    
    # Сохранение
    os.makedirs(out_dir, exist_ok=True)
    
    if prompt or class_label is not None:
        filename = f"{class_name}_{n}samples_{steps}steps.png"
    else:
        filename = f"generated_{n}samples_{steps}steps.png"
    
    grid = make_grid(imgs, nrow=int(n**0.5))
    out_path = os.path.join(out_dir, filename)
    save_image(grid, out_path)
    
    print(f"\n[OK] Образцы сохранены: {out_path}")
    
    # Отдельные изображения
    individual_dir = os.path.join(out_dir, f"{class_name}_individual" if (prompt or class_label is not None) else "generated_individual")
    os.makedirs(individual_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        save_image(img, os.path.join(individual_dir, f'sample_{i:03d}.png'))
    print(f"[OK] Отдельные изображения: {individual_dir}/")
    
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Генерация изображений с текстовыми промптами')
    ap.add_argument('--config', type=str, default='configs/cifar10_stage_b_64x64.yaml')
    ap.add_argument('--ckpt', type=str, required=True, help='Путь к чекпоинту')
    ap.add_argument('--out', type=str, default='samples', help='Директория для сохранения')
    ap.add_argument('--prompt', type=str, default=None, 
                    help='Текстовый промпт (например: "cat", "bird", "a cat sitting")')
    ap.add_argument('--class', type=int, default=None, choices=range(10),
                    help='Метка класса (0-9): 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck')
    ap.add_argument('--n', type=int, default=4, help='Количество образцов')
    ap.add_argument('--steps', type=int, default=100, help='Количество DDIM шагов')
    args = ap.parse_args()
    
    # Обработка аргумента --class (Python резервирует class как ключевое слово)
    class_label = getattr(args, 'class', None)
    
    main(args.config, args.ckpt, args.out, args.prompt, class_label, args.n, args.steps)

