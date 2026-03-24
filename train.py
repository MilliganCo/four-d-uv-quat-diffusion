# train.py
from __future__ import annotations

import os, math, argparse, yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from datasets.cifar10 import make_loader
from diffusion.schedules import Diffusion
from models.unet4d import UNet4D
from ops.fold_unfold_4d import image_to_4d, uv_grad_l2
from utils.logger import JSONLLogger
from ops.quat_utils import apply_quaternion_weight_norm


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)





def main(cfg):
    torch.manual_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        try:
            # Уменьшаем использование VRAM до 60% для безопасности
            torch.cuda.set_per_process_memory_fraction(0.6, torch.cuda.current_device())
            # Очищаем кэш перед началом
            torch.cuda.empty_cache()
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')

        # Устанавливаем временную директорию на диск E для swap файлов
        import os
        temp_dir = os.path.join('E:', 'tmp_pytorch')
        os.makedirs(temp_dir, exist_ok=True)
        os.environ['TMPDIR'] = temp_dir
        os.environ['TMP'] = temp_dir
        os.environ['TEMP'] = temp_dir

    # Уменьшаем num_workers для экономии памяти
    num_workers = min(cfg.get('num_workers', 4), 2)
    dl = make_loader(cfg['batch_size'], num_workers, train=True)


    # X=H/s, Y=W/s
    s = cfg['stride_s']
    X = cfg['image_size'] // s
    Y = cfg['image_size'] // s


    model = UNet4D(in_channels=cfg['channels'], base_channels=cfg['base_channels'],
                   channel_mults=tuple(cfg['channel_mults']), num_res_blocks=cfg['num_res_blocks'],
                   uv_kernel=cfg['uv_kernel']).to(device)


    diff = Diffusion(T=cfg['num_timesteps'], schedule=cfg['beta_schedule'], device=device)


    opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    ema = EMA(model, decay=cfg['ema_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['mixed_precision'] and device=='cuda')


    os.makedirs(cfg['output_dir'], exist_ok=True)
    logger = JSONLLogger(cfg.get('log_jsonl', os.path.join(cfg['output_dir'], 'log.jsonl')))
    os.makedirs(cfg.get('previews_dir', os.path.join(cfg['output_dir'], 'previews')), exist_ok=True)


    # Загрузка чекпоинта для продолжения обучения
    global_step = 0
    resume_ckpt = cfg.get('resume_checkpoint', None)
    if resume_ckpt is None:
        # Автоматически ищем последний чекпоинт
        ckpt_last_path = os.path.join(cfg['output_dir'], 'ckpt_last.pt')
        if os.path.exists(ckpt_last_path):
            resume_ckpt = ckpt_last_path
            print(f"Найден чекпоинт для продолжения: {ckpt_last_path}")
    
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Загрузка чекпоинта: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            print("Модель загружена")
        if 'ema' in ckpt:
            ema.shadow = ckpt['ema']
            print("EMA веса загружены")
        if 'opt' in ckpt:
            opt.load_state_dict(ckpt['opt'])
            print("Оптимизатор загружен")
        if 'step' in ckpt:
            global_step = ckpt['step']
            print(f"Продолжение с шага: {global_step}")
    else:
        print("Начинаем обучение с нуля")
    
    gradient_accumulation_steps = cfg.get('gradient_accumulation_steps', 1)
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Эффективный batch size: {cfg['batch_size'] * gradient_accumulation_steps}")
    
    for epoch in range(9999999):
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        opt.zero_grad(set_to_none=True)  # Сбрасываем градиенты в начале эпохи
        accumulated_loss = 0.0
        last_loss_eps = 0.0
        last_loss_uv = 0.0
        
        for batch_idx, imgs in enumerate(pbar):
            imgs = imgs.to(device)
            # 4D фолдинг с случайным (δU,δV)
            du = torch.randint(0, s, ()).item()
            dv = torch.randint(0, s, ()).item()
            x4 = image_to_4d(imgs, s=s, du=du, dv=dv)


            B = x4.size(0)
            t = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(x4)
            x_t = diff.q_sample(x4, t, noise)


            try:
                with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                eps_pred = model(x_t, t.float())
                loss_eps = (noise - eps_pred).pow(2).mean()
                loss_uv = cfg['lambda_uv'] * uv_grad_l2(x_t)
                    loss = (loss_eps + loss_uv) / gradient_accumulation_steps  # Нормализуем loss для accumulation
                    accumulated_loss += loss.item() * gradient_accumulation_steps
                    last_loss_eps = float(loss_eps.detach())
                    last_loss_uv = float(loss_uv.detach())

            scaler.scale(loss).backward()
                
                # Обновляем веса только после накопления нужного количества градиентов
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping для стабильности
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
            scaler.step(opt)
            scaler.update()
                    opt.zero_grad(set_to_none=True)
                    
                    # Обновляем глобальный шаг только после реального обновления весов
                    global_step += 1
                    
                    # Очистка кэша CUDA периодически
                    if device == 'cuda' and global_step % 100 == 0:
                        torch.cuda.empty_cache()
                    
                    # проекция кватернионных ядер на единичную норму
                    if cfg.get('quat_weight_norm', False):
                        with torch.no_grad():
                            apply_quaternion_weight_norm(model, eps=cfg.get('quat_norm_eps', 1e-8))
                    
                    # Обновляем EMA только после реального обновления весов
            ema.update(model)

                    # Используем накопленный loss для отображения
                    current_loss = accumulated_loss / gradient_accumulation_steps if accumulated_loss > 0 else 0.0
                    pbar.set_postfix({
                        'loss': current_loss,
                        'loss_eps': last_loss_eps,
                        'loss_uv': last_loss_uv,
                        'step': global_step
                    })
                    
                    # Логируем только после реального обновления весов
                    logger.log(step=global_step,
                               loss=current_loss,
                               loss_eps=last_loss_eps,
                               loss_uv=last_loss_uv,
                               lr=float(opt.param_groups[0]['lr']))
                    
                    accumulated_loss = 0.0
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[ОШИБКА ПАМЯТИ] Шаг {global_step}, batch {batch_idx}")
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    # Пропускаем этот батч и продолжаем
                    opt.zero_grad(set_to_none=True)
                    accumulated_loss = 0.0
                    continue
                else:
                    raise e

            if global_step > 0 and global_step % int(cfg.get('sample_every', 2000)) == 0:
                try:
                    model.eval()
                    ema.copy_to(model)
                    with torch.no_grad():
                        # Очищаем память перед генерацией
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        imgs = diff.ddim_sample(model,
                                                x_shape=(cfg.get('preview_n', 16), cfg['channels'], X, Y),
                                                steps=int(cfg.get('sample_steps_preview', 20)),
                                                s=s, shift_avg=True, device=device)
                        imgs = imgs.clamp(-1,1); imgs = (imgs + 1)/2
                        grid = make_grid(imgs, nrow=int(cfg.get('preview_n', 16)**0.5))
                        out_path = os.path.join(cfg.get('previews_dir', os.path.join(cfg['output_dir'], 'previews')), f'step_{global_step}.png')
                        save_image(grid, out_path)
                    model.train()
                    # Очищаем память после генерации
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Ошибка при генерации предпросмотра на шаге {global_step}: {e}")
                    model.train()
                    if device == 'cuda':
                        torch.cuda.empty_cache()


            # Сохранение чекпоинтов каждые 1000 шагов (с очисткой памяти)
            if global_step > 0 and global_step % 1000 == 0:
                try:
                ckpt = {
                    'model': model.state_dict(),
                    'ema': ema.shadow,
                    'opt': opt.state_dict(),
                    'cfg': cfg,
                    'step': global_step,
                }
                    checkpoint_path = os.path.join(cfg['output_dir'], f'ckpt_{global_step}.pt')
                    # Сохраняем на диск E если возможно
                    if os.path.exists('E:'):
                        # Используем временную директорию на E для больших файлов
                        temp_checkpoint = os.path.join('E:', 'tmp_pytorch', f'ckpt_{global_step}.pt')
                        torch.save(ckpt, temp_checkpoint)
                        # Копируем на основную директорию
                        import shutil
                        shutil.copy2(temp_checkpoint, checkpoint_path)
                    else:
                        torch.save(ckpt, checkpoint_path)
                    # Также сохраняем последний чекпоинт
                    last_checkpoint = os.path.join(cfg['output_dir'], 'ckpt_last.pt')
                    torch.save(ckpt, last_checkpoint)
                    # Очищаем память после сохранения
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Ошибка при сохранении чекпоинта на шаге {global_step}: {e}")
                    # Продолжаем обучение даже если сохранение не удалось

            # Сохранение чекпоинтов каждые 20000 шагов (специальные чекпоинты)
            checkpoint_interval = cfg.get('checkpoint_interval', 20000)
            if global_step > 0 and global_step % checkpoint_interval == 0:
                try:
                    print(f"\n[Сохранение чекпоинта] Шаг {global_step}...")
                ckpt = {
                    'model': model.state_dict(),
                    'ema': ema.shadow,
                        'opt': opt.state_dict(),
                    'cfg': cfg,
                        'step': global_step,
                    }
                    checkpoint_path = os.path.join(cfg['output_dir'], f'ckpt_{global_step}.pt')
                    
                    # Сохраняем на диск E если возможно для экономии места на C
                    if os.path.exists('E:'):
                        temp_checkpoint = os.path.join('E:', 'tmp_pytorch', f'ckpt_{global_step}.pt')
                        torch.save(ckpt, temp_checkpoint)
                        import shutil
                        shutil.copy2(temp_checkpoint, checkpoint_path)
                    else:
                        torch.save(ckpt, checkpoint_path)
                    
                    # Также обновляем последний чекпоинт
                    last_checkpoint = os.path.join(cfg['output_dir'], 'ckpt_last.pt')
                    torch.save(ckpt, last_checkpoint)
                    
                    print(f"[Чекпоинт сохранен] Шаг {global_step}: {checkpoint_path}")
                    
                    # Очищаем память после сохранения
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                        # Показываем использование памяти
                        if torch.cuda.is_available():
                            mem_allocated = torch.cuda.memory_allocated() / 1024**3
                            mem_reserved = torch.cuda.memory_reserved() / 1024**3
                            print(f"[VRAM] Выделено: {mem_allocated:.2f} GB, Зарезервировано: {mem_reserved:.2f} GB")
                except Exception as e:
                    print(f"КРИТИЧЕСКАЯ ОШИБКА при сохранении чекпоинта на шаге {global_step}: {e}")
                    print("Продолжаем обучение, но чекпоинт не сохранен!")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/cifar10.yaml')
    ap.add_argument('--resume', type=str, default=None, help='Путь к чекпоинту для продолжения обучения')
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.resume:
        cfg['resume_checkpoint'] = args.resume
    main(cfg)
