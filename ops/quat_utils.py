# ops/quat_utils.py
from __future__ import annotations

import torch, torch.nn as nn
from typing import Iterable


QUAT_NAMES = ('A','B','C','D')


def normalize_quaternion_kernels(module: nn.Module, eps: float = 1e-8):
    """Ищет в модуле параметры .A/.B/.C/.D (Conv2d) и нормирует их так,
    чтобы для каждого канала ||[A,B,C,D]||_F = 1.
    Работает для depthwise Conv2d: weight.shape == [C,1,K,K]."""
    for name in QUAT_NAMES:
        if not hasattr(module, name):
            return
    A, B, C, D = module.A.weight.data, module.B.weight.data, module.C.weight.data, module.D.weight.data
    # [C,1,K,K] -> [C,K,K]
    def flat(W):
        return W.view(W.size(0), -1)
    stacked = torch.stack([flat(A), flat(B), flat(C), flat(D)], dim=1)  # [C,4,K*K]
    norm = torch.linalg.vector_norm(stacked, dim=(1,2), keepdim=True).clamp_min(eps)  # [C,1,1]
    A.div_(norm.view(-1,1,1,1)); B.div_(norm.view(-1,1,1,1)); C.div_(norm.view(-1,1,1,1)); D.div_(norm.view(-1,1,1,1))




def apply_quaternion_weight_norm(model: nn.Module, eps: float = 1e-8):
    for m in model.modules():
        if hasattr(m, 'A') and hasattr(m, 'B') and hasattr(m, 'C') and hasattr(m, 'D'):
            normalize_quaternion_kernels(m, eps)

