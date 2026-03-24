# ops/conv_uv_quat.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuaternionUVConv(nn.Module):
    """

    Кватернионный оператор по (U,V) для каждого канала независимо.

    Реализация через 4 depthwise Conv2d по UV и линейную комбинацию (матрица Гамильтона).



    Вход:  x [B, C, X, Y, U, V]

    Выход: y [B, C, X, Y, U, V]

    """
    def __init__(self, channels: int, kernel_size: int = 1, bias: bool = False):
        super().__init__()
        assert kernel_size in (1, 3), "kernel_size должен быть 1 или 3"
        padding = kernel_size // 2
        # depthwise по каналам, работаем только над UV (будем разворачивать в forward)
        self.A = nn.Conv2d(channels, channels, kernel_size, padding=padding,
                           groups=channels, bias=bias, padding_mode='circular')
        self.B = nn.Conv2d(channels, channels, kernel_size, padding=padding,
                           groups=channels, bias=bias, padding_mode='circular')
        self.C = nn.Conv2d(channels, channels, kernel_size, padding=padding,
                           groups=channels, bias=bias, padding_mode='circular')
        self.D = nn.Conv2d(channels, channels, kernel_size, padding=padding,
                           groups=channels, bias=bias, padding_mode='circular')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, U, V = x.shape
        x0 = x[..., 0, 0]
        x1 = x[..., 0, 1]
        x2 = x[..., 1, 0]
        x3 = x[..., 1, 1]
        # свертки по UV делаем, превратив U,V в spatial, а C — в каналы
        # N = B*X*Y, Cin = C, H=U, W=V
        x_uv = x.permute(0, 2, 3, 1, 4, 5).reshape(B*X*Y, C, U, V)
        x0_uv = x_uv[:, :, 0:1, 0:1].expand(-1, -1, U, V)  # тождественно, чтобы свернуть по UV ядрами
        x1_uv = x_uv[:, :, 0:1, 1:2].expand(-1, -1, U, V)
        x2_uv = x_uv[:, :, 1:2, 0:1].expand(-1, -1, U, V)
        x3_uv = x_uv[:, :, 1:2, 1:2].expand(-1, -1, U, V)


        Ax0 = self.A(x0_uv)
        Bx1 = self.B(x1_uv)
        Cx2 = self.C(x2_uv)
        Dx3 = self.D(x3_uv)


        # четыре выхода в форме [N,C,U,V]
        y0 = Ax0 - Bx1 - Cx2 - Dx3
        y1 = self.B(x0_uv) + self.A(x1_uv) - self.D(x2_uv) + self.C(x3_uv)
        y2 = self.C(x0_uv) + self.D(x1_uv) + self.A(x2_uv) - self.B(x3_uv)
        y3 = self.D(x0_uv) - self.C(x1_uv) + self.B(x2_uv) + self.A(x3_uv)


        # возвращаем в [B,C,X,Y,U,V]
        def back(y):
            return y.reshape(B, X, Y, C, U, V).permute(0, 3, 1, 2, 4, 5).contiguous()
        y0, y1, y2, y3 = map(back, (y0, y1, y2, y3))
        y = torch.stack([y0[...,0,0], y1[...,0,1], y2[...,1,0], y3[...,1,1]], dim=-1)
        # stack вернул [B,C,X,Y,4]; вернём в [B,C,X,Y,2,2]
        y = y.view(B, C, X, Y, 2, 2)
        return y
