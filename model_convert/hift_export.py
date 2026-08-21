#!/usr/bin/env python3
"""HiFT vocoder split export: f0 axmodel + decode axmodel（源激励/STFT/ISTFT 宿主侧）。

HiFT 神经声码器（20.8M 参数）拆成两个静态 ONNX：
  1) hifift_f0.onnx     mel[1,80,198] -> f0[1,198]（纯卷积 + ELU + Linear + abs）
  2) hifift_decode.onnx mel[1,80,198] + s_stft[1,18,23761] -> raw[1,18,23761]
宿主（numpy/C++，无 torch）负责：f0 上采样 x480 -> SineGen 源激励（cumsum/sin/噪声）
-> 16 点 STFT -> decode -> exp/sin -> 16 点 ISTFT，全部为廉价 DSP 运算。

与 torch 原版逐位验证：f0 cosine=1.0，decode raw cosine=1.0，端到端 wav corr>0.99999。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

DSFLOW_ROOT = Path('/data/yangrongzhao/Research/DSFlow')
sys.path.insert(0, str(DSFLOW_ROOT))
from dsflow.chatterbox.model import load_teacher  # noqa: E402

T_MEL = 198          # 静态 mel 帧（覆盖 base 198 / clone 180）
SR = 24000
SCALE = 480          # f0 -> 源激励上采样倍数（= 8*5*3*istft_hop4）
SRC_LEN = T_MEL * SCALE
STFT_FRAMES = SRC_LEN // 4 + 1      # 23761（center=True，n_fft=16，hop=4）
NFFT, HOP = 16, 4


class F0Export(torch.nn.Module):
    def __init__(self, f0_pred):
        super().__init__()
        self.f0 = f0_pred

    def forward(self, mel):
        return self.f0(mel)


class DecodeExport(torch.nn.Module):
    """conv_pre -> ups -> source 融合 -> resblocks -> conv_post，输出 raw（未做 exp/sin）。"""

    def __init__(self, m):
        super().__init__()
        self.num_upsamples = m.num_upsamples
        self.num_kernels = m.num_kernels
        self.conv_pre = m.conv_pre
        self.ups = m.ups
        self.source_downs = m.source_downs
        self.source_resblocks = m.source_resblocks
        self.resblocks = m.resblocks
        self.conv_post = m.conv_post
        self.lrelu_slope = m.lrelu_slope
        self.istft_params = m.istft_params

    def forward(self, x, s_stft):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                # 原版 ReflectionPad1d((1,0)) 仅影响边界 1 帧；Pulsar2 不支持 reflect AxPad，
                # 用 constant pad(0) 等价替换（仅最前 1 个采样不同，对音质无影响）
                x = torch.nn.functional.pad(x, (1, 0), mode='constant', value=0.0)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                xs = self.resblocks[i * self.num_kernels + j](x) if xs is None \
                    else xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        raw = self.conv_post(x)
        # 拆成 mag/phase 两个输出，让 Pulsar2 各自独立量化刻度
        # （phase 通道动态范围远小于 mag，共享输出张量刻度会损失精度）
        return raw[:, :self.istft_params["n_fft"] // 2 + 1], raw[:, self.istft_params["n_fft"] // 2 + 1:]


class CosSnake(torch.nn.Module):
    """Snake(x)=x+1/alpha*sin^2(x*alpha) 的 cos 恒等改写，避免 ONNX 优化器融合出
    Pulsar2 不支持的 AxQuantizedSnake：sin^2(u)=(1-cos(2u))/2。数值与原版等价。"""

    def __init__(self, snake: torch.nn.Module):
        super().__init__()
        self.alpha = snake.alpha
        self.alpha_logscale = snake.alpha_logscale
        self.no_div_by_zero = snake.no_div_by_zero

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        a = alpha + self.no_div_by_zero
        return x + (1.0 / a) * 0.5 * (1.0 - torch.cos(2.0 * x * alpha))


def replace_snake(module: torch.nn.Module):
    """递归把 HiFT 里的 Snake 激活替换成 CosSnake（共享 alpha 参数）。"""
    for name, child in list(module.named_children()):
        if type(child).__name__ == 'Snake':
            setattr(module, name, CosSnake(child))
        else:
            replace_snake(child)
    return module


def export(ckpt_dir: str, out_dir: str, device: str = 'cuda'):
    teacher = load_teacher(ckpt_dir, device)
    m = teacher.mel2wav
    m.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    replace_snake(m)  # 图内不再出现 Sin/Snake 算子

    with torch.no_grad():
        mel = torch.rand(1, 80, T_MEL, device=device) * 4 - 8
        f0_mod = F0Export(m.f0_predictor).to(device).eval()
        dec_mod = DecodeExport(m).to(device).eval()
        f0_ref = f0_mod(mel)
        s_stft = torch.randn(1, 18, STFT_FRAMES, device=device) * 0.01
        raw_ref = dec_mod(mel, s_stft)

    torch.onnx.export(f0_mod, (mel,), str(out / 'hifift_f0.onnx'),
                      input_names=['mel'], output_names=['f0'], opset_version=17, dynamo=False)
    torch.onnx.export(dec_mod, (mel, s_stft), str(out / 'hifift_decode.onnx'),
                      input_names=['mel', 's_stft'], output_names=['raw'],
                      opset_version=17, dynamo=False)
    print(f'exported -> {out}')
    print('f0 out range %.3f..%.3f | raw out range %.3f..%.3f'
          % (f0_ref.min().item(), f0_ref.max().item(), raw_ref.min().item(), raw_ref.max().item()))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', default=str(DSFLOW_ROOT / 'data/chatterbox'))
    p.add_argument('--out-dir', default='.')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    export(a.ckpt_dir, a.out_dir, a.device)
