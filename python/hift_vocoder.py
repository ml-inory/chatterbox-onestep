#!/usr/bin/env python3
"""HiFT 神经声码器（NPU 版）：f0/decode 两个 AXMODEL + 宿主侧 DSP（纯 numpy，无 torch）。

HiFT 被拆成两个静态模型：
  hifift_f0.axmodel      mel[1,80,198] -> f0[1,198]（纯卷积，U8）
  hifift_decode.axmodel  mel[1,80,198] + s_stft[1,18,23761] -> raw_mag[1,9,23761],
                         raw_phase[1,9,23761]（U16，双输出独立量化）

宿主侧 DSP（本模块）：f0 最近邻上采样 x480 -> SineGen 源激励（cumsum/sin/噪声）
-> 16 点 STFT -> decode -> exp/sin -> 16 点 ISTFT -> clamp。

与 torch 原版 HiFT 逐位验证：stft max diff ~5.6e-9、istft ~1.3e-7、源激励 ~4.9e-10、
端到端 wav corr > 0.99999（fp32）；板端 U16 编译后 wav corr ~0.92、>4kHz 能量与 torch 持平。
"""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path

import numpy as np

SR = 24000
T_MEL = 198              # 静态 mel 帧
PAD_VAL = -11.0          # mel 尾部补静音值（log10）
NFFT, HOP = 16, 4
HARMONICS = 9            # 8 次谐波 + 基频
SINE_AMP = 0.1
NOISE_STD = 0.003
VOICED_THRESHOLD = 10.0
UPSAMPLE_SCALE = 480     # f0 -> 源激励上采样倍数
STFT_FRAMES = T_MEL * UPSAMPLE_SCALE // HOP + 1   # 23761
AUDIO_LIMIT = 0.99

# periodic hann == torch hann_window(16)
HANN16 = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(NFFT) / NFFT))).astype(np.float64)


def _reflect_pad1d(x: np.ndarray, p: int) -> np.ndarray:
    return np.pad(x, (p, p), mode="reflect")


def f0_upsample(f0: np.ndarray) -> np.ndarray:
    """(1,T) Hz -> (1,T*480) 最近邻上采样。"""
    return np.repeat(f0, UPSAMPLE_SCALE, axis=-1)


def sine_source(f0_up: np.ndarray, phase: np.ndarray, noise: np.ndarray,
                linear_w: np.ndarray, linear_b: np.ndarray) -> np.ndarray:
    """SineGen + SourceModuleHnNSF：f0_up(1,T*480) -> 源激励 s(1,1,T*480)。"""
    F_mat = np.stack([f0_up * (h + 1) / SR for h in range(HARMONICS)], axis=1)
    theta = 2.0 * math.pi * (np.cumsum(F_mat, axis=-1) % 1.0)
    sine = SINE_AMP * np.sin(theta + phase)
    uv = (f0_up > VOICED_THRESHOLD).astype(np.float64)
    noise_amp = uv * NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0
    sine_wavs = sine * uv + noise_amp * noise
    return np.tanh(np.einsum("oh,bht->bot", linear_w, sine_wavs) + linear_b.reshape(1, 1, -1))


def source_stft(s: np.ndarray) -> np.ndarray:
    """源激励 (1,1,L) -> s_stft (1,18,F) real/imag 拼接。"""
    x = s[0, 0]
    xp = _reflect_pad1d(x, NFFT // 2)
    frames = np.lib.stride_tricks.sliding_window_view(xp, NFFT)[::HOP] * HANN16
    spec = np.fft.rfft(frames, n=NFFT, axis=1)
    return np.concatenate([spec.real.T, spec.imag.T], axis=0)[None].astype(np.float32)


def _istft(mag: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """mag/ph (F,9) -> wav (L,)：16 点 ISTFT + 去 center pad。"""
    F = mag.shape[0]
    frames = np.fft.irfft(mag * np.exp(1j * ph), n=NFFT, axis=1) * HANN16
    n = (F - 1) * HOP + NFFT
    idx = np.arange(F)[:, None] * HOP + np.arange(NFFT)[None, :]
    out = np.bincount(idx.ravel(), weights=frames.ravel(), minlength=n)
    wsum = np.bincount(idx.ravel(), weights=np.tile(HANN16 * HANN16, F), minlength=n)
    y = np.divide(out, wsum, out=np.zeros_like(out), where=wsum > 1e-8)
    return y[NFFT // 2:n - NFFT // 2]


class HiftVocoder:
    """NPU HiFT 声码器：f0 + decode 两个 axmodel（默认本地 AxEngineExecutionProvider）。"""

    def __init__(self, f0_model: str | Path, decode_model: str | Path,
                 linear_w: np.ndarray, linear_b: np.ndarray, providers=None):
        import axengine as axe
        self.f0 = axe.InferenceSession(str(f0_model), providers=providers or ["AxEngineExecutionProvider"])
        self.dec = axe.InferenceSession(str(decode_model), providers=providers or ["AxEngineExecutionProvider"])
        self.linear_w = np.asarray(linear_w, dtype=np.float64).reshape(1, HARMONICS)
        self.linear_b = np.asarray(linear_b, dtype=np.float64).reshape(-1)
        self.out_names = [o.name for o in self.dec.get_outputs()]

    def _run(self, sess, feeds):
        return sess.run(None, {k: np.ascontiguousarray(v) for k, v in feeds.items()})

    def synth(self, mel: np.ndarray, valid_frames: int) -> np.ndarray:
        """mel(1,80,T_valid) -> wav (T_valid*480,)，尾部按 valid_frames 截断。"""
        mel = np.asarray(mel, dtype=np.float32)
        T = mel.shape[2]
        mel_pad = np.full((1, 80, T_MEL), PAD_VAL, dtype=np.float32)
        mel_pad[:, :, :T] = mel
        rng = np.random.default_rng()
        phase = rng.uniform(-np.pi, np.pi, size=(1, HARMONICS, 1))
        phase[:, 0, :] = 0.0
        noise = rng.standard_normal((1, HARMONICS, T_MEL * UPSAMPLE_SCALE))

        f0 = self._run(self.f0, {"mel": mel_pad})[0].astype(np.float32)
        s = sine_source(f0_upsample(f0), phase, noise, self.linear_w, self.linear_b)
        s_stft = source_stft(s)
        outs = self._run(self.dec, {"mel": mel_pad, "s_stft": s_stft})
        if len(outs) == 2 and len(self.out_names) == 2:
            mag_raw, ph_raw = outs[0][0], outs[1][0]
        else:
            raw = outs[0][0]
            mag_raw, ph_raw = raw[:9], raw[9:]
        mag = np.exp(np.asarray(mag_raw, dtype=np.float64))
        ph = np.sin(np.asarray(ph_raw, dtype=np.float64))
        wav = np.clip(_istft(mag.T, ph.T), -AUDIO_LIMIT, AUDIO_LIMIT)
        return wav[: int(valid_frames) * UPSAMPLE_SCALE].astype(np.float32)


def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
