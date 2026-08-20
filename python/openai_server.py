#!/usr/bin/env python3
"""OpenAI-Compatible 板端 TTS 服务示例（POST /v1/audio/speech，torch-free）。

仅依赖 numpy + pyaxengine（AX 芯片）：
  input(speech tokens) ──S3Gen AXMODEL(axengine)──> mel ──numpy Griffin-Lim──> wav

说明：
  - 本服务输出的是 S3Gen 的"语音 token -> 音频"能力；若需要 text -> audio，
    在宿主用 T3（torch，官方 chatterbox）把文本转成 speech tokens 后再调用本服务
    （或直接把文本换成语义 token 序列）。板端本身不依赖 torch。
  - mel_basis_24k.npy（S3Gen mel 前端滤波矩阵）与 default_embedding.npy（内置音色）
    随包提供；voice 参数可传自定义 192 维 xvector。
  - response_format 支持 wav（默认）/ mel（JSON 调试）。纯标准库写 WAV，不依赖 ffmpeg。

用法（板端）：
  python3 openai_server.py --model models/model.axmodel --port 8000
  python3 openai_client.py --tokens "12,34,56" --out out.wav
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

SR = 24000
N_FFT = 1920
HOP = 480
WINDOW = np.hanning(N_FFT).astype(np.float32)


def _load_pkg_assets(here: Path):
    mel_basis = np.load(here / "mel_basis_24k.npy")  # (80, 961)
    emb = np.load(here / "default_embedding.npy")  # (1, 192)
    # 最小二乘伪逆：mel 幅度 -> 线性谱幅度
    inv = np.linalg.pinv(mel_basis.astype(np.float64)).astype(np.float32)
    return mel_basis, inv, emb


def _stft(x: np.ndarray) -> np.ndarray:
    """numpy STFT（center=False，hop=480, n_fft=1920）-> (961, T) 复数谱"""
    n = (len(x) - N_FFT) // HOP + 1
    frames = np.stack([x[i * HOP:i * HOP + N_FFT] for i in range(n)])  # (T, 1920)
    return np.fft.rfft(frames * WINDOW, axis=1).T  # (961, T)


def _istft(spec: np.ndarray) -> np.ndarray:
    """numpy ISTFT（overlap-add + COLA 归一化）"""
    T = spec.shape[1]
    n = (T - 1) * HOP + N_FFT
    out = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    frames = np.fft.irfft(spec.T, n=N_FFT, axis=1)  # (T, 1920)
    for i in range(T):
        s = i * HOP
        out[s:s + N_FFT] += frames[i] * WINDOW
        wsum[s:s + N_FFT] += WINDOW ** 2
    eps = 1e-8
    return (out / np.maximum(wsum, eps)).astype(np.float32)


def griffin_lim(mel_log10: np.ndarray, mel_inv: np.ndarray, iters: int = 32) -> np.ndarray:
    """log10 mel(1,80,T) -> 24kHz 波形（纯 numpy，无 torch）"""
    M = np.power(10.0, mel_log10[0].astype(np.float64))  # (80,T) 线性 mel 幅度
    V = np.maximum(mel_inv @ M, 0.0)  # (961,T) 线性谱幅度
    phase = np.random.rand(*V.shape) * 2.0 * math.pi
    spec = V * np.exp(1j * phase)
    for _ in range(iters):
        x = _istft(spec)
        X = _stft(x)
        spec = V * np.exp(1j * np.angle(X))
    x = _istft(spec)
    peak = np.max(np.abs(x)) + 1e-8
    return (x / peak).astype(np.float32)


def mel_to_wav(mel_log10: np.ndarray, mel_inv: np.ndarray) -> bytes:
    x = griffin_lim(mel_log10, mel_inv)
    pcm = (x * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class TTSApp:
    def __init__(self, args):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from chatterbox_s3gen_onestep_sdk.inference import ModelSession
        from chatterbox_s3gen_onestep_sdk.preprocess import preprocess

        self.session = ModelSession(args.model)
        self.clone_session = ModelSession(args.clone_model) if getattr(args, "clone_model", "") else None
        self.preprocess = preprocess
        here = Path(__file__).resolve().parent
        self.mel_basis, self.mel_inv, self.default_emb = _load_pkg_assets(here)
        # clone 模型默认 prompt（可用 extract_voice_embedding.py 生成后放在同目录）
        self.default_prompt = None
        if self.clone_session:
            pt_path, pf_path = here / "ref_prompt_token.npy", here / "ref_prompt_feat.npy"
            if pt_path.exists() and pf_path.exists():
                self.default_prompt = (np.load(pt_path), np.load(pf_path))

    def synthesize(self, tokens: list[int], voice=None, fmt="wav") -> bytes:
        tokens = np.clip(np.asarray(tokens, dtype=np.int32).reshape(1, -1), 0, 6560)
        tlen = np.asarray([tokens.shape[1]], dtype=np.int32)
        if tokens.shape[1] < 256:
            pad = np.zeros((1, 256 - tokens.shape[1]), dtype=np.int32)
            tokens = np.concatenate([tokens, pad], axis=1)
        embedding = None if voice in (None, "default") else (voice.get("embedding") if isinstance(voice, dict) else voice)
        emb = self.default_emb if embedding is None else np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if self.clone_session:
            # 完整克隆路径（5 输入）：tokens_all = prompt_token + 生成 token（宿主侧拼接，
            # 避免图内 Concat——AX650 NPU 对 AxConcat 有 wdma bug）+ embedding + z + prompt_feat
            pt, pf = self.default_prompt if self.default_prompt is not None else (None, None)
            if isinstance(voice, dict):
                pt = np.asarray(voice.get("prompt_token"), dtype=np.int32).reshape(1, -1) if voice.get("prompt_token") else pt
                pf = np.asarray(voice.get("prompt_feat"), dtype=np.float32).reshape(1, -1, 80) if voice.get("prompt_feat") else pf
            if pt is None or pf is None:
                raise ValueError("clone 模型需要 prompt：请用 extract_voice_embedding.py 生成 ref_prompt_token/ref_prompt_feat.npy 放同目录，或请求 voice.prompt_token/prompt_feat")
            gen = np.asarray(tokens, dtype=np.int32).reshape(-1)[:99]
            if len(gen) < 99:
                gen = np.pad(gen, (0, 99 - len(gen)), mode="edge")
            pt_pad = np.zeros(157, dtype=np.int32)
            pt_pad[: min(pt.shape[1], 157)] = pt[:, :157].reshape(-1)
            tokens_all = np.concatenate([pt_pad, gen]).reshape(1, 256)
            gen_len = min(int(tlen[0]), 99)
            token_len_all = np.asarray([157 + gen_len], dtype=np.int32)
            pf_pad = np.zeros((1, 314, 80), dtype=np.float32)
            pf_pad[:, : min(pf.shape[1], 314)] = pf[:, :314, :]
            z = np.random.randn(1, 80, 512).astype(np.float32)
            feeds = [tokens_all, token_len_all, emb, z, pf_pad]
            raw = self.clone_session.run_named(feeds)
        else:
            feeds = self.preprocess(tokens, tlen, emb, None)
            raw = self.session.run_named(feeds)
        mel = np.asarray(raw[0], dtype=np.float32)[:, :, :int(tlen[0]) * 2]  # (1,80,T)
        if fmt == "mel":
            return json.dumps({"mel": mel[0].tolist()}).encode("utf-8")
        wav = mel_to_wav(mel, self.mel_inv)
        return wav


def make_handler(app: TTSApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("[openai-server] %s\n" % (fmt % args))

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [{
                    "id": "chatterbox-onestep", "object": "model", "created": 0, "owned_by": "axera",
                }]})
            else:
                self._json(404, {"error": {"message": "not found", "type": "invalid_request_error", "code": None}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/audio/speech":
                return self._json(404, {"error": {"message": "not found", "type": "invalid_request_error", "code": None}})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))
            except Exception:
                return self._json(400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error", "code": None}})
            inp = req.get("input")
            fmt = str(req.get("response_format", "wav"))
            if isinstance(inp, str):
                inp = [int(t) for t in inp.split(",") if t.strip()]
            if not isinstance(inp, list) or not inp or not all(isinstance(t, int) for t in inp):
                return self._json(400, {"error": {"message": "input must be a non-empty list of S3 token ids (or comma-separated string)", "type": "invalid_request_error", "code": None}})
            if fmt not in ("wav", "mel"):
                return self._json(400, {"error": {"message": f"response_format '{fmt}' not supported (wav|mel)", "type": "invalid_request_error", "code": None}})
            voice = req.get("voice")
            try:
                audio = app.synthesize(inp, voice=voice, fmt=fmt)
            except Exception as e:  # noqa: BLE001
                return self._json(500, {"error": {"message": f"synthesis failed: {e}", "type": "server_error", "code": None}})
            ctype = {"wav": "audio/wav", "mel": "application/json"}[fmt]
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)

    return Handler


def main():
    p = argparse.ArgumentParser(description="OpenAI-Compatible 板端 S3Gen TTS（torch-free）")
    p.add_argument("--model", default="models/model.axmodel")
    p.add_argument("--clone-model", default="", help="完整克隆模型 model_clone.axmodel（可选）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    app = TTSApp(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"[openai-server] listening on http://{args.host}:{args.port} (torch-free, axengine)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
