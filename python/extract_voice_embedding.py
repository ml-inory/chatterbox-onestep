#!/usr/bin/env python3
"""从参考音频提取 192 维音色 embedding（宿主侧，torch + 官方 S3Gen.speaker_encoder）。

本 AXMODEL 以 192 维 xvector 作为音色条件（embedding 级克隆/音色迁移）。
完整官方克隆还含 10s 参考 prompt 条件，当前板端 AXMODEL 未导出该路径。

用法：
  python3 extract_voice_embedding.py --wav ref.wav --ckpt-dir /path/to/chatterbox_models \
      --out ref_embedding.npy

（ckpt-dir 需含 s3gen.safetensors；参考音频建议 6-10s 清晰人声。）

然后调用板端 OpenAI 服务时带上 voice.embedding：
  curl -X POST http://<board>:8000/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"input":[12,34,56],"voice":{"embedding":[...192 个 float...]},"response_format":"wav"}'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def main():
    p = argparse.ArgumentParser(description="提取参考音频的 192 维音色 embedding")
    p.add_argument("--wav", required=True, help="参考音频（任意采样率，内部重采样到 16k）")
    p.add_argument("--ckpt-dir", required=True, help="Chatterbox 模型目录（含 ve.safetensors）")
    p.add_argument("--out", default="voice_embedding.npy")
    args = p.parse_args()

    import torch
    from chatterbox.models.s3gen import S3Gen, S3GEN_SR
    from safetensors.torch import load_file

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(Path(args.ckpt_dir) / "s3gen.safetensors"), strict=False)
    s3gen.eval()

    wav, sr = librosa.load(args.wav, sr=S3GEN_SR, mono=True)
    with torch.inference_mode():
        ref_dict = s3gen.embed_ref(wav, S3GEN_SR, device="cpu")
        emb = ref_dict["embedding"].numpy().astype(np.float32)  # (1,192)
    np.save(args.out, emb)
    print(f"OK: embedding {emb.shape} -> {args.out}")
    print("板端 OpenAI 调用示例：")
    print(
        "  curl -X POST http://<board>:8000/v1/audio/speech -H 'Content-Type: application/json' "
        "-d " + json.dumps({"input": [12, 34, 56], "voice": {"embedding": emb.reshape(-1).tolist()},
                             "response_format": "wav"})[:160] + " ..."
    )


if __name__ == "__main__":
    main()
