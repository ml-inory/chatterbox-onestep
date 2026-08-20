#!/usr/bin/env python3
"""从参考音频提取音色条件（宿主侧，torch + 官方 S3Gen.embed_ref）。

输出三个文件（clone 完整克隆模型 model_clone.axmodel 需要全部三个）：
  ref_embedding.npy      192 维 xvector（基础模型 / clone 模型均需要）
  ref_prompt_token.npy   S3 prompt token [1,157]（clone 模型需要）
  ref_prompt_feat.npy    prompt mel [1,314,80]（clone 模型需要）

基础模型（model.axmodel）为 embedding 级克隆；clone 模型（model_clone.axmodel，带 prompt
条件）为完整官方克隆路径。

用法：
  python3 extract_voice_embedding.py --wav ref.wav --ckpt-dir /path/to/chatterbox_models \
      --out-dir .

（ckpt-dir 需含 s3gen.safetensors；参考音频会 pad/截断到固定 6.28s，对应静态
prompt 314 mel 帧 / 157 token。）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def main():
    p = argparse.ArgumentParser(description="提取参考音频的音色条件（embedding + prompt）")
    p.add_argument("--wav", required=True, help="参考音频（任意采样率，内部重采样到 24k）")
    p.add_argument("--ckpt-dir", required=True, help="Chatterbox 模型目录（含 s3gen.safetensors）")
    p.add_argument("--out-dir", default=".", help="输出目录（embedding + prompt token/feat）")
    args = p.parse_args()

    import torch
    from chatterbox.models.s3gen import S3Gen, S3GEN_SR
    from safetensors.torch import load_file

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(Path(args.ckpt_dir) / "s3gen.safetensors"), strict=False)
    s3gen.eval()

    wav, _ = librosa.load(args.wav, sr=S3GEN_SR, mono=True)
    prompt_samples = int(6.28 * S3GEN_SR)
    if len(wav) > prompt_samples:
        wav = wav[:prompt_samples]
    elif len(wav) < prompt_samples:
        wav = np.pad(wav, (0, prompt_samples - len(wav)))
    with torch.inference_mode():
        ref_dict = s3gen.embed_ref(wav, S3GEN_SR, device="cpu")
        emb = ref_dict["embedding"].numpy().astype(np.float32)      # (1,192)
        p_tok = ref_dict["prompt_token"].numpy().astype(np.int32)   # (1,T)
        p_feat = ref_dict["prompt_feat"].numpy().astype(np.float32)  # (1,T',80)
    # 静态对齐：prompt_token [1,157]、prompt_feat [1,314,80]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pt = np.zeros((1, 157), dtype=np.int32)
    pt[:, : min(p_tok.shape[1], 157)] = p_tok[:, :157]
    pf = np.zeros((1, 314, 80), dtype=np.float32)
    pf[:, : min(p_feat.shape[1], 314)] = p_feat[:, :314]
    np.save(out / "ref_embedding.npy", emb)
    np.save(out / "ref_prompt_token.npy", pt)
    np.save(out / "ref_prompt_feat.npy", pf)
    print(f"OK: embedding {emb.shape}, prompt_token {pt.shape}, prompt_feat {pf.shape} -> {out}")
    print("板端 OpenAI 调用示例（clone 模型）：")
    print(
        "  curl -X POST http://<board>:8000/v1/audio/speech -H 'Content-Type: application/json' "
        "-d " + json.dumps({"input": [12, 34, 56],
                             "voice": {"embedding": emb.reshape(-1).tolist(),
                                       "prompt_token": pt.reshape(-1).tolist(),
                                       "prompt_feat": pf.reshape(-1).tolist()},
                             "response_format": "wav"})[:200] + " ..."
    )


if __name__ == "__main__":
    main()
