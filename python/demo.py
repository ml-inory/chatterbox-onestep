#!/usr/bin/env python3
"""NPU 端到端推理 demo：tokens/token_len/embedding/z -> mel[1,80,512]（仅依赖 numpy + axengine）。"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatterbox_s3gen_onestep_sdk.inference import ModelSession
from chatterbox_s3gen_onestep_sdk.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]
session = ModelSession(str(ROOT / "models" / "model.axmodel"))
inp = ROOT / "sample_input"
feeds = [np.load(inp / f) for f in ["tokens.npy", "token_len.npy", "embedding.npy", "z.npy"]]
raw = session.run_named(preprocess(*feeds))
mel = np.asarray(raw[0], dtype=np.float32)
np.save(ROOT / "output_mel.npy", mel)
print(f"OK: mel {mel.shape} backend={session.backend} saved=output_mel.npy")
