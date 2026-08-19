#!/usr/bin/env python3
"""OpenAI-Compatible 客户端示例：把 S3 speech tokens POST 到 /v1/audio/speech 生成 wav。

用法：
  python3 openai_client.py --tokens "12,34,56" --out out.wav
  python3 openai_client.py --tokens-npy sample_input/tokens.npy --out out.wav --fmt mp3
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description="OpenAI-Compatible 板端 TTS client")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/audio/speech")
    p.add_argument("--tokens", default="", help="逗号分隔的 S3 token id 列表")
    p.add_argument("--tokens-npy", default="", help="或从 npy 读取 token 序列")
    p.add_argument("--out", default="output.wav")
    p.add_argument("--fmt", default="wav", choices=["wav", "mel"])
    p.add_argument("--model", default="chatterbox-onestep")
    args = p.parse_args()

    if args.tokens:
        tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    elif args.tokens_npy:
        arr = np.load(args.tokens_npy).reshape(-1)
        tokens = [int(t) for t in arr if t > 0]
    else:
        print("请提供 --tokens 或 --tokens-npy")
        sys.exit(1)

    body = json.dumps({
        "model": args.model, "input": tokens, "voice": "default", "response_format": args.fmt,
    }).encode("utf-8")
    req = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            audio = resp.read()
    except urllib.error.HTTPError as e:
        print("server error:", e.code, e.read().decode()[:500])
        sys.exit(1)
    Path(args.out).write_bytes(audio)
    print(f"OK: {len(audio)} bytes -> {args.out}")


if __name__ == "__main__":
    main()
