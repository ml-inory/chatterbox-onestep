import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatterbox_s3gen_onestep_sdk.inference import ModelSession
from chatterbox_s3gen_onestep_sdk.postprocess import postprocess
from chatterbox_s3gen_onestep_sdk.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description="chatterbox-s3gen-onestep inference example")
    parser.add_argument("--model", required=True, help="AXMODEL 路径")
    parser.add_argument("--input", nargs="+", required=True, help="输入 npy（每输入一个，与 model_meta 顺序一致）")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    args = parser.parse_args()

    arrays = [np.load(p).astype(np.float32) for p in args.input]
    session = ModelSession(args.model)
    feeds = preprocess(*arrays)
    raw = session.run_named(feeds)
    result = postprocess(*raw)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(raw):
        np.save(out_dir / f"output_{i}.npy", np.asarray(arr, dtype=np.float32))
    try:
        json.dumps(result)
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except TypeError:
        np.save(out_dir / "result.npy", np.asarray(result, dtype=np.float32))

    print("backend:", session.backend)
    print("inputs:", session.input_names)
    print("outputs:", session.output_names)
    print("saved to:", out_dir)


if __name__ == "__main__":
    main()
