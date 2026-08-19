#!/usr/bin/env bash
set -euo pipefail

echo "=== 运行推理 ==="
python3 python/demo.py
# ./cpp/build/model_example models/model.axmodel
