#!/usr/bin/env bash
set -euo pipefail

echo "=== 安装依赖 ==="
pip install -r python/chatterbox_s3gen_onestep_sdk/requirements.txt

echo "C++ SDK: 请先安装 AX650 BSP SDK，然后："
# export AX_RUNTIME_ROOT=/path/to/axruntime
# mkdir -p cpp/build && cd cpp/build
# cmake .. -DCMAKE_TOOLCHAIN_FILE=${AX_RUNTIME_ROOT}/toolchain.cmake
# make -j$(nproc)

echo "✅ 环境准备完成"
