# chatterbox-s3gen-onestep C++ SDK

- 输入（与 model_meta.json 一致）: tokens[1, 256], token_len[1], embedding[1, 192], z[1, 80, 512]
- 输出（与 model_meta.json 一致）: mel[1, 80, 512]
- 直接链接 AX Engine runtime（`ax_engine`/`ax_sys`），目标: AX650

```bash
cmake -S cpp -B cpp/build-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=cpp/toolchain-aarch64.cmake \
  -DAX_RUNTIME_ROOT=/path/to/ax/runtime
cmake --build cpp/build-aarch64
```
