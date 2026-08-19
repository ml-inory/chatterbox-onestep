# 预编译产物（aarch64 / AX650）

由 `examples/` 源码交叉编译（aarch64-none-linux-gnu-g++ 9.2，Release + fast-math），链接 AX Engine：

- `openai_server` —— OpenAI-Compatible TTS 服务（`--model ../../models/model.axmodel --assets ../assets --port 8000`）
- `openai_client` —— 对应客户端
- `model_example` —— SDK 基础示例
- `libchatterbox_s3gen_onestep_sdk.a` —— C++ SDK 静态库

板端运行需 `LD_LIBRARY_PATH=/soc/lib`（AX Engine runtime）。
