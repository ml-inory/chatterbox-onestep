# chatterbox-s3gen-onestep AXMODEL

精度 cosine ≈ 0.9932  |  推理耗时 181.08 ms  |  模型大小 124.7 MB  |  实时率 RTF ≈ 0.018

## 源码 / 复现

想要复现导出与编译流程、查看或修改 Python/C++ SDK 源码（含 OpenAI-Compatible server/client）、
运行 DSP 自测？源码仓库：**https://github.com/ml-inory/chatterbox-onestep**

- `model_convert/`：ONNX 导出 + Pulsar2 编译脚本（可复现本 AXMODEL）
- `python/` / `cpp/`：完整 SDK 源码与示例
- `cpp/tests/`：FFT/Griffin-Lim 本地自测

## 性能与实时率（RTF）

| 场景 | 耗时 | RTF |
| --- | --- | --- |
| AX650C 板端 NPU（本模型，单步） | 181 ms / 512 帧（10.24s 音频） | **≈ 0.018** |
| NVIDIA L4 GPU（同模型，单步） | 47.2 ms | ≈ 0.0046 |
| 教师 10 步（L4，参考） | ~540 ms/句 | ≈ 0.06 |

RTF = 生成耗时 / 音频时长，< 1 即快于实时。本模型 S3Gen 单步在板端生成 10 秒音频的 mel 只需约
0.18 秒；相比教师 10 步，单步化带来约 10 倍加速（RTF 0.06 → 0.018，同板卡口径）。

## 快速开始（只需两步）

### 1. 安装环境

```bash
bash setup.sh
```

### 2. 跑推理

```bash
bash run.sh
```

## 目录说明

| 目录 | 用途 |
|------|------|
| `models/` | AXMODEL 模型文件 + 元信息 |
| `python/` | Python SDK（pyaxengine）|
| `cpp/` | C++ SDK（AX Engine runtime）|
| `model_convert/` | 模型导出 & 编译脚本（可复现）|
| `reports/` | 各阶段报告 |
| `setup.sh` | 一键安装依赖 |
| `run.sh` | 一键运行推理 |

## 常见问题

**Q: import 报错找不到 pyaxengine？**
A: 运行 `bash setup.sh` 会自动安装。

**Q: 怎么在自己的代码里用？**
A: 参考 `python/demo.py`，核心就 3 行：

```python
from chatterbox-s3gen-onestep_sdk import ModelSDK
sdk = ModelSDK("models/model.axmodel")
result = sdk.run(your_input)
```

**Q: 想自己重新编译？**
A: 进入 `model_convert/`，确保 Pulsar2 可用后运行 `bash compile_pulsar2.sh`。
   原始编译使用 Docker 镜像 `docker-registry.aitsw.axera-tech.com/pulsar2:7.0`。

## 示例音频（学生单步 vs 教师 10 步，同噪声 A/B）

三句话的端到端合成示例：**学生 1 步（本模型）** 与 **教师 10 步（官方）** 对比，均为 24kHz WAV、响度归一化。

### 句 1：Technology is best when it brings people together.

<audio controls src="examples/audio/s0_student1.wav"></audio>
<audio controls src="examples/audio/s0_teacher10.wav"></audio>

- 学生 1 步：[s0_student1.wav](examples/audio/s0_student1.wav)
- 教师 10 步：[s0_teacher10.wav](examples/audio/s0_teacher10.wav)

### 句 2：The quick brown fox jumps over the lazy dog.

<audio controls src="examples/audio/s1_student1.wav"></audio>
<audio controls src="examples/audio/s1_teacher10.wav"></audio>

- 学生 1 步：[s1_student1.wav](examples/audio/s1_student1.wav)
- 教师 10 步：[s1_teacher10.wav](examples/audio/s1_teacher10.wav)

### 句 3：A journey of a thousand miles begins with a single step.

<audio controls src="examples/audio/s2_student1.wav"></audio>
<audio controls src="examples/audio/s2_teacher10.wav"></audio>

- 学生 1 步：[s2_student1.wav](examples/audio/s2_student1.wav)
- 教师 10 步：[s2_teacher10.wav](examples/audio/s2_teacher10.wav)

## OpenAI-Compatible 服务（板端 torch-free）

提供 OpenAI 风格的 `POST /v1/audio/speech`（另含 `GET /v1/models`）：输入 S3 speech token 序列，
输出 wav（`response_format=wav`）或 mel（`response_format=mel`）。**Python 与 C++ 两版均可在 AX 板上运行，
不依赖 torch / ffmpeg / FFTW**；文本 → token 的 T3 环节在宿主（torch）完成后调用本服务。

### Python 版（依赖 numpy + pyaxengine）

```bash
python3 python/openai_server.py --model models/model.axmodel --port 8000
python3 python/openai_client.py --tokens-npy sample_input/tokens.npy --out out.wav
```

### C++ 版（自实现 Bluestein FFT + Griffin-Lim，仅依赖 AX Engine）

> 本仓库 `cpp/bin/` 提供 aarch64 预编译产物（Release + fast-math）；C++ 完整源码见
> GitHub：https://github.com/ml-inory/chatterbox-onestep

```bash
mkdir -p cpp/build && cd cpp/build
cmake .. -DCMAKE_TOOLCHAIN_FILE=${AX_RUNTIME_ROOT}/toolchain.cmake -DAX_RUNTIME_ROOT=${AX_RUNTIME_ROOT}
make -j$(nproc)
./openai_server --model ../../models/model.axmodel --assets ../assets --port 8000
./openai_client --url http://127.0.0.1:8000/v1/audio/speech --tokens-file <tokens_int32.bin> --out out.wav
```

不想自己编译也可以直接用 `cpp/bin/` 里的预编译产物（板端加 `LD_LIBRARY_PATH=/soc/lib`）：

```bash
./cpp/bin/openai_server --model models/model.axmodel --assets cpp/assets --port 8000
```

说明：
- 板端实测：Python wav 请求约 10s（numpy Griffin-Lim），C++ wav 请求约 23s（20 次迭代）；
- `cpp/assets/` 提供 `mel_inv_24k.bin`（线性谱逆矩阵）与 `default_embedding.bin`（内置音色）；
- DSP 核心见 `cpp/include/dsp.hpp`，本地自测见 `cpp/tests/gl_test.cpp`（FFT 对 numpy 误差约 1e-5）。

## 音色克隆（Voice Cloning，embedding 级）

本模型以 192 维 xvector 音色 embedding 为条件，支持"参考音频 → 换音色"的 embedding 级克隆
（完整官方克隆还含 10s 参考 prompt 条件，当前板端 AXMODEL 未导出该路径，需要可另行定制）。

### 完整克隆（model_clone.axmodel，prompt 条件）

`models/model_clone.axmodel` 为完整官方克隆路径：参考音频的 **prompt 条件**（prompt mel +
prompt token）与音色 embedding 一起输入，还原参考说话人更完整的音色细节。

- 输入（5 个）：`tokens_all[1,256]`（prompt_token 157 + 生成区 token ≤99，宿主侧拼接）、
  `token_len_all[1]`、`embedding[1,192]`、`z[1,80,512]`、`prompt_feat[1,314,80]`
- 输出：mel[1,80,198]（生成区，截取 token_len_all−157 帧 ×2）
- 单句上限 99 个 S3 token（约 4-5 秒）；NPU3 三核编译
- 提示：prompt_token 与生成 token 必须在宿主侧拼接（图内 Concat 会触发 AX650 NPU 的
  `AxConcat` wdma bug，见 `reports/` 与 model_convert 注释）
- 单步学生对噪声 z 敏感（个别 z 会静音）：server 的 clone 路径默认做 **4 次 z 平均**
  （`--clone-z-ensemble` 可调），消除偶发静音

**步骤 1（宿主，torch）**：从参考音频提取 embedding

```bash
python3 python/extract_voice_embedding.py --wav ref.wav \
    --ckpt-dir /path/to/chatterbox_models --out ref_embedding.npy
```

**步骤 2（板端）**：调用 OpenAI 服务时带上 `voice.embedding`

```bash
curl -X POST http://<board>:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d "$(python3 - <<'PY'
import json, numpy as np
emb = np.load('ref_embedding.npy').reshape(-1).tolist()
print(json.dumps({"input": [12,34,56], "voice": {"embedding": emb}, "response_format": "wav"}))
PY
)"
```

> `input` 为 S3 speech tokens（文本 → token 的 T3 在宿主完成）；也可在 `python/openai_client.py`
> 基础上扩展，直接读 `ref_embedding.npy` 填 `voice.embedding`。

### 克隆示例音频

用 LJ001-0001 参考人声克隆的板端合成结果（模型：model_clone.axmodel）：

**参考音频（被克隆的人声）**

<audio controls src="examples/audio/clone_reference.wav"></audio>

- [clone_reference.wav](examples/audio/clone_reference.wav)

**克隆合成结果（3.66s，24kHz）**

<audio controls src="examples/audio/clone_output.wav"></audio>

- [clone_output.wav](examples/audio/clone_output.wav)
