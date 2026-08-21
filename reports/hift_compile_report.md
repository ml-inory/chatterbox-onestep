# HiFT 声码器编译报告

- image: docker-registry.aitsw.axera-tech.com/pulsar2:7.0
- target: AX650 / NPU3
- 方案：HiFT 神经声码器拆成两个静态模型，源激励/STFT/ISTFT 宿主侧（numpy/C++，无 torch）

## hifift_f0.axmodel（mel -> f0）
- input: mel[1,80,198]（FP32）；output: f0[1,198]
- U8 / MinMax / 31 组真实 mel 校准；MACs 647.3M
- 模型 12.48 MB（axmodel 3.49 MB）
- 板端 cosine vs ONNX：0.9972（9 样本均值）

## hifift_decode.axmodel（mel + s_stft -> raw_mag/raw_phase）
- inputs: mel[1,80,198], s_stft[1,18,23761]；outputs: raw_mag[1,9,23761], raw_phase[1,9,23761]
- U16（Conv/ConvTranspose/Gemm/MatMul = U16/S8/U16）+ smooth quant；MACs 59.97G
- 模型 66.89 MB（axmodel 27.15 MB）
- 板端 raw cosine vs ONNX：0.987–0.996；端到端 wav corr ~0.92
- >4kHz 能量 0.28（≈ torch 0.27），频谱质心 ~2500（旧 GL 1294）

## 关键改造
1. Snake 激活改写为 cos 恒等式（sin²=(1−cos2u)/2），避免 AxQuantizedSnake 无 builder；
2. ReflectionPad1d 改 constant pad（仅边界 1 帧）；
3. 输出拆 mag/phase 双张量，独立量化刻度；
4. C++ 源激励相位累加用 float32，与 torch 原版逐位对齐（float64 会漂移）。

## 验证
- ONNX vs torch：f0 cosine 1.0、decode raw cosine 1.0（maxdiff 5e-4）
- 宿主 DSP vs torch：stft 5.6e-9、istft 1.3e-7、源激励 3.4e-8
- 板端 Python/C++ server 端到端：clone melcorr ~0.74-0.78（旧 GL 0.51）
