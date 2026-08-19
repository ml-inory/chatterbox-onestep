# Run On Board Report
- board: 10.126.35.203
- chip_type: AX650C_CHIP
- python_shape: [1, 80, 512]
- cosine_vs_onnx: 0.9924230363581176
- mae_vs_onnx: 0.5717893838882446
- latency_ms_mean: 181.08
- latency_ms_min: 180.95
- peak_rss_mb: 167.6

- 方法: Python SDK（NPU-only, axengine 2.12.0s）板端端到端，5 次推理取均值
