# Export Report

- Model: chatterbox-s3gen-onestep
- ONNX: model.onnx
- Opset: 17
- Torch-ONNX cosine: 1.000000
- Inputs: tokens[1, 256], token_len[1], embedding[1, 192], z[1, 80, 512]
- Outputs: mel[1, 80, 512]
- Calibration: real 业务数据（tokens:128, token_len:128, embedding:128, z:128 样本）

## Export attempts

- [ok] torch.onnx.export(dynamo=False, opset=17): 
- [ok] static shape 检查: 全部静态
- [ok] onnxruntime 推理: 
- [ok] torch-vs-ORT 对分 mel: cosine=1.000000
