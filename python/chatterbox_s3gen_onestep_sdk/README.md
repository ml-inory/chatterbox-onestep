        # chatterbox-s3gen-onestep Python SDK

        - 输入（与 model_meta.json 一致）: tokens[1, 256], token_len[1], embedding[1, 192], z[1, 80, 512]
        - 输出（与 model_meta.json 一致）: mel[1, 80, 512]
        - 预处理: S3 token 序列（int32，pad 到 256，裁剪到 [0,6560]）、token_len(int32)、192 维 x-vector 音色嵌入(float32)、随机噪声 z[1,80,512]（缺省自动生成）
        - 后处理: 输出 mel[1,80,512]，按 token_len*2 截取有效帧，其余丢弃
        - 示例输入: export/sample_input.npy

> NPU 专用发布版：端到端 NPU 验证已通过，SDK 仅依赖 pyaxengine（无 onnxruntime/torch/transformers 回退）。


        ```bash
        LD_LIBRARY_PATH=/soc/lib PYTHONPATH=$PWD/python python3 chatterbox_s3gen_onestep_sdk/example.py           --model models/model.axmodel --input input.npy --output-dir output
        ```
