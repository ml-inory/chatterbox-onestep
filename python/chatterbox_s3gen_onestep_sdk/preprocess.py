import numpy as np

T_MAX = 256
L_MAX = 512


def preprocess(tokens, token_len, embedding, z=None):
    """S3 tokens(int32, 裁剪到 [0,6560]) + 长度 + 音色嵌入(float32) + 噪声(可选) -> 模型输入。"""
    tokens = np.clip(np.asarray(tokens, dtype=np.int32).reshape(1, T_MAX), 0, 6560)
    token_len = np.asarray(token_len, dtype=np.int32).reshape(1)
    embedding = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, -1)
    if z is None:
        z = np.random.randn(1, 80, L_MAX).astype(np.float32)
    z = np.ascontiguousarray(z, dtype=np.float32).reshape(1, 80, L_MAX)
    return [np.ascontiguousarray(tokens), token_len, embedding, z]