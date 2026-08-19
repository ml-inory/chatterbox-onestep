import numpy as np


def postprocess(mel, token_len):
    """mel[1,80,512] -> 按 token_len*2 截取有效帧。"""
    mel = np.asarray(mel, dtype=np.float32)
    n = int(np.asarray(token_len).reshape(-1)[0]) * 2
    return mel[..., :n]