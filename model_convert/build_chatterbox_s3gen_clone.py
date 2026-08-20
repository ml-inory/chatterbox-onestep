"""Magnetar load script: distilled one-step Chatterbox S3Gen WITH prompt cloning path.

Convention: ``build()`` returns ``(model, example_inputs)``.
Model: (tokens_all, token_len_all, embedding, z, prompt_feat) -> mel (static).
prompt_token 与生成 token 在宿主侧拼接为 tokens_all[1,256]（157 prompt + 99 生成，2 的幂），
prompt_feat[1,314,80] 单独输入做条件；图内不包含 Concat（AxConcat 在 AX650 NPU 有 wdma bug）。
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

T_MAX = 99     # 生成区 speech tokens（157 prompt + 99 生成 = 256 token，2 的幂）
PT_MAX = 157   # prompt tokens（官方内置音色，宿主侧拼入 tokens_all）
TOK_ALL = T_MAX + PT_MAX          # 256（2 的幂，匹配基础模型已验证形状）
MEL_PROMPT = 314                  # prompt mel 帧 = 2 * PT_MAX
L_MAX = MEL_PROMPT + 2 * T_MAX    # 512


class OneStepCloneExport(nn.Module):
    """One-step S3Gen with prompt conditioning:
    (tokens_all, token_len_all, embedding, z, prompt_feat) -> mel 生成区。"""

    def __init__(self, student):
        super().__init__()
        self.flow = student.flow

    def forward(self, tokens_all, token_len_all, embedding, z, prompt_feat):
        from chatterbox.models.s3gen.utils.mask import make_pad_mask

        B = tokens_all.size(0)
        tok_all = torch.clamp(tokens_all, 0, 6560)             # (1, 256)
        len_all = token_len_all.to(torch.int64)
        emb = nn.functional.normalize(embedding, dim=1)
        emb = self.flow.spk_embed_affine_layer(emb)
        mask = (~make_pad_mask(len_all, TOK_ALL)).unsqueeze(-1).to(embedding)
        token_emb = self.flow.input_embedding(tok_all.long()) * mask
        h, h_masks = self.flow.encoder(token_emb, len_all)
        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
        h = self.flow.encoder_proj(h)
        mask2 = (~make_pad_mask(h_lengths, L_MAX)).unsqueeze(1).to(h)
        mu = h.transpose(1, 2).contiguous()
        conds = torch.zeros(B, L_MAX, 80, device=z.device, dtype=z.dtype)
        conds[:, :MEL_PROMPT, :] = prompt_feat  # prompt_feat: (1, 314, 80)
        conds = conds.transpose(1, 2)           # (1, 80, 826)
        t = torch.zeros(B, device=z.device, dtype=z.dtype)
        _, x0 = self.flow.decoder.estimator(z, mask2, mu, t, emb, conds)
        return x0[:, :, MEL_PROMPT:].reshape(1, 80, 2 * T_MAX)  # 固定静态 shape (1,80,198)


def build(device: str = "cpu"):
    from dsflow.chatterbox.model import OneStepS3Gen, load_teacher
    import chatterbox.models.s3gen.utils.mask as mask_utils

    mask_utils.add_optional_chunk_mask = lambda xs, masks, *a, **k: masks

    def _patch_rel_attn(attn, T: int):
        h, d = attn.h, attn.d_k

        def forward_qkv(query, key, value):
            q = attn.linear_q(query).view(1, T, h, d)
            k = attn.linear_k(key).view(1, T, h, d)
            v = attn.linear_v(value).view(1, T, h, d)
            return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        def forward_attention(v, scores, mask):
            if mask is not None and mask.size(2) > 0:
                m = mask.unsqueeze(1).eq(0)[:, :, :, :T]
                # NPU 软max 内核不支持 -inf；用大负数 -1e9 遮罩
                scores = scores.masked_fill(m, -1e9)
                attn_weights = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
            else:
                attn_weights = torch.softmax(scores, dim=-1)
            x = torch.matmul(attn_weights, v)
            x = x.transpose(1, 2).contiguous().view(1, T, h * d)
            return attn.linear_out(x)

        def rel_shift(x):
            zero_pad = torch.zeros(1, h, T, 1, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([zero_pad, x], dim=-1).reshape(1, h, 2 * T, T)
            return x_padded[:, :, 1:].reshape(1, h, T, 2 * T - 1)[:, :, :, :T]

        def forward(query, key, value, mask, pos_emb, cache):
            q, k, v = forward_qkv(query, key, value)
            q = q.transpose(1, 2)
            new_cache = torch.cat((k, v), dim=-1)
            p = attn.linear_pos(pos_emb).view(1, -1, h, d).transpose(1, 2)
            q_u = (q + attn.pos_bias_u.to(q.device)).transpose(1, 2)
            q_v = (q + attn.pos_bias_v.to(q.device)).transpose(1, 2)
            matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
            matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
            if matrix_ac.shape != matrix_bd.shape:
                matrix_bd = rel_shift(matrix_bd)
            scores = (matrix_ac + matrix_bd) / math.sqrt(d)
            return forward_attention(v, scores, mask), new_cache

        attn.forward_qkv = forward_qkv
        attn.forward_attention = forward_attention
        attn.rel_shift = rel_shift
        attn.forward = forward

    def _make_static_attn_processor(T: int, B: int = 1):
        class StaticAttnProcessor:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
                if attention_mask is not None:
                    attention_mask = attention_mask.view(B, 1, 1, T).expand(B, attn.heads, 1, T)
                query = attn.to_q(hidden_states)
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                head_dim = attn.to_q.out_features // attn.heads
                query = query.view(B, T, attn.heads, head_dim).transpose(1, 2)
                key = key.view(B, T, attn.heads, head_dim).transpose(1, 2)
                value = value.view(B, T, attn.heads, head_dim).transpose(1, 2)
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(B, T, attn.heads * head_dim)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states
        return StaticAttnProcessor()

    origin = Path(os.environ.get("MAGNETAR_ORIGIN", Path(__file__).parent))
    teacher = load_teacher(origin, device)
    student = OneStepS3Gen(teacher).to(device)
    state = torch.load(origin / "student_state.pt", map_location=device, weights_only=True)
    student.load_state_dict(state)
    student.eval()
    # 总序列 413 token（encoders）/ 826 mel 帧（up_encoders、decoder）
    for layer in student.flow.encoder.encoders:
        _patch_rel_attn(layer.self_attn, TOK_ALL)
    for layer in student.flow.encoder.up_encoders:
        _patch_rel_attn(layer.self_attn, L_MAX)
    # decoder 的 diffusers Attention 替换为字面 shape processor（T=826），
    # 与原版逐位等价（numpy 验证 cosine=1.0），并让 ONNX 为静态 Reshape（Pulsar2 需要）。
    proc = _make_static_attn_processor(L_MAX)
    for name, mod in student.flow.decoder.estimator.named_modules():
        if type(mod).__module__.startswith("diffusers") and type(mod).__name__ == "Attention":
            mod.set_processor(proc)
    model = OneStepCloneExport(student).to(device).eval()

    tokens_all = torch.randint(4, 6560, (1, TOK_ALL), dtype=torch.int32, device=device)
    token_len_all = torch.tensor([TOK_ALL], dtype=torch.int32, device=device)
    embedding = torch.randn(1, 192, device=device)
    z = torch.randn(1, 80, L_MAX, device=device)
    prompt_feat = torch.randn(1, MEL_PROMPT, 80, device=device)
    example_inputs = (tokens_all, token_len_all, embedding, z, prompt_feat)
    return model, example_inputs
