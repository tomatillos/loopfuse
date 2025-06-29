# not actual tests!
# mostly for debugging / a reference of lots of attention variants

import logging

import torch

from loopfuse import loopfuse_backend

logging.basicConfig(level=logging.DEBUG)


def test_attention(B, H, S, D):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Q = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_unmasked_attention(B, H, S, D):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        scores = torch.softmax(QK, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Q = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)
    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_gqa_attention(B, Hq, S, D):
    def fn(q, k, v):
        Hq = q.shape[1]
        Hk = k.shape[1]
        k = k.repeat_interleave(Hq // Hk, dim=1)
        v = v.repeat_interleave(Hq // Hk, dim=1)
        QK = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ v

    dtype = torch.bfloat16
    Hk = Hq // 8
    Q = torch.randn(B, Hq, S, D, dtype=dtype)
    K = torch.randn(B, Hk, S, D, dtype=dtype)
    V = torch.randn(B, Hk, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)
    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_modified_attn(B, H, S, D):
    def fn(Q, q_bias, K, scores_bias, V, out_bias):
        QK = (Q + q_bias) @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        QK = QK * 3.5 + scores_bias
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V + out_bias

    dtype = torch.bfloat16
    Q = torch.randn(B, H, S, D, dtype=dtype)
    q_bias = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    scores_bias = torch.randn(B, H, S, S, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    out_bias = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)
    prg_str = compiled_fn(Q, q_bias, K, scores_bias, V, out_bias)
    print(prg_str.code)


def test_attention_explicit_softmax(B, H, S, D):
    import math

    def _softmax(X):
        Xf = X.to(torch.float32)
        Xf_scaled = Xf * 1 / math.log(2)  # for efficient exp2 (mul fuses with sm_scale)
        X_mik, X_mi = torch.ops.loopfuse.scan_max(Xf_scaled)
        expX = torch.exp2(Xf_scaled - X_mik) * torch.exp2(
            X_mik - X_mi
        )  # compiler hint for online softmax
        S = torch.sum(expX, dim=-1, keepdim=True)
        out = expX / (S + 1)
        return out

    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = _softmax(masked_attn)
        return scores.to(Q.dtype) @ V

    dtype = torch.bfloat16
    Q = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_MLA(B, H, S, D):
    def fn(Q_nope, Q_pe, K_nope, K_pe, V):
        qk_nope = Q_nope @ K_nope.transpose(-2, -1)
        qk_pe = Q_pe @ K_pe.transpose(-2, -1)
        qk = qk_nope + qk_pe
        qk *= (Q_nope.shape[-1] + Q_pe.shape[-1]) ** (-0.5)
        masked_qk = torch.ops.loopfuse.causal_mask(qk)
        scores = torch.softmax(masked_qk, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    D_nope = D
    D_pe = D // 8
    Q_nope = torch.randn(B, H, S, D_nope, dtype=dtype)
    Q_pe = torch.randn(B, H, S, D_pe, dtype=dtype)
    K_nope = torch.randn(B, H, S, D_nope, dtype=dtype)
    K_pe = torch.randn(B, H, S, D_pe, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q_nope, 3)
    torch._dynamo.mark_static(Q_pe, 3)
    torch._dynamo.mark_dynamic(Q_nope, 2)
    torch._dynamo.mark_static(V, 3)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q_nope, Q_pe, K_nope, K_pe, V)
    print(prg_str.code)


def test_decoding(B, H, S, D):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        scores = torch.softmax(QK, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Q = torch.randn(B, H, 1, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_gqa_decoding_alt(B, H, S, D):
    def fn(Q, K, V):
        group_size = H // Hk
        K = K.repeat_interleave(group_size, dim=1)
        V = V.repeat_interleave(group_size, dim=1)
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        scores = torch.softmax(QK, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Hk = H // 8
    Q = torch.randn(B, H, 1, D, dtype=dtype)
    K = torch.randn(B, Hk, S, D, dtype=dtype)
    V = torch.randn(B, Hk, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)


def test_gqa_decoding(B, H, S, D):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        scores = torch.softmax(QK, dim=-1)
        return (scores @ V).view(B, H, 1, D)

    dtype = torch.bfloat16
    group_size = 8
    Hk = H // group_size
    Q = torch.randn(B, H, 1, D, dtype=dtype)
    Q = Q.view(B, Hk, group_size, D)
    K = torch.randn(B, Hk, S, D, dtype=dtype)
    V = torch.randn(B, Hk, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)


def test_attention_softcap(B, H, S, D):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        QK = torch.tanh(QK / 25) * 25
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Q = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(
        fn, backend=loopfuse_backend, dynamic=True, mode="max-autotune"
    )

    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_attention_sliding_mask(B, H, S, D):
    S = 64

    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.sliding_mask(QK, 32)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Q = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)
    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_add():
    def fn(x, y):
        return x + y

    x = torch.randn(64, 128, dtype=torch.float32)
    y = torch.randn(64, 128, dtype=torch.float32)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)
    prg_str = compiled_fn(x, y)
    print(prg_str.code)


def test_attention_no_transpose(B, H, S, D):
    def fn(Q, K, V):
        QK = Q @ K * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Q = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, H, D, S, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


# todo: this is broken
def test_attention_reshape(B, H, S, D):
    def fn(Q, K, V):
        Q = Q.reshape(*V.shape)
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Q = torch.randn(B * H, S, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 2)
    torch._dynamo.mark_dynamic(Q, 1)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_simplified_attn(B, H, S, D):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1)
        m = torch.max(QK, dim=-1, keepdim=True)[0]  # + torch.sum(QK)
        QK = QK / m
        out = QK @ V
        return out

    dtype = torch.bfloat16
    Q = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_fuse_matmul_matmul(M, K, N):
    def fn(a, b, c):
        return a @ b @ c

    dtype = torch.bfloat16
    a = torch.randn(M, K, dtype=dtype)
    b = torch.randn(K, N, dtype=dtype)
    c = torch.randn(N, M, dtype=dtype)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(a, b, c)
    print(prg_str.code)


def test_prefix_attention(B, H, Sq, Sp, D):
    def fn(Q, Kp, Vp, Ks, Vs):
        QKp = Q @ Kp.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        QKs = Q @ Ks.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn_prefix = QKp
        masked_attn_suffix = torch.ops.loopfuse.causal_mask(QKs)

        # todo: need proper softmax combine
        scores_prefix = torch.softmax(masked_attn_prefix, dim=-1)
        scores_suffix = torch.softmax(masked_attn_suffix, dim=-1)
        return scores_prefix @ Vp + scores_suffix @ Vs

    dtype = torch.bfloat16
    Q = torch.randn(B, H, Sq, D, dtype=dtype)
    Kp = torch.randn(1, H, Sp, D, dtype=dtype)
    Vp = torch.randn(1, H, Sp, D, dtype=dtype)
    Ks = torch.randn(B, H, Sq, D, dtype=dtype)
    Vs = torch.randn(B, H, Sq, D, dtype=dtype)

    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)
    prg = compiled_fn(Q, Kp, Vp, Ks, Vs)
    print(prg.code)


def test_attention_sliding_mask_decoding(B, H, S, D):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.sliding_mask(QK, 256)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Q = torch.randn(B, H, 1, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    # torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_gqa_softcap_sliding_mask(B, H, S, D):
    SOFTCAP = 20
    WINDOW_SIZE = 512

    def loopfuse_fn(Q, K, V):
        Hq = Q.shape[1]
        Hk = K.shape[1]
        K = K.repeat_interleave(Hq // Hk, dim=1)
        V = V.repeat_interleave(Hq // Hk, dim=1)
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        QK = torch.tanh(QK / SOFTCAP) * SOFTCAP
        masked_attn = torch.ops.loopfuse.sliding_mask(QK, WINDOW_SIZE)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    dtype = torch.bfloat16
    Hk = H // 8
    Q = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, Hk, S, D, dtype=dtype)
    V = torch.randn(B, Hk, S, D, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    compiled_fn = torch.compile(loopfuse_fn, backend=loopfuse_backend, dynamic=True)

    prg_str = compiled_fn(Q, K, V)
    print(prg_str.code)


def test_paged_gqa_decoding(B, Hq, S, D, dtype):
    group_size = 8
    Hkv = Hq // group_size

    N_blocks = B * S * 10

    K_cache = torch.randn(N_blocks, Hkv, D, dtype=dtype)
    V_cache = torch.randn(N_blocks, Hkv, D, dtype=dtype)

    blocks_per_seq = S
    K_block_table = torch.randint(0, N_blocks, (B, blocks_per_seq), dtype=torch.int32)
    V_block_table = torch.randint(0, N_blocks, (B, blocks_per_seq), dtype=torch.int32)

    Q = torch.randn(B, Hkv, group_size, D, dtype=dtype)

    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)

    @torch.compile(backend=loopfuse_backend, dynamic=True)
    def loopfuse_fn(Q, K_cache, K_block_table, V_cache, V_block_table):
        K = torch.ops.loopfuse.paged_load(K_cache, K_block_table).transpose(-3, -2)
        V = torch.ops.loopfuse.paged_load(V_cache, V_block_table).transpose(-3, -2)
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        scores = torch.softmax(QK, dim=-1)
        return scores @ V

    prg_str = loopfuse_fn(Q, K_cache, K_block_table, V_cache, V_block_table)


if __name__ == "__main__":
    B, H, S, D = 8, 32, 1024, 128
    Sq = 512
    Sp = 2048
    # test_attention(B, H, S, D)
    # test_unmasked_attention(B, H, S, D)
    # test_gqa_attention(B, H, S, D)
    # test_modified_attn(B, H, S, D)
    # test_attention_explicit_softmax(B, H, S, D)
    # test_MLA(B, H, S, D)
    # test_decoding(B, H, S, D)
    test_gqa_decoding(B, H, S, D)
    # test_attention_softcap(B, H, S, D)
    # test_attention_sliding_mask(B, H, S, D)
    # test_add()
    # test_attention_no_transpose(B, H, S, D)
    # test_attention_reshape(B, H, S, D)
    # test_prefix_attention(B, H, Sq, Sp, D)
    # test_attention_sliding_mask_decoding(B, H, S, D)
    # test_simplified_attn(B, H, S, D)
    # test_fuse_matmul_matmul(1024, 1024, 1024)
    # test_gqa_softcap_sliding_mask(B, H, S, D)
    # test_paged_gqa_decoding(B, H, S, D, torch.bfloat16)
