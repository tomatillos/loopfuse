import torch
import pytest

from loopfuse import loopfuse_backend

DEVICE = "cuda"
if torch.cuda.get_device_capability(0)[0] <= 7.5:
    DTYPES = [torch.float32]
else:
    DTYPES = [torch.bfloat16]


def check_accuracy(fn, inputs, rtol=1e-4, atol=1e-4):
    ref_output = fn(*inputs)
    torch._dynamo.reset()
    fn_output = torch.compile(fn, backend=loopfuse_backend, dynamic=True, mode=None)(
        *inputs
    )
    try:
        torch.testing.assert_close(ref_output, fn_output, rtol=rtol, atol=atol)
    except Exception as e:
        print(e)
        print(ref_output)
        print("-" * 100)
        print(fn_output)
        raise e


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("M", [1024])
@pytest.mark.parametrize("N", [128])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fuse_add_3dim(B, M, N, dtype):
    def fn(a, b, c):
        return a + b + c

    _inputs = [torch.randn(B, M, N, device=DEVICE, dtype=dtype) for _ in range(3)]
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("M", [1024])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fuse_add_2dim(B, M, dtype):
    def fn(a, b, c):
        return a + b + c

    _inputs = [torch.randn(B, M, device=DEVICE, dtype=dtype) for _ in range(3)]
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("M", [1024])
@pytest.mark.parametrize("N", [128])
@pytest.mark.parametrize("K", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fuse_matmul_add(M, N, K, dtype):
    def fn(a, b, c):
        return a @ b + c

    _a = torch.randn(M, K, device=DEVICE, dtype=dtype)
    torch._dynamo.mark_static(_a, 1)
    _b = torch.randn(K, N, device=DEVICE, dtype=dtype)
    _c = torch.randn(M, N, device=DEVICE, dtype=dtype)
    _inputs = (_a, _b, _c)
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("M", [1024])
@pytest.mark.parametrize("N", [128])
@pytest.mark.parametrize("K", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fuse_matmul_add_3dim(B, M, N, K, dtype):
    def fn(a, b, c):
        return a @ b + c

    _a = torch.randn(B, M, K, device=DEVICE, dtype=dtype)
    torch._dynamo.mark_static(_a, 2)
    _b = torch.randn(B, K, N, device=DEVICE, dtype=dtype)
    _c = torch.randn(B, M, N, device=DEVICE, dtype=dtype)
    _inputs = (_a, _b, _c)
    check_accuracy(fn, _inputs)


# @pytest.mark.parametrize("M", [1024])
# @pytest.mark.parametrize("K", [64])
# @pytest.mark.parametrize("N", [128])
# @pytest.mark.parametrize("D", [64])
# @pytest.mark.parametrize("dtype", DTYPES)
# def test_fuse_double_matmul(M, K, N, D, dtype):
#     def fn(a, b, c):
#         return a @ b @ c

#     _a = torch.randn(M, K, device=DEVICE, dtype=dtype)
#     _b = torch.randn(K, N, device=DEVICE, dtype=dtype)
#     _c = torch.randn(N, D, device=DEVICE, dtype=dtype)
#     _inputs = (_a, _b, _c)
#     check_accuracy(
#         fn, _inputs, rtol=1e-3, atol=1e-3
#     )  # fma will make fused have higher accuracy


# @pytest.mark.skip("ooms")
# @pytest.mark.parametrize("M", [256])
# @pytest.mark.parametrize("K", [128])
# @pytest.mark.parametrize("N", [128])
# @pytest.mark.parametrize("D", [128])
# @pytest.mark.parametrize("dtype", DTYPES)
# def test_fuse_generalized_matmul(M, K, N, D, dtype):
#     def fn(a, b, c, d, e):
#         return (a @ b + c) @ d @ e

#     _a = torch.randn(M, K, device=DEVICE, dtype=dtype)
#     _b = torch.randn(K, N, device=DEVICE, dtype=dtype)
#     _c = torch.randn(M, N, device=DEVICE, dtype=dtype)
#     _d = torch.randn(N, D, device=DEVICE, dtype=dtype)
#     _e = torch.randn(D, K, device=DEVICE, dtype=dtype)
#     # ?
#     torch._dynamo.mark_static(_a, 1)
#     torch._dynamo.mark_static(_d, 1)
#     _inputs = (_a, _b, _c, _d, _e)
#     check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [8])
@pytest.mark.parametrize("S", [256, 593])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fuse_standard_attn(B, H, S, D, dtype):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    Q = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    K = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    V = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    _inputs = (Q, K, V)
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("Hq", [8])
@pytest.mark.parametrize("Hk", [4])
@pytest.mark.parametrize("S", [1024])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fuse_gqa_attn(B, Hq, Hk, S, D, dtype):
    def fn(Q, K, V):
        Hq = Q.shape[1]
        Hk = K.shape[1]
        K = K.repeat_interleave(Hq // Hk, dim=1)
        V = V.repeat_interleave(Hq // Hk, dim=1)
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    Q = torch.randn(B, Hq, S, D, device=DEVICE, dtype=dtype)
    K = torch.randn(B, Hk, S, D, device=DEVICE, dtype=dtype)
    V = torch.randn(B, Hk, S, D, device=DEVICE, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    _inputs = (Q, K, V)
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [8])
@pytest.mark.parametrize("S", [1024])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fuse_modified_attn(B, H, S, D, dtype):
    def fn(Q, K, V, q_bias, scores_bias, out_bias):
        QK = (Q + q_bias) @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        QK = QK * 3.5 + scores_bias
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V + out_bias

    Q = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    K = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    V = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    q_bias = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    scores_bias = torch.randn(B, H, S, S, device=DEVICE, dtype=dtype)
    out_bias = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    _inputs = (Q, K, V, q_bias, scores_bias, out_bias)
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [8])
@pytest.mark.parametrize("S", [1024])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fuse_softmax_matmul(B, H, S, D, dtype):
    def fn(x, y):
        return torch.softmax(x, dim=-1) @ y

    x = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    y = torch.randn(B, H, D, S, device=DEVICE, dtype=dtype)
    torch._dynamo.mark_static(x, 3)
    torch._dynamo.mark_dynamic(x, 2)
    _inputs = (x, y)
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [8])
@pytest.mark.parametrize("S", [1024])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_attention_sliding_mask(B, H, S, D, dtype):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.sliding_mask(QK, 256)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    Q = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    K = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    V = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    _inputs = (Q, K, V)
    check_accuracy(fn, _inputs)


def test_paged_add():
    """Tests paged_load(X_cache, X_block_table) + 1"""

    def fn(buffer, block_table):
        loaded = torch.ops.loopfuse.paged_load(buffer, block_table)
        return loaded + 1

    N = 1000
    S = 8
    D = 32
    dtype = torch.float32
    buffer = torch.randn(N, D, device=DEVICE, dtype=dtype)
    block_table = torch.randint(0, N, (S,), device=DEVICE, dtype=torch.int64)

    _inputs = (buffer, block_table)
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [8])
@pytest.mark.parametrize("S", [1024])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_paged_attention(B, H, S, D, dtype):
    """Tests paged attention using paged_load for K cache"""

    def fn(Q, K_cache, K_block_table, V):
        K = torch.ops.loopfuse.paged_load(K_cache, K_block_table).transpose(1, 2)
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    N_blocks = B * S * 2
    K_cache = torch.randn(N_blocks, H, D, device=DEVICE, dtype=dtype)
    K_block_table = torch.randint(0, N_blocks, (B, S), device=DEVICE, dtype=torch.int32)
    Q = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    V = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)

    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)

    _inputs = (Q, K_cache, K_block_table, V)
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("Hq", [8])
@pytest.mark.parametrize("Hk", [4])
@pytest.mark.parametrize("S", [256])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_paged_attention_gqa(B, Hq, Hk, S, D, dtype):
    """Tests paged GQA using paged_load for K cache"""

    def fn(Q, K_cache, K_block_table, V):
        K = torch.ops.loopfuse.paged_load(K_cache, K_block_table).transpose(1, 2)
        Hq_ = Q.shape[1]
        Hk_ = K.shape[1]
        K = K.repeat_interleave(Hq_ // Hk_, dim=1)
        V = V.repeat_interleave(Hq_ // Hk_, dim=1)
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(QK)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ V

    N_blocks = B * S * 2
    K_cache = torch.randn(N_blocks, Hk, D, device=DEVICE, dtype=dtype)
    K_block_table = torch.randint(0, N_blocks, (B, S), device=DEVICE, dtype=torch.int32)
    Q = torch.randn(B, Hq, S, D, device=DEVICE, dtype=dtype)
    V = torch.randn(B, Hk, S, D, device=DEVICE, dtype=dtype)

    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)

    _inputs = (Q, K_cache, K_block_table, V)
    check_accuracy(fn, _inputs)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [8])
@pytest.mark.parametrize("S", [256])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_simplified_attn(B, H, S, D, dtype):
    def fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1)
        m = torch.max(QK, dim=-1, keepdim=True)[0] + torch.sum(QK, dim=-1, keepdim=True)
        QK = QK / (m + 1e-3)  # for stability
        out = QK @ V
        return out

    Q = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    K = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    V = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)
    _inputs = (Q, K, V)
    check_accuracy(fn, _inputs, rtol=1e-2)
