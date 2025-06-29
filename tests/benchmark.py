from functools import partial
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import triton

from loopfuse.torch_backend.interface import loopfuse_backend

# for custom tanh
from torch._inductor.lowering import make_pointwise, register_lowering
from torch._inductor.virtualized import ops


device = "cuda"

mode = "max-autotune"
# mode = ""


def helper_benchmark_fn(fn, inputs):
    torch._dynamo.reset()
    fn_time = triton.testing.do_bench(lambda: fn(*inputs))
    return fn_time


def compare_outputs(*outputs):
    if len(outputs) < 2:
        print("Need at least 2 outputs to compare")
        return

    reference = outputs[0]
    all_match = True

    for i, output in enumerate(outputs[1:], 1):
        try:
            torch.testing.assert_close(reference, output, atol=1e-2, rtol=1e-3)
            print(f"Output {i} matches reference")
        except Exception as e:
            print(f"Output {i} does NOT match reference: {e}")
            all_match = False

    if all_match:
        print("All outputs match!")
    else:
        print("Some outputs do not match!")


# fair test for flexattention


@torch.library.custom_op("approx::tanh", mutates_args=())
def _tanh_approx(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


@_tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


def _tanh_approx_lowering(inp):
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)


register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)


def benchmark_causal_attn(shape, dtype):
    B, H, S, D = shape
    inputs = (
        torch.randn(B, H, S, D, device=device, dtype=dtype),  # q
        torch.randn(B, H, S, D, device=device, dtype=dtype),  # k
        torch.randn(B, H, S, D, device=device, dtype=dtype),  # v
    )
    torch._dynamo.mark_static(inputs[0], 3)
    torch._dynamo.mark_dynamic(inputs[0], 2)

    @torch.compile(backend=loopfuse_backend, dynamic=True, mode=mode)
    def loopfuse_fn(q, k, v):
        qk = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)
        masked_attn = torch.ops.loopfuse.causal_mask(qk)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ v

    sdpa_fn = lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )

    # flexattention
    def flex_causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(flex_causal, B=None, H=None, Q_LEN=S, KV_LEN=S)

    @torch.compile(backend="inductor", dynamic=True)
    def flex_fn(q, k, v):
        return flex_attention(q, k, v, block_mask=block_mask)

    print("Comparing outputs for causal attention...")
    loopfuse_output = loopfuse_fn(*inputs)
    sdpa_output = sdpa_fn(*inputs)
    flex_output = flex_fn(*inputs)

    compare_outputs(sdpa_output, loopfuse_output, flex_output)

    loopfuse_time = helper_benchmark_fn(loopfuse_fn, inputs)
    sdpa_time = helper_benchmark_fn(sdpa_fn, inputs)
    flex_time = helper_benchmark_fn(flex_fn, inputs)

    tflop_count = 2 * B * S**2 * D * H / 1e12

    loopfuse_tflops = tflop_count / (loopfuse_time * 1e-3)
    sdpa_tflops = tflop_count / (sdpa_time * 1e-3)
    flex_tflops = tflop_count / (flex_time * 1e-3)

    print(f"Loopfuse: {loopfuse_tflops:.2f} TFLOPS/s")
    print(f"SDPA: {sdpa_tflops:.2f} TFLOPS/s")
    print(f"Flex: {flex_tflops:.2f} TFLOPS/s")

    return {
        "name": "Causal",
        "loopfuse (TFLOPs/s)": loopfuse_tflops,
        "torch.compile (TFLOPs/s)": sdpa_tflops,
        "flex attention (TFLOPs/s)": flex_tflops,
    }


def benchmark_softcapped_causal_attn(shape, dtype):
    B, H, S, D = shape
    inputs = (
        torch.randn(B, H, S, D, device=device, dtype=dtype),  # q
        torch.randn(B, H, S, D, device=device, dtype=dtype),  # k
        torch.randn(B, H, S, D, device=device, dtype=dtype),  # v
    )
    torch._dynamo.mark_static(inputs[0], 3)
    torch._dynamo.mark_dynamic(inputs[0], 2)

    SOFTCAP = 20

    ### loopfuse
    @torch.compile(backend=loopfuse_backend, dynamic=True, mode=mode)
    def loopfuse_fn(q, k, v):
        qk = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)
        softcapped_qk = torch.tanh(qk / SOFTCAP) * SOFTCAP
        masked_attn = torch.ops.loopfuse.causal_mask(softcapped_qk)
        scores = torch.softmax(masked_attn, dim=-1)
        return scores @ v

    ### torch
    causal_mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))

    @torch.compile(backend="inductor", dynamic=True)
    def torch_fn(q, k, v, causal_mask):
        qk = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)
        softcapped_qk = SOFTCAP * torch.ops.approx.tanh(qk / SOFTCAP)
        softcapped_qk = softcapped_qk.masked_fill(~causal_mask, float("-inf"))
        scores = torch.softmax(softcapped_qk, dim=-1)
        return scores @ v

    ### flex attention
    def flex_causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def score_mod(score, b, h, q_idx, kv_idx):
        return SOFTCAP * torch.ops.approx.tanh(score / SOFTCAP)

    block_mask = create_block_mask(flex_causal, B=None, H=None, Q_LEN=S, KV_LEN=S)

    @torch.compile(backend="inductor", dynamic=True)
    def flex_fn(q, k, v):
        return flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)

    print("Comparing outputs for softcapped causal attention...")
    loopfuse_output = loopfuse_fn(*inputs)
    torch_output = torch_fn(*inputs, causal_mask)
    flex_output = flex_fn(*inputs)

    compare_outputs(torch_output, loopfuse_output, flex_output)

    ### benchmark

    loopfuse_time = helper_benchmark_fn(loopfuse_fn, inputs)
    torch_time = helper_benchmark_fn(torch_fn, (*inputs, causal_mask))
    flex_time = helper_benchmark_fn(flex_fn, inputs)

    tflop_count = 2 * B * S**2 * D * H / 1e12

    loopfuse_tflops = tflop_count / (loopfuse_time * 1e-3)
    torch_tflops = tflop_count / (torch_time * 1e-3)
    flex_tflops = tflop_count / (flex_time * 1e-3)

    print(f"Softcapped Loopfuse: {loopfuse_tflops:.2f} TFLOPS/s")
    print(f"Softcapped Torch: {torch_tflops:.2f} TFLOPS/s")
    print(f"Softcapped Flex: {flex_tflops:.2f} TFLOPS/s")

    return {
        "name": "Causal + Softcap",
        "loopfuse (TFLOPs/s)": loopfuse_tflops,
        "torch.compile (TFLOPs/s)": torch_tflops,
        "flex attention (TFLOPs/s)": flex_tflops,
    }


# gqa + sliding mask + softcap
def benchmark_gqa_sliding_mask_softcap_attn(shape, dtype):
    B, Hq, S, D = shape
    Hk = Hq // 8
    inputs = (
        torch.randn(B, Hq, S, D, device=device, dtype=dtype),  # Q
        torch.randn(B, Hk, S, D, device=device, dtype=dtype),  # K
        torch.randn(B, Hk, S, D, device=device, dtype=dtype),  # V
    )
    torch._dynamo.mark_static(inputs[0], 3)
    torch._dynamo.mark_dynamic(inputs[0], 2)

    SOFTCAP = 20
    WINDOW_SIZE = 1024

    ### loopfuse
    @torch.compile(backend=loopfuse_backend, dynamic=True, mode=mode)
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

    ### torch
    i = torch.arange(S, device=device).unsqueeze(1)
    j = torch.arange(S, device=device).unsqueeze(0)

    sliding_mask = (i >= j) & ((i - j) < WINDOW_SIZE)

    @torch.compile(backend="inductor", dynamic=True)
    def torch_fn(Q, K, V, sliding_mask):
        Hq = Q.shape[1]
        Hk = K.shape[1]
        K = K.repeat_interleave(Hq // Hk, dim=1)
        V = V.repeat_interleave(Hq // Hk, dim=1)
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        QK = torch.ops.approx.tanh(QK / SOFTCAP) * SOFTCAP
        QK = QK.masked_fill(~sliding_mask, float("-inf"))
        scores = torch.softmax(QK, dim=-1)
        return scores @ V

    # flex
    def sliding_window_causal(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        window_mask = q_idx - kv_idx < WINDOW_SIZE
        return causal_mask & window_mask

    def score_mod(score, b, h, q_idx, kv_idx):
        return SOFTCAP * torch.ops.approx.tanh(score / SOFTCAP)

    block_mask = create_block_mask(
        sliding_window_causal, B=None, H=None, Q_LEN=S, KV_LEN=S
    )

    @torch.compile(backend="inductor", dynamic=True)
    def flex_fn(Q, K, V):
        return flex_attention(
            Q, K, V, block_mask=block_mask, score_mod=score_mod, enable_gqa=True
        )

    print("Comparing outputs for GQA sliding mask softcap attention...")
    loopfuse_output = loopfuse_fn(*inputs)
    torch_output = torch_fn(*inputs, sliding_mask)
    flex_output = flex_fn(*inputs)

    compare_outputs(torch_output, loopfuse_output, flex_output)

    loopfuse_time = helper_benchmark_fn(loopfuse_fn, inputs)
    torch_time = helper_benchmark_fn(torch_fn, (*inputs, sliding_mask))
    flex_time = helper_benchmark_fn(flex_fn, inputs)

    # use window_size for flop count
    # s*w block, minus the 0.5*w^2 triangle at start of seq
    w = WINDOW_SIZE
    attn_unmasked = 2 * S * w - w**2
    tflop_count = 2 * B * Hq * attn_unmasked * D / 1e12

    loopfuse_tflops = tflop_count / (loopfuse_time * 1e-3)
    torch_tflops = tflop_count / (torch_time * 1e-3)
    flex_tflops = tflop_count / (flex_time * 1e-3)

    print(f"GQA Sliding Mask Softcap Loopfuse: {loopfuse_tflops:.2f} TFLOPS/s")
    print(f"GQA Sliding Mask Softcap SDPA: {torch_tflops:.2f} TFLOPS/s")
    print(f"GQA Sliding Mask Softcap Flex: {flex_tflops:.2f} TFLOPS/s")

    return {
        "name": "Sliding window + GQA + Softcap",
        "loopfuse (TFLOPs/s)": loopfuse_tflops,
        "torch.compile (TFLOPs/s)": torch_tflops,
        "flex attention (TFLOPs/s)": flex_tflops,
    }


# decoding


def benchmark_paged_decoding(shape, dtype):
    B, H, S, D = shape

    N_blocks = B * S * 2

    K_cache = torch.randn(N_blocks, H, D, device=device, dtype=dtype)
    V_cache = torch.randn(N_blocks, H, D, device=device, dtype=dtype)

    blocks_per_seq = S
    K_block_table = torch.randint(
        0, N_blocks, (B, blocks_per_seq), device=device, dtype=torch.int32
    )
    V_block_table = torch.randint(
        0, N_blocks, (B, blocks_per_seq), device=device, dtype=torch.int32
    )

    Q = torch.randn(B, H, 1, D, device=device, dtype=dtype)

    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)

    ### loopfuse
    @torch.compile(backend=loopfuse_backend, dynamic=True, mode=mode)
    def loopfuse_fn(Q, K_cache, K_block_table, V_cache, V_block_table):
        K = torch.ops.loopfuse.paged_load(K_cache, K_block_table).transpose(-3, -2)
        V = torch.ops.loopfuse.paged_load(V_cache, V_block_table).transpose(-3, -2)

        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        scores = torch.softmax(QK, dim=-1)
        return scores @ V

    ### torch reference implementation
    def load_from_paged_cache(cache, block_table):
        B, num_blocks = block_table.shape
        H, D = cache.shape[-2:]

        gathered = cache[block_table]
        gathered = gathered.transpose(1, 2)

        return gathered

    @torch.compile(backend="inductor", dynamic=True)
    def torch_fn(Q, K_cache, K_block_table, V_cache, V_block_table):
        K = load_from_paged_cache(K_cache, K_block_table)
        V = load_from_paged_cache(V_cache, V_block_table)

        return torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
        )

    inputs = (Q, K_cache, K_block_table, V_cache, V_block_table)

    print("Comparing outputs for paged decoding attention...")
    loopfuse_output = loopfuse_fn(*inputs)
    torch_output = torch_fn(*inputs)

    compare_outputs(torch_output, loopfuse_output)

    ### benchmark
    loopfuse_time = helper_benchmark_fn(loopfuse_fn, inputs)
    torch_time = helper_benchmark_fn(torch_fn, inputs)

    mem_bw_count = 2 * 2 * B * H * S * D / 1e9

    loopfuse_mem_bw = mem_bw_count / (loopfuse_time * 1e-3)
    torch_mem_bw = mem_bw_count / (torch_time * 1e-3)

    print(f"Paged Decoding Loopfuse: {loopfuse_mem_bw:.2f} GB/s")
    print(f"Paged Decoding Torch: {torch_mem_bw:.2f} GB/s")

    return {
        "name": "Paged decoding",
        "loopfuse (GB/s)": loopfuse_mem_bw,
        "torch.compile (GB/s)": torch_mem_bw,
    }


def benchmark_paged_decoding_GQA(shape, dtype):
    B, Hq, S, D = shape
    group_size = 8
    Hkv = Hq // group_size

    N_blocks = B * S * 2

    K_cache = torch.randn(N_blocks, Hkv, D, device=device, dtype=dtype)
    V_cache = torch.randn(N_blocks, Hkv, D, device=device, dtype=dtype)

    blocks_per_seq = S
    K_block_table = torch.randint(
        0, N_blocks, (B, blocks_per_seq), device=device, dtype=torch.int32
    )
    V_block_table = torch.randint(
        0, N_blocks, (B, blocks_per_seq), device=device, dtype=torch.int32
    )

    Q = torch.randn(B, Hq, 1, D, device=device, dtype=dtype)

    ### loopfuse
    @torch.compile(backend=loopfuse_backend, dynamic=True, mode=mode)
    def loopfuse_fn(Q, K_cache, K_block_table, V_cache, V_block_table):
        K = torch.ops.loopfuse.paged_load(K_cache, K_block_table).transpose(-3, -2)
        V = torch.ops.loopfuse.paged_load(V_cache, V_block_table).transpose(-3, -2)
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        scores = torch.softmax(QK, dim=-1)
        return scores @ V

    ### torch reference implementation
    def load_from_paged_cache(cache, block_table):
        B, num_blocks = block_table.shape
        H, D = cache.shape[-2:]

        gathered = cache[block_table]
        gathered = gathered.transpose(1, 2)

        return gathered

    @torch.compile(backend="inductor", dynamic=True)
    def torch_fn(Q, K_cache, K_block_table, V_cache, V_block_table):
        K = load_from_paged_cache(K_cache, K_block_table)
        V = load_from_paged_cache(V_cache, V_block_table)

        return torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            enable_gqa=True,
        )

    inputs = (Q, K_cache, K_block_table, V_cache, V_block_table)

    print("Comparing outputs for GQA + Paged Decoding attention...")
    # temp shape fixes until fix shape bug!
    Q_loopfuse = Q.view(B, Hkv, group_size, D)
    torch._dynamo.mark_static(Q_loopfuse, 3)
    torch._dynamo.mark_dynamic(Q_loopfuse, 2)
    loopfuse_inputs = (Q_loopfuse, K_cache, K_block_table, V_cache, V_block_table)
    loopfuse_output = loopfuse_fn(*loopfuse_inputs).view(B, Hq, 1, D)
    torch_output = torch_fn(*inputs)

    compare_outputs(torch_output, loopfuse_output)

    ### benchmark
    loopfuse_time = helper_benchmark_fn(loopfuse_fn, loopfuse_inputs)
    torch_time = helper_benchmark_fn(torch_fn, inputs)

    mem_bw_count = 2 * 2 * B * Hkv * S * D / 1e9

    loopfuse_mem_bw = mem_bw_count / (loopfuse_time * 1e-3)
    torch_mem_bw = mem_bw_count / (torch_time * 1e-3)

    print(f"GQA + Paged Decoding Loopfuse: {loopfuse_mem_bw:.2f} GB/s")
    print(f"GQA + Paged Decoding Torch: {torch_mem_bw:.2f} GB/s")

    return {
        "name": "Paged decoding + GQA",
        "loopfuse (GB/s)": loopfuse_mem_bw,
        "torch.compile (GB/s)": torch_mem_bw,
    }


def benchmark_decoding(shape, dtype):
    B, H, S, D = shape

    Q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)

    torch._dynamo.mark_static(Q, 3)
    torch._dynamo.mark_dynamic(Q, 2)

    inputs = (Q, K, V)

    ### loopfuse
    @torch.compile(backend=loopfuse_backend, dynamic=True, mode=mode)
    def loopfuse_fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        scores = torch.softmax(QK, dim=-1)
        return scores @ V

    ### torch reference implementation
    @torch.compile(backend="inductor", dynamic=True)
    def torch_fn(Q, K, V):
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    print("Comparing outputs for decoding attention...")
    loopfuse_output = loopfuse_fn(*inputs)
    torch_output = torch_fn(*inputs)

    compare_outputs(torch_output, loopfuse_output)

    ### benchmark
    loopfuse_time = helper_benchmark_fn(loopfuse_fn, inputs)
    torch_time = helper_benchmark_fn(torch_fn, inputs)

    mem_bw_count = 2 * 2 * B * H * S * D / 1e9

    loopfuse_mem_bw = mem_bw_count / (loopfuse_time * 1e-3)
    torch_mem_bw = mem_bw_count / (torch_time * 1e-3)

    print(f"Decoding Loopfuse: {loopfuse_mem_bw:.2f} GB/s")
    print(f"Decoding Torch: {torch_mem_bw:.2f} GB/s")

    return {
        "name": "Decoding",
        "loopfuse (GB/s)": loopfuse_mem_bw,
        "torch.compile (GB/s)": torch_mem_bw,
    }


def benchmark_decoding_gqa(shape, dtype):
    B, Hq, S, D = shape
    group_size = 8
    Hkv = Hq // group_size

    Q = torch.randn(B, Hq, 1, D, device=device, dtype=dtype)
    K = torch.randn(B, Hkv, S, D, device=device, dtype=dtype)
    V = torch.randn(B, Hkv, S, D, device=device, dtype=dtype)

    inputs = (Q, K, V)

    ### loopfuse
    @torch.compile(backend=loopfuse_backend, dynamic=True, mode=mode)
    def loopfuse_fn(Q, K, V):
        QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
        scores = torch.softmax(QK, dim=-1)
        return scores @ V

    ### torch reference implementation
    @torch.compile(backend="inductor", dynamic=True)
    def torch_fn(Q, K, V):
        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, enable_gqa=True
        )

    print("Comparing outputs for GQA decoding attention...")
    Q_loopfuse = Q.view(B, Hkv, group_size, D)
    torch._dynamo.mark_static(Q_loopfuse, 3)
    torch._dynamo.mark_dynamic(Q_loopfuse, 2)
    loopfuse_inputs = (Q_loopfuse, K, V)
    loopfuse_output = loopfuse_fn(*loopfuse_inputs).view(B, Hq, 1, D)
    torch_output = torch_fn(*inputs)

    compare_outputs(torch_output, loopfuse_output)

    ### benchmark
    loopfuse_time = helper_benchmark_fn(loopfuse_fn, loopfuse_inputs)
    torch_time = helper_benchmark_fn(torch_fn, inputs)

    mem_bw_count = 2 * B * Hkv * S * D / 1e9

    loopfuse_mem_bw = mem_bw_count / (loopfuse_time * 1e-3)
    torch_mem_bw = mem_bw_count / (torch_time * 1e-3)

    print(f"GQA Decoding Loopfuse: {loopfuse_mem_bw:.2f} GB/s")
    print(f"GQA Decoding Torch: {torch_mem_bw:.2f} GB/s")

    return {
        "name": "Decoding GQA",
        "loopfuse (GB/s)": loopfuse_mem_bw,
        "torch.compile (GB/s)": torch_mem_bw,
    }


def print_markdown_table(title, results):
    if not results:
        return

    headers = list(results[0].keys())
    if "name" in headers:
        headers.remove("name")
        headers.insert(0, "Variant")
        results = [{**d, "Variant": d["name"]} for d in results]

    print(f"### {title}")
    print("| " + " | ".join(h for h in headers) + " |")
    print("|" + "---|" * len(headers))

    for res in results:
        numeric_values = [v for v in res.values() if isinstance(v, (int, float))]
        max_val = max(numeric_values) if numeric_values else None

        row = []
        for h in headers:
            val = res.get(h)
            if isinstance(val, (int, float)):
                formatted_val = f"{val:.0f}"
                if val == max_val:
                    row.append(f"**{formatted_val}**")
                else:
                    row.append(formatted_val)
            else:
                row.append(str(val) if val is not None else "N/A")
        print("| " + " | ".join(row) + " |")
    print("\n")


@torch.inference_mode()
def run_benchmarks():
    prefill_results = []
    prefill_fn_list = [
        benchmark_causal_attn,
        benchmark_softcapped_causal_attn,
        benchmark_gqa_sliding_mask_softcap_attn,
    ]
    for fn in prefill_fn_list:
        prefill_results.append(fn((8, 32, 2048, 128), torch.bfloat16))

    decoding_results = []
    decode_fn_list = [
        benchmark_decoding,
        benchmark_paged_decoding,
        benchmark_paged_decoding_GQA,
    ]
    for fn in decode_fn_list:
        decoding_results.append(fn((64, 32, 2048, 128), torch.bfloat16))

    print("\n--- Results ---")
    print_markdown_table("Prefill Benchmarks", prefill_results)
    print_markdown_table("Decoding Benchmarks", decoding_results)


if __name__ == "__main__":
    run_benchmarks()
