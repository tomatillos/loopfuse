# loopfuse

[Demo video.](https://www.loom.com/share/0d491fee18cb459da598e6a99cf37185?sid=d578d626-34b7-4723-97d0-d642c9b0e096)

A tile-based compiler that generates fused Triton kernels from scratch.

By mirroring the underlying optimisation ideas in FlashAttention, `loopfuse` can codegen efficient Triton kernels for a whole range of attention variants.

The key features are:
- Fusion of adjacent operations to reduce HBM accesses
- Generalisation of online softmax
- Sparsity-aware compilation to skip masked blocks

## Why? 
When torch.compile can't pattern match with FlashAttention, performance drops. FlexAttention is great, but it's still templated + handwritten.

`loopfuse` generates fused kernels from scratch, using general, reusable compiler passes - which means you can modify the graph and still get performant kernels.

For example, GQA + sliding window + logit softcapping:

| Method | Performance (A100 40GB)|
|--------|-------------|
| **loopfuse** | **108 TFLOPs/s**|
| FlexAttention | 105 TFLOPs/s |
| torch.compile | 15 TFLOPs/s |


It also aims to output readable code - to use as a starting point for manual improvement.

## Key techniques

The basic approach is repeated fusion of adjacent operations to minimise HBM reads/writes. Capture torch graph, convert to a tile-based loop ir, repeatedly fuse operations (being careful of dependencies). Tilings of tensors are first-class citizens, all operations happen on tiles.

Ideas needed to get to performant attention kernels:
- Transforming sum reductions into an 'online' version gets us attention fusion (cf. online softmax)
- Sparsity aware compilation - skip masked blocks by reasoning about which loop steps have no effect on output

Using these building blocks, we can handle many attention modifications, compositions and shapes, for instance:
- MHA, GQA, MLA, prefill, decode
- Paged Attention (fuse the k/v load)
- Sliding window attention, skipping computation over masked blocks
- Score modifications: logit softcapping, softmax variants, etc.

## Benchmarks

Benchmarks run on A100 40GB. (tests/benchmark.py)

### Prefill
Config: Batch=8, Heads=32, Seqlen=2048, Head Dim=128
| Variant| loopfuse (TFLOPs/s) | torch.compile (TFLOPs/s) | flex attention (TFLOPs/s) |
|---|---|---|---|
| Causal | 157 | **182** | 122 |
| Causal + Softcap | **122** | 21 | 115 |
| Sliding window + GQA + Softcap | **108** | 15 | 105 |


Maintains high performance with attention modifications! Not quite the speed of cuda flash attention, but at least as fast as FlexAttention (a templated Triton implementation).


### Decode
Config: Batch=64, Heads=32, Q_len=1, K_len=2048, Head Dim=128

| Variant | loopfuse (GB/s) | torch.compile (GB/s) |
|---|---|---|
| Decoding | **1407** | 1345 |
| Paged decoding | **1383** | 447 |
| Paged decoding + GQA | **1243** | 307 |


## Usage
Implemented as a custom backend to `torch.compile`, so easy interface with PyTorch.

```python
@torch.compile(backend=loopfuse_backend, dynamic=True)
def attn(Q, K, V):
    QK = Q @ K.transpose(-2, -1) * (Q.shape[-1] ** -0.5)
    masked_attn = torch.ops.loopfuse.causal_mask(QK)
    scores = torch.softmax(masked_attn, dim=-1)
    return scores @ V
```

## Technical details 


### Online softmax generalisation
Idea is to turn sum reductions into online sum reductions, and see if that allows fusion.

There is an equivalence:
```
s = sum(ri * xi)
<=>
for i:
  s *= r_{i-1} / r_i
  s += xi
s *= r_n
```
(write out a few terms if not convinced)

In some cases the ratio of the ri's cancels out a term that is blocking fusion. We are free to choose the factorisation `ri*xi`, so we try to factorise all of the 'blocking' terms into `ri`. When such a factorisation exists, and the ratio of the `ri`'s no longer contains any blocking terms, we gain a new fusion opportunity!

### Sparsity analysis

If our mask is algebraically defined / known at compile-time, we can calculate conditions on the indices that correspond to 'no-ops' i.e. no changes to the outputs. We do this by propagating compile-time known values through the graph, deducing things like masked blocks being `-inf`, `exp(-inf) = 0` and `0 @ x = 0`.


## Future directions 

### What's missing from FlashAttention
1. TMA support for Hopper+ (not too hard to extend to this though)
2. Backwards pass!


### Other possible additions:
1. More sparsity/constant analysis (e.g. when we can skip doing any mask logic because it's all True)
2. Fine-grained quantization + wider support for dtypes + accumulation types (maybe annotation)
4. Extra features: chunked prefill, bigger pages in paged attention
5. Split-k & persistent kernels (otherwise grid size is too small in small batch size decoding)
