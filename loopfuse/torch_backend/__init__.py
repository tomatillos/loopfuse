import torch


@torch.library.custom_op("loopfuse::causal_mask", mutates_args=())
def causal_mask(a: torch.Tensor) -> torch.Tensor:
    seq_len = a.shape[-1]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=a.device, dtype=torch.bool))
    return a.masked_fill(~mask, float("-inf"))


@torch.library.register_fake("loopfuse::causal_mask")
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("loopfuse::sliding_mask", mutates_args=())
def sliding_mask(a: torch.Tensor, window_size: int) -> torch.Tensor:
    seq_len = a.shape[-1]
    row_idx = torch.arange(seq_len, device=a.device)[:, None]
    col_idx = torch.arange(seq_len, device=a.device)[None, :]
    # mask = (col_idx <= row_idx) & (col_idx > row_idx - window_size)
    mask = col_idx > row_idx - window_size
    return a.masked_fill(~mask, float("-inf"))


@torch.library.register_fake("loopfuse::sliding_mask")
def _(x: torch.Tensor, window_size: int) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("loopfuse::scan_max", mutates_args=())
def scan_max(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cummax_result = torch.cummax(a, dim=-1)
    running_maxs = cummax_result.values
    final_max = torch.max(a, dim=-1, keepdim=True).values
    return running_maxs, final_max


@torch.library.register_fake("loopfuse::scan_max")
def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    running_maxs = torch.empty_like(x)
    final_max = torch.max(x, dim=-1, keepdim=True).values
    return running_maxs, final_max


@torch.library.custom_op("loopfuse::paged_load", mutates_args=())
def paged_load(buffer: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """
    block_table: int[*S]
    buffer: float[N, *T]
    returns: float[*S, *T]
    where out[*s, *t] = buffer[block_table[*s], *t]
    """
    S = block_table.shape
    N, *T = buffer.shape
    return buffer[block_table]


@torch.library.register_fake("loopfuse::paged_load")
def _(buffer: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    S = block_table.shape
    N, *T = buffer.shape
    return buffer.new_empty((*S, *T))
