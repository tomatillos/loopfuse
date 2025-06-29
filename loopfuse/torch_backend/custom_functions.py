import math

import torch


def softmax_decomposed(X):
    Xf = X.to(torch.float32)
    Xf_scaled = (
        Xf * 1 / math.log(2)
    )  # to use efficient exp2 (mul can fuse with earlier muls)
    X_mij, X_mi = torch.ops.loopfuse.scan_max(Xf_scaled)
    expX = torch.exp2(Xf_scaled - X_mij) * torch.exp2(
        X_mij - X_mi
    )  # compiler hint for online softmax
    S = torch.sum(expX, dim=-1, keepdim=True)
    out = expX / S
    return out.to(X.dtype)
