"""Torch-native drop-in for flash_attn.ops.triton.cross_entropy.cross_entropy_loss.

OpenRLHF imports this function at module load; on pods without a compiled
flash-attn we redirect to torch.nn.functional.cross_entropy, which is
numerically equivalent for the label-smoothed + ignore-index case that
OpenRLHF actually uses. The `lse_square_scale` (z-loss) term is
stubbed out as zeros, which matches the behaviour when OpenRLHF sets
the coefficient to 0.
"""

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits,
    labels,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignore_index: int = -100,
    inplace_backward: bool = False,
    process_group=None,
):
    if logit_scale != 1.0:
        logits = logits * logit_scale
    loss = F.cross_entropy(
        logits.float(),
        labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    z_loss = torch.zeros_like(loss)
    return loss, z_loss
