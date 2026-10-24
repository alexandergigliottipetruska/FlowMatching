import torch
import torch.nn as nn

def sample_path(x0, x1, t=None):
    """
    Args:
        x0: Source samples, shape (B, C, H, W)
        x1: Target samples, shape (B, C, H, W)
        t: Optional timesteps, (B,)

    Returns:
        t: sampled or provided timesteps, (B,)
        xt: intermediate samples, (B, C, H, W)
        ut: target velocities, (B, C, H, W)
    """
    # Sample t if not provide
    if t is None:
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype) # size (B, )

    # Broadcast t to (B, 1, 1, 1)
    t = t.view(x0.shape[0], 1, 1, 1)

    # Intermediate samples
    xt = (1 - t) * x0 + t * x1

    # Target velocities
    ut = x1 - x0

    return t.view(-1), xt, ut




