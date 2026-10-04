import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def span_masking(
    z: torch.Tensor,
    mask_token: Optional[torch.Tensor] = None,
    mask_ratio: float = 0.8,
    min_span: int = 3,
    max_span: int = 10,
    fill_mode: str = "mask",   # "mask" | "interpolate"
    noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Perform random span masking on a batch of temporal embeddings.

    Args:
        z:            [B, T, D] input embeddings
        mask_token:   [D] or [1,1,D] tensor used to fill masked spans (for fill_mode='mask')
        mask_ratio:   approximate fraction of time steps to mask
        min_span:     minimum span length (frames)
        max_span:     maximum span length (frames)
        fill_mode:    'mask' (fill with mask_token) or 'interpolate' (linear interpolate)
        noise_std:    add optional Gaussian noise to interpolated spans

    Returns:
        z_masked:     masked embeddings [B, T, D]
        mask:         boolean mask [B, T], True = masked positions
    """
    B, T, D = z.shape
    device = z.device
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    # decide number of masked frames per batch element
    total_mask = int(T * mask_ratio)

    for b in range(B):
        num_masked = 0
        while num_masked < total_mask:
            span = torch.randint(min_span, max_span + 1, (1,)).item()
            start = torch.randint(0, max(1, T - span), (1,)).item()
            end = min(T, start + span)
            mask[b, start:end] = True
            num_masked += (end - start)

    z_masked = z.clone()

    if fill_mode == "mask":
        if mask_token is None:
            # default: learnable or mean token; here zeros
            mask_token = torch.zeros(1, 1, D, device=device)
        if mask_token.ndim == 1:
            mask_token = mask_token.view(1, 1, D)
        z_masked[mask.unsqueeze(-1).expand_as(z)] = mask_token.expand(B, T, D)[mask.unsqueeze(-1).expand_as(z)]

    elif fill_mode == "interpolate":
        # fill masked spans by linear interpolation between nearest unmasked neighbors
        for b in range(B):
            idx = torch.arange(T, device=device)
            valid = ~mask[b]
            if valid.sum() < 2:
                continue  # skip if everything masked
            # interpolate each dimension independently
            z_valid = z[b, valid]
            idx_valid = idx[valid].float()
            idx_all = idx.float()
            z_interp = F.interpolate(
                z_valid.unsqueeze(0).transpose(1, 2),
                size=T,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)[0]
            if noise_std > 0:
                z_interp += torch.randn_like(z_interp) * noise_std
            z_masked[b, mask[b]] = z_interp[mask[b]]

    else:
        raise ValueError("fill_mode must be 'mask' or 'interpolate'")

    return z_masked, mask


def mask_one_keep_n_safe(z: torch.Tensor, n: int, mask_value: float = 0.0) -> torch.Tensor:
    """
    Mask embeddings with pattern: keep 1 frame, then mask n frames, repeat,
    but always keep the FIRST and LAST frames unmasked.

    Args:
        z:          [T, D] or [B, T, D] embedding tensor
        n:          number of masked frames after each unmasked frame (n >= 0)
        mask_value: value to fill masked frames (default = 0.0)

    Returns:
        z_masked:   same shape as z, with masked frames replaced by mask_value
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    if z.ndim == 2:
        T, D = z.shape
        B = 1
    elif z.ndim == 3:
        B, T, D = z.shape
    else:
        raise ValueError("z must be 2D ([T,D]) or 3D ([B,T,D])")

    # Base pattern: keep 1, mask n → [0, 1, 1, ..., 1]
    base = torch.tensor([0] + [1]*n, dtype=torch.bool, device=z.device)
    pattern = base.repeat((T + len(base) - 1) // len(base))[:T]  # [T]

    # Ensure first and last are always unmasked
    pattern[0] = False
    pattern[-1] = False

    if z.ndim == 3:
        mask = pattern.unsqueeze(0).expand(B, T)
    else:
        mask = pattern

    z_masked = z.clone()
    z_masked[mask] = mask_value
    return z_masked


def mask_one_keep_n_safe_randoffset(
    z: torch.Tensor,
    n: int = 9,
    mask_value: float = 0.0,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, None]:
    """
    Keep 1 frame, then mask n frames, repeating; first/last frames always unmasked.
    Random offset is sampled per batch row; all frames in a row share that offset.

    Args:
        z: [T, D] or [B, T, D] embeddings
        n: number of masked frames after each kept frame (n >= 0)
        mask_value: value to fill masked frames
        generator: optional torch.Generator for reproducibility

    Returns:
        z_masked: same shape as z, with masked frames replaced by mask_value
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    # Normalize shapes to [B, T, D]
    unbatched = False
    if z.ndim == 2:
        z = z.unsqueeze(0)
        unbatched = True
    elif z.ndim != 3:
        raise ValueError("z must be [T,D] or [B,T,D]")

    B, T, D = z.shape
    device = z.device

    # Base pattern over one period: [keep=0] + [mask=1]*n
    L = n + 1
    base = torch.tensor([0] + [1] * n, dtype=torch.bool, device=device)  # shape [L]

    # Random per-row offsets in [0, L-1]
    if L > 1:
        offsets = torch.randint(low=0, high=L, size=(B,), device=device, generator=generator)
    else:
        offsets = torch.zeros(B, dtype=torch.long, device=device)  # n=0 → no masking

    # Build mask via modulo indexing: mask[b, t] = base[(t + offset[b]) % L]
    t_idx = torch.arange(T, device=device).unsqueeze(0)              # [1, T]
    idx = (t_idx + offsets.view(B, 1)) % L                           # [B, T]
    mask = base[idx]                                                 # [B, T] (bool)

    # Ensure first and last frames are never masked
    mask[:, 0] = False
    mask[:, -1] = False

    # Apply mask
    z_masked = z.clone()
    # Expand mask to [B, T, D] and fill
    z_masked[mask.unsqueeze(-1).expand(B, T, D)] = mask_value

    return z_masked.squeeze(0) if unbatched else z_masked, None

def mask_keep_a_mask_b(
    z: torch.Tensor,
    a: int=10,
    b: int=40,
    use_random_offset: bool = True,
    mask_value: float = 0.0,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, None]:
    """
    Periodic masking with pattern: keep 'a' frames (0), then mask 'b' frames (1), repeat.
    First and last frames are always kept (unmasked).

    Args:
        z:  [T, D] or [B, T, D] embeddings
        a:  number of consecutive unmasked frames (>= 0)
        b:  number of consecutive masked frames (>= 0)
        use_random_offset: if True, sample a random phase per batch row
        mask_value: value to write into masked frames
        generator: optional torch.Generator for reproducibility

    Returns:
        z_masked: tensor with the same shape as z, masked in-place
    """
    if a < 0 or b < 0:
        raise ValueError("a and b must be >= 0")
    if a == 0 and b == 0:
        # Nothing to do, return as-is
        return z

    # Normalize to [B, T, D]
    unbatched = False
    if z.ndim == 2:
        z = z.unsqueeze(0)
        unbatched = True
    elif z.ndim != 3:
        raise ValueError("z must be 2D ([T,D]) or 3D ([B,T,D])")

    B, T, D = z.shape
    device = z.device

    L = a + b  # period length
    if L == 0:
        # Degenerate: treat as no masking
        return z.squeeze(0) if unbatched else z

    # Build one period: first 'a' unmasked (0), then 'b' masked (1)
    base = torch.cat([
        torch.zeros(a, dtype=torch.bool, device=device),
        torch.ones(b, dtype=torch.bool, device=device)
    ])  # shape [L]

    # Per-row offsets
    if use_random_offset and L > 1:
        offsets = torch.randint(0, L, (B,), device=device, generator=generator)
    else:
        offsets = torch.zeros(B, dtype=torch.long, device=device)

    # Expand to [B, T] via modulo indexing
    t_idx = torch.arange(T, device=device).unsqueeze(0)         # [1, T]
    idx = (t_idx + offsets.view(B, 1)) % L                      # [B, T]
    mask = base[idx]                                            # [B, T] bool

    # Ensure first and last frames are always unmasked
    mask[:, 0] = False
    mask[:, -1] = False

    # Apply mask
    z_masked = z.clone()
    z_masked[mask.unsqueeze(-1).expand(B, T, D)] = mask_value

    return z_masked.squeeze(0) if unbatched else z_masked, None



# ---------------- Example usage ----------------
if __name__ == "__main__":
    # torch.manual_seed(0)
    B, T, D = 3, 12, 1
    z = torch.arange(B*T*D).view(B, T, D).float()
    out, _ = mask_one_keep_n_safe_randoffset(z, n=2, mask_value=0.0)
    print("Masked shape:", out.shape)
    # First and last frames per row should remain unmasked (non-zero if original was non-zero)
    print("Row 0:")
    print( out[0, :, :])
    print("Row 1:")
    print( out[1, :, :])
    print("Row 2:")
    print(out[2, :, :])