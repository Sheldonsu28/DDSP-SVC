import torch
from torch import Tensor


def range_compression(f0:Tensor, low:float, high:float, slient_mask:Tensor) -> tuple[Tensor, float, float] :
    """
    Compress shift and rescale f0 so max(f0) = high, min(f0) = low
    f0 : Float Tensor of size [1, T, 1], unit is in hz
    high: int, exmaple: 1060 (hz)
    low: int, exmaple: 20 (hz)
    silent mask: Boolean Tensor of size [1, T, 1], mask out unsounded part
    returns:
        new_f0: f0 in new range
        original_high: highest value of original f0
        original_low: lowest value of original f0
    """
    # print("max ", f0.max(), "min ", f0.min())
    voiced = f0[~slient_mask]
    original_high = voiced.max().item()
    original_low = voiced.min().item()
    
    if low <= original_low and original_high <= high:
        return f0, original_high, original_low
    
    if original_high == original_low:
        # print('silence')
        midpoint = (high + low) / 2
        new_f0 = torch.full_like(f0, midpoint)
        return new_f0, original_high, original_low

    scale = (high - low) / (original_high - original_low)
    new_f0 = (f0 - original_low) * scale + low
    # new_f0 = torch.where(slient_mask, f0, new_f0)

    return new_f0, original_high, original_low
