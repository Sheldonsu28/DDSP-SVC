import torch
import torch.nn as nn
import torch.nn.functional as F

def hz_to_cents(f0_hz, eps=1e-8):
    return 1200.0 * torch.log2(torch.clamp(f0_hz, min=eps))

def cents_to_hz(cents):
    return 2.0 ** (cents / 1200.0)

class Lowpass1D(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        L = kernel if kernel % 2 == 1 else (kernel + 1)
        win = torch.hann_window(L)
        filt = (win / win.sum()).view(1, 1, L)
        self.register_buffer("filt", filt)
        self.pad = L // 2
    def forward(self, x):  # (B,T,1)
        return F.conv1d(x.transpose(1,2), self.filt, padding=self.pad).transpose(1,2)

class F0PatternPredictor(nn.Module):
    """
    Predicts a band-limited cents residual from style/aux.
    Train to match:
      A) residual-to-GT:  cents(f0_gt) - cents(f0_base)
      or
      B) fluctuation-only: cents(f0_gt) - MA(cents(f0_gt))
    Inference: add predicted residual (clamped) to cents(f0_base).
    """
    def __init__(self, style_dim=64, aux_dim=0, hidden=128,
                 max_semitone=0.3, lp_kernel=5, mode="fluctuation"):
        super().__init__()
        assert mode in ("residual", "fluctuation")
        self.mode = mode
        self.max_cents = max_semitone * 100.0
        self.lp = Lowpass1D(kernel=lp_kernel)
        in_dim = style_dim + aux_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)  # (B,T,1) cents residual (pre-bound)
        )

    def forward(self, f0_base_hz, style_seq, aux_seq=None):
        """
        f0_base_hz: (B,T,1) base f0 to be nudged (content/analysis path)
        style_seq : (B,T,style_dim)
        aux_seq   : (B,T,aux_dim) or None
        returns:
          residual_cnt: (B,T,1) bounded, smoothed cents residual
          f0_hat_hz   : (B,T,1) f0 after applying residual to f0_base
        """
        cond = style_seq if aux_seq is None else torch.cat([style_seq, aux_seq], dim=-1)
        raw = self.mlp(cond)                                       # (B,T,1)
        # bound then low-pass (straight-through trick to block HF gaming)
        bounded = torch.tanh(raw) * self.max_cents                 # (B,T,1)
        smooth = self.lp(bounded)
        residual_cnt = smooth + (bounded - smooth).detach()        # ST low-pass

        base_cents = hz_to_cents(f0_base_hz)
        f0_hat_hz  = cents_to_hz(base_cents + residual_cnt)
        return residual_cnt, f0_hat_hz

# ---------- Training helpers ----------

def moving_average_cents(cents, win=31):
    L = win if win % 2 == 1 else (win+1)
    filt = torch.ones(1,1,L, device=cents.device) / L
    pad = L//2
    return F.conv1d(cents.transpose(1,2), filt, padding=pad).transpose(1,2)

def residual_targets(f0_gt_hz, f0_base_hz, mode="fluctuation", ma_win=31):
    """
    Returns target cents residuals for supervision.
    mode='residual'    -> cents(gt) - cents(base)
    mode='fluctuation' -> cents(gt) - MA(cents(gt))
    """
    gt_c = hz_to_cents(f0_gt_hz)
    if mode == "residual":
        base_c = hz_to_cents(f0_base_hz)
        tgt = gt_c - base_c
    else:
        trend = moving_average_cents(gt_c, win=ma_win)
        tgt = gt_c - trend
    return tgt  # (B,T,1)

def pattern_loss(residual_pred, residual_tgt, voiced_mask=None,
                 w_l1=1.0, w_tv=0.005, w_curv=0.002):
    """
    Supervise residual in cents; add anti-jitter regularizers.
    """
    if voiced_mask is not None:
        m = voiced_mask
        l1 = (m * (residual_pred - residual_tgt).abs()).sum() / (m.sum() + 1e-6)
    else:
        l1 = (residual_pred - residual_tgt).abs().mean()

    # 1st and 2nd derivatives (anti-jitter)
    d1 = residual_pred[:,1:] - residual_pred[:,:-1]
    d2 = d1[:,1:] - d1[:,:-1]
    if voiced_mask is not None:
        m1 = voiced_mask[:,1:] * voiced_mask[:,:-1]
        m2 = m1[:,1:] * m1[:,:-1]
        tv  = (m1 * d1.abs()).sum() / (m1.sum() + 1e-6)
        cur = (m2 * d2.pow(2)).sum() / (m2.sum() + 1e-6)
    else:
        tv  = d1.abs().mean()
        cur = d2.pow(2).mean()

    return w_l1*l1 + w_tv*tv + w_curv*cur


def volume_mask_from_rms(volume, threshold_db=-40.0):
    """
    Create a binary mask from RMS volume in dB.

    Args:
        volume: Tensor of shape (B, T, 1) or (B, T)
                RMS amplitude in [0,1] (from Volume_Extractor).
        threshold_db: float
                Frames below this dB threshold are masked to 0.

    Returns:
        mask: Tensor of same shape as volume (0 or 1).
    """
    # Ensure shape (B, T, 1)
    if volume.dim() == 2:
        volume = volume.unsqueeze(-1)

    # Convert RMS amplitude -> dB
    volume_db = 20 * torch.log10(volume + 1e-8)  # (B, T, 1)

    # Compare to threshold
    mask = (volume_db >= threshold_db).float()

    return mask
