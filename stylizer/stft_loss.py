# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F
import torchaudio


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag.clamp(min=1e-5)), torch.log(x_mag.clamp(min=1e-5)))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, device, fft_size=1024, shift_size=120, win_length=600, window="hann_window", sc_weight=1, weight=1):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.sc_weight = sc_weight
        self.weight = weight
        self.window = getattr(torch, window)(win_length).to(device)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag) * self.weight
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag) * self.weight

        return sc_loss * self.sc_weight, (1-self.sc_weight) * mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 device,
                 resolutions,
                 window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            resolutions (list): List of (FFT size, hop size, window length).
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl, sc_weight, weight in resolutions:
            self.stft_losses += [STFTLoss(device, fs, ss, wl, window, sc_weight, weight)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return 1 * ( sc_loss + mag_loss)
    
    
    
def hz_to_mel_htk(f):
    return 2595.0 * torch.log10(1.0 + f / 700.0)

def mel_to_hz_htk(m):
    return 700.0 * (10.0**(m / 2595.0) - 1.0)

def mel_bin_centers(n_mels, fmin, fmax, device):
    m_lo = hz_to_mel_htk(torch.tensor(fmin, device=device))
    m_hi = hz_to_mel_htk(torch.tensor(fmax, device=device))
    m = torch.linspace(m_lo, m_hi, n_mels, device=device)     # (M,)
    f = mel_to_hz_htk(m)                                      # (M,)
    return f, m  # (M,), (M,)

# ---------- soft harmonic mask in Mel space ----------
def harmonic_mask_mel(
    f0_hz,                   # (B, T) F0 in Hz (0 for unvoiced)
    n_mels=128,
    fmin=30.0,
    fmax=22050.0,
    max_harm=24,
    bw_cents=80.0,           # hard cutoff half-width around each harmonic
    soft_sigma_cents=40.0,   # Gaussian sigma in cents (soft spread)
):
    """
    Returns mask M in [0,1] with shape (B, M, T), peaking along harmonic lines.
    """
    if f0_hz.ndim == 3:
        f0_hz = f0_hz.squeeze(-1)
    assert f0_hz.ndim == 2, "f0_hz should be (B, T)"
    B, T = f0_hz.shape
    device = f0_hz.device

    mel_cent_freq_hz, _ = mel_bin_centers(n_mels, fmin, fmax, device)  # (M,)
    f_bin = mel_cent_freq_hz.view(1, -1, 1).clamp_min(1e-6)            # (1, M, 1)

    H = torch.arange(1, max_harm + 1, device=device).view(1, 1, -1)    # (1,1,H)
    f0 = f0_hz.view(B, 1, T).clamp_min(1e-6)                           # (B,1,T)
    f_h = (H.view(1, 1, -1, 1) * f0.view(B, 1, 1, T))                  # (B,1,H,T)

    valid = (f_h <= fmax).float()
    f_bin_b = f_bin.view(1, -1, 1, 1)                                  # (1,M,1,1)
    f_h_b   = f_h.expand(B, 1, -1, T)                                  # (B,1,H,T)
    valid   = valid.expand(B, 1, -1, T)                                # (B,1,H,T)

    # Cents distance: 1200 * |log2(f_bin / f_h)|
    eps = 1e-8
    cents = 1200.0 * torch.log2((f_bin_b + eps) / (f_h_b + eps)).abs() # (B,M,H,T)

    gauss = torch.exp(-0.5 * (cents / soft_sigma_cents) ** 2) * valid  # (B,M,H,T)
    if bw_cents is not None and bw_cents > 0:
        gauss = gauss * (cents <= bw_cents).float()

    M = gauss.max(dim=2).values                                        # (B,M,T)
    # zero out unvoiced frames
    uv = (f0_hz <= 0.0).float().view(B, 1, T)
    M = M * (1.0 - uv)
    return M.clamp(0.0, 1.0)

# ---------- harmonic-emphasis loss on *log*-Mel ----------
def harmonic_emphasis_mel_loss_logged(
    logmel_pred,        # (B, M, T)  <-- already log-compressed (e.g., dynamic_range_compression_torch)
    logmel_gt,          # (B, M, T)  <-- already log-compressed
    f0_hz,              # (B, T) aligned to Mel frames
    fmin=30.0,
    fmax=22050.0,
    max_harm=20,
    bw_cents=80.0,
    soft_sigma_cents=40.0,
    reduction='mean',
    normalize_by_mask=True,
    off_harmonic_weight=0.3,  # >0 to lightly penalize off-harmonic error
    off_mask_blur=0.1         # in cents; if >0, expands the anti-mask slightly
):
    """
    L1 on log-Mel residuals weighted by a pitch-tracked harmonic mask.
    Optionally adds a weak off-harmonic penalty to discourage broadband hiss.
    """
    assert logmel_pred.shape == logmel_gt.shape, "pred/gt log-Mel must match"
    logmel_gt = logmel_gt.transpose(1, 2)
    logmel_pred = logmel_pred.transpose(1, 2)
    B, M, T = logmel_pred.shape
    # Build harmonic mask on Mel
    harm_mask = harmonic_mask_mel(
        f0_hz=f0_hz, n_mels=M, fmin=fmin, fmax=fmax,
        max_harm=max_harm, bw_cents=bw_cents, soft_sigma_cents=soft_sigma_cents
    )  # (B,M,T)
    # Main (on-harmonic) term
    on_err = (logmel_pred - logmel_gt).abs() * harm_mask  # (B,M,T)

    if normalize_by_mask:
        on = on_err.sum() / harm_mask.sum().clamp_min(1.0)
    else:
        on = on_err.mean() if reduction == 'mean' else on_err.sum()

    # Optional: off-harmonic penalty (very light) to discourage hiss
    if off_harmonic_weight > 0.0:
        if off_mask_blur > 0.0:
            # Widen mask a bit by using a looser sigma (cheap “blur”)
            wide_mask = harmonic_mask_mel(
                f0_hz=f0_hz, n_mels=M, fmin=fmin, fmax=fmax,
                max_harm=max_harm, bw_cents=bw_cents + off_mask_blur,
                soft_sigma_cents=soft_sigma_cents + off_mask_blur/2
            )
            off_mask = (1.0 - wide_mask).clamp(0.0, 1.0)
        else:
            off_mask = (1.0 - harm_mask).clamp(0.0, 1.0)

        off_err = (logmel_pred - logmel_gt).abs() * off_mask
        if normalize_by_mask:
            off = off_err.sum() / off_mask.sum().clamp_min(1.0)
        else:
            off = off_err.mean() if reduction == 'mean' else off_err.sum()
        return on + off_harmonic_weight * off

    return on
    
    
    
def hnr_loss_mel(
    mel_pred,                # (B, M, T)  Mel (log or linear)
    mel_gt,                  # (B, M, T)
    f0_hz,                   # (B, T)     aligned to Mel frames
    *,
    mel_is_log=True,         # True if mel_* are log-scaled already
    mel_rep="power",         # "power" or "magnitude" for the underlying linear meaning
    fmin=30.0,
    fmax=22050.0,
    max_harm=20,
    bw_cents=80.0,
    soft_sigma_cents=40.0,
    voiced_thresh=0.0,       # treat f0<=this as unvoiced
    reduction="mean",
    time_smoothing=0,        # e.g., 3 to median-filter HNR over time
):
    """
    Compute per-frame HNR for pred & GT from Mel, then L2 on HNR curves.
    HNR(frame) = log( E_harm / E_noise ), energies measured on Mel with a harmonic mask.
    """
    mel_gt = mel_gt.transpose(1, 2)
    mel_pred = mel_pred.transpose(1, 2)
    f0_hz = f0_hz.squeeze(-1)
    assert mel_pred.shape == mel_gt.shape
    B, M, T = mel_pred.shape
    device = mel_pred.device
    eps = 1e-8

    # Convert to linear Mel energies
    if mel_is_log:
        mel_lin_p = mel_pred.exp()
        mel_lin_g = mel_gt.exp()
    else:
        mel_lin_p = mel_pred
        mel_lin_g = mel_gt

    # If Mel represents magnitude, convert to power for energy ratios (more standard for HNR)
    if mel_rep.lower().startswith("mag"):
        mel_lin_p = mel_lin_p ** 2
        mel_lin_g = mel_lin_g ** 2

    # Harmonic mask on Mel
    harm_mask = harmonic_mask_mel(
        f0_hz.clamp_min(voiced_thresh),
        n_mels=M, fmin=fmin, fmax=fmax,
        max_harm=max_harm, bw_cents=bw_cents, soft_sigma_cents=soft_sigma_cents
    )  # (B,M,T)

    # To avoid bias from different mask areas, compute MEAN energy inside/outside masks
    area_h = harm_mask.sum(dim=1).clamp_min(1.0)          # (B,T)
    area_n = (1.0 - harm_mask).sum(dim=1).clamp_min(1.0)  # (B,T)

    Eh_p = (mel_lin_p * harm_mask).sum(dim=1) / area_h     # (B,T)
    En_p = (mel_lin_p * (1.0 - harm_mask)).sum(dim=1) / area_n
    Eh_g = (mel_lin_g * harm_mask).sum(dim=1) / area_h
    En_g = (mel_lin_g * (1.0 - harm_mask)).sum(dim=1) / area_n

    # Per-frame HNR in natural log (or use 10*log10 for dB)
    HNR_p = torch.log((Eh_p + eps) / (En_p + eps))         # (B,T)
    HNR_g = torch.log((Eh_g + eps) / (En_g + eps))

    # Optional robust smoothing over time (median) to reduce occasional F0 blips
    if time_smoothing and time_smoothing > 1:
        k = time_smoothing
        pad = k // 2
        # simple median filter via unfold
        def medfilt(x):
            xu = x.unfold(dimension=1, size=k, step=1)  # (B, T-k+1, k)
            # pad at edges by replication
            left = x[:, :1].repeat(1, pad)
            right = x[:, -1:].repeat(1, pad)
            xpad = torch.cat([left, x, right], dim=1)
            xu = xpad.unfold(1, k, 1)
            return xu.median(dim=-1).values
        HNR_p = medfilt(HNR_p)
        HNR_g = medfilt(HNR_g)

    # Mask out unvoiced frames in loss
    voiced = (f0_hz > voiced_thresh).float()
    if voiced.shape[1] != HNR_p.shape[1]:
        # simple resize if needed; better is to resample f0 earlier to match Mel frames
        voiced = torch.nn.functional.interpolate(
            voiced.unsqueeze(1), size=HNR_p.shape[1], mode="nearest"
        ).squeeze(1)

    diff = (HNR_p - HNR_g) ** 2  # L2 is common; use .abs() for L1 if preferred
    diff = diff * voiced  # zero out unvoiced frames

    if reduction == "mean":
        denom = voiced.sum().clamp_min(1.0)
        return diff.sum() / denom
    elif reduction == "sum":
        return diff.sum()
    else:
        return diff
    
    
    
def stft_energy_ratio(
    Y_h: torch.Tensor,
    Y_n: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute per-frame noise fraction r = E_n / (E_h + E_n) from complex STFTs.

    Args:
        Y_h: harmonic STFT, shape (B, F, T), complex or (..., 2, F, T) stacked real/imag
        Y_n: noise STFT,     shape (B, F, T), complex or (..., 2, F, T) stacked real/imag
        voiced_mask: optional (B, T) boolean/float mask; if provided, r is still returned
                     for all frames, but you can use it to compute voiced stats downstream.
        eps: numerical stability.

    Returns:
        r: (B, T) tensor of per-frame noise fraction.
    """
    # Y_h = _ensure_complex(Y_h)
    # Y_n = _ensure_complex(Y_n)

    # Power per time frame = sum over frequency bins of |Y|^2
    Eh = (Y_h.real**2 + Y_h.imag**2).sum(dim=-2)  # (B, T)
    En = (Y_n.real**2 + Y_n.imag**2).sum(dim=-2)  # (B, T)

    r = En / (Eh + En + eps)  # (B, T)

    return r


def summarize_ratio(
    r: torch.Tensor,
) -> dict:
    """
    Quick stats for your dashboard/logs.

    Args:
        r: (B, T) per-frame ratios.
        voiced_mask: optional (B, T) boolean/float mask.

    Returns:
        dict of overall and (if provided) voiced-only medians/means.
    """
    stats = {
        "mean_all": r.mean().item(),
        "median_all": r.median().item()
    }
    return stats


bands_hz = [
    (0.0,   200.0),    # LF "support" / fundamental region
    (200.0, 800.0),    # low-mid body / lower formants
    (800.0, 3000.0),   # mid (F2–F3 region)
    (3000.0, 6000.0),  # presence / brightness
    (6000.0, 16000.0), # "air" / hiss / breath
]
band_names = ["LF", "LowMid", "Mid", "Presence", "Air"]



def hz_to_bin(hz: float, sr: int, n_fft: int) -> int:
    return int(round(hz * n_fft / sr))

@torch.no_grad()
def stft_bandwise_ratios(
    Y_h: torch.Tensor,
    Y_n: torch.Tensor,
    sr: int,
    n_fft: int,
    eps: float = 1e-8,
    bands_hz=bands_hz,
) -> torch.Tensor:
    """
    Returns r_band: (B, T, BANDS) where r = En_band / (Eh_band + En_band)
    """
    # Y_h = _ensure_complex(Y_h)
    # Y_n = _ensure_complex(Y_n)
    magsq_h = (Y_h.real**2 + Y_h.imag**2)    # (B,F,T)
    magsq_n = (Y_n.real**2 + Y_n.imag**2)

    B, F, T = magsq_h.shape
    outs = []
    for (f_lo, f_hi) in bands_hz:
        lo = max(0, hz_to_bin(f_lo, sr, n_fft))
        hi = min(F - 1, hz_to_bin(f_hi, sr, n_fft))
        if hi < lo: lo = hi
        Eh = magsq_h[:, lo:hi+1, :].sum(dim=-2)  # (B,T)
        En = magsq_n[:, lo:hi+1, :].sum(dim=-2)
        r  = En / (Eh + En + eps)                # (B,T)
        outs.append(r.unsqueeze(-1))
    return torch.cat(outs, dim=-1)               # (B,T,BANDS)

@torch.no_grad()
def summarize_bandwise(
    r_band: torch.Tensor,          # (B,T,BANDS)
) -> dict:
    """
    Returns dict with overall + voiced-only mean/median/p90 per band.
    """
    B, T, K = r_band.shape
    out = {"overall": {}, "voiced": {}}
    # overall
    rb = r_band.reshape(-1, K)  # (B*T, K)
    out["overall"]["mean"]   = rb.mean(dim=0).tolist()
    out["overall"]["median"] = rb.median(dim=0).values.tolist()
    out["overall"]["p90"]    = rb.kthvalue(int(0.90 * rb.shape[0]), dim=0).values.tolist()
    # voiced
    out["voiced"] = {"mean": None, "median": None, "p90": None}
    return out

def fmt_row(label, vals):
    return f"{label:10s} | " + "  ".join(f"{v:6.3f}" if v is not None else "  None" for v in vals)



class DriftFieldConfig:
    temperature: float = 0.02
    normalize_over_x: bool = True
    mask_self_negatives: bool = False
    self_mask_value: float = 1e6
    eps: float = 1e-12

DRIFT_CONFIG = DriftFieldConfig()
    
def compute_affinity_matrices(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # _validate_inputs(
    #     x=x,
    #     y_pos=y_pos,
    #     y_neg=y_neg,
    #     negative_log_weights=negative_log_weights,
    #     generated_negative_count=generated_negative_count,
    # )

    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)

    generated_count = generated_negative_count if generated_negative_count is not None else y_neg.shape[0]
    if config.mask_self_negatives and generated_count > 0:
        diag_count = min(x.shape[0], generated_count, y_neg.shape[0])
        diagonal = torch.arange(diag_count, device=x.device)
        dist_neg = dist_neg.clone()
        dist_neg[diagonal, diagonal] = dist_neg[diagonal, diagonal] + config.self_mask_value

    logit_pos = -(dist_pos / config.temperature)
    logit_neg = -(dist_neg / config.temperature)
    if negative_log_weights is not None:
        logit_neg = logit_neg + negative_log_weights.view(1, -1)

    logits = torch.cat([logit_pos, logit_neg], dim=1)
    row_affinity = torch.softmax(logits, dim=-1)

    if config.normalize_over_x:
        col_affinity = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(row_affinity * col_affinity, min=config.eps))
    else:
        affinity = row_affinity

    n_pos = y_pos.shape[0]
    return affinity[:, :n_pos], affinity[:, n_pos:]
    
    
def compute_drift_components(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    affinity_pos, affinity_neg = compute_affinity_matrices(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    weight_pos = affinity_pos * affinity_neg.sum(dim=1, keepdim=True)
    weight_neg = affinity_neg * affinity_pos.sum(dim=1, keepdim=True)

    drift_pos = weight_pos @ y_pos
    drift_neg = weight_neg @ y_neg
    return drift_pos, drift_neg

def compute_v(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> torch.Tensor:
    drift_pos, drift_neg = compute_drift_components(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    return drift_pos - drift_neg


def drift_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    negative_log_weights: torch.Tensor | None = None):
    B = x.shape[0]
    x = x.reshape(B, -1)
    y_pos = y_pos.reshape(B, -1)
    v = compute_v(
        x=x,
        y_pos=y_pos,
        y_neg=x,
        config = DRIFT_CONFIG,
        negative_log_weights=negative_log_weights,
    )
    x_drifted = (x + v).detach()
    return ((x - x_drifted)**2).mean()