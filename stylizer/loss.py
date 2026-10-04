# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F

# ---------------------------
# Utilities: STFT & windows
# ---------------------------

def _hann(n: int, device=None, dtype=None):
    return torch.hann_window(n, periodic=True, device=device, dtype=dtype)

def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int, center: bool = True) -> torch.Tensor:
    """
    x: (B, T) waveform in [-1,1]
    returns |STFT| : (B, F, TT)
    """
    B, T = x.shape
    window = _hann(win, device=x.device, dtype=x.dtype)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                   window=window, center=center, return_complex=True)
    return torch.abs(X)  # (B, F, TT)

# ---------------------------
# Mel filter, MFCC (no deps)
# ---------------------------

def _hz_to_mel(hz: torch.Tensor, htk: bool = False):
    if htk:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)
    # Slaney
    f_0 = 0.0
    f_sp = 200.0 / 3
    brkfrq = 1000.0
    brkpt = (brkfrq - f_0) / f_sp
    logstep = math.log(6.4) / 27.0
    mel = torch.empty_like(hz)
    # linear below 1kHz, log above
    lin = hz < brkfrq
    mel[lin] = (hz[lin] - f_0) / f_sp
    mel[~lin] = brkpt + torch.log(hz[~lin] / brkfrq) / logstep
    return mel

def _mel_to_hz(mel: torch.Tensor, htk: bool = False):
    if htk:
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)
    # Slaney
    f_0 = 0.0
    f_sp = 200.0 / 3
    brkfrq = 1000.0
    brkpt = (brkfrq - f_0) / f_sp
    logstep = math.log(6.4) / 27.0
    hz = torch.empty_like(mel)
    lin = mel < brkpt
    hz[lin] = f_0 + f_sp * mel[lin]
    hz[~lin] = brkfrq * torch.exp(logstep * (mel[~lin] - brkpt))
    return hz

def build_mel_filter(sr: int, n_fft: int, n_mels: int,
                     fmin: float = 0.0, fmax: Optional[float] = None,
                     device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Returns Mel filter bank (F, n_mels) that maps linear magnitude/power to Mel.
    """
    if fmax is None: fmax = sr / 2
    # freq bins (only non-redundant)
    Fbins = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sr / 2, Fbins, device=device, dtype=dtype)
    # mel points
    m_min = _hz_to_mel(torch.tensor([fmin], device=device, dtype=dtype))
    m_max = _hz_to_mel(torch.tensor([fmax], device=device, dtype=dtype))
    m_pts = torch.linspace(m_min.item(), m_max.item(), n_mels + 2, device=device, dtype=dtype)
    f_pts = _mel_to_hz(m_pts)
    # create triangular filters
    fb = torch.zeros(Fbins, n_mels, device=device, dtype=dtype)
    for m in range(n_mels):
        f_l, f_c, f_u = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        # rising slope
        left = (fft_freqs - f_l) / (f_c - f_l + 1e-8)
        # falling slope
        right = (f_u - fft_freqs) / (f_u - f_c + 1e-8)
        fb[:, m] = torch.clamp(torch.minimum(left, right), min=0.0)
    # Slaney norm (area normalization)
    enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
    fb = fb * enorm.unsqueeze(0)
    return fb  # (F, n_mels)

def dct_matrix(n_mels: int, n_mfcc: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Orthonormal DCT-II matrix (n_mfcc, n_mels) to multiply log-mel (B, n_mels, T).
    """
    n = torch.arange(n_mels, device=device, dtype=dtype)
    k = torch.arange(n_mfcc, device=device, dtype=dtype).unsqueeze(1)
    M = torch.cos(math.pi / n_mels * (n + 0.5) * k)
    M[0] *= (1.0 / math.sqrt(2.0))
    M *= math.sqrt(2.0 / n_mels)
    return M  # (n_mfcc, n_mels)

# ---------------------------
# Band helpers (FFT & STFT)
# ---------------------------

def hz_to_fft_bin(hz: float, sr: int, n_fft: int) -> int:
    return int(round(hz * n_fft / sr))

def ensure_complex(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x): return x
    if x.dim() >= 3 and x.size(-3) == 2 and not torch.is_complex(x):
        return torch.complex(x.select(-3, 0), x.select(-3, 1))
    raise ValueError("Expected complex (B,F,T) or stacked real/imag (...,2,F,T).")

@torch.no_grad()
def stft_bandwise_ratios(
    Y_h: torch.Tensor,
    Y_n: torch.Tensor,
    bands_hz: List[Tuple[float, float]],
    sr: int,
    n_fft: int,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Returns r_band: (B, T, BANDS) where r = En_band / (Eh_band + En_band)
    """
    Y_h = ensure_complex(Y_h)
    Y_n = ensure_complex(Y_n)
    magsq_h = (Y_h.real**2 + Y_h.imag**2)
    magsq_n = (Y_n.real**2 + Y_n.imag**2)

    B, F, T = magsq_h.shape
    outs = []
    for (f_lo, f_hi) in bands_hz:
        lo = max(0, hz_to_fft_bin(f_lo, sr, n_fft))
        hi = min(F - 1, hz_to_fft_bin(f_hi, sr, n_fft))
        if hi < lo: lo = hi
        Eh = magsq_h[:, lo:hi+1, :].sum(dim=-2)  # (B,T)
        En = magsq_n[:, lo:hi+1, :].sum(dim=-2)
        r  = En / (Eh + En + eps)
        outs.append(r.unsqueeze(-1))
    return torch.cat(outs, dim=-1)  # (B,T,BANDS)

# ---------------------------
# Core loss pieces
# ---------------------------

def spectral_convergence(Y: torch.Tensor, Yhat: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Frobenius norm ratio per-example, then mean over batch & resolutions.
    Y, Yhat: (B, F, T)
    """
    num = torch.linalg.norm(Y - Yhat, ord="fro", dim=(1, 2))
    den = torch.linalg.norm(Y, ord="fro", dim=(1, 2)).clamp_min(eps)
    return (num / den).mean()

def log_mag_L1(Y: torch.Tensor, Yhat: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return F.l1_loss(torch.log(Yhat + eps), torch.log(Y + eps))

def mrstft_loss(
    y: torch.Tensor, yhat: torch.Tensor,
    resolutions: List[Tuple[int, int, int]],
    sc_weight: float = 0.5, logmag_weight: float = 0.5
) -> torch.Tensor:
    """
    resolutions: list of (n_fft, hop, win)
    """
    losses = []
    for (n_fft, hop, win) in resolutions:
        Y     = stft_mag(y,    n_fft, hop, win)  # (B,F,T)
        Y_hat = stft_mag(yhat, n_fft, hop, win)
        sc  = spectral_convergence(Y, Y_hat)
        lm  = log_mag_L1(Y, Y_hat)
        losses.append(sc_weight * sc + logmag_weight * lm)
    return sum(losses) / len(losses)

def mel_spectrogram_from_mag(mag: torch.Tensor, mel_fb: torch.Tensor, power: bool = False) -> torch.Tensor:
    """
    mag: (B, F, T) linear magnitude
    mel_fb: (F, n_mels)
    returns (B, n_mels, T)
    """
    X = mag.pow(2.0) if power else mag
    return torch.matmul(X.transpose(1, 2), mel_fb).transpose(1, 2)

def mel_loss_bandweighted(
    y: torch.Tensor, yhat: torch.Tensor,
    n_fft: int, hop: int, win: int,
    mel_fb: torch.Tensor,                 # (F, n_mels)
    mel_band_weights: torch.Tensor,       # (n_mels,) positive weights
    use_log_mel: bool = False,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Weighted MSE on (log-)Mel.
    """
    Y     = stft_mag(y,    n_fft, hop, win)
    Y_hat = stft_mag(yhat, n_fft, hop, win)
    Mel     = mel_spectrogram_from_mag(Y,     mel_fb)  # (B, M, T)
    Mel_hat = mel_spectrogram_from_mag(Y_hat, mel_fb)
    if use_log_mel:
        Mel     = torch.log(Mel + eps)
        Mel_hat = torch.log(Mel_hat + eps)
    w = mel_band_weights.view(1, -1, 1)
    return ((w * (Mel_hat - Mel) ** 2).mean())

def mfcc_envelope_loss(
    y: torch.Tensor, yhat: torch.Tensor,
    n_fft: int, hop: int, win: int,
    mel_fb: torch.Tensor,            # (F, n_mels)
    dct_mat: torch.Tensor,           # (n_mfcc, n_mels)
    n_mfcc_keep: int = 20,
    lifter: Optional[torch.Tensor] = None, # (n_mfcc_keep,)
    eps: float = 1e-7
) -> torch.Tensor:
    """
    MFCC from log-mel. Use only low-quefrency coeffs (envelope).
    """
    Y     = stft_mag(y,    n_fft, hop, win)
    Y_hat = stft_mag(yhat, n_fft, hop, win)
    Mel     = torch.matmul(mel_spectrogram_from_mag(Y,     mel_fb).transpose(1,2), dct_mat.T).transpose(1,2)
    Mel_hat = torch.matmul(mel_spectrogram_from_mag(Y_hat, mel_fb).transpose(1,2), dct_mat.T).transpose(1,2)
    # NOTE: dct_mat expects log-mel input; apply log before DCT
    LM     = torch.log(mel_spectrogram_from_mag(Y, mel_fb)     + eps)
    LM_hat = torch.log(mel_spectrogram_from_mag(Y_hat, mel_fb) + eps)
    MFCC     = torch.matmul(LM.transpose(1,2), dct_mat.T).transpose(1,2)      # (B, n_mfcc, T)
    MFCC_hat = torch.matmul(LM_hat.transpose(1,2), dct_mat.T).transpose(1,2)
    MFCC     = MFCC[:, :n_mfcc_keep, :]
    MFCC_hat = MFCC_hat[:, :n_mfcc_keep, :]
    if lifter is not None:
        MFCC     = MFCC * lifter.view(1, -1, 1)
        MFCC_hat = MFCC_hat * lifter.view(1, -1, 1)
    return F.l1_loss(MFCC_hat, MFCC)

# ---------------------------
# Noise-budget regularizer
# ---------------------------

@dataclass
class NoiseBudgetConfig:
    bands_hz: List[Tuple[float, float]]
    tau_per_band: List[float]           # target r for each band
    weight_per_band: Optional[List[float]] = None  # relative weight per band
    sr: int = 44100
    n_fft_for_budget: int = 1024
    voiced_only: bool = True

def noise_budget_loss(
    Yh: torch.Tensor, Yn: torch.Tensor,      # harmonic & noise STFTs (B,F,T) complex or (...,2,F,T)
    cfg: NoiseBudgetConfig,
    voiced_mask: Optional[torch.Tensor] = None,  # (B,T) bool/float
) -> torch.Tensor:
    r_band = stft_bandwise_ratios(Yh, Yn, cfg.bands_hz, cfg.sr, cfg.n_fft_for_budget)  # (B,T,K)
    K = r_band.shape[-1]
    tau = torch.tensor(cfg.tau_per_band, device=r_band.device, dtype=r_band.dtype).view(1,1,K)
    w   = torch.ones_like(tau) if cfg.weight_per_band is None else \
          torch.tensor(cfg.weight_per_band, device=r_band.device, dtype=r_band.dtype).view(1,1,K)
    err = w * (r_band - tau) ** 2  # (B,T,K)
    if cfg.voiced_only and (voiced_mask is not None):
        vm = (voiced_mask > 0.5).float().unsqueeze(-1)  # (B,T,1)
        num = (err * vm).sum()
        den = vm.sum().clamp_min(1.0)
        return num / den
    return err.mean()

# ---------------------------
# Master loss (compose all)
# ---------------------------

@dataclass
class DDSPComboLossConfig:
    # Mel
    mel_n_fft: int = 1024
    mel_hop: int = 256
    mel_win: int = 768
    use_log_mel: bool = False
    # MR-STFT
    mr_resolutions: Tuple[Tuple[int,int,int], ...] = ((2048,512,1536),(1024,256,768),(512,128,384),(256,64,192))
    # MFCC
    n_mels: int = 128
    n_mfcc: int = 40
    n_mfcc_keep: int = 20
    # Weights
    w_mel: float = 1.0
    w_mrstft: float = 0.25
    w_mfcc: float = 0.2
    w_noise_budget: float = 0.15  # start modest; tune 0.1–0.2
    # Sample rate & mel ranges
    sr: int = 44100
    fmin: float = 0.0
    fmax: Optional[float] = None

class DDSPComboLoss(torch.nn.Module):
    def __init__(self,
                 cfg: DDSPComboLossConfig,
                 mel_fb: Optional[torch.Tensor] = None,           # (F, n_mels) or None -> auto-build
                 dct_m: Optional[torch.Tensor] = None,            # (n_mfcc, n_mels) or None -> auto-build
                 mel_band_weights: Optional[torch.Tensor] = None  # (n_mels,) weights; if None -> build from bands
                 ):
        super().__init__()
        self.cfg = cfg
        # Build mel filters
        Fbins = cfg.mel_n_fft // 2 + 1
        if mel_fb is None:
            mel_fb = build_mel_filter(cfg.sr, cfg.mel_n_fft, cfg.n_mels, cfg.fmin, cfg.fmax,
                                      device="cpu", dtype=torch.float32)  # register later to device
        if dct_m is None:
            dct_m = dct_matrix(cfg.n_mels, cfg.n_mfcc, device="cpu", dtype=torch.float32)
        self.register_buffer("mel_fb", mel_fb)   # (F, M)
        self.register_buffer("dct_m", dct_m)     # (n_mfcc, M)
        # Default Mel band weights: emphasize LF/LowMid/Mid/Presence
        if mel_band_weights is None:
            # Define bands in Hz for weighting: LF, LowMid, Mid, Presence, Air
            # bands = [(0.,200.), (200.,800.), (800.,3000.), (3000.,6000.), (6000., 16000.)]
            # Get mel bin center freqs to map bands→mel bins (rough)
            # m_edges = torch.linspace(0, cfg.n_mels-1, cfg.n_mels)
            # Approximate mel centers via inverse of evenly spaced mel points:
            m_min = _hz_to_mel(torch.tensor([cfg.fmin if cfg.fmin else 0.0]))
            m_max = _hz_to_mel(torch.tensor([cfg.fmax if cfg.fmax else cfg.sr/2]))
            mel_points = torch.linspace(m_min.item(), m_max.item(), cfg.n_mels)
            mel_hz = _mel_to_hz(mel_points)
            w = torch.ones(cfg.n_mels)
            def apply_w(flo, fhi, mult):
                mask = (mel_hz >= flo) & (mel_hz < fhi)
                w[mask] = w[mask] * mult
            apply_w(0., 200.,    2.0)   # LF
            apply_w(200., 800.,  2.0)   # LowMid
            apply_w(800., 3000., 2.0)   # Mid
            apply_w(3000., 6000.,1.5)   # Presence
            apply_w(6000., 16000.,1.0)  # Air
            mel_band_weights = w
        self.register_buffer("mel_band_w", mel_band_weights)

        # Default NoiseBudget config (can override at call)
        self.default_nb_cfg = NoiseBudgetConfig(
            bands_hz=[(0.,200.), (200.,800.), (800.,3000.), (3000.,6000.), (6000.,16000.)],
            tau_per_band=[0.22,   0.22,       0.30,         0.35,          0.50],
            weight_per_band=[1.5, 1.5,        2.0,          1.2,           0.8],
            sr=cfg.sr,
            n_fft_for_budget=1024,
            voiced_only=True
        )

    def forward(self,
                y: torch.Tensor, yhat: torch.Tensor,                 # (B,T) waveforms
                Yh: torch.Tensor, Yn: torch.Tensor,                  # (B,F,T) complex STFTs (harmonic/noise)
                voiced_mask: Optional[torch.Tensor] = None,          # (B,T) bool/float
                nb_cfg: Optional[NoiseBudgetConfig] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with individual terms and total.
        """
        cfg = self.cfg
        mel_fb = self.mel_fb.to(y.device, y.dtype)
        dct_m  = self.dct_m.to(y.device, y.dtype)
        mel_w  = self.mel_band_w.to(y.device, y.dtype)

        # Mel
        L_mel = mel_loss_bandweighted(
            y, yhat,
            cfg.mel_n_fft, cfg.mel_hop, cfg.mel_win,
            mel_fb, mel_w, use_log_mel=cfg.use_log_mel
        )

        # MFCC envelope
        L_mfcc = mfcc_envelope_loss(
            y, yhat,
            cfg.mel_n_fft, cfg.mel_hop, cfg.mel_win,
            mel_fb, dct_m,
            n_mfcc_keep=cfg.n_mfcc_keep
        )

        # MR-STFT
        L_mrstft = mrstft_loss(y, yhat, list(cfg.mr_resolutions), sc_weight=0.5, logmag_weight=0.5)

        # Noise budget (voiced-only by default)
        nb_cfg = nb_cfg or self.default_nb_cfg
        L_nb = noise_budget_loss(Yh, Yn, nb_cfg, voiced_mask=voiced_mask)

        total = (cfg.w_mel * L_mel
                 + cfg.w_mfcc * L_mfcc
                 + cfg.w_mrstft * L_mrstft
                 + cfg.w_noise_budget * L_nb)

        return {
            "total": total,
            "mel": L_mel.detach(),
            "mfcc": L_mfcc.detach(),
            "mrstft": L_mrstft.detach(),
            "noise_budget": L_nb.detach()
        }
