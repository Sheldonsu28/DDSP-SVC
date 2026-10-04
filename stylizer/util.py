import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from reflow.lynxnet2 import LYNXNet2
from reflow.reflow import RectifiedFlow

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0, **kwargs)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.padding, 0)))

class ConformerBlock(nn.Module):
    def __init__(self, dim_model=768, ff_multiplier=4.0, conv_kernel_size=31, num_heads=8, dropout=0.1, local_window=12, useCasualConv=False):
        super().__init__()

        self.local_window = local_window

        ff_dim = int(dim_model * ff_multiplier)

        # Feedforward Module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim_model),
            nn.Dropout(dropout)
        )

        # Multi-Head Self Attention
        self.attn_norm = nn.LayerNorm(dim_model)
        self.attn = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout, batch_first=True)

        # Convolution Module
        self.conv_norm = nn.LayerNorm(dim_model)
        if useCasualConv:
            self.conv = nn.Sequential(
                CausalConv1d(dim_model, 2 * dim_model, kernel_size=1),
                nn.GLU(dim=1),
                CausalConv1d(dim_model, dim_model, kernel_size=conv_kernel_size, groups=dim_model),
                nn.BatchNorm1d(dim_model),
                nn.SiLU(),
                CausalConv1d(dim_model, dim_model, kernel_size=1),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(dim_model, 2 * dim_model, kernel_size=1),
                nn.GLU(dim=1),
                nn.Conv1d(dim_model, dim_model, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=dim_model),
                # CausalConv1d(dim_model, dim_model, kernel_size=conv_kernel_size, groups=dim_model),
                nn.BatchNorm1d(dim_model),
                nn.SiLU(),
                nn.Conv1d(dim_model, dim_model, kernel_size=1),
            )
        self.conv_dropout = nn.Dropout(dropout)

        # Feedforward Module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim_model),
            nn.Dropout(dropout)
        )

        self.final_norm = nn.LayerNorm(dim_model)
        
    def _make_local_mask(self, seq_len, window_size, device):
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = 0.0
        return mask

    def forward(self, x):
        B, T, D = x.size()
        # Feedforward 1
        x = x + 0.5 * self.ff1(x)

        # Multi-head self-attention with local mask
        attn_input = self.attn_norm(x)
        attn_mask = self._make_local_mask(T, self.local_window, x.device)  # (T, T)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=attn_mask)
        x = x + attn_out

        # Convolution
        conv_input = self.conv_norm(x).transpose(1, 2)  # (B, D, T)
        conv_out = self.conv(conv_input).transpose(1, 2)  # (B, T, D)
        x = x + self.conv_dropout(conv_out)

        # Feedforward 2
        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)

class ConformerBlock(nn.Module):
    def __init__(self, dim_model=768, ff_multiplier=4.0, conv_kernel_size=31, num_heads=8, dropout=0.1, local_window=12, causal=False):
        super().__init__()

        self.local_window = local_window
        self.casual = causal

        ff_dim = int(dim_model * ff_multiplier)

        # Feedforward Module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim_model),
            nn.Dropout(dropout)
        )

        # Multi-Head Self Attention
        self.attn_norm = nn.LayerNorm(dim_model)
        self.attn = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout, batch_first=True)

        # Convolution Module
        self.conv_norm = nn.LayerNorm(dim_model)
    
        self.conv = nn.Sequential(
            nn.Conv1d(dim_model, 2 * dim_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim_model, dim_model, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=dim_model) if not causal else CausalConv1d(dim_model, dim_model, kernel_size=conv_kernel_size, groups=dim_model),
            # CausalConv1d(dim_model, dim_model, kernel_size=conv_kernel_size, groups=dim_model),
            nn.BatchNorm1d(dim_model),
            nn.SiLU(),
            nn.Conv1d(dim_model, dim_model, kernel_size=1),
        )
        self.conv_dropout = nn.Dropout(dropout)

        # Feedforward Module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim_model),
            nn.Dropout(dropout)
        )

        self.final_norm = nn.LayerNorm(dim_model)
        
    def _make_local_mask(self, seq_len, window_size, device):
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = False
        return mask

    def forward(self, x, cond=None):
        B, T, D = x.size()
        # Feedforward 1
        x = x + 0.5 * self.ff1(x)

        # Multi-head self-attention with local mask
        attn_input = self.attn_norm(x)
        if not self.casual:
            attn_mask = self._make_local_mask(T, self.local_window, x.device)  # (T, T)
        else:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(sz=T)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=attn_mask, is_causal=self.casual, need_weights=False)
        x = x + attn_out

        # Convolution
        conv_input = self.conv_norm(x).transpose(1, 2)  # (B, D, T)
        conv_out = self.conv(conv_input).transpose(1, 2)  # (B, T, D)
        x = x + self.conv_dropout(conv_out)

        # Feedforward 2
        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)

    
def kl_budget_loss(mu, logvar, target=1.0, scale=1.0):
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # (B, T, D)
    # kl = kl.sum(dim=-1).mean()  # average over time and batch
    excess = (kl - target)
    return scale * (excess ** 2)


def simple_kl_loss(post_mu, post_logvar, prior_mu, prior_logvar):
    """
    Calculates KL( Posterior || Learned_Prior )
    All inputs shape: [Batch, Dim, Length]
    """
    # 1. Variance term
    # log(var2 / var1) = log_var2 - log_var1
    # size : [B, L, D]
    # print(f"LogVar Mean: {post_logvar.mean().item()}")
    var_term = prior_logvar - post_logvar  
    
    # 2. Mean shift term
    mean_term = (post_mu - prior_mu).pow(2) / torch.exp(prior_logvar)
    
    # 3. Trace term
    trace_term = torch.exp(post_logvar - prior_logvar)
    
    # --- THE FIX IS HERE ---
    # Formula: 0.5 * ( log(var2/var1) - 1 + var1/var2 + (m1-m2)^2/var2 )
    # Note: We use +var_term, because var_term is already log(var2) - log(var1)
    
    kl_element_wise = 0.5 * (var_term + mean_term + trace_term - 1)
    
    return kl_element_wise.sum(dim=2).mean()


def vae_kl_loss(mu, logvar):
    """
    Calculates KL( q(z|x) || N(0, 1) )
    
    Args:
        mu (Tensor): Mean predicted by encoder
        logvar (Tensor): Log-Variance predicted by encoder
    """
    # The "Reference" is N(0, 1), which allows this specific simplification:
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Usually we average over the batch
    return kl_div.mean()


def mod_kl_loss1_stable(
    z_p,          # [B, D, L] samples AFTER flow (forward) or BEFORE (if you're doing the inverse term)
    logs_q,       # [B, D, L] log-std of the "q" (source) Gaussian
    m_p, logs_p,  # [B, D, L] mean & log-std of the "p" (target) Gaussian
    total_logdet, # [B] log|det J| for the SAME direction that produced z_p
    z_mask,       # [B, D, L] 1/0 mask
    logdet_is_forward: bool = True
):
    """
    Monte-Carlo KL with change-of-variables correction, mean-reduced and mask-aware.

    We compute (per element):
      term = (logs_p - logs_q - 0.5) + 0.5 * ((z_p - m_p)^2) * exp(-2*logs_p)
    and average it over masked elements. This corresponds (up to constants) to
    E_q[...] for the Gaussian pieces. Then we add/subtract a *mean-per-element*
    logdet term so its scale matches.
    """
    # Casts
    z_p    = z_p.float()
    logs_q = logs_q.float()
    m_p    = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    # Per-element gaussian part
    elem = (logs_p - logs_q - 0.5) + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    elem = elem * z_mask

    # Mean over masked elements to keep scale stable
    denom = torch.clamp(z_mask.sum(), min=1.0)
    kl_mean = elem.sum() / denom  # scalar

    # Normalize logdet to per-element mean so it’s commensurate
    # total_logdet: [B] -> mean over batch
    logdet_batch_mean = total_logdet.mean()

    # Count masked elements per batch on average (avoid per-batch variance in scale)
    B = z_p.shape[0]
    elems_per_batch_mean = denom / B

    # per-element mean logdet
    logdet_per_elem_mean = logdet_batch_mean / torch.clamp(elems_per_batch_mean, min=1.0)

    # Change-of-vars: if total_logdet is forward (for z_fwd), we SUBTRACT it.
    # If it's inverse (for z_bkw with inverse mapping), we ADD it (since inverse logdet is -forward).
    if logdet_is_forward:
        loss = kl_mean - logdet_per_elem_mean
    else:
        loss = kl_mean + logdet_per_elem_mean

    return loss


def kl_loss_new(z_p, logs_q, m_p, logs_p, total_logdet, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    total_logdet: [b] - total_logdet summed over each batch
    """
    # B, L, _ = logs_p.shape
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
    kl = logs_p - logs_q - 0.5
    
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    # add total_logdet (Negative LL)
    kl -= torch.sum(total_logdet)
    l = kl / torch.sum(z_mask)
    return l


_LOG2PI = 1.8378770664093453  # log(2π)

def log_diag_normal_sum(z, mu, logs, mask=None):
    """
    Log N(z; mu, diag(exp(2*logs))) summed over (D,L), per sample.
    z, mu, logs: [B, L, D] or [B, D, L] (must match exactly).
    mask (optional): same shape as z, with 1/0.
    Returns: [B]
    """
    # ensure same dtype/device
    z, mu, logs = z.float(), mu.float(), logs.float()
    elem = -logs - 0.5 * _LOG2PI - 0.5 * ((z - mu) ** 2) * torch.exp(-2.0 * logs)
    if mask is not None:
        elem = elem * mask.float()
    return elem.sum(dim=(1, 2))

def count_mask(mask):
    # number of elements per sample for normalization
    return mask.float().sum(dim=(1, 2)).clamp(min=1.0)

def flow_kl_forward_mc(z0, z, mu_q, logs_q, mu_p, logs_p, logdet_fwd, mask=None):
    """
    KL(q_f || p) Monte-Carlo:
      E_{z0~q0}[ log q0(z0) - log p(z) - log|det J_fwd(z0)| ]
    All tensors are [B, L, D]. mask is [B, L, D] of 1/0.
    Returns scalar (average per masked element).
    """
    log_q0 = log_diag_normal_sum(z0, mu_q, logs_q, mask)     # [B]
    log_p  = log_diag_normal_sum(z,  mu_p, logs_p,  mask)     # [B]
    # log q_f(z) = log q0(z0) - log|det J_fwd(z0)|
    per_sample = (log_q0 - log_p - logdet_fwd)                # [B]
    denom = count_mask(mask)                                  # [B]
    per_elem = per_sample / denom
    return per_elem.mean()

def flow_kl_inverse_mc(y, z_inv, mu_p, logs_p, mu_q, logs_q, logdet_inv, mask=None):
    """
    KL(p || q_f) Monte-Carlo:
      E_{y~p}[ log p(y) - (log q0(z_inv) + log|det J_inv(y)|) ]
    where z_inv = f^{-1}(y).
    """
    log_p  = log_diag_normal_sum(y,     mu_p, logs_p,  mask)  # [B]
    log_q0 = log_diag_normal_sum(z_inv, mu_q, logs_q, mask)   # [B]
    # log q_f(y) = log q0(z_inv) + log|det J_inv(y)|
    per_sample = (log_p - (log_q0 + logdet_inv))              # [B]
    denom = count_mask(mask)
    per_elem = per_sample / denom
    return per_elem.mean()


def kl_div_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def nll_loss(
    z_0: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    log_det_j: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the Negative Log-Likelihood (NLL) for a Glow-style model.

    Args:
        z_0 (torch.Tensor): The latent vector after the inverse flow.
        m_p (torch.Tensor): Mean of the prior distribution.
        logs_p (torch.Tensor): Log-variance of the prior distribution.
        log_det_j (torch.Tensor): The log-determinant from the inverse flow.
    """
    z_0, m_p, logs_p = z_0.float(), m_p.float(), logs_p.float()

    # NLL of the latent z_0 under the prior Gaussian N(m_p, logs_p)
    nll_prior = 0.5 * (math.log(2 * math.pi) + 2 * logs_p + ((z_0 - m_p)**2) * torch.exp(-2.0 * logs_p))
    nll_prior = torch.sum(nll_prior, dim=[1, 2])
    nll_prior = torch.mean(nll_prior)
    
    # The log-determinant from the flow
    log_det_loss = torch.mean(log_det_j)
    
    # Total NLL = NLL of prior - log_det_j
    total_loss = nll_prior - log_det_loss

    return total_loss



def mod_kl_loss1(z_p, logs_q, m_p, logs_p, total_logdet, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    total_logdet: [b] - total_logdet summed over each batch
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    # add total_logdet (Negative LL)
    kl -= torch.sum(total_logdet)
    l = kl / torch.sum(z_mask)
    return l



def vits_kl_loss(z_p, logs_q, m_p, logs_p, total_logdet):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    total_logdet: [b] - total_logdet summed over each batch
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return (torch.sum(kl) - torch.sum(total_logdet)) / z_p.numel()


def vits_kl_loss2(z_p, logs_q, m_p, logs_p, total_logdet, useTopK=True, alpha=1.0):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    total_logdet: [b] - total_logdet summed over each batch
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    
    if useTopK:
        with torch.no_grad():
            # print(loss_map.shape)
            mu = kl.mean()
            sigma = kl.std()

            # Dynamic Threshold
            threshold = mu + (alpha * sigma)
        
            # Create Boolean Mask
            masks = kl > threshold
        if masks.sum() > 0:
            # Mean of only the "hard" pixels
            masked_loss = kl[masks].mean()
    else:
        # Fallback for perfect batches (avoid NaN)
        masked_loss = torch.tensor(0.0, device=kl.device, dtype=kl.dtype)
    
    return kl.mean() + masked_loss
    

def mod_kl_loss(z_p, logs_q, m_p, logs_p, total_logdet, target=1.5, weight=1.):
    z_p, logs_q, m_p, logs_p = z_p.float(), logs_q.float(), m_p.float(), logs_p.float()

    # 1. Calculate the actual KL divergence (mean per-dimension)
    kl_elementwise = logs_p - logs_q - 0.5
    kl_elementwise += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    mean_kl_div = torch.mean(kl_elementwise)

    # 2. Calculate the primary VITS loss component (KL - log_det)
    log_det_loss = torch.mean(total_logdet)
    primary_loss = mean_kl_div - log_det_loss
    # print(mean_kl_div, log_det_loss)
    # 3. Calculate the budget loss
    budget_loss = (mean_kl_div - target).pow(2)
    # print(mean_kl_div, budget_loss, primary_loss)
    # 4. Combine for the final total loss and return
    total_loss = primary_loss + (weight * budget_loss)
    # print(primary_loss, budget_loss)
    return total_loss

def mod_kl_loss2(mu_q, logvar_q, mu_p, logvar_p, total_logdet, target=0.55, weight=1., use_inverse_logdet=True):
    """
    Closed-form KL(q||p) where q = N(mu_q, diag(var_q)), p = N(mu_p, diag(var_p)),
    with logvar_* = log(var_*). total_logdet is typically the *inverse* log-det
    reported by your flow layer (see Option A you tried).
    """
    # convert to vars
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    # per-dim KL, then mean over all dims/time/batch (same reduction you had)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    mean_kl_div = kl.mean()

    # logdet convention
    log_det_term = total_logdet.mean()
    primary_loss = mean_kl_div - log_det_term if use_inverse_logdet else mean_kl_div + log_det_term

    budget_loss = mean_kl_div
    return primary_loss + weight * budget_loss

def kl_loss_between_gaussians(mu_r, logvar_r, mu_s, logvar_s):
    """
    KL( N(mu_r, var_r) || N(mu_s, var_s) )
    All inputs: (batch, time, latent_dim)
    """
    return 0.5 * torch.mean(
        logvar_s - logvar_r                                      # log(σ_s / σ_r)
        + (logvar_r.exp() + (mu_r - mu_s).pow(2)) / logvar_s.exp()  # (σ_r² + (μ_r-μ_s)²) / σ_s²
        - 1
    )


def close_form_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence: KL( q(z|x) || N(0,I) ).
    Averaged over batch and latent dims so it scales consistently
    regardless of latent_dim or batch size.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def mel_centroid(mel_mag, mel_freqs_hz):
    num = (mel_mag * mel_freqs_hz.view(1, 1, -1)).sum(dim=1)  # [B,T]
    den = mel_mag.sum(dim=1).clamp_min(1e-8)
    return num / den

def dynamic_range_decompression_torch(x, C=1):
    s = torch.exp(x) / C
    return torch.exp(x) / C

def freq_smooth_mel(mel, k=9):
    # mel: [B,M,T]
    mel_t = mel.transpose(1,2)  # [B,T,M]
    env_t = torch.nn.functional.avg_pool1d(mel_t, kernel_size=k, stride=1, padding=k//2)
    return env_t.transpose(1,2) 

def centroid_loss_from_mels(y_mel_mag, x_mel_mag,mel_freqs_hz,
                            voiced_mask=None, mode="L1", weight=1.0):
    cy = mel_centroid(dynamic_range_decompression_torch(y_mel_mag), mel_freqs_hz)
    cx = mel_centroid(dynamic_range_decompression_torch(x_mel_mag), mel_freqs_hz)
    x_mel = freq_smooth_mel(dynamic_range_decompression_torch(x_mel_mag))
    y_mel = freq_smooth_mel(dynamic_range_decompression_torch(y_mel_mag))
    cy = (y_mel * mel_freqs_hz.view(1,1,-1)).sum(1) / y_mel.sum(1).clamp_min(1e-8)  # [B,T]
    cx = (x_mel * mel_freqs_hz.view(1,1,-1)).sum(1) / x_mel.sum(1).clamp_min(1e-8)

    energy = x_mel.sum(1)  # [B,T]
    mask = (energy > 1e-6).float()

    mae_hz = (cy - cx).abs()
    mae_hz = (mae_hz * mask).sum() / mask.sum().clamp_min(1.0)
    
    assert (x_mel >= 0).all() and (y_mel >= 0).all()

    # 1) Are x_mel and y_mel literally the same tensor?
    print("ptr x:", x_mel.data_ptr(), "ptr y:", y_mel.data_ptr())

    # 2) How different are they?
    diff_mean = (y_mel - x_mel).abs().mean().item()
    diff_max  = (y_mel - x_mel).abs().max().item()
    print(f"mel |Δ| mean={diff_mean:.6g}, max={diff_max:.6g}")

    # 3) Are waveforms identical?
    if 'x' in locals() and 'y' in locals():
        print("wave allclose:", torch.allclose(x, y))

    # 4) Is y_mel just a scaled version of x_mel (per frame)?
    B, M, T = x_mel.shape
    scale = (y_mel.sum(1) / x_mel.sum(1).clamp_min(1e-8))        # [B,T]
    y_hat = x_mel * scale.unsqueeze(1)
    shape_err = (y_mel - y_hat).abs().mean().item()
    print(f"per-frame scaling shape_err mean={shape_err:.6g}") 

    # print({
    #     "centroid_gt_mean_Hz": float((cx*mask).sum()/mask.sum()),
    #     "centroid_gen_mean_Hz": float((cy*mask).sum()/mask.sum()),
    #     "centroid_MAE_Hz": float(mae_hz),
    #     "mel_freqs_max": float(mel_freqs_hz.max()),
    #     "mel_is_linear": bool(y_mel.min() >= 0 and y_mel.max() > 1e-6),
    # })
    if mode.upper() == "L2":
        d = (cy - cx)**2
    else:
        d = (cy - cx).abs()
    if voiced_mask is not None:
        d = d * voiced_mask  # [B,T]
        loss = d.sum() / voiced_mask.sum().clamp_min(1.0)
    else:
        loss = d.mean()
    return weight * loss

import torch

def spectral_tilt_loss(mel_bt_m, mel_freqs_hz):
    """
    Compute spectral tilt (slope of log magnitude vs log frequency) per frame.

    Args:
        mel_bt_m: [B, T, M] linear mel magnitudes
        mel_freqs_hz: [M] mel bin center freqs in Hz

    Returns:
        slope: [B, T] per-frame tilt value
    """
    eps = 1e-8
    B, T, M = mel_bt_m.shape

    # log frequency axis
    logf = torch.log(mel_freqs_hz.clamp_min(30.0))  # [M]
    logf = (logf - logf.mean()) / (logf.std() + eps)  # normalize

    # normalize weights per frame
    w = mel_bt_m / mel_bt_m.sum(dim=-1, keepdim=True).clamp_min(eps)  # [B,T,M]

    logm = torch.log(mel_bt_m + eps)  # [B,T,M]

    # expectations
    Ew_logf   = (w * logf.view(1,1,M)).sum(-1)         # [B,T]
    Ew_logm   = (w * logm).sum(-1)                     # [B,T]
    Ew_logflm = (w * logf.view(1,1,M) * logm).sum(-1)  # [B,T]

    var_logf  = (w * (logf.view(1,1,M)**2)).sum(-1) - Ew_logf**2
    cov       = Ew_logflm - Ew_logf * Ew_logm

    slope     = cov / var_logf.clamp_min(eps)  # [B,T]
    return slope


# def kl_loss1(z_p, logs_q, m_p, logs_p, total_logdet):
#     """
#     z_p, logs_q: [b, h, t_t]
#     m_p, logs_p: [b, h, t_t]
#     total_logdet: [b] - total_logdet summed over each batch
#     """
#     # print('================')
#     # print(torch.isnan(z_p).any(), torch.isnan(logs_q).any(), torch.isnan(m_p).any(), torch.isnan(logs_p).any(), torch.isnan(total_logdet).any())
#     # z_p = z_p.float()
#     logs_q = logs_q.float()
#     m_p = m_p.float()
#     logs_p = logs_p.float()
#     # kl = torch.clip(logs_p, max=1e2, min=-1e2) -  torch.clip(logs_q, max=1e2, min=-1e2) - 0.5
#     kl = logs_p - logs_q - 0.5
#     # print(torch.isnan(kl).any())
#     kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
#     # print(torch.isnan(kl).any())
#     # kl = torch.mean(torch.clip(kl, max=1e2, min=-1e2))
#     kl = torch.mean(kl)
#     # print(torch.isnan(kl).any())
#     # add total_logdet (Negative LL)
#     kl -= torch.mean(total_logdet)
#     # print(torch.isnan(kl).any())
#     l = kl
#     return l

import torch
from torch.distributions import Normal, kl_divergence

def kl_loss_distributions(mu1, std, mu2, std2):
    """
    Computes the KL divergence between two diagonal Gaussian distributions.
    
    Args:
        mu1, var1: Mean and variance tensors for distribution P. Shape: [B, D]
        mu2, var2: Mean and variance tensors for distribution Q. Shape: [B, D]
        
    Returns:
        kl_loss: KL divergence per row. Shape: [B]
    """
    # PyTorch's Normal distribution takes standard deviation, not variance
    # Add a small epsilon to prevent taking the sqrt of zero or negative numbers
    eps = 1e-8
    p = Normal(mu1, std)
    q = Normal(mu2, std2)
    
    # kl_divergence computes element-wise KL of shape [B, D]
    # We sum over the D dimension (dim=1) to get shape [B]
    return kl_divergence(p, q).mean()

def kl_loss2(z_p, logs_q, m_p, logs_p, total_logdet):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    total_logdet: [b] - total_logdet summed over each batch
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    z_mask = torch.ones_like(kl)
    kl = torch.sum(kl * z_mask)
    # add total_logdet (Negative LL)
    kl -= torch.sum(total_logdet)
    l = kl / torch.sum(z_mask)
    return l

def vits_kl_loss_with_free_bits(z_ps, logs_q, m_p, logs_p, total_logdet, kl_budget=1.0, budget_weight=1.0):
    """
    VITS-style KL loss with a "free bits" budget.

    Args:
        z_ps (Tensor): Latent sample from the posterior encoder q(z|mel).
        logs_q (Tensor): Log variance of the posterior.
        m_p (Tensor): Mean of the prior encoder p(z|content).
        logs_p (Tensor): Log variance of the prior.
        total_logdet (Tensor): The log determinant from the normalizing flow.
        kl_budget (float): The "free bits" budget. KL divergence below this is not penalized.
        budget_weight (float): A weight for the KL term.
    """
    # Ensure all tensors are float for calculation
    z_ps, logs_q, m_p, logs_p = z_ps.float(), logs_q.float(), m_p.float(), logs_p.float()

    # 1. Calculate the actual KL divergence KL(q || p)
    # This measures the difference between the posterior (from mel) and the prior (from content)
    kl_elementwise = (logs_p - logs_q - 0.5) + (0.5 * ((z_ps - m_p)**2) * torch.exp(-2.0 * logs_p))
    mean_kl_div = torch.mean(kl_elementwise)

    # 2. Apply the "free bits" budget
    # We only apply a loss if the KL divergence exceeds our budget.
    # This prevents posterior collapse while still encouraging efficiency.
    kl_loss = torch.maximum(torch.tensor(0.), mean_kl_div - kl_budget)

    # 3. Calculate the VITS flow loss component
    # This encourages the flow to be volume-preserving and helps align the distributions.
    log_det_loss = torch.mean(total_logdet)

    # 4. Combine for the final loss
    # The goal is to minimize the KL (above the budget) and maximize the log_det (which is minimizing -log_det)
    total_loss = budget_weight * (kl_loss - log_det_loss)
    
    return total_loss # Return mean_kl_div for monitoring


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    return 0.5 * ((logvar_p - logvar_q) + (torch.exp(logvar_q) + (mu_q - mu_p)**2) * torch.exp(-logvar_p) - 1.0).mean()

def compute_flow_loss(
    mu_pr, logvar_pr,    # → prior params p(z|c)
    mu_ps, logvar_ps,    # → posterior params q(z|x)
    z_fwd, log_det_fwd,  # → forward flow outputs
    z_bkw, log_det_bkw,  # → inverse flow outputs
    target_kl=1.5,
    kl_weight=1.0
):
    """
    Returns:
      loss_total:   combined forward + reverse NLL + KL‐budget
      stats:       dict of individual components for logging
    """
    # B = z_fwd.shape[0]

    # 1) Forward (data → latent) NLL
    #    prior is N(0,1): log p(z) = -0.5*(z^2 + log(2π))
    # clamp_logdet = 10.0
    # log_det_fwd = torch.clamp(log_det_fwd, -clamp_logdet, clamp_logdet)
    # log_det_bkw = torch.clamp(log_det_bkw, -clamp_logdet, clamp_logdet)
    prior_ll_fwd = -0.5 * (z_fwd.pow(2).sum(dim=[1,2]) + z_fwd[0].numel() * math.log(2 * math.pi))
    nll_fwd     = -(prior_ll_fwd + log_det_fwd).mean()

    # 2) Reverse (latent → data) NLL
    #    prior p(z|c) = N(mu_pr, var=exp(logvar_pr))
    #    so log p(z_bkw) = -0.5 * [ (z - mu)^2/σ^2 + logvar + log(2π) ]
    diff2 = (z_bkw - mu_pr).pow(2)
    prior_ll_bkw = -0.5 * ((diff2 * torch.exp(-logvar_pr) + logvar_pr + math.log(2*math.pi))
                           .sum(dim=[1,2]))
    nll_bkw = -(prior_ll_bkw + log_det_bkw).mean()

    # 3) KL budgets (optional regularization)
    kl_fwd = gaussian_kl(mu_q=mu_ps, logvar_q=logvar_ps, mu_p=mu_pr, logvar_p=logvar_pr)
    kl_bkw = gaussian_kl(mu_q=mu_pr, logvar_q=logvar_pr, mu_p=mu_ps, logvar_p=logvar_ps)

    budget_fwd = (kl_fwd - target_kl).pow(2)
    budget_bkw = (kl_bkw - target_kl).pow(2)

    # 4) Total loss
    loss_total = 0.00005 * (nll_fwd + nll_bkw) + kl_weight * (budget_fwd + budget_bkw)

    # stats = {
    #     "nll_fwd": nll_fwd.item(),
    #     "nll_bkw": nll_bkw.item(),
    #     "kl_fwd":  kl_fwd.item(),
    #     "kl_bkw":  kl_bkw.item(),
    #     "budget_fwd": budget_fwd.item(),
    #     "budget_bkw": budget_bkw.item(),
    #     "loss_total": loss_total.item()
    # }
    # print( stats)
    return loss_total


# def mod_kl_loss2(z_p, logs_q, m_p, logs_p, target=1.5, weight=1.):
#     z_p, logs_q, m_p, logs_p = z_p.float(), logs_q.float(), m_p.float(), logs_p.float()

#     # 1. Calculate the actual KL divergence (mean per-dimension)
#     kl_elementwise = logs_p - logs_q - 0.5
#     kl_elementwise += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
#     mean_kl_div = torch.mean(kl_elementwise)

#     # 2. Calculate the primary VITS loss component (KL - log_det)
#     # log_det_loss = torch.mean(total_logdet)
#     # primary_loss = (mean_kl_div - log_det_loss)
#     # print(mean_kl_div, log_det_loss)
#     # 3. Calculate the budget loss
#     budget_loss = (mean_kl_div - target).pow(2)
#     # print(mean_kl_div, budget_loss, primary_loss)
#     # 4. Combine for the final total loss and return
#     total_loss = (weight * budget_loss)
#     # print(primary_loss, budget_loss)
#     return total_loss
    

def gaussian_deviation_loss(input, mean_vector, sigma_vector, reduction='mean', factor=0):
    """
    Penalizes deviation from a mean vector using variance (sigma^2).

    Args:
        input (torch.Tensor): Input tensor of shape [B, D]
        mean_vector (torch.Tensor): Mean vector of shape [D]
        sigma_vector (torch.Tensor): Variance vector (sigma^2), shape [D]
        reduction (str): 'mean', 'sum', or 'none'

    Returns:
        torch.Tensor: Loss value (scalar or vector depending on reduction)
    """
    # Compute squared deviation normalized by variance
    deviation = (input - mean_vector) ** 2
    normalized = deviation / (sigma_vector + 1e-8)

    # Sum over feature dimension
    loss = torch.sum(normalized, dim=-1)

    # Apply reduction
    if reduction == 'mean':
        return loss.mean() * factor
    elif reduction == 'sum':
        return loss.sum() * factor
    else:
        return loss * factor  # shape [B]
    
def create_reflow(version=0):
    if version == 1:
        return RectifiedFlow(LYNXNet2(in_dims=768, dim_cond=192, n_layers=4, n_chans=896, kernel_size=5, use_wn=True, lite=False), out_dims=768, train_embed=True)
    if version == 2:
        return RectifiedFlow(LYNXNet2(in_dims=768, dim_cond=256, n_layers=4, n_chans=896, kernel_size=5, use_wn=True, lite=False), out_dims=768, train_embed=True)
    return RectifiedFlow(LYNXNet2(in_dims=768, dim_cond=768, n_layers=4, n_chans=896, kernel_size=5, use_wn=True, lite=False), out_dims=768, train_embed=True)


def create_post_processor():
    return RectifiedFlow(LYNXNet2(in_dims=128, dim_cond=128, n_layers=6, n_chans=1024, kernel_size=3, use_wn=False, lite=False), out_dims=128, train_embed=False)


def random_rotation_scale(z, max_angle=0.05, scale_range=(0.95, 1.05)):
    # z: [B, T, D]
    B, T, D = z.shape
    R = torch.eye(D, device=z.device).unsqueeze(0).repeat(B, 1, 1)
    # create small skew-symmetric random matrices
    A = torch.randn(B, D, D, device=z.device)
    K = A - A.transpose(1, 2)
    R = torch.matrix_exp(max_angle * K)
    s = torch.empty(B, 1, 1, device=z.device).uniform_(*scale_range)
    z_aug = torch.bmm(z, R.transpose(1, 2)) * s
    return z_aug, R, s

import torch


def random_piecewise_time_warp(
    x: torch.Tensor,
    zone_size: int = 50,
    max_frame_offset: float = 5.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Piecewise-linear random time warp.

    Each full `zone_size` interval has its boundary randomly shifted by
    Uniform(-max_frame_offset, +max_frame_offset).

    The first and last boundaries of the warped region are fixed, so:
      - total sequence length stays unchanged
      - there is no accumulated/global timing drift
      - any trailing region shorter than `zone_size` is untouched

    Args:
        x:
            Tensor of shape [B, T, D], or [T, D].

        zone_size:
            Number of frames per warp zone.
            Example: 50 frames = 500 ms for a 10 ms SSL hop.

        max_frame_offset:
            Maximum displacement of each internal boundary, in frames.
            Example: 5 = +/-50 ms for a 10 ms SSL hop.

        generator:
            Optional torch.Generator for reproducibility.

    Returns:
        Tensor with the same shape as x.
    """

    squeeze_batch = False

    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze_batch = True

    if x.ndim != 3:
        raise ValueError(
            f"Expected [B, T, D] or [T, D], got {tuple(x.shape)}"
        )

    if zone_size <= 0:
        raise ValueError("zone_size must be > 0")

    if max_frame_offset < 0:
        raise ValueError("max_frame_offset must be >= 0")

    # This guarantees q[i+1] > q[i] for every possible random draw.
    if 2 * max_frame_offset >= zone_size:
        raise ValueError(
            "Need 2 * max_frame_offset < zone_size "
            "to guarantee a monotonic warp."
        )

    B, T, D = x.shape

    # Number of complete zones.
    num_zones = T // zone_size

    # With only one complete zone, both of its boundaries must stay fixed,
    # so there is nothing to warp.
    if num_zones <= 1 or max_frame_offset == 0:
        return x.squeeze(0) if squeeze_batch else x

    # Everything after this point is an incomplete tail and remains unchanged.
    warp_end = num_zones * zone_size

    device = x.device

    # ------------------------------------------------------------
    # Original boundaries:
    #
    # p = [0, 50, 100, 150, ...]
    # ------------------------------------------------------------
    p = (
        torch.arange(
            num_zones + 1,
            device=device,
            dtype=torch.float32,
        )
        * float(zone_size)
    )

    # ------------------------------------------------------------
    # Perturbed boundaries:
    #
    # q_i = p_i + U(-offset, +offset)
    #
    # First and last boundary are fixed.
    # Each sample in the batch gets an independent warp.
    # ------------------------------------------------------------
    q = p.unsqueeze(0).expand(B, -1).clone()

    offsets = (
        torch.rand(
            B,
            num_zones - 1,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 2.0
        - 1.0
    ) * max_frame_offset

    q[:, 1:-1] += offsets

    # ------------------------------------------------------------
    # Output frame coordinates.
    #
    # We only warp [0, warp_end).
    # ------------------------------------------------------------
    t = torch.arange(
        warp_end,
        device=device,
        dtype=torch.float32,
    )

    t = t.unsqueeze(0).expand(B, -1)

    # ------------------------------------------------------------
    # For each output frame t, determine which warped interval:
    #
    #     [q_i, q_{i+1}]
    #
    # contains it.
    #
    # searchsorted does this for the entire batch at once.
    # ------------------------------------------------------------
    zone_idx = torch.searchsorted(
        q[:, 1:].contiguous(),
        t,
        right=True,
    )

    q0 = torch.gather(q, 1, zone_idx)
    q1 = torch.gather(q, 1, zone_idx + 1)

    # Original interval always has length `zone_size`.
    p0 = zone_idx.to(torch.float32) * float(zone_size)

    # ------------------------------------------------------------
    # Inverse piecewise-linear mapping:
    #
    # destination:
    #     q_i -------- t -------- q_{i+1}
    #
    # source:
    #     p_i -------- s -------- p_{i+1}
    #
    # s = p_i +
    #     (t - q_i) / (q_{i+1} - q_i) * zone_size
    # ------------------------------------------------------------
    src_pos = (
        p0
        + (t - q0)
        * (float(zone_size) / (q1 - q0))
    )

    # ------------------------------------------------------------
    # Linear interpolation in the original SSL sequence.
    # ------------------------------------------------------------
    src0 = torch.floor(src_pos).long()
    src1 = src0 + 1

    src0.clamp_(0, T - 1)
    src1.clamp_(0, T - 1)

    alpha = src_pos - src0.to(src_pos.dtype)
    alpha = alpha.to(dtype=x.dtype)

    batch_idx = torch.arange(
        B,
        device=device,
    )[:, None]

    x0 = x[batch_idx, src0]
    x1 = x[batch_idx, src1]

    warped = torch.lerp(
        x0,
        x1,
        alpha.unsqueeze(-1),
    )

    # ------------------------------------------------------------
    # Preserve incomplete tail exactly.
    # ------------------------------------------------------------
    if warp_end < T:
        warped = torch.cat(
            [
                warped,
                x[:, warp_end:],
            ],
            dim=1,
        )

    if squeeze_batch:
        warped = warped.squeeze(0)

    return warped
    
import torch
import torch.nn.functional as F

def gaussian_blur_1d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Applies a 1D Gaussian blur to an input tensor.
    
    Args:
        x (Tensor): Input tensor of shape (batch_size, channels, sequence_length)
        kernel_size (int): Size of the blurring kernel (should be an odd integer)
        sigma (float): Standard deviation of the Gaussian distribution
        
    Returns:
        Tensor: Blurred output tensor with the same shape as x
    """
    x = x.transpose(1, 2)
    # 1. Create a 1D grid centered at 0
    radius = kernel_size // 2
    x_grid = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    
    # 2. Compute the unnormalized Gaussian values
    kernel = torch.exp(-0.5 * (x_grid / sigma) ** 2)
    
    # 3. Normalize the kernel so all elements sum to 1
    kernel = kernel / kernel.sum()
    
    # 4. Reshape kernel for conv1d: (out_channels, in_channels/groups, kernel_width)
    channels = x.shape[1]
    kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
    
    # 5. Apply reflection padding to keep the input size intact
    padded_x = F.pad(x, (radius, radius), mode="reflect")
    
    # 6. Apply depthwise 1D convolution
    return F.conv1d(padded_x, kernel, groups=channels).transpose(1, 2)

# --- Example Usage ---
# Batch size = 1, Channels = 2 (e.g., stereo audio), Sequence length = 10
# signal = torch.randn(1, 2, 10) 
# blurred_signal = gaussian_blur_1d(signal, kernel_size=5, sigma=1.0)

# print("Original Signal Shape:", signal.shape)
# print("Blurred Signal Shape:", blurred_signal.shape)
