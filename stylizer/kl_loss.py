import torch

_LOG2PI = 1.8378770664093453  # log(2π)

# def log_diag_normal_sum_safe(z, mu, logvar, mask=None,
#                              LOGVAR_MIN=-14.0, LOGVAR_MAX=10.0):
#     """
#     Stable log N(z; mu, diag(exp(logvar))) summed over time & dims, per sample.
#     Shapes: [B, L, D] everywhere.
#     """
#     z = z.float(); mu = mu.float(); logvar = logvar.float()
#     # Clamp log-variance to avoid 1/var overflow under tiny variances
#     logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)
#     inv_var = torch.exp(-logvar)                 # = 1/var
#     elem = -0.5 * (logvar + _LOG2PI + (z - mu)**2 * inv_var)  # [B,L,D]
#     if mask is not None:
#         elem = elem * mask.float()
#     # guard against accidental NaN/Inf
#     elem = torch.nan_to_num(elem, neginf=-1e6, posinf=-1e6)
#     return elem.sum(dim=(1, 2))                  # [B]

def _count_mask(mask, B):
    if mask is None:
        return torch.full((B,), 1.0, device='cuda' if torch.cuda.is_available() else None)
    return mask.float().sum(dim=(1, 2)).clamp(min=1.0)


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


# def flow_kl_forward_mc_safe(z0, z, mu_q, logvar_q, mu_p, logvar_p, logdet_fwd, mask=None):
#     """
#     KL(q_f || p) = E_{z0~q0}[ log q0(z0) - log p(z) - log|det J_fwd(z0)| ]
#     All tensors: [B, L, D]; logdet_fwd: [B]
#     """
#     B = z0.shape[0]
#     log_q0 = log_diag_normal_sum_safe(z0, mu_q, logvar_q, mask)  # [B]
#     log_p  = log_diag_normal_sum_safe(z,  mu_p, logvar_p, mask)  # [B]
#     per_sample = log_q0 - log_p - logdet_fwd                     # [B]
#     denom = _count_mask(mask, B)                                 # [B]
#     per_elem = per_sample / denom
#     # guard
#     per_elem = torch.nan_to_num(per_elem, neginf=1e6, posinf=1e6)
#     return per_elem.mean()

# def flow_kl_inverse_mc_safe(y, z_inv, mu_p, logvar_p, mu_q, logvar_q, logdet_inv, mask=None):
#     """
#     KL(p || q_f) = E_{y~p}[ log p(y) - (log q0(z_inv) + log|det J_inv(y)|) ]
#     """
#     B = y.shape[0]
#     log_p  = log_diag_normal_sum_safe(y,     mu_p, logvar_p, mask)   # [B]
#     log_q0 = log_diag_normal_sum_safe(z_inv, mu_q, logvar_q, mask)   # [B]
#     per_sample = log_p - (log_q0 + logdet_inv)                       # [B]
#     denom = _count_mask(mask, B)
#     per_elem = per_sample / denom
#     per_elem = torch.nan_to_num(per_elem, neginf=1e6, posinf=1e6)
#     return per_elem.mean()

def flow_kl_forward_mc_safe(z0, z, mu_q, logvar_q, mu_p, logvar_p, logdet_fwd, mask_BLD=None):
    log_q0 = log_diag_normal_sum_safe(z0, mu_q, logvar_q, mask_BLD)  # [B]
    log_p  = log_diag_normal_sum_safe(z,  mu_p, logvar_p,  mask_BLD) # [B]
    per_sample = log_q0 - log_p - logdet_fwd                         # [B]
    denom = denom_from_mask_BLD(mask_BLD, z0)                        # [B] ~= L*D
    per_elem = per_sample / denom
    per_elem = torch.nan_to_num(per_elem, neginf=1e6, posinf=1e6)
    return per_elem.mean()

def flow_kl_inverse_mc_safe(y, z_inv, mu_p, logvar_p, mu_q, logvar_q, logdet_inv, mask_BLD=None):
    log_p  = log_diag_normal_sum_safe(y,     mu_p, logvar_p, mask_BLD)  # [B]
    log_q0 = log_diag_normal_sum_safe(z_inv, mu_q, logvar_q, mask_BLD)  # [B]
    per_sample = log_p - (log_q0 + logdet_inv)                          # [B]
    denom = denom_from_mask_BLD(mask_BLD, y)                             # [B]
    per_elem = per_sample / denom
    per_elem = torch.nan_to_num(per_elem, neginf=1e6, posinf=1e6)
    return per_elem.mean()


def log_diag_normal_sum_safe(z, mu, logvar, mask_BLD=None,
                             LOGVAR_MIN=-14.0, LOGVAR_MAX=10.0):
    z = z.float(); mu = mu.float(); logvar = logvar.float()
    logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)
    inv_var = torch.exp(-logvar)
    elem = -0.5 * (logvar + 1.8378770664093453 + (z - mu)**2 * inv_var)  # log(2π)
    if mask_BLD is not None:
        elem = elem * mask_BLD
    elem = torch.nan_to_num(elem, neginf=-1e6, posinf=-1e6)
    return elem.sum(dim=(1, 2))


def make_mask_BLD(z_like, mask):
    # z_like: [B, L, D]
    B, L, D = z_like.shape
    if mask is None:
        return z_like.new_ones(B, L, D)

    if mask.dim() == 2:               # [B, L]
        return mask.unsqueeze(-1).expand(B, L, D)
    if mask.dim() == 3:
        if mask.shape == (B, L, 1):   # [B, L, 1]
            return mask.expand(B, L, D)
        if mask.shape == (B, 1, L):   # [B, 1, L]
            return mask.transpose(1, 2).expand(B, L, D)
        if mask.shape == (B, L, D):   # already BLD
            return mask
    raise ValueError(f"Unexpected mask shape {tuple(mask.shape)} for z_like {tuple(z_like.shape)}")

def denom_from_mask_BLD(mask_BLD, z_like):
    if mask_BLD is None:
        B, L, D = z_like.shape
        return torch.full((B,), float(L * D), device=z_like.device)
    return mask_BLD.float().sum(dim=(1, 2)).clamp(min=1.0)

#kl_loss = mod_kl_loss1_stable(z_fwd.transpose(1, 2) , 0.5*logvar_pr.transpose(1, 2), mu_pr.transpose(1, 2),  0.5*logvar_ps.transpose(1, 2), log_det_fwd, mask.expand_as(z_fwd.transpose(1, 2))) + 0.5 * mod_kl_loss1_stable(z_bkw.transpose(1, 2),  0.5*logvar_ps.transpose(1, 2), mu_ps.transpose(1, 2),  0.5*logvar_pr.transpose(1, 2), log_det_bkw, mask.expand_as(z_fwd.transpose(1, 2)), False)

# def kl_loss_vits(z_p, logs_q, m_p, logs_p, total_logdet, z_mask):
#     """
#     z_p, logs_q: [b, h, t_t]
#     m_p, logs_p: [b, h, t_t]
#     total_logdet: [b] - total_logdet summed over each batch
#     """
#     z_p = z_p.float()
#     logs_q = logs_q.float()
#     m_p = m_p.float()
#     logs_p = logs_p.float()
#     z_mask = z_mask.float()

#     kl = logs_p - logs_q - 0.5
#     kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
#     kl = torch.sum(kl * z_mask)
#     # add total_logdet (Negative LL)
#     kl -= torch.sum(total_logdet)
#     l = kl / torch.sum(z_mask)
#     return l