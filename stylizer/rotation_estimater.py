import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- utils

def _num_skew_params(d: int) -> int:
    # number of free params for skew-symmetric dxd: d*(d-1)/2
    return d * (d - 1) // 2

def vector_to_skew(vec: torch.Tensor, d: int) -> torch.Tensor:
    """
    vec: (B, d*(d-1)//2)
    returns K: (B, d, d) with K^T = -K
    """
    B = vec.shape[0]
    K = vec.new_zeros(B, d, d)
    idx = 0
    for i in range(d):
        for j in range(i+1, d):
            v = vec[:, idx]
            K[:, i, j] = v
            K[:, j, i] = -v
            idx += 1
    return K

# ---------- Rotation via matrix exponential on so(d)

class SOdExp(nn.Module):
    """
    Maps unconstrained params -> rotation R in SO(d) using the Lie exponential:
        R = exp( cap * skew(vec) )
    'cap' (≈ max_angle) gates the Frobenius norm so rotations stay small & stable.
    """
    def __init__(self, d: int, max_angle: float = 0.2):
        super().__init__()
        self.d = d
        self.max_angle = float(max_angle)

    @property
    def num_params(self) -> int:
        return _num_skew_params(self.d)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        theta: (B, P) unconstrained, P = d*(d-1)//2
        Returns R: (B, d, d)
        """
        B = theta.shape[0]
        K = vector_to_skew(theta, self.d)                                # (B,d,d)
        # cap rotation magnitude smoothly with tanh
        # scale each sample so that ||K||_F is bounded by max_angle
        # (You can also use a single global cap; this per-sample variant is flexible.)
        frob = torch.linalg.norm(K, dim=(1,2), keepdim=True) + 1e-12
        scale = (self.max_angle * torch.tanh(frob)) / frob               # (B,1,1)
        Kc = K * scale
        # exact rotation via matrix exponential
        R = torch.matrix_exp(Kc)                                         # (B,d,d)
        # Numerical guard: enforce det +1 (very rare drift)
        # If det<0 (reflection), flip the last column.
        detR = torch.linalg.det(R)
        mask = (detR < 0).view(-1)
        if mask.any():
            Rm = R[mask]
            Rm[:, :, -1] = -Rm[:, :, -1]
            R = R.clone()
            R[mask] = Rm
        return R

# ---------- Rotation via a few Givens rotations (fast/structured)

class GivensStack(nn.Module):
    """
    Compose K small 2D rotations (Givens) -> R in SO(d).
    Indices for planes are fixed (round-robin); angles are predicted.
    """
    def __init__(self, d: int, n_givens: int = 16, max_angle: float = 0.2):
        super().__init__()
        self.d = d
        self.n_givens = int(n_givens)
        self.max_angle = float(max_angle)

        # Pre-pick a list of (i,j) plane indices (deterministic)
        pairs = []
        i, j = 0, 1
        for k in range(self.n_givens):
            pairs.append((i, j))
            j = (j + 1) % self.d
            if j == i:
                j = (j + 1) % self.d
            if j <= i:
                i = (i + 1) % self.d
                if i == j: j = (j + 1) % self.d
        self.register_buffer("pairs", torch.tensor(pairs, dtype=torch.long))  # (K,2)

    @property
    def num_params(self) -> int:
        return self.n_givens

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        angles: (B, K) unconstrained; we cap each via tanh * max_angle.
        Returns R: (B, d, d)
        """
        B = angles.shape[0]
        capped = self.max_angle * torch.tanh(angles)       # (B,K)
        R = torch.eye(self.d, device=angles.device).expand(B, self.d, self.d).clone()
        for k in range(self.n_givens):
            i, j = self.pairs[k].tolist()
            theta = capped[:, k]                           # (B,)
            c = torch.cos(theta)
            s = torch.sin(theta)
            # Apply Givens from the right: modify columns i and j
            Ri = R[:, :, i].clone()
            Rj = R[:, :, j].clone()
            R[:, :, i] = c.unsqueeze(-1) * Ri + s.unsqueeze(-1) * Rj
            R[:, :, j] = -s.unsqueeze(-1) * Ri + c.unsqueeze(-1) * Rj
        return R

# ---------- Positive scale with bound

class PositiveScale(nn.Module):
    """
    s = exp( clamp ) with bounded log-scale to keep things near 1.
    """
    def __init__(self, max_log_scale: float = 0.1):
        super().__init__()
        self.max_log_scale = float(max_log_scale)

    def forward(self, log_s_raw: torch.Tensor) -> torch.Tensor:
        # log_s in [-max, max]
        log_s = self.max_log_scale * torch.tanh(log_s_raw)
        return torch.exp(log_s)  # (B, 1) or (B, 1, 1)

# ---------- One-stop rotation + scaling layer

class RotationScaleLayer(nn.Module):
    """
    Predict (or take as input) rotation parameters and scale, produce R in SO(d) and s>0,
    then transform embeddings: z' = (z @ R^T) * s
    - method='exp': full Lie-exponential (robust, exact orthogonal)
    - method='givens': faster, uses K plane rotations
    """
    def __init__(self, d: int, method: str = 'givens',
                 max_angle: float = 0.15, max_log_scale: float = 0.1,
                 givens_K: int = 128):
        super().__init__()
        self.d = d
        self.method = method
        if method == 'exp':
            self.rot = SOdExp(d, max_angle)
            self.rot_params = self.rot.num_params
        elif method == 'givens':
            self.rot = GivensStack(d, n_givens=givens_K, max_angle=max_angle)
            self.rot_params = self.rot.num_params
        else:
            raise ValueError("method must be 'exp' or 'givens'")
        self.scaler = PositiveScale(max_log_scale)

    def forward(self, z: torch.Tensor, rot_params: torch.Tensor, log_s_raw: torch.Tensor):
        """
        z: (B, T, d)
        rot_params: (B, P) where P = rot.num_params
        log_s_raw: (B, 1) or (B, 1, 1)
        Returns:
          z_out: (B, T, d), R: (B,d,d), s: (B,1,1)
        """
        B, T, d = z.shape
        assert d == self.d
        R = self.rot(rot_params)                       # (B,d,d)
        s = self.scaler(log_s_raw.view(B, -1))         # (B,1)
        s = s.view(B, 1, 1)
        z_rot = torch.matmul(z, R.transpose(1, 2))     # (B,T,d)
        z_out = z_rot * s
        return z_out, R, s



class RotScalePredictor(nn.Module):
    """
    Input: features H of shape (B, T, Hdim)
    Output: rot_params (B,P), log_s_raw (B,1)
    """
    def __init__(self, Hdim: int, rot_param_dim: int, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(Hdim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_rot = nn.Linear(hidden, rot_param_dim)
        self.head_s   = nn.Linear(hidden, 1)

    def forward(self, H: torch.Tensor):
        # Mean-pool across time (swap in attention pooling if you prefer)
        h = H.mean(dim=1)
        h = self.mlp(h)
        return self.head_rot(h), self.head_s(h)


def rotation_from_two_vectors(
    x: torch.Tensor, 
    y: torch.Tensor, 
    eps: float = 1e-9, 
    anti_parallel_tol: float = 1e-7
):
    """
    Returns an orthogonal rotation matrix R (..., N, N) that maps x -> y via the
    minimal rotation in the plane spanned by {x, y}, acting as identity elsewhere.
    All pairwise angles in R^N are preserved (R is orthogonal).

    Args:
        x, y: (..., N) with the same shape
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape (..., N). Got {x.shape} vs {y.shape}")
    if x.ndim < 1:
        raise ValueError("x and y must be at least 1D tensors of shape (..., N)")

    *batch, N = x.shape
    device, dtype = x.device, x.dtype
    I = torch.eye(N, device=device, dtype=dtype).expand(*batch, N, N)

    x_norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    y_norm = torch.linalg.norm(y, dim=-1, keepdim=True).clamp_min(eps)
    u = x / x_norm
    v = y / y_norm

    c = (u * v).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)   # cos θ
    w = v - c * u                                            # component ⟂ to u
    s = torch.linalg.norm(w, dim=-1, keepdim=True)           # sin θ ≥ 0

    theta = torch.atan2(s.squeeze(-1), c.squeeze(-1))        # angle in [0, π]

    # Handle anti-parallel: pick a stable orthogonal direction to u
    need_alt = (s.squeeze(-1) <= anti_parallel_tol) & (c.squeeze(-1) < 0.0)

    # pick coordinate least aligned with u
    idx = torch.argmin(torch.abs(u), dim=-1)                 # (...,)
    eyeN = torch.eye(N, device=device, dtype=dtype)
    e = eyeN.index_select(0, idx.view(-1)).view(*batch, N)   # (..., N)
    u_k = torch.gather(u, -1, idx[..., None])                # (..., 1)
    a = e - u_k * u
    a = a / (torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(eps))

    w_hat = torch.where(need_alt[..., None], a, w / (s + eps))  # (..., N)

    uuT = u[..., :, None] @ u[..., None, :]
    wwT = w_hat[..., :, None] @ w_hat[..., None, :]
    wuT = w_hat[..., :, None] @ u[..., None, :]
    uwT = u[..., :, None] @ w_hat[..., None, :]

    R = I + (c[..., None, None] - 1.0) * (uuT + wwT) + s[..., None, None] * (wuT - uwT)

    # Near-parallel (θ ~ 0): just identity to avoid jitter
    near_parallel = (s.squeeze(-1) <= anti_parallel_tol) & (c.squeeze(-1) >= 0.0)
    if torch.any(near_parallel):
        R = torch.where(near_parallel.view(*batch, 1, 1), I, R)

    return R, theta
    
    
def rotation_align_subspaces(
    U: torch.Tensor, 
    V: torch.Tensor, 
    assume_orthonormal: bool = False, 
    eps: float = 1e-9
):
    """
    Construct an orthogonal rotation R (..., N, N) that maps the k-dim subspace span(U)
    onto span(V) by aligning their principal vectors. When k=1 this reduces to (A).
    The construction composes mutually-orthogonal 2D plane rotations; result is
    a global rigid rotation (preserves all pairwise angles and dot products).

    Args:
        U: (..., N, k) - basis (not necessarily orthonormal) for source subspace
        V: (..., N, k) - basis (not necessarily orthonormal) for target subspace
        assume_orthonormal: if True, skip QR orthonormalization
    Returns:
        R:  (..., N, N) orthogonal rotation matrix
        thetas: (..., k) principal angles (radians, in [0, π/2])
    """
    if U.shape != V.shape or U.ndim < 2:
        raise ValueError(f"U and V must have same shape (..., N, k). Got {U.shape} vs {V.shape}")

    *batch, N, k = U.shape
    device, dtype = U.device, U.dtype

    # Orthonormalize columns for numerical stability unless told otherwise
    if not assume_orthonormal:
        # QR with economic mode
        U, _ = torch.linalg.qr(U, mode='reduced')  # (..., N, k)
        V, _ = torch.linalg.qr(V, mode='reduced')  # (..., N, k)

    # Principal angles/vectors between subspaces via SVD of U^T V
    # U^T V = P (cosΘ) Q^T
    M = torch.matmul(U.transpose(-2, -1), V)          # (..., k, k)
    P, S, Qh = torch.linalg.svd(M)                    # S = singular values in [0,1] = cos(theta_i)
    Q = Qh.transpose(-2, -1)

    # Rotate U,V into principal bases
    U1 = torch.matmul(U, P)                           # (..., N, k) columns = {u_i}
    V1 = torch.matmul(V, Q)                           # (..., N, k) columns = {v_i}

    # Principal angles (radians)
    cos_t = S.clamp(-1.0, 1.0)
    thetas = torch.acos(cos_t)                        # (..., k)

    # Compose commuting plane rotations over each principal pair (u_i, v_i)
    I = torch.eye(N, device=device, dtype=dtype).expand(*batch, N, N)
    R = I.clone()

    for i in range(k):
        u = U1[..., :, i]                             # (..., N)
        v = V1[..., :, i]                             # (..., N)

        # re-use (A) but as a helper that builds a plane rotation in the {u,v} plane
        Ri, _ = rotation_from_two_vectors(u, v, eps=eps)
        R = torch.matmul(Ri, R)                       # left-compose; planes are orthogonal so order is safe

    return R, thetas
