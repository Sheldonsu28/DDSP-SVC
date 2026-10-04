import torch
import torch.nn as nn
import torch.nn.functional as F

from stylizer.wavenet import WN

class Flip(nn.Module):
    def forward(self, x, condition=None):
        y = torch.flip(x, dims=[-1])
        logdet = x.new_zeros(x.size(0))
        return y, logdet
    def inverse(self, y, condition=None):
        x = torch.flip(y, dims=[-1])
        logdet = y.new_zeros(y.size(0))
        return x, logdet


class TCNBlock(nn.Module):
    """
    Lightweight gated TCN block (depthwise-dilated conv + GLU + residual).
    Input/Output: [B, C, T]
    """
    def __init__(self, channels, kernel_size=5, dilation=1, p_dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv1d(channels, channels, kernel_size,
                            padding=padding, dilation=dilation, groups=channels)
        self.pw = nn.Conv1d(channels, 2 * channels, 1)  # to GLU
        self.out = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(p_dropout)
        self.act = nn.GLU(dim=1)  # expects 2*C from self.pw
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x, mask=None):  # x: [B,C,T]; mask: [B,1,T] or None
        y = self.dw(x)
        y = self.pw(y)
        y = self.act(y)  # -> [B, C, T]
        y = self.out(y)
        if mask is not None:
            y = y * mask
        y = self.dropout(y)
        return self.norm(x + y)


class TemporalSNACCoupling(nn.Module):
    """
    Conditional affine coupling with:
      - Temporal perception via a small TCN stack on x0
      - SNAC speaker normalization (affine norm using pooled speaker embedding)
      - Optional mean-only mode (no scaling)
    I/O: z [B, L, D], condition [B, L, gin]
    """
    def __init__(self,
                 latent_dim: int,
                 gin_channels: int,
                 hidden_channels: int = 256,
                 n_layers: int = 4,
                 kernel_size: int = 5,
                 base_dilation: int = 2,
                 p_dropout: float = 0.0,
                 clamp_tanh: float = 3.0,
                 snac_m_max: float = 5.0,   # bound speaker mean shift per channel
                 snac_v_max: float = 2.0,
                 mean_only: bool = False):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even for half/half split"
        self.D = latent_dim
        self.H = hidden_channels
        self.half = latent_dim // 2
        self.clamp = clamp_tanh
        self.mean_only = mean_only
        self.snac_m_max = snac_m_max
        self.snac_v_max = snac_v_max

        # Pre 1x1 to lift x0 channels
        self.pre = nn.Conv1d(self.half, hidden_channels, 1)

        # Temporal stack on x0 (context over time)
        # blocks = []
        # for i in range(n_layers):
        #     dilation = base_dilation ** i
        #     blocks.append(TCNBlock(hidden_channels, kernel_size, dilation, p_dropout))
        # self.tcn = nn.ModuleList(blocks)
        self.wn = WN(hidden_channels, 5, 1, 4, 0, 0.1)
        self.cond_norm = nn.LayerNorm(gin_channels, elementwise_affine=False)

        # Post 1x1 to produce stats for x1
        out_channels = self.half if mean_only else 2 * self.half
        self.post = nn.Conv1d(hidden_channels, out_channels, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

        # SNAC: speaker-normalized affine stats for both halves (use pooled speaker)
        # We mirror the residual layer behavior: produce mean & log-var for HALF channels.
        self.snac = nn.Linear(gin_channels, 2 * self.half)
        nn.init.zeros_(self.snac.weight)
        nn.init.zeros_(self.snac.bias)

    @staticmethod
    def _to_bct(x):  # [B,L,C] -> [B,C,T]
        return x.transpose(1, 2)

    @staticmethod
    def _to_blt(x):  # [B,C,T] -> [B,L,C]
        return x.transpose(1, 2)
    
    def _snac_stats(self, condition, T, device, dtype):
        """
        Pool -> LayerNorm -> Linear -> tanh clamps.
        Returns speaker_m, speaker_v shaped [B, half, 1].
        """
        # condition: [B, L, gin]
        g = condition.mean(dim=1)                 # [B, gin]
        g = self.cond_norm(g)                     # stabilize scale
        s = self.snac(g)                          # [B, 2*half]
        m, v = torch.split(s, [self.half, self.half], dim=-1)

        # HARD CLAMPS (critical fix)
        m = torch.tanh(m) * self.snac_m_max       # [-m_max, m_max]
        v = torch.tanh(v) * self.snac_v_max       # [-v_max, v_max]  (log-std)

        m = m.unsqueeze(-1)                       # [B, half, 1]
        v = v.unsqueeze(-1)                       # [B, half, 1]
        return m.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor, condition: torch.Tensor, x_mask: torch.Tensor = None):
        """
        Forward (non-reverse) pass.
        z: [B, L, D], condition: [B, L, gin], optional x_mask: [B, 1, L] or [B, L, 1]
        Returns: y [B, L, D], logdet [B]
        """
        B, L, D = z.shape
        assert D == self.D
        if x_mask is None:
            x_mask_bct = z.new_ones(B, 1, L)
        else:
            x_mask_bct = x_mask.transpose(1, 2) if x_mask.size(1) == L else x_mask  # -> [B,1,L]

        # Split & permute to conv layout
        x0, x1 = z[:, :, :self.half], z[:, :, self.half:]
        x0_bct = self._to_bct(x0)  # [B, half, L]
        x1_bct = self._to_bct(x1)  # [B, half, L]

        # SNAC speaker stats (broadcast over time)
        speaker_m, speaker_v = self._snac_stats(condition, L, z.device, z.dtype)
        # Normalize x0 for stats net
        x0_norm = (x0_bct - speaker_m) * torch.exp(-speaker_v)
        x0_norm = x0_norm * x_mask_bct  # mask-safe

        # Temporal stack to compute stats for x1
        h = self.pre(x0_norm) * x_mask_bct
        # for blk in self.tcn:
        #     h = blk(h, mask=x_mask_bct)
        h = self.wn(h)
        stats = self.post(h) * x_mask_bct  # [B, out, L]

        if self.mean_only:
            m = stats
            logs = torch.zeros_like(m)
        else:
            m, logs = torch.split(stats, [self.half, self.half], dim=1)
            logs = self.clamp * torch.tanh(logs)  # stabilize

        # Normalize x1, apply affine, keep it in normalized space (SNAC-style)
        x1_norm = (x1_bct - speaker_m) * torch.exp(-speaker_v)
        y1_bct = (m + x1_norm * torch.exp(logs)) * x_mask_bct

        # Compose output
        y = torch.cat([x0_bct, y1_bct], dim=1)
        y = self._to_blt(y)

        # Log-det: sum(logs) - sum(speaker_v) over transformed half & time
        # Use the same mask as above
        logdet = (logs * x_mask_bct).sum(dim=(1, 2)) - (speaker_v.expand(-1, -1, L) * x_mask_bct).sum(dim=(1, 2))
        return y, logdet

    def inverse(self, y: torch.Tensor, condition: torch.Tensor, x_mask: torch.Tensor = None):
        """
        Inverse pass.
        """
        B, L, D = y.shape
        if x_mask is None:
            x_mask_bct = y.new_ones(B, 1, L)
        else:
            x_mask_bct = x_mask.transpose(1, 2) if x_mask.size(1) == L else x_mask

        y0, y1 = y[:, :, :self.half], y[:, :, self.half:]
        y0_bct = self._to_bct(y0)
        y1_bct = self._to_bct(y1)

        speaker_m, speaker_v = self._snac_stats(condition, L, y.device, y.dtype)

        # Recompute stats from normalized y0
        y0_norm = (y0_bct - speaker_m) * torch.exp(-speaker_v)
        y0_norm = y0_norm * x_mask_bct

        h = self.pre(y0_norm) * x_mask_bct
        # for blk in self.tcn:
        #     h = blk(h, mask=x_mask_bct)
        h = self.wn(h)
        stats = self.post(h) * x_mask_bct

        if self.mean_only:
            m = stats
            logs = torch.zeros_like(m)
        else:
            m, logs = torch.split(stats, [self.half, self.half], dim=1)
            logs = self.clamp * torch.tanh(logs)

        # Invert affine, then de-normalize (SNAC inverse)
        x1_norm = (y1_bct - m) * torch.exp(-logs)
        x1_bct = (speaker_m + x1_norm * torch.exp(speaker_v)) * x_mask_bct

        x = torch.cat([y0_bct, x1_bct], dim=1)
        x = self._to_blt(x)

        # Inverse log-det: -sum(logs) + sum(speaker_v)
        logdet = (-logs * x_mask_bct).sum(dim=(1, 2)) + (speaker_v.expand(-1, -1, L) * x_mask_bct).sum(dim=(1, 2))
        return x, logdet

    @torch.no_grad()
    def zero_post_weights(self):
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)
    
        
class ConditionalNormalizingFlowSNAC(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, num_layers: int = 4,
                 hidden_channels: int = 256, n_tcn_layers: int = 4, kernel_size: int = 5,
                 base_dilation: int = 2, p_dropout: float = 0.0,
                 clamp_tanh: float = 3.0, mean_only: bool = False):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                TemporalSNACCoupling(
                    latent_dim=latent_dim,
                    gin_channels=condition_dim,
                    hidden_channels=hidden_channels,
                    n_layers=n_tcn_layers,
                    kernel_size=kernel_size,
                    base_dilation=base_dilation,
                    p_dropout=p_dropout,
                    clamp_tanh=clamp_tanh,
                    mean_only=mean_only,
                )
            )
            layers.append(Flip())
        self.layers = nn.ModuleList(layers)

    def forward(self, z, condition, x_mask=None):
        log_det_total = z.new_zeros(z.size(0))
        for layer in self.layers:
            if isinstance(layer, Flip):
                z, log_det = layer(z, condition=None)
            else:
                z, log_det = layer(z, condition, x_mask=x_mask)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, y, condition, x_mask=None):
        log_det_total = y.new_zeros(y.size(0))
        for layer in reversed(self.layers):
            if isinstance(layer, Flip):
                y, log_det = layer.inverse(y, condition=None)
            else:
                y, log_det = layer.inverse(y, condition, x_mask=x_mask)
            log_det_total = log_det_total + log_det
        return y, log_det_total

    def zeros_layer(self):
        for layer in self.layers:
            if hasattr(layer, "zero_post_weights"):
                layer.zero_post_weights()



class TemporalWN(nn.Module):
    """
    Lightweight WaveNet-style temporal stack over time dimension.
    Input/Output: [B, H, T], preserves channels.
    """
    def __init__(self, channels, kernel_size=5, dilation_rate=2, n_layers=4, p_dropout=0.0, causal=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.causal = causal
        self.dilation_rate = dilation_rate
        d = 1
        for _ in range(n_layers):
            padding = (kernel_size - 1) * d if causal else ((kernel_size - 1) * d) // 2
            conv = nn.Conv1d(channels, 2 * channels, kernel_size, dilation=d, padding=padding)
            self.blocks.append(nn.ModuleDict({
                "conv": conv,
                "drop": nn.Dropout(p_dropout),
            }))
            d = d * dilation_rate

        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # x: [B, H, T]
        h = x
        d = 1
        for b in self.blocks:
            y = b["conv"](h)
            a, g = y.chunk(2, dim=1)           # gated activation
            y = torch.tanh(a) * torch.sigmoid(g)
            y = b["drop"](y)
            h = h + y                           # residual
            d *= self.dilation_rate
        return self.out_proj(h)


class ConditionalSNACResidualCouplingLayer2(nn.Module):
    """
    Residual-style affine coupling with:
      - SNAC speaker normalization (point 2)
      - Correct logdet incl. SNAC term (point 4)
      - mean_only option (point 5)
      - Temporal perception via TemporalWN over time

    Expects inputs [B, L, D], condition [B, L, C].
    Optional mask: [B, L] or [B, L, 1] (if provided).
    """
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_channels: int = 256,
        kernel_size: int = 5,
        dilation_rate: int = 2,
        n_layers: int = 4,
        p_dropout: float = 0.0,
        mean_only: bool = False,
        center_volume: bool = False,  # typically False with SNAC; you can set True if you still want zero-mean log_s
        causal: bool = False,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim (channels) must be even"
        self.D = latent_dim
        self.C = condition_dim
        self.H = hidden_channels
        self.mean_only = mean_only
        self.center_volume = center_volume

        self.half = latent_dim // 2
        # pre -> temporal WN -> post, all in [B, C, T] format
        self.pre = nn.Conv1d(self.half, hidden_channels, 1)
        self.enc = TemporalWN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout, causal=causal)
        out_ch = self.half if mean_only else 2 * self.half
        self.post = nn.Conv1d(hidden_channels, out_ch, 1)

        # SNAC speaker affine: from pooled speaker vec -> (m, v) for half channels
        self.snac = nn.Conv1d(condition_dim, 2 * self.half, 1)

        # Init last layers near-identity
        nn.init.zeros_(self.post.weight); nn.init.zeros_(self.post.bias)
        # nn.init.zeros_(self.snac.weight); nn.init.zeros_(self.snac.bias)

    def _make_mask(self, x, mask):
        # x: [B, L, D]; mask -> [B, 1, T]
        B, L, _ = x.shape
        if mask is None:
            return x.new_ones(B, 1, L)
        if mask.dim() == 2:
            return mask.unsqueeze(1).to(dtype=x.dtype)
        if mask.size(-1) == 1:
            return mask.transpose(1, 2).to(dtype=x.dtype)  # [B, 1, L]
        if mask.size(1) == L:  # [B, L, D?] unlikely
            return mask.transpose(1, 2)[..., :1].to(dtype=x.dtype)
        return x.new_ones(B, 1, L)

    def _pool_condition(self, condition, mask_1xT):
        # condition: [B, L, C], mask_1xT: [B, 1, L]
        B, L, C = condition.shape
        cond = condition.transpose(1, 2)                   # [B, C, L]
        if mask_1xT is not None:
            w = mask_1xT                                   # [B, 1, L]
            denom = w.sum(dim=2, keepdim=True).clamp_min(1.0)
            pooled = (cond * w).sum(dim=2, keepdim=True) / denom  # [B, C, 1]
        else:
            pooled = cond.mean(dim=2, keepdim=True)        # [B, C, 1]
        return pooled                                      # [B, C, 1]

    def forward(self, z: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor = None):
        """
        z: [B, L, D], condition: [B, L, C], optional mask: [B, L] or [B, L, 1]
        returns y: [B, L, D], logdet: [B]
        """
        B, L, D = z.shape
        assert D == self.D and condition.shape[0] == B and condition.shape[1] == L

        # Prepare formats
        x = z.transpose(1, 2)                              # [B, D, L]
        x0, x1 = x.split([self.half, self.half], dim=1)    # [B, half, L] each
        m1 = self._make_mask(z, mask)                      # [B, 1, L]

        # SNAC: speaker stats
        g_vec = self._pool_condition(condition, m1)        # [B, C, 1]
        speaker = self.snac(g_vec)                         # [B, 2*half, 1]
        speaker_m, speaker_v = speaker.chunk(2, dim=1)     # [B, half, 1] each

        # Normalize x0 for stats
        x0_norm = (x0 - speaker_m) * torch.exp(-speaker_v) * m1

        # Temporal stats network
        h = self.pre(x0_norm) * m1
        h = self.enc(h) * m1
        stats = self.post(h) * m1                          # [B, half or 2*half, L]

        if self.mean_only:
            m = stats                                      # [B, half, L]
            logs = torch.zeros_like(m)
        else:
            m, logs = stats.split(self.half, dim=1)        # [B, half, L] each

        # Optional zero-mean log-scale per sample (usually False with SNAC)
        if not self.mean_only and self.center_volume:
            logs = logs - logs.mean(dim=(1, 2), keepdim=True)

        # Normalize x1 with speaker before affine
        x1_norm = (x1 - speaker_m) * torch.exp(-speaker_v) * m1

        # Affine transform (forward)
        if self.mean_only:
            y1 = (m + x1_norm) * m1
            logdet = -torch.sum(speaker_v.expand(-1, -1, L) * m1, dim=(1, 2))  # only SNAC contributes
        else:
            y1 = (m + x1_norm * torch.exp(logs)) * m1
            logdet = (
                torch.sum(logs * m1, dim=(1, 2))
                - torch.sum(speaker_v.expand(-1, -1, L) * m1, dim=(1, 2))
            )

        y = torch.cat([x0, y1], dim=1).transpose(1, 2)     # back to [B, L, D]
        return y, logdet

    def inverse(self, y: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor = None):
        B, L, D = y.shape
        x = y.transpose(1, 2)
        y0, y1 = x.split([self.half, self.half], dim=1)
        m1 = self._make_mask(y, mask)

        g_vec = self._pool_condition(condition, m1)        # [B, C, 1]
        speaker = self.snac(g_vec)                         # [B, 2*half, 1]
        speaker_m, speaker_v = speaker.chunk(2, dim=1)     # [B, half, 1]

        # Recompute stats from y0 (note: we must use y0 normalized as in forward)
        y0_norm = (y0 - speaker_m) * torch.exp(-speaker_v) * m1

        h = self.pre(y0_norm) * m1
        h = self.enc(h) * m1
        stats = self.post(h) * m1

        if self.mean_only:
            m = stats
            logs = torch.zeros_like(m)
        else:
            m, logs = stats.split(self.half, dim=1)

        if not self.mean_only and self.center_volume:
            logs = logs - logs.mean(dim=(1, 2), keepdim=True)

        # Invert affine on normalized space
        if self.mean_only:
            x1_norm = (y1 - m) * m1
            logdet = torch.sum(speaker_v.expand(-1, -1, L) * m1, dim=(1, 2))
        else:
            x1_norm = (y1 - m) * torch.exp(-logs) * m1
            logdet = (
                -torch.sum(logs * m1, dim=(1, 2))
                + torch.sum(speaker_v.expand(-1, -1, L) * m1, dim=(1, 2))
            )

        # Denormalize to original space
        x1 = (speaker_m + x1_norm * torch.exp(speaker_v)) * m1
        x = torch.cat([y0, x1], dim=1).transpose(1, 2)
        return x, logdet

    @torch.no_grad()
    def zero_post_weights(self):
        # keeps your zeros_layer() API
        self.post.weight.zero_()
        self.post.bias.zero_()
        self.snac.weight.zero_()
        self.snac.bias.zero_()
        
        
class ConditionalNormalizingFlow4(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, num_layers: int = 4, **kwargs):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ConditionalSNACResidualCouplingLayer2(
                latent_dim=latent_dim,
                condition_dim=condition_dim,
                hidden_channels=256,
                kernel_size=3,
                dilation_rate=2,
                n_layers=3,
                p_dropout=0.1,
                center_volume=False,      # usually False with SNAC
                causal=False,             # set True if you need strict causality
                **kwargs
            ))
            layers.append(Flip())         # your existing Flip with zero logdet
        self.layers = nn.ModuleList(layers)

    def forward(self, z, condition, mask=None):
        log_det = z.new_zeros(z.size(0))
        for layer in self.layers:
            # Flip ignores mask; coupling uses mask if provided
            if isinstance(layer, ConditionalSNACResidualCouplingLayer2):
                z, ld = layer(z, condition, mask)
            else:
                z, ld = layer(z, condition)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, y, condition, mask=None):
        log_det = y.new_zeros(y.size(0))
        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalSNACResidualCouplingLayer2):
                y, ld = layer.inverse(y, condition, mask)
            else:
                y, ld = layer.inverse(y, condition)
            log_det = log_det + ld
        return y, log_det

    def zeros_layer(self):
        for layer in self.layers:
            if hasattr(layer, "zero_post_weights"):
                layer.zero_post_weights()

