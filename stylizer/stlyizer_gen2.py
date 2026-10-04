"""
Lightweight frontend pre-processor for a frozen DDSP-SVC model.

Inputs (all length-T aligned at 100 Hz / hop 160 @ 16kHz):
    whisper:    (B, T, 1280)
    contentvec: (B, T, 768)
    hubert:     (B, T, 256)
    spk_emb:    (B, 192)         # from speaker verification model

Outputs:
    cv_out:        (B, T, 768)   # replacement for ContentVec, fed to DDSP as `units`
    formant_shift: (B, T, 1)     # fed to DDSP as `aug_shift`
    aux:           dict          # auxiliary outputs for training losses
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Helper modules
# ----------------------------------------------------------------------

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on speaker embedding."""

    def __init__(self, d_cond, d_model):
        super().__init__()
        self.to_gamma_beta = nn.Linear(d_cond, 2 * d_model)
        # Initialize so FiLM starts as identity: gamma=1, beta=0
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, cond):  # x: (B, T, D); cond: (B, D_cond)
        gb = self.to_gamma_beta(cond)              # (B, 2D)
        gamma, beta = gb.chunk(2, dim=-1)          # each (B, D)
        gamma = gamma.unsqueeze(1) + 1.0           # start at 1
        beta = beta.unsqueeze(1)                   # start at 0
        return gamma * x + beta


# class RMSNorm(nn.Module):
#     """
#     Root-mean-square LayerNorm (LLaMA-style).

#     Normalizes by the RMS over the last (feature) dimension with a learned
#     per-channel gain and *no* mean-centering or bias. The normalization is
#     computed in float32 and cast back, which keeps it numerically stable
#     under the bf16 mixed-precision path used throughout this model.
#     """

#     def __init__(self, d_model, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d_model))

#     def forward(self, x):
#         # dtype = x.dtype
#         # x = x.float()
#         x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         return x * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward (LLaMA-style): down(SiLU(gate(x)) * up(x)).

    The gate and up projections share a single fused input Linear. All
    projections are bias-free. Dropout is applied to the gated hidden state
    and to the output, matching the two dropout sites of the MLP it replaces.
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.w_in(x).chunk(2, dim=-1)
        h = self.drop(F.silu(gate) * up)
        return self.drop(self.w_out(h))


class ConformerBlock2(nn.Module):
    """
    GPT/LLaMA-style Conformer block + FiLM speaker conditioning at the end.

    Keeps the Conformer macro-structure (macaron FFN -> MHSA -> conv -> macaron
    FFN) but adopts modern decoder-block conventions internally:
      * RMSNorm instead of LayerNorm (pre-norm on every sub-module),
      * SwiGLU feed-forward instead of a SiLU MLP,
      * bias-free Linear/Conv projections,
      * GPT-2 scaled residual init: the projections writing into the residual
        stream are downscaled by 1/sqrt(2 * n_layers) so residual variance
        stays bounded with depth. Pass ``n_layers`` = stack depth for the
        depth-correct scaling (defaults to 1, a mild 1/sqrt(2) downscale).

    RoPE / local-window / causal masking semantics are unchanged, so it stays
    a drop-in replacement for the previous ``ConformerBlock2`` signature.
    """

    def __init__(self, d_model=512, n_heads=8, ffn_mult=2,
                 conv_kernel=15, dropout=0.1, d_cond=512, local_window=12,
                 causal=False, rope=None, n_layers=1):
        super().__init__()
        d_ff = d_model * ffn_mult

        # Macaron FFN 1 (half-step) -- RMSNorm + SwiGLU
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1 = SwiGLU(d_model, d_ff, dropout)

        # Self-attention
        # NOTE: We use F.scaled_dot_product_attention instead of
        # nn.MultiheadAttention. SDPA has fused kernels (FlashAttention /
        # mem-efficient attention) that are significantly faster in bf16
        # on Ampere/Ada GPUs. Projections are bias-free (LLaMA-style).
        self.norm_attn =nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop_p = dropout
        # RoPE applied to q/k inside attention (shared across blocks). May be
        # None to fall back to no positional information.
        self.rope = rope

        # Conv module
        # NOTE: GroupNorm instead of BatchNorm1d -- BN running stats are
        # unreliable under bf16 mixed precision. GroupNorm has no running
        # stats and is numerically stable in bf16.
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv_pw1 = nn.Conv1d(d_model, 2 * d_model, 1, bias=False)  # GLU expand
        self.conv_dw = nn.Conv1d(d_model, d_model, conv_kernel,
                                 padding=conv_kernel // 2, groups=d_model,
                                 bias=False)
        self.conv_gn = nn.GroupNorm(num_groups=32, num_channels=d_model)
        self.conv_pw2 = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.conv_drop = nn.Dropout(dropout)

        # Macaron FFN 2 -- RMSNorm + SwiGLU
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2 = SwiGLU(d_model, d_ff, dropout)

        self.local_window = local_window
        self.causal = causal

        self.norm_final = nn.LayerNorm(d_model)
        self.film = FiLM(d_cond, d_model)

        # GPT-2 scaled residual init: downscale every projection that writes
        # into the residual stream so the accumulated residual variance stays
        # bounded as depth grows.
        self._init_residual_scale(n_layers)

        # Lazily-built local-attention mask. During training every step uses
        # the same chunk length / window / causal config, so the mask is built
        # on the first batch and reused thereafter. A single slot is kept; if a
        # later batch arrives with a different config (e.g. longer inference
        # sequence or new device) the mask is rebuilt in place.
        self._mask_cache = None
        self._mask_key = None

    def _make_local_mask(self, seq_len, window_size, device, causal=False):
        key = (seq_len, window_size, causal, device)
        if self._mask_key == key:
            return self._mask_cache
        # Vectorized build: diff[i, j] = j - i.
        idx = torch.arange(seq_len, device=device)
        diff = idx[None, :] - idx[:, None]
        if causal:
            # Position i may attend to [i - window_size, i] (never the future).
            allowed = (diff <= 0) & (diff >= -window_size)
        else:
            # Symmetric window: |j - i| <= window_size.
            allowed = diff.abs() <= window_size
        # Same convention as before: True = outside window (matches original).
        mask = ~allowed
        self._mask_cache = mask
        self._mask_key = key
        return mask

    @torch.no_grad()
    def _init_residual_scale(self, n_layers):
        # GPT-2 style: scale each residual-writing projection by
        # 1/sqrt(2 * n_layers). attn_out, both SwiGLU output projections and
        # the conv output pointwise all feed directly into the residual stream.
        scale = 1.0 / math.sqrt(2.0 * max(1, n_layers))
        for module in (self.attn_out, self.ff1.w_out,
                       self.ff2.w_out, self.conv_pw2):
            module.weight.mul_(scale)

    def forward(self, x, cond=None):
        # Macaron FFN 1
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        # Attention via fused SDPA
        h = self.norm_attn(x)
        B, T, _ = h.shape
        qkv = self.attn_qkv(h)  # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Rotary position embedding: rotate q/k so the q-k dot product encodes
        # relative position. Applied here, before SDPA, on each head.
        if self.rope is not None:
            q, k = self.rope(q, k)
        # Attention masking:
        #  - causal + local_window > 0: causal sliding window, each frame
        #    attends only to [i - window, i].
        #  - causal + local_window == 0: full causal (use SDPA's fast path).
        #  - non-causal + local_window > 0: symmetric local window.
        #  - otherwise: full attention.
        is_causal = False
        if self.causal and self.local_window > 0:
            attn_mask = self._make_local_mask(T, self.local_window, x.device,
                                              causal=True)  # (T, T)
        elif self.causal:
            attn_mask = None
            is_causal = True  # SDPA builds the causal mask internally
        elif self.local_window > 0:
            attn_mask = self._make_local_mask(T, self.local_window, x.device)  # (T, T)
        else:
            attn_mask = None
        # bf16-friendly fused kernel
        h = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=is_causal,
            attn_mask=attn_mask
        )
        h = h.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, D)
        h = self.attn_out(h)
        x = x + h

        # Conv (transpose for Conv1d which expects (B, C, T))
        h = self.norm_conv(x).transpose(1, 2)
        h = self.conv_pw1(h)
        h = F.glu(h, dim=1)
        h = self.conv_dw(h)
        h = self.conv_gn(h)
        h = F.silu(h)
        h = self.conv_pw2(h)
        h = self.conv_drop(h).transpose(1, 2)
        x = x + h

        # Macaron FFN 2
        x = x + 0.5 * self.ff2(self.norm_ff2(x))

        x = self.norm_final(x)
        if not (cond is None):
            x = self.film(x, cond)
        return x


class ConformerEncoderLayer(nn.Module):
    """
    One bidirectional Conformer encoder layer -- the block repeated N times
    inside ``conformerEncoder``.

    Same modern internals as ``ConformerBlock2`` (macaron SwiGLU FFN -> MHSA ->
    conv module -> macaron SwiGLU FFN, LayerNorm pre-norm on every sub-module,
    bias-free projections, GPT-2 scaled residual init, RoPE on q/k) but with
    full *bidirectional* attention and no FiLM speaker conditioning -- i.e.
    BERT-style. There is no causal / local-window masking; an optional boolean
    ``attn_mask`` (broadcastable to ``(B, H, T, T)``, ``True`` = attend) is
    forwarded to SDPA so the encoder can mask padding.
    """

    def __init__(self, d_model=512, n_heads=8, ffn_mult=2,
                 conv_kernel=15, dropout=0.1, rope=None, n_layers=1):
        super().__init__()
        d_ff = d_model * ffn_mult

        # Macaron FFN 1 (half-step) -- LayerNorm + SwiGLU
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1 = SwiGLU(d_model, d_ff, dropout)

        # Bidirectional self-attention via fused SDPA (bias-free projections).
        self.norm_attn = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop_p = dropout
        # RoPE applied to q/k inside attention (shared across layers). May be
        # None to fall back to no positional information.
        self.rope = rope

        # Conv module (GroupNorm instead of BatchNorm1d for bf16 stability).
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv_pw1 = nn.Conv1d(d_model, 2 * d_model, 1, bias=False)  # GLU expand
        self.conv_dw = nn.Conv1d(d_model, d_model, conv_kernel,
                                 padding=conv_kernel // 2, groups=d_model,
                                 bias=False)
        self.conv_gn = nn.GroupNorm(num_groups=32, num_channels=d_model)
        self.conv_pw2 = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.conv_drop = nn.Dropout(dropout)

        # Macaron FFN 2 -- LayerNorm + SwiGLU
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2 = SwiGLU(d_model, d_ff, dropout)

        self.norm_final = nn.LayerNorm(d_model)

        # GPT-2 scaled residual init: downscale every projection that writes
        # into the residual stream so the accumulated residual variance stays
        # bounded as depth grows.
        self._init_residual_scale(n_layers)

    @torch.no_grad()
    def _init_residual_scale(self, n_layers):
        # GPT-2 style: scale each residual-writing projection by
        # 1/sqrt(2 * n_layers). attn_out, both SwiGLU output projections and
        # the conv output pointwise all feed directly into the residual stream.
        scale = 1.0 / math.sqrt(2.0 * max(1, n_layers))
        for module in (self.attn_out, self.ff1.w_out,
                       self.ff2.w_out, self.conv_pw2):
            module.weight.mul_(scale)

    def forward(self, x, attn_mask=None):
        # Macaron FFN 1
        x = x + 0.5 * self.ff1(self.norm_ff1(x))

        # Bidirectional multi-head self-attention via fused SDPA
        h = self.norm_attn(x)
        B, T, _ = h.shape
        qkv = self.attn_qkv(h)  # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Rotary position embedding: rotate q/k so the q-k dot product encodes
        # relative position. Applied here, before SDPA, on each head.
        if self.rope is not None:
            q, k = self.rope(q, k)
        # bf16-friendly fused kernel. attn_mask (if given) is a boolean mask
        # where True = attend; full attention when None.
        h = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            attn_mask=attn_mask,
        )
        h = h.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, D)
        h = self.attn_out(h)
        x = x + h

        # Conv (transpose for Conv1d which expects (B, C, T))
        h = self.norm_conv(x).transpose(1, 2)
        h = self.conv_pw1(h)
        h = F.glu(h, dim=1)
        h = self.conv_dw(h)
        h = self.conv_gn(h)
        h = F.silu(h)
        h = self.conv_pw2(h)
        h = self.conv_drop(h).transpose(1, 2)
        x = x + h

        # Macaron FFN 2
        x = x + 0.5 * self.ff2(self.norm_ff2(x))

        x = self.norm_final(x)
        return x
    
class EncoderLayers(nn.Module):
     def __init__(self, input_dim=192, ff_multiplier=4, nhead=4,  window=16, dropout=0.1, n_blocks=2):
        #try content_dim = 256
        super().__init__()

        self.encoder_blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dropout=dropout, batch_first=True, dim_feedforward=input_dim * ff_multiplier), num_layers=n_blocks)



class conformerEncoder(nn.Module):
    """
    BERT-style bidirectional Conformer encoder.

    Input embedding -> positional encoding -> N x [ bidirectional Conformer
    encoder layer ] -> encoded sequence, following the canonical Transformer /
    BERT encoder stack. The N repeated layers are ``ConformerEncoderLayer``
    (full bidirectional attention, no causal masking, no speaker conditioning).

    Positional information is supplied by a single shared RoPE instance applied
    to q/k inside every layer's attention -- the rotary equivalent of the
    additive positional encoding in the classic diagram -- so no absolute PE is
    added to the input embedding.

    Args:
        d_model:     model / residual width.
        n_layers:    number of stacked encoder layers (the "Nx").
        n_heads:     attention heads.
        ffn_mult:    macaron FFN hidden multiple.
        conv_kernel: depthwise conv kernel size.
        dropout:     dropout used inside the layers and on the input embedding.
        d_input:     if given and != d_model, an input projection ("input
                     embedding") maps (B, T, d_input) -> (B, T, d_model). Leave
                     None when the caller already provides d_model-wide vectors.

    forward(x, key_padding_mask=None):
        x:                (B, T, d_input) if d_input set, else (B, T, d_model).
        key_padding_mask: optional bool (B, T), True = real token, False = pad.
        returns:          (B, T, d_model)
    """

    def __init__(self, d_model=512, n_layers=6, n_heads=8, ffn_mult=2,
                 conv_kernel=15, dropout=0.1, d_input=None):
        super().__init__()

        # ---- Input embedding ----
        if d_input is not None and d_input != d_model:
            self.embed = nn.Sequential(nn.Linear(d_input, d_model),
                                       nn.LayerNorm(d_model))
        else:
            self.embed = None
        self.in_drop = nn.Dropout(dropout)

        # ---- Positional encoding: single shared RoPE (per-head dim) ----
        self.rope = RotaryPositionalEmbedding(d_model // n_heads)

        # ---- N x bidirectional encoder layers ----
        self.layers = nn.ModuleList([
            ConformerEncoderLayer(d_model, n_heads, ffn_mult, conv_kernel,
                                  dropout, rope=self.rope, n_layers=n_layers)
            for _ in range(n_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        if self.embed is not None:
            x = self.embed(x)
        x = self.in_drop(x)

        # Turn a (B, T) key-padding mask (True = keep) into an SDPA attention
        # mask. Shape (B, 1, 1, T) broadcasts over heads and query positions,
        # masking padded *keys* for every query; True = attend.
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]

        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class RoformerBlock(nn.Module):
    """
    GPT/LLaMA-style RoFormer encoder layer: a pre-norm Transformer block
    (MHSA + FFN) whose only source of position information is RoPE applied to
    q/k inside attention. Compared to `ConformerBlock2` it drops the
    convolution module and the macaron half-step FFNs, so it is lighter and
    purely global; use it where local convolutional inductive bias is not
    needed.

    Shares the same modern decoder-block internals as `ConformerBlock2`:
    RMSNorm (pre-norm), a SwiGLU feed-forward, bias-free projections, and
    GPT-2 scaled residual init (1/sqrt(2 * n_layers)). The constructor
    signature mirrors `ConformerBlock2` (same masking / RoPE / FiLM
    conventions) so the two are drop-in interchangeable inside the trunk.
    """

    def __init__(self, d_model=512, n_heads=8, ffn_mult=4,
                 dropout=0.1, d_cond=512, local_window=0,
                 causal=False, rope=None, n_layers=1):
        super().__init__()
        d_ff = d_model * ffn_mult

        # Self-attention (same fused-SDPA rationale as ConformerBlock2).
        # Bias-free projections (LLaMA-style).
        self.norm_attn =nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop_p = dropout
        # RoPE applied to q/k inside attention (shared across blocks). May be
        # None to fall back to no positional information.
        self.rope = rope

        # Position-wise FFN (single, full-step -- not macaron) -- SwiGLU
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

        self.local_window = local_window
        self.causal = causal
        self.d_cond = d_cond

        self.norm_final = nn.LayerNorm(d_model)
        if self.d_cond > 0:
            self.film = FiLM(d_cond, d_model)
        else:
            self.film = None

        # GPT-2 scaled residual init: downscale the two projections that write
        # into the residual stream (attention out, FFN out) so residual
        # variance stays bounded with depth. Pass n_layers = stack depth for
        # the depth-correct scaling (defaults to 1).
        self._init_residual_scale(n_layers)

        # Lazily-built local-attention mask. During training every step uses
        # the same chunk length / window / causal config, so the mask is built
        # on the first batch and reused thereafter. A single slot is kept; if a
        # later batch arrives with a different config (e.g. longer inference
        # sequence or new device) the mask is rebuilt in place.
        self._mask_cache = None
        self._mask_key = None

    def _make_local_mask(self, seq_len, window_size, device, causal=False):
        key = (seq_len, window_size, causal, device)
        if self._mask_key == key:
            return self._mask_cache
        # Vectorized build: diff[i, j] = j - i.
        idx = torch.arange(seq_len, device=device)
        diff = idx[None, :] - idx[:, None]
        if causal:
            # Position i may attend to [i - window_size, i] (never the future).
            allowed = (diff <= 0) & (diff >= -window_size)
        else:
            # Symmetric window: |j - i| <= window_size.
            allowed = diff.abs() <= window_size
        # Same convention as before: True = outside window (matches original).
        mask = ~allowed
        self._mask_cache = mask
        self._mask_key = key
        return mask

    @torch.no_grad()
    def _init_residual_scale(self, n_layers):
        # GPT-2 style: scale each residual-writing projection by
        # 1/sqrt(2 * n_layers). attn_out and the SwiGLU output projection both
        # feed directly into the residual stream.
        scale = 1.0 / math.sqrt(2.0 * max(1, n_layers))
        for module in (self.attn_out, self.ffn.w_out):
            module.weight.mul_(scale)

    def forward(self, x, cond=None):
        # Multi-head self-attention via fused SDPA
        h = self.norm_attn(x)
        B, T, _ = h.shape
        qkv = self.attn_qkv(h)  # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Rotary position embedding: rotate q/k so the q-k dot product encodes
        # relative position. Applied here, before SDPA, on each head.
        if self.rope is not None:
            q, k = self.rope(q, k)
        # Attention masking (identical semantics to ConformerBlock2):
        #  - causal + local_window > 0: causal sliding window.
        #  - causal + local_window == 0: full causal (SDPA fast path).
        #  - non-causal + local_window > 0: symmetric local window.
        #  - otherwise: full attention.
        is_causal = False
        if self.causal and self.local_window > 0:
            attn_mask = self._make_local_mask(T, self.local_window, x.device,
                                              causal=True)  # (T, T)
        elif self.causal:
            attn_mask = None
            is_causal = True  # SDPA builds the causal mask internally
        elif self.local_window > 0:
            attn_mask = self._make_local_mask(T, self.local_window, x.device)  # (T, T)
        else:
            attn_mask = None
        # bf16-friendly fused kernel
        h = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=is_causal,
            attn_mask=attn_mask
        )
        h = h.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, D)
        h = self.attn_out(h)
        x = x + h

        # Position-wise FFN (full residual step)
        x = x + self.ffn(self.norm_ffn(x))

        x = self.norm_final(x)
        if self.d_cond > 0:
            x = self.film(x, cond)
        return x
    





class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embedding (RoPE).

    Unlike additive positional encodings, RoPE injects position by *rotating*
    the query/key vectors inside attention, so it is applied per-head on
    `head_dim` rather than added to the model-width input. Relative position
    then falls out of the q-k dot product automatically.

    Caches cos/sin tables up to `init_len` and transparently grows (and
    re-caches) when a longer sequence arrives, so training on short chunks
    stays fast while inference on full-length songs just works.
    """

    def __init__(self, dim, base=1000.0, init_len=4096):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension (head_dim) must be even"
        self.dim = dim
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        # buffers so they move with .to(device) / .cuda()
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        cos, sin = self._build(init_len)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def _build(self, length):
        t = torch.arange(length, dtype=torch.float, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)        # (L, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)      # (L, dim)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        # q, k: (B, H, T, head_dim)
        T = q.size(-2)
        if T > self.cos.size(0):
            # Grow the cached tables with headroom to avoid repeated realloc.
            new_len = max(T, self.cos.size(0) * 2)
            self.cos, self.sin = self._build(new_len)
        cos = self.cos[:T].to(dtype=q.dtype)         # (T, head_dim)
        sin = self.sin[:T].to(dtype=q.dtype)
        # (T, head_dim) broadcasts against (B, H, T, head_dim)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with unbounded length.

    Kept for modules that use the additive-PE convention (e.g. those built on
    the older `ConformerBlock`, which does not consume RoPE). New attention
    stacks based on `ConformerBlock2` should prefer `RotaryPositionalEmbedding`.

    Caches a precomputed buffer up to `init_len` for the common case, and
    transparently expands (and re-caches) on the fly when a longer sequence
    arrives. This means training on short chunks stays fast, while inference
    on full-length songs (which can exceed the initial buffer) just works.
    """

    def __init__(self, d_model, init_len=4096):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(init_len, d_model)
        # register as buffer so it moves with .to(device) / .cuda()
        self.register_buffer('pe', pe, persistent=False)

    @staticmethod
    def _build_pe(length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe.unsqueeze(0)  # (1, L, D)

    def forward(self, x):
        T = x.size(1)
        if T > self.pe.size(1):
            # Grow the cached buffer to at least T (with some headroom to
            # avoid repeated reallocation if length keeps creeping up).
            new_len = max(T, self.pe.size(1) * 2)
            new_pe = self._build_pe(new_len, self.d_model).to(
                device=x.device, dtype=self.pe.dtype
            )
            self.pe = new_pe  # replace buffer in-place on the module
        return x + self.pe[:, :T]


# ----------------------------------------------------------------------
# Main frontend model
# ----------------------------------------------------------------------

class LeakageReductionFrontend(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_blocks=4,
        n_heads=8,
        ffn_mult=2,
        conv_kernel=15,
        dropout=0.1,
        d_whisper=1280,
        d_contentvec=768,
        d_hubert=256,
        d_spk=192,
        formant_clip=6.0,
        n_aug_buckets=32,  # for adversarial speaker classifier
        # Input noise: noise level is expressed as a fraction of each stream's
        # running feature std, so noise scales with the natural feature
        # magnitudes regardless of which SSL model produced them. Set per
        # stream to balance robustness vs. retained information.
        noise_whisper=0.10,
        noise_contentvec=0.15,
        noise_hubert=0.10,
    ):
        super().__init__()
        self.formant_clip = formant_clip
        self.d_contentvec = d_contentvec

        # Per-stream noise levels (fractions of feature std). Stored as a
        # buffer-like attribute that can be modified externally (e.g. for
        # warmup scheduling). Set to 0.0 to disable noise on a stream.
        self.noise_whisper = float(noise_whisper)
        self.noise_contentvec = float(noise_contentvec)
        self.noise_hubert = float(noise_hubert)

        # ---------------- Input projections ----------------
        self.proj_w = nn.Sequential(nn.Linear(d_whisper, d_model),
                                    nn.LayerNorm(d_model))
        self.proj_cv = nn.Sequential(nn.Linear(d_contentvec, d_model),
                                     nn.LayerNorm(d_model))
        self.proj_hs = nn.Sequential(nn.Linear(d_hubert, d_model),
                                     nn.LayerNorm(d_model))
        self.spk_mlp = nn.Sequential(
            nn.Linear(d_spk, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

        # ---------------- Gated fusion ----------------
        self.gate = nn.Linear(3 * d_model, 3)

        # ---------------- Conformer trunk ----------------
        # Single RoPE instance shared across blocks (parameter-free, just
        # caches cos/sin tables). Operates on per-head dim = d_model // n_heads.
        self.rope = RotaryPositionalEmbedding(d_model // n_heads)
        self.blocks = nn.ModuleList([
            ConformerBlock2(d_model, n_heads, ffn_mult, conv_kernel,
                           dropout, d_cond=d_model, rope=self.rope,
                           n_layers=n_blocks)
            for _ in range(n_blocks)
        ])

        # ---------------- Output head 1: ContentVec replacement ----------------
        # Gated residual: cv_out = (1 - g) * cv_in + g * cv_pred
        self.cv_pred = nn.Linear(d_model, d_contentvec)
        self.cv_gate = nn.Linear(d_model, d_contentvec)
        # Bias gate to start near 0 (heavily favor identity at init)
        nn.init.constant_(self.cv_gate.bias, -2.0)

        # ---------------- Output head 2: Formant shift ----------------
        self.fs_pre = nn.Linear(d_model, 64)
        self.fs_out = nn.Linear(64, 1)
        # Init last layer to zero -> formant_shift starts at 0 -> identity
        nn.init.zeros_(self.fs_out.weight)
        nn.init.zeros_(self.fs_out.bias)
        self.fs_smooth = nn.Conv1d(1, 1, kernel_size=9, padding=4, bias=False)
        nn.init.constant_(self.fs_smooth.weight, 1.0 / 9.0)

        # ---------------- Auxiliary heads (train-only) ----------------
        # These heads exist purely to shape the encoder's hidden state to
        # explicitly contain prosodic information (F0 + energy/volume).
        # They are not used at inference. The short gradient path from each
        # head back to `x` forces the representation to be linearly decodable
        # for these prosodic features, which in turn lets the main cv_pred
        # head inject prosody into cv_out more cleanly.

        # F0 prediction in linear Hz. Softplus guarantees f0 > 0, with a bias
        # to start predictions in a musically reasonable range (~200 Hz).
        self.aux_f0 = nn.Sequential(
            nn.Linear(d_model, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.aux_f0[-1].weight)
        nn.init.constant_(self.aux_f0[-1].bias, 200.0)

        # Volume / energy prediction. Output is a single positive scalar per
        # frame via softplus. Helps the encoder represent phonation timing
        # (attack/sustain/release) and amplitude envelope, both strong
        # prosodic / speaker-style cues.
        # Volume here is windowed RMS amplitude (linear, not dB), typically
        # 0.01-0.15 for voiced singing, with silence at ~0. We init bias to
        # the inverse-softplus of 0.05 so softplus(bias) ~= 0.05 at init,
        # matching the typical voiced mean.
        self.aux_volume = nn.Sequential(
            nn.Linear(d_model, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.aux_volume[-1].weight)
        # inverse softplus: b = log(exp(target) - 1)
        _vol_init_target = 0.05
        _vol_init_bias = math.log(math.exp(_vol_init_target) - 1)
        nn.init.constant_(self.aux_volume[-1].bias, _vol_init_bias)

        # Adversarial speaker / augmentation classifier on cv_out
        # Operates on time-pooled cv_out via gradient reversal
        self.adv_clf = nn.Sequential(
            nn.Linear(d_contentvec, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, n_aug_buckets),
        )

    def _inject_noise(self, x, noise_level):
        """
        Add Gaussian noise scaled to a fraction of each feature dimension's
        own std. Computed per (batch, channel) over the time axis so that
        the noise tracks the natural magnitude of each feature.

        Disabled in eval mode.
        """
        if not self.training or noise_level <= 0.0:
            return x
        # Per-channel std along time, then broadcast over time
        feat_std = x.std(dim=1, keepdim=True).detach()  # (B, 1, D)
        noise = torch.randn_like(x) * feat_std * noise_level
        return x + noise

    def forward(self, whisper, contentvec, hubert, spk_emb,
                grl_lambda=0.1, return_aux=True):
        # Optional input noise injection (training-time only). Scales with
        # each feature's own std so per-stream noise levels are comparable.
        whisper = self._inject_noise(whisper, self.noise_whisper)
        contentvec_clean = contentvec  # keep clean copy for residual gate
        contentvec = self._inject_noise(contentvec, self.noise_contentvec)
        hubert = self._inject_noise(hubert, self.noise_hubert)

        # Project all streams to d_model
        h_w = self.proj_w(whisper)
        h_cv = self.proj_cv(contentvec)
        h_hs = self.proj_hs(hubert)
        spk = self.spk_mlp(spk_emb)                       # (B, D)

        # Gated fusion
        g = F.softmax(self.gate(torch.cat([h_w, h_cv, h_hs], dim=-1)),
                      dim=-1)                              # (B, T, 3)
        fused = (g[..., 0:1] * h_w
                 + g[..., 1:2] * h_cv
                 + g[..., 2:3] * h_hs)
        # Position is injected later via RoPE inside each attention block,
        # so no additive positional encoding is applied here.
        x = fused

        # Conformer trunk with FiLM speaker conditioning
        for blk in self.blocks:
            x = blk(x, spk)

        # ----- ContentVec replacement (gated residual) -----
        # NOTE: the residual pass-through uses the CLEAN contentvec, not the
        # noised version. Otherwise we'd be passing noise into DDSP whenever
        # the gate is closed. The noised version only affects what the
        # encoder sees internally.
        cv_pred = self.cv_pred(x)
        gate = torch.sigmoid(self.cv_gate(x))
        cv_out = (1.0 - gate) * contentvec_clean + gate * cv_pred

        # ----- Formant shift -----
        fs = F.silu(self.fs_pre(x))
        fs = self.fs_out(fs)                               # (B, T, 1)
        fs = self.fs_smooth(fs.transpose(1, 2)).transpose(1, 2)
        # Soft bound: linear near 0, asymptotes near ±formant_clip
        c = self.formant_clip
        fs = c * torch.tanh(fs / c)

        if not return_aux:
            return cv_out, fs

        # ----- Auxiliary outputs -----
        # F0 in linear Hz, guaranteed positive via softplus.
        f0_raw = self.aux_f0(x)                            # (B, T, 1)
        f0_pred = F.softplus(f0_raw)                       # > 0, in Hz

        # Volume / energy, guaranteed positive via softplus.
        # Unit matches whatever batch['volume'] is (linear amplitude, RMS, etc.).
        vol_raw = self.aux_volume(x)                       # (B, T, 1)
        vol_pred = F.softplus(vol_raw)                     # > 0

        # Adversarial classifier on time-mean of cv_out
        cv_pooled = cv_out.mean(dim=1)                     # (B, 768)
        cv_rev = grad_reverse(cv_pooled, grl_lambda)
        adv_logits = self.adv_clf(cv_rev)                  # (B, n_buckets)

        aux = {
            'cv_pred': cv_pred,        # raw prediction (for diagnostics)
            'cv_gate': gate,           # per-frame, per-channel gate values
            'f0_pred': f0_pred,        # in linear Hz, > 0
            'vol_pred': vol_pred,      # volume / energy, > 0
            'adv_logits': adv_logits,
        }
        return cv_out, fs, aux


# ----------------------------------------------------------------------
# Parameter count sanity check
# ----------------------------------------------------------------------

if __name__ == '__main__':
    model = LeakageReductionFrontend()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:>12,}")
    print(f"Trainable params: {trainable:>12,}")

    # Quick forward smoke test
    B, T = 2, 200
    whisper = torch.randn(B, T, 1280)
    cv = torch.randn(B, T, 768)
    hs = torch.randn(B, T, 256)
    spk = torch.randn(B, 192)

    cv_out, fs, aux = model(whisper, cv, hs, spk)
    print(f"cv_out:        {tuple(cv_out.shape)}")
    print(f"formant_shift: {tuple(fs.shape)}  range=[{fs.min():.3f}, {fs.max():.3f}]")
    print(f"f0_pred:       {tuple(aux['f0_pred'].shape)}  range=[{aux['f0_pred'].min():.1f}, {aux['f0_pred'].max():.1f}] Hz")
    print(f"vol_pred:      {tuple(aux['vol_pred'].shape)}  range=[{aux['vol_pred'].min():.3f}, {aux['vol_pred'].max():.3f}]")
    print(f"adv_logits:    {tuple(aux['adv_logits'].shape)}")