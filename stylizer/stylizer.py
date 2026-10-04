import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddsp.model_conformer_naive import ConformerNaiveEncoder
from reflow.lynxnet2 import LYNXNet2
from reflow.reflow import RectifiedFlow
from stylizer import commons
from stylizer.attentions import LocalGlobalMultiheadAttention
from stylizer.flow import ConditionalNormalizingFlow, ConditionalNormalizingFlow2, ConditionalNormalizingFlow3, ConditionalNormalizingFlowMod2, ConditionalNormalizingFlowWN, ResidualCouplingBlock, ResidualCouplingBlockVits, ResidualCouplingBlockVits2, ResidualCouplingBlockVits3, ResidualCouplingLayerVits, WNConditionalNormalizingFlow
from stylizer.key_shifter import F0PatternPredictor
from stylizer.layers import ConditionalNormalizingFlow4, ConditionalNormalizingFlowSNAC
from stylizer.modules_grl import GradientReversal, SpeakerClassifier
from stylizer.rotation_estimater import RotScalePredictor, RotationScaleLayer
from stylizer.span_masking import mask_keep_a_mask_b
from stylizer.stlyizer_gen2 import ConformerBlock2, FiLM, RoformerBlock, RotaryPositionalEmbedding, SinusoidalPositionalEncoding
from stylizer.util import CausalConv1d, ConformerBlock, create_reflow, gaussian_blur_1d, random_rotation_scale
from torch.nn.utils import weight_norm, spectral_norm

from stylizer.vits.utils import f0_to_coarse

class GradientScaler(torch.autograd.Function):
    factor = 1.0
    @staticmethod
    def forward(ctx, input):
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        factor = GradientScaler.factor
        return factor.view(-1, 1, 1, 1)*grad_output

from stylizer.vqvae import VQVAE, ResidualVQ
from stylizer.wavenet import WN

    
class StylizerRefiner(nn.Module):
    def __init__(self, dim=768, num_layers=2, dropout=0.1, use_norm=False, usePrelu=False):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                weight_norm(nn.Linear(dim, dim)) if use_norm else nn.Linear(dim, dim),
                nn.PReLU() if usePrelu else nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.refine = nn.Sequential(*layers)
        # self.refine = LSTMProjectionLayer(input_dim=dim, num_layers=num_layers, dropout=dropout, bidirectional=False)
        self.residual_proj = weight_norm(nn.Linear(dim, dim)) if use_norm else nn.Linear(dim, dim)

    def forward(self, stylized_feats):
        refined = self.refine(stylized_feats)
        residual = self.residual_proj(refined)
        return stylized_feats + residual


class CnnSpeakerClassifier2(nn.Module):
    """
    A lightweight, CNN-based speaker classifier that focuses on local features.

    Args:
        input_dim (int): The dimension of the input embeddings (e.g., 256).
        num_channels (int): The number of channels in the CNN layers.
        output_dim (int): The dimension of the final speaker embedding (e.g., 256).
    """
    def __init__(self, input_dim=256, num_channels=256, output_dim=256, lambda_reversal=2.0):
        super(CnnSpeakerClassifier2, self).__init__()
        # Assuming you have this module defined elsewhere
        self.grad_revers = GradientReversal(lambda_reversal=lambda_reversal)
        # A sequence of 1D convolutional layers
        self.cnn = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=input_dim, out_channels=num_channels, kernel_size=5, padding=2)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=5, padding=2)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(in_channels=num_channels, out_channels=output_dim, kernel_size=5, padding=2)),
        )
                
        # Global average pooling to get a fixed-size vector
        # self.pooling = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(num_channels, output_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor of shape [B, T, D].
        """
        x = self.grad_revers(x)
        
        # Reshape for Conv1d: [B, D, T]
        x = x.transpose(1, 2)
        
        outputs = self.cnn(x)
        outputs = torch.mean(outputs, dim=-1)
        return outputs
    
class MelVaeEncoderWN(nn.Module):
    """Posterior Encoder: Takes target mel and encodes it to a latent z."""
    def __init__(self, mel_bins: int = 128, latent_dim: int = 192, cnn_channels: int = 512, rnn_hidden_dim: int = 256, speaker_dim=192, useCausual=False, conv_kernel_size=31, layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(mel_bins, latent_dim, kernel_size=1, padding='same'),
        )
        self.wn_projection = WN(latent_dim, conv_kernel_size, 1, layers, gin_channels=speaker_dim)
        self.proj = nn.Conv1d(latent_dim, latent_dim*2, 1)
    

    def forward(self, mel: torch.Tensor, g=torch.Tensor):
        x = mel.transpose(1, 2)
        g = g.transpose(1, 2)
        x = self.cnn(x)
        x = self.wn_projection(x, g)
        x = self.proj(x)
        mu, log = torch.split(x, self.latent_dim, dim=1)
        return mu.transpose(1, 2), log.transpose(1, 2) 
    
class MelVaeEncoderWN2(nn.Module):
    """Posterior Encoder: Takes target mel and encodes it to a latent z."""
    def __init__(self, mel_bins: int = 128, latent_dim: int = 192, cnn_channels: int = 512, rnn_hidden_dim: int = 256, speaker_dim=192, useCausual=False, conv_kernel_size=5, layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(mel_bins, latent_dim, kernel_size=1, padding='same'),
        )
        self.wn_projection = WN(latent_dim, conv_kernel_size, 1, layers, gin_channels=speaker_dim)
        self.proj = nn.Conv1d(latent_dim, latent_dim*2, 1)
    

    def forward(self, mel: torch.Tensor, g=torch.Tensor):
        x = mel.transpose(1, 2)
        g = g.transpose(1, 2)
        x = self.cnn(x)
        x = self.wn_projection(x, g)
        x = self.proj(x)
        mu, log = torch.split(x, self.latent_dim, dim=1)
        return mu.transpose(1, 2), log.transpose(1, 2)
    
    
class MelVaeEncoderWNMod2(nn.Module):
    """Posterior Encoder: Takes target mel and encodes it to a latent z."""
    def __init__(self, mel_bins: int = 128, latent_dim: int = 512, out_dim: int=768, rnn_hidden_dim: int = 256, speaker_dim=192, useCausual=False, conv_kernel_size=31, layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(mel_bins, latent_dim)
        self.encoder = ConformerBlock2(d_model=latent_dim, n_heads=8, conv_kernel=15, ffn_mult=4, dropout=0.1, local_window=12, d_cond=speaker_dim, causal=False, rope=None)
        self.mu_proj = nn.Linear(latent_dim, latent_dim)
        self.logvar_proj = nn.Linear(latent_dim, latent_dim)
    

    def forward(self, x: torch.Tensor, g=torch.Tensor):

        x = self.proj(x)
        encoeded = self.encoder(x, g)
        mu, log = self.mu_proj(encoeded), self.logvar_proj(encoeded)
        return mu, log
    
class ModifiedSoftVcStylizer4(nn.Module):
    def __init__(self, input_dim=2304, content_dim=192, speaker_dim=192, lambda_reversal=1.0, kernel_size=5, num_heads=2, window=8, dropout=0.1):
        #try content_dim = 256
        super().__init__()
        self.pre = nn.Conv1d(input_dim, content_dim, kernel_size=5, padding=2)
        self.encoder = ConformerBlock(content_dim, num_heads, kernel_size, 4, dropout, window)
        self.mu_proj = nn.Linear(content_dim, content_dim)
        self.logvar_proj = nn.Linear(content_dim, content_dim)
        self.classifer = CnnSpeakerClassifier2(content_dim, speaker_dim, speaker_dim, lambda_reversal=lambda_reversal)

    def forward(self, contentvec_feats, whisper_feats, hubert, f0, infer=False):  # (B, T, 2304) 
        input_feats = torch.cat([contentvec_feats, whisper_feats, hubert], dim=-1)
        
        latent_proj = self.pre(input_feats.transpose(1, 2)).transpose(1, 2)
        encodeded = self.encoder(latent_proj)
        mu = self.mu_proj(encodeded)
        logvar = self.logvar_proj(encodeded)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # B, _, L = z.shape
        # mask = torch.unsqueeze(commons.sequence_mask(torch.ones((B), device=z.device) + L, z.size(2)), 1).to(
        #     contentvec_feats.dtype
        # )
        if infer:
           spk_pred = None
        else:
            spk_pred = self.classifer(encodeded)
           
        return z, mu, logvar, spk_pred, None
    
class ModifiedSoftVcStylizerMod5(nn.Module):
    def __init__(self, input_dim=2304, content_dim=192, speaker_dim=192, lambda_reversal=1.0, kernel_size=5, num_heads=4, window=8, dropout=0.1, n_blocks=3):
        #try content_dim = 256
        super().__init__()
        self.pre = nn.Conv1d(input_dim, content_dim, kernel_size=5, padding=2)
        self.encoder_blocks = nn.Sequential(*[
            RoformerBlock(
                content_dim, num_heads, 4, 0.1, d_cond=0, rope=None, local_window=window)
            for _ in range(n_blocks)
        ])
        self.mu_proj = nn.Linear(content_dim, content_dim)
        self.logvar_proj = nn.Linear(content_dim, content_dim)
        self.classifer = CnnSpeakerClassifier2(content_dim, speaker_dim, speaker_dim, lambda_reversal=lambda_reversal)

    def forward(self, contentvec_feats, whisper_feats, hubert, f0, infer=False):  # (B, T, 2304) 
        input_feats = torch.cat([contentvec_feats, whisper_feats, hubert], dim=-1)
        
        latent_proj = self.pre(input_feats.transpose(1, 2)).transpose(1, 2)
        encodeded = self.encoder_blocks(latent_proj)
        mu = self.mu_proj(encodeded)
        logvar = self.logvar_proj(encodeded)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if infer:
           spk_pred = None
        else:
            spk_pred = self.classifer(encodeded)
           
        return z, mu, logvar, spk_pred, None
    
class ModifiedSoftVcStylizerMod6(nn.Module):
    def __init__(self, input_dim=2304, content_dim=192, speaker_dim=192, lambda_reversal=1.0, kernel_size=5, num_heads=4, window=16, dropout=0.1, n_blocks=2):
        #try content_dim = 256
        super().__init__()
        self.pre = nn.Conv1d(input_dim, content_dim, kernel_size=5, padding=2)
        self.encoder_blocks = nn.Sequential(*[
            RoformerBlock(
                content_dim, num_heads, 4, 0.1, d_cond=0, rope=None, local_window=8)
            for i in range(n_blocks)
        ])
        self.mu_proj = weight_norm(nn.Linear(content_dim, content_dim))
        self.logvar_proj = nn.Linear(content_dim, content_dim)
        self.classifer = CnnSpeakerClassifier2(content_dim, speaker_dim, speaker_dim, lambda_reversal=lambda_reversal)

    def forward(self, contentvec_feats, whisper_feats, hubert, f0, infer=False):  # (B, T, 2304) 
        input_feats = torch.cat([contentvec_feats, whisper_feats, hubert], dim=-1)
        latent_proj = self.pre(input_feats.transpose(1, 2)).transpose(1, 2)
        encodeded = self.encoder_blocks(latent_proj)
        mu = self.mu_proj(encodeded)
        logvar = self.logvar_proj(encodeded)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if infer:
            # z = mu
            spk_pred = None
        else:
            spk_pred = self.classifer(encodeded)
           
        return z, mu, logvar, spk_pred, None
    

class ModifiedSoftVcStylizerMod7(nn.Module):
    def __init__(self, input_dim=2304, content_dim=192, speaker_dim=192, lambda_reversal=1.0, kernel_size=5, num_heads=4, window=16, dropout=0.1, n_blocks=2):
        #try content_dim = 256
        super().__init__()
        self.pre = nn.Conv1d(input_dim, content_dim, kernel_size=11, padding=5)
        self.encoder_blocks = nn.Sequential(*[
            RoformerBlock(
                content_dim, num_heads, 4, 0.0, d_cond=0, rope=None, local_window=window)
            for _ in range(n_blocks)
        ])
        # self.mu_proj = nn.Linear(content_dim, content_dim)
        # self.logvar_proj = nn.Linear(content_dim, content_dim)
        self.classifer = CnnSpeakerClassifier2(content_dim, speaker_dim, speaker_dim, lambda_reversal=lambda_reversal)

    def forward(self, contentvec_feats, whisper_feats, hubert, f0, infer=False):  # (B, T, 2304) 
        input_feats = torch.cat([whisper_feats], dim=-1)
        latent_proj = self.pre(input_feats.transpose(1, 2)).transpose(1, 2)
        encodeded = self.encoder_blocks(latent_proj)
        
        if infer:
           spk_pred = None
        else:
            spk_pred = self.classifer(encodeded)
            
        return encodeded, spk_pred
    
    
class ModifiedSoftVcStylizerMod8(nn.Module):
    def __init__(self, input_dim=2304, content_dim=192, speaker_dim=192, lambda_reversal=1.0, kernel_size=5, num_heads=4, window=16, dropout=0.1, n_blocks=2):
        #try content_dim = 256
        super().__init__()
        self.pre = nn.Conv1d(input_dim, content_dim, kernel_size=5, padding=2)
        self.encoder_blocks = nn.Sequential(*[
            ConformerBlock2(d_model=content_dim, n_heads=num_heads, conv_kernel=5, ffn_mult=4, dropout=dropout, local_window=(i+1)*16, d_cond=0, causal=False, rope=None)
            for i in range(n_blocks)
        ])
       
        self.mu_proj = weight_norm(nn.Linear(content_dim, content_dim))
        self.logvar_proj = nn.Linear(content_dim, content_dim)
        self.classifer = CnnSpeakerClassifier2(content_dim, speaker_dim, speaker_dim, lambda_reversal=lambda_reversal)

    def forward(self, contentvec_feats, whisper_feats, hubert, f0, infer=False):  # (B, T, 2304) 
        input_feats = torch.cat([contentvec_feats, whisper_feats, hubert], dim=-1)
        latent_proj = self.pre(input_feats.transpose(1, 2)).transpose(1, 2)
        encodeded = self.encoder_blocks(latent_proj)
        mu = self.mu_proj(encodeded)
        logvar = self.logvar_proj(encodeded)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if infer:
            # z = mu
            spk_pred = None
        else:
            spk_pred = self.classifer(encodeded)
           
        return z, mu, logvar, spk_pred, None
    
    
class ModifiedSoftVcStylizerMod9(nn.Module):
    def __init__(self, input_dim=2304, content_dim=192, speaker_dim=192, lambda_reversal=1.0, kernel_size=5, num_heads=4, window=16, dropout=0.1, n_blocks=2):
        #try content_dim = 256
        super().__init__()
        self.pre = nn.Conv1d(input_dim, content_dim, kernel_size=5, padding=2)
        self.encoder_blocks = nn.Sequential(*[
            ConformerBlock2(d_model=content_dim, n_heads=num_heads, conv_kernel=5, ffn_mult=4, dropout=dropout, local_window=8, d_cond=0, causal=False, rope=None)
            for i in range(n_blocks)
        ])
        
        self.mu_proj = weight_norm(nn.Linear(content_dim, content_dim))
        self.logvar_proj = nn.Linear(content_dim, content_dim)
        self.classifer = CnnSpeakerClassifier2(content_dim, speaker_dim, speaker_dim, lambda_reversal=lambda_reversal)
   
    def forward(self, contentvec_feats, whisper_feats, hubert, f0, infer=False):  # (B, T, 2304) 
        input_feats = torch.cat([whisper_feats, hubert], dim=-1)
        latent_proj = self.pre(input_feats.transpose(1, 2)).transpose(1, 2)
        encodeded = self.encoder_blocks(latent_proj)
        mu = self.mu_proj(encodeded)
        logvar = self.logvar_proj(encodeded)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if infer:
            # z = mu
            spk_pred = None
        else:
            spk_pred = self.classifer(encodeded)
            
        return z, mu, logvar, spk_pred, None

    

class FormantShiftHeadSimple(nn.Module):
    """
    ContentVec + F0 -> per-frame formant shift, zero-centered per sample.
    No mask, no bounding (output can take any real value).

    Inputs:
      content: (B, T, D_c)
      f0_hz:   (B, T) or (B, T, 1)

    Output:
      shift:   (B, T, 1), zero-mean per sample
    """
    def __init__(
        self,
        d_content: int,
        hidden: int = 384,
        fix_rms: float | None = None,  # e.g., 0.5 to stabilize strength; None = no RMS fix
        eps: float = 1e-6,
        apply_norm=True
    ):
        super().__init__()
        self.fix_rms = fix_rms
        self.eps = eps

        in_dim = d_content + 2
        if apply_norm:
            self.mlp = nn.Sequential(
                weight_norm(nn.Linear(in_dim, hidden, bias=True)),
                nn.SiLU(),
                weight_norm(nn.Linear(hidden, hidden, bias=True)),
                nn.SiLU(),
                weight_norm(nn.Linear(hidden, 1, bias=False)),  # bias=False avoids DC offset
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden, bias=True),
                nn.SiLU(),
                nn.Linear(hidden, hidden, bias=True),
                nn.SiLU(),
                nn.Linear(hidden, 1, bias=False),  # bias=False avoids DC offset
            )
         # stable start
         
    def zscore_global(self, x:torch.Tensor, eps=1e-6):
        m = x.mean(dim=(0, 1), keepdim=True)
        s = x.std(dim=(0, 1), keepdim=True).clamp_min(eps)
        return (x - m) / s

    def forward(self, content, f0_hz:torch.Tensor):
        if f0_hz.dim() == 3:  # (B,T,1) -> (B,T)
            f0 = f0_hz.squeeze(-1)
        else:
            f0 = f0_hz

        # logF0 + delta
        f0_safe = f0.clamp_min(1.0)
        logf0 = torch.log(f0_safe)
        dlogf0 = F.pad(logf0[:, 1:] - logf0[:, :-1], (1, 0))

        # global z-score for scalars

        logf0_n  = self.zscore_global(logf0)[..., None]   # (B,T,1)
        dlogf0_n = self.zscore_global(dlogf0)[..., None]  # (B,T,1)

        x = torch.cat([content, logf0_n, dlogf0_n], dim=-1)  # (B,T,Dc+2)
        # Predict raw shift
        y = self.mlp(x)  # (B,T,1)
        # Zero-mean per sample
        y = y - y.mean(dim=1, keepdim=True)
        # Optional: fix RMS
        if self.fix_rms is not None:
            rms = torch.sqrt((y * y).mean(dim=1, keepdim=True) + self.eps)
            y = y * (self.fix_rms / (rms + self.eps))

        return y
    
    def init_zero(self):
        nn.init.zeros_(self.mlp[-1].weight)

class GlowVcStylizerWN5(nn.Module):
    def __init__(self, input_dim=2304, latent_dim=256, mel_bins=128, decoder_dim=768, num_flow_layers=4, decoder_layers=1, spk_dim=192):
        super().__init__()
        # try decoder_layers = 2 next
        self.latent_dim = latent_dim
        # self.contentvec_projection = nn.Linear(768, 256)
        self.spk_dim = spk_dim
        # Prior Encoder (your stylizer)
        self.prior_encoder = ModifiedSoftVcStylizer4(input_dim=input_dim, content_dim=latent_dim, speaker_dim=spk_dim)
        
        # Posterior Encoder
        self.posterior_encoder = MelVaeEncoderWN(mel_bins=mel_bins, latent_dim=latent_dim, useCausual=False, conv_kernel_size=15, speaker_dim=spk_dim)
        
        # Conditional Normalizing Flow
        self.flow = ConditionalNormalizingFlow(latent_dim=latent_dim, condition_dim=spk_dim, num_layers=num_flow_layers)
        
        self.formant_estimator = FormantShiftHeadSimple(decoder_dim, apply_norm=False)
                
        self.refiner = StylizerRefiner()
        
        self.decoder_input_proj = nn.Linear(latent_dim + spk_dim, decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            # new conv_kernel_size=31
            ConformerBlock(dim_model=decoder_dim, num_heads=4, conv_kernel_size=15, ff_multiplier=1, dropout=0.1) # try ff = 1
            for _ in range(decoder_layers)
        ])
        self.output_proj = nn.Linear(decoder_dim, decoder_dim)
        # self.reflow = RectifiedFlow(LYNXNet2(in_dims=decoder_dim, dim_cond=latent_dim, n_layers=6, n_chans=1024, kernel_size=5, use_wn=False, lite=False), out_dims=768, train_embed=True)

        
    def forward(self, contentvec_feats, whisper_feats, hubert_feats, speaker_feat, mel, f0, vol, infer=False, noise_fac=1e-4, t_start=0.0, infer_step=50, alpha=0., latent_only=False):
        B, L, _ = contentvec_feats.shape
        # contentvec_copy = contentvec_feats
        # contentvec_feats = self.contentvec_projection(contentvec_feats)
        if infer:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 0
            whisper_feats += torch.randn_like(whisper_feats) * noise_fac
            hubert_feats += torch.randn_like(hubert_feats) * 0
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = speaker_feat.unsqueeze(0).unsqueeze(0).expand(-1, L, -1)
            else:
                speaker_feat = speaker_feat.unsqueeze(1).expand(-1, L, -1)
        else:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 2
            whisper_feats += torch.randn_like(whisper_feats)
            hubert_feats += torch.randn_like(hubert_feats) * 2
            # speaker_feat = speaker_feat.unsqueeze(1).expand(-1, L, -1)
            speaker_feat = torch.broadcast_to(speaker_feat.unsqueeze(1), (B, L, self.spk_dim))
        if not infer:
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, f0, infer=False)
            mu_ps, logvar_ps = self.posterior_encoder(mel, speaker_feat)
            z_ps = mu_ps + torch.randn_like(mu_ps) * torch.exp(logvar_ps)
            z_fwd, log_det_fwd = self.flow(z_pr, speaker_feat)
            z_bkw, log_det_bkw = self.flow.inverse(z_ps, condition=speaker_feat)
        else:
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats,f0,infer=True)
            # print(z_pr.shape, speaker_feat.shape)
            z_fwd, log_det_fwd = self.flow(z_pr, speaker_feat)
            mu_ps= None 
            logvar_ps = None
            z_ps = None
            z_bkw = None
            log_det_bkw = None
            spk_pred = None
        # print(z_fwd.shape, speaker_feat.shape)
        if latent_only:
            return z_fwd
        s = torch.cat([z_fwd, speaker_feat], dim=-1)
        x = self.decoder_input_proj(s)
        for block in self.decoder_blocks:
            x = block(x)
        decode_proj = self.output_proj(x)
        reflow_loss = None
        stylized_feats = self.refiner(decode_proj)
        formant = self.formant_estimator(stylized_feats, f0)
        # multiplier = self.f0_estimator(stylized_feats, f0)
        # new_f0 = f0 / torch.pow(2, multiplier / 12.0)
        return stylized_feats, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask
    
class F0Embedding(nn.Module):
    """
    Embed continuous f0 (Hz) using:
    1. Sinusoidal encoding on log-frequency (captures continuous pitch incl. cents)
    2. Learned octave + pitch-class embeddings (captures musical structure)
    3. Learned unvoiced token for f0 == 0

    Expected range: ~65 Hz (C2) to ~1060 Hz (C6).
    """

    def __init__(
        self,
        d_model: int=64,
        f0_min: float = 65.0,
        f0_max: float = 1060.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.f0_min = f0_min
        self.f0_max = f0_max

        self.sin_dim = d_model // 2
        self.octave_embed = nn.Embedding(12, d_model // 4)
        self.pitch_class_embed = nn.Embedding(12, d_model // 4)
        self.unvoiced_embed = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """f0: [B, T] frequency in Hz, 0.0 for unvoiced/rap frames."""
        if len(f0.shape) == 3:
            f0 = f0.squeeze(dim=-1)
        device = f0.device
        voiced = f0 > 64  # [B, T]

        # Clamp to expected range so log2 is finite for unvoiced frames too
        f0_safe = f0.clamp(min=self.f0_min, max=self.f0_max)
        midi_cont = 69.0 + 12.0 * torch.log2(f0_safe / 440.0)  # [B, T]

        # 1. Sinusoidal encoding on continuous MIDI value
        half = self.sin_dim // 2
        div_term = torch.exp(
            torch.arange(0, half, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / half)
        )
        midi_float = midi_cont.float().unsqueeze(-1)  # [B, T, 1]
        sin_emb = torch.cat([
            torch.sin(midi_float * div_term),
            torch.cos(midi_float * div_term),
        ], dim=-1)  # [B, T, sin_dim]

        # 2. Quantize to nearest semitone for categorical embeddings
        midi_int = midi_cont.round().long().clamp(0, 127)
        octave = (midi_int // 12).clamp(0, 11)
        pitch_class = midi_int % 12
        oct_emb = self.octave_embed(octave)        # [B, T, d/4]
        pc_emb = self.pitch_class_embed(pitch_class)  # [B, T, d/4]

        # 3. Concatenate and project
        combined = torch.cat([sin_emb, oct_emb, pc_emb], dim=-1)  # [B, T, d_model]
        out = self.proj(combined)
        # Replace unvoiced frames with the learned unvoiced vector
        out = torch.where(voiced.unsqueeze(-1), out, self.unvoiced_embed)
        return out


class GlowVcStylizerWN5Mod3(nn.Module):
    def __init__(self, input_dim=2304, latent_dim=256, mel_bins=128, decoder_dim=768, num_flow_layers=4, decoder_layers=1, spk_dim=192):
        super().__init__()
        # try decoder_layers = 2 next
        self.latent_dim = latent_dim
        # self.contentvec_projection = nn.Linear(768, 256)
        self.spk_dim = spk_dim
        # Prior Encoder (your stylizer)
        self.prior_encoder = ModifiedSoftVcStylizer4(input_dim=input_dim, content_dim=latent_dim, speaker_dim=spk_dim)
        
        self.spk_mlp = nn.Sequential(
            nn.Linear(spk_dim, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim),
        )
        
        self.formant_clip = 6
        
        # Posterior Encoder
        self.posterior_encoder = MelVaeEncoderWN(mel_bins=mel_bins, latent_dim=latent_dim, useCausual=False, conv_kernel_size=15, speaker_dim=spk_dim, layers=2)
        
        # Conditional Normalizing Flow
        self.flow = ConditionalNormalizingFlow(latent_dim=latent_dim, condition_dim=spk_dim, num_layers=num_flow_layers)
                        
        self.refiner = StylizerRefiner()
        
        self.decoder_input_proj = nn.Linear(latent_dim, decoder_dim)
        self.pos_enc = SinusoidalPositionalEncoding(decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            # new conv_kernel_size=31
            ConformerBlock(dim_model=decoder_dim, num_heads=4, conv_kernel_size=15, ff_multiplier=1, dropout=0.1, local_window=8), # try ff = 1,
            ConformerBlock(dim_model=decoder_dim, num_heads=4, conv_kernel_size=31, ff_multiplier=1, dropout=0.1, local_window=31) # try ff = 1
            # for _ in range(decoder_layers)
        ])
        self.output_proj = nn.Linear(decoder_dim, decoder_dim)
        
        
         # ---------------- Output head 2: Formant shift ----------------
        self.fs_pre = nn.Linear(decoder_dim, 64)
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
            nn.Linear(decoder_dim, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.aux_f0[-1].weight)
        nn.init.constant_(self.aux_f0[-1].bias, 200.0)
        
        
        self.aux_volume = nn.Sequential(
            nn.Linear(decoder_dim, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.aux_volume[-1].weight)
        # inverse softplus: b = log(exp(target) - 1)
        _vol_init_target = 0.05
        _vol_init_bias = math.log(math.exp(_vol_init_target) - 1)
        nn.init.constant_(self.aux_volume[-1].bias, _vol_init_bias)

        
    def forward(self, contentvec_feats, whisper_feats, hubert_feats, speaker_feat, mel, f0, vol, infer=False, noise_fac=1e-4, t_start=0.0, infer_step=50, alpha=0., latent_only=False):
        B, L, _ = contentvec_feats.shape
        # contentvec_copy = contentvec_feats
        # contentvec_feats = self.contentvec_projection(contentvec_feats)
        if infer:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 0
            whisper_feats += torch.randn_like(whisper_feats) * noise_fac
            hubert_feats += torch.randn_like(hubert_feats) * 0
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                spk_emb = self.spk_mlp(speaker_feat.unsqueeze(0))
                speaker_feat = speaker_feat.unsqueeze(0).unsqueeze(0).expand(-1, L, -1)
            else:
                spk_emb = self.spk_mlp(speaker_feat)
                speaker_feat = speaker_feat.unsqueeze(1).expand(-1, L, -1)
        else:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 2
            whisper_feats += torch.randn_like(whisper_feats)
            hubert_feats += torch.randn_like(hubert_feats) * 2
            spk_emb = self.spk_mlp(speaker_feat)
            speaker_feat = torch.broadcast_to(speaker_feat.unsqueeze(1), (B, L, self.spk_dim))
        if not infer:
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, spk_emb, infer=False)
            mu_ps, logvar_ps = self.posterior_encoder(mel, speaker_feat)
            z_ps = mu_ps + torch.randn_like(mu_ps) * torch.exp(logvar_ps)
            z_fwd, log_det_fwd = self.flow(z_pr, speaker_feat)
            z_bkw, log_det_bkw = self.flow.inverse(z_ps, condition=speaker_feat)
        else:
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, spk_emb, infer=True)
            z_fwd, log_det_fwd = self.flow(z_pr, speaker_feat)
            mu_ps= None 
            logvar_ps = None
            z_ps = None
            z_bkw = None
            log_det_bkw = None
            spk_pred = None

        if latent_only:
            return z_fwd
        
        x = self.pos_enc(self.decoder_input_proj(z_fwd))
        
        fused = torch.zeros_like(contentvec_feats)
        for block in self.decoder_blocks:
            fused += block(x, spk_emb)

        decode_proj = self.output_proj(fused)
        reflow_loss = None
        stylized_feats = self.refiner(decode_proj)
        
        formant = F.silu(self.fs_pre(fused))
        formant = self.fs_out(formant)                               # (B, T, 1)
        formant = self.fs_smooth(formant.transpose(1, 2)).transpose(1, 2)
        # Soft bound: linear near 0, asymptotes near ±formant_clip
        c = self.formant_clip
        formant = c * torch.tanh(formant / c)
        
        f0_raw = self.aux_f0(fused)                       
        f0_pred = F.softplus(f0_raw)
        
        vol_raw = self.aux_volume(fused)                       
        vol_pred = F.softplus(vol_raw)
        
        return stylized_feats, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, f0_pred, log_det_bkw, spk_pred, reflow_loss,  formant, None, vol_pred

class GlowVcStylizerWN5Mod4(nn.Module):
    def __init__(self, input_dim=2304, latent_dim=384, mel_bins=128, decoder_dim=768, num_flow_layers=2, decoder_layers=1, spk_dim=192, train_formant=True):
        super().__init__()
        # try decoder_layers = 2 next
        self.latent_dim = latent_dim
        # self.contentvec_projection = nn.Linear(768, 256)
        self.spk_dim = spk_dim
        # Prior Encoder (your stylizer)
        self.prior_encoder = ModifiedSoftVcStylizerMod5(input_dim=input_dim, content_dim=latent_dim, speaker_dim=spk_dim)
        # Posterior Encoder
        # self.posterior_encoder = MelVaeEncoderWN(mel_bins=mel_bins, latent_dim=latent_dim, useCausual=False, conv_kernel_size=15, speaker_dim=spk_dim)
        self.train_formant = train_formant
        self.formant_estimator = FormantShiftHeadSimple(decoder_dim, apply_norm=False)
                
        self.refiner = StylizerRefiner()
        
        self.decoder_input_proj = weight_norm(nn.Linear(latent_dim, decoder_dim))
        # RoPE injects position inside each decoder block's attention (per-head
        # dim = decoder_dim // n_heads), replacing the old additive PE. Shared
        # across blocks since it is parameter-free (just cached cos/sin tables).
        self.rope = RotaryPositionalEmbedding(decoder_dim // 8)
        self.decoder_blocks = nn.ModuleList([
            # new conv_kernel_size=31
            ConformerBlock2(d_model=decoder_dim, n_heads=8, conv_kernel=127, ffn_mult=1, dropout=0.1, local_window=-1, d_cond=spk_dim, causal=False, rope=self.rope),
            ConformerBlock2(d_model=decoder_dim, n_heads=8, conv_kernel=63, ffn_mult=1, dropout=0.1, local_window=64, d_cond=spk_dim, causal=False, rope=self.rope), # try ff = 1
            ConformerBlock2(d_model=decoder_dim, n_heads=8, conv_kernel=31, ffn_mult=1, dropout=0.1, local_window=32, d_cond=spk_dim, causal=False, rope=self.rope),
            ConformerBlock2(d_model=decoder_dim, n_heads=8, conv_kernel=15, ffn_mult=1, dropout=0.1, d_cond=spk_dim, causal=False, rope=self.rope),
            # ConformerBlock(dim_model=decoder_dim, num_heads=4, conv_kernel_size=31, ff_multiplier=1, dropout=0.1, local_window=31) # try ff = 1
            # for _ in range(decoder_layers)
        ])
        self.output_proj = weight_norm(nn.Linear(decoder_dim, decoder_dim))
        self.reflow = RectifiedFlow(LYNXNet2(in_dims=decoder_dim, dim_cond=latent_dim, n_layers=6, n_chans=1024, kernel_size=5, use_wn=False, lite=False), out_dims=768, train_embed=True)

        
    def forward(self, contentvec_feats, whisper_feats, hubert_feats, speaker_feat, mel, f0, vol, infer=False, noise_fac=1e-4, t_start=0.0, infer_step=50, alpha=0., latent_only=False):
        B, L, _ = contentvec_feats.shape
        # contentvec_copy = contentvec_feats
        # contentvec_feats = self.contentvec_projection(contentvec_feats)
        if infer:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 0
            whisper_feats += torch.randn_like(whisper_feats) * noise_fac
            hubert_feats += torch.randn_like(hubert_feats) * 0
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = speaker_feat.unsqueeze(0)

        else:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 2
            whisper_feats += torch.randn_like(whisper_feats)
            hubert_feats += torch.randn_like(hubert_feats) * 2
            # speaker_feat = speaker_feat.unsqueeze(1).expand(-1, L, -1)
            # speaker_feat = torch.broadcast_to(speaker_feat.unsqueeze(1), (B, L, self.spk_dim))
        if not infer:
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, f0, infer=False)
            z_bkw, log_det_bkw = None, None
            mu_ps, logvar_ps = None, None
            z_ps = None
        else:
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats,f0,infer=True)
            # print(z_pr.shape, speaker_feat.shape)
            mu_ps= None 
            logvar_ps = None
            z_ps = None
            z_bkw = None
            log_det_bkw = None
            spk_pred = None
        z_fwd = z_pr
        if latent_only:
            return z_fwd
        # Position is now injected via RoPE inside each decoder block's
        # attention, so no additive positional encoding is applied here.
        x = self.decoder_input_proj(z_pr)
        # fused = torch.zeros_like(x)  # accumulator at decoder_dim, not input_dim
        for block in self.decoder_blocks:
            x = block(x, speaker_feat)
        decode_proj = self.output_proj(x)
        reflow_loss = None
        stylized_feats = self.refiner(decode_proj)
        if self.train_formant:
            formant = self.formant_estimator(stylized_feats, f0)
        else:
            formant = 0
        # multiplier = self.f0_estimator(stylized_feats, f0)
        # new_f0 = f0 / torch.pow(2, multiplier / 12.0)
        return stylized_feats, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, None, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask
    
    
class GlowVcStylizerWN5Mod6(nn.Module):
    def __init__(self, input_dim=2304, latent_dim=256, mel_bins=128, decoder_dim=768, num_flow_layers=4, decoder_layers=1, spk_dim=192, train_formant=True):
        super().__init__()
        # try decoder_layers = 2 next
        self.latent_dim = latent_dim
        # self.contentvec_projection = nn.Linear(768, 256)
        self.spk_dim = spk_dim
        # Prior Encoder (your stylizer)
        self.prior_encoder = ModifiedSoftVcStylizerMod6(input_dim=input_dim, content_dim=latent_dim, speaker_dim=spk_dim, n_blocks=4)
        # Posterior Encoder
        self.posterior_encoder = MelVaeEncoderWN2(mel_bins=mel_bins, latent_dim=latent_dim, useCausual=False, conv_kernel_size=5, speaker_dim=spk_dim)
        self.train_formant = train_formant
        self.flow = ResidualCouplingBlockVits2(latent_dim, latent_dim, 5, 1, num_flow_layers, gin_channels=spk_dim)
                
        self.refiner = StylizerRefiner()
        
        self.decoder_input_proj = weight_norm(nn.Linear(latent_dim, decoder_dim))
        # RoPE injects position inside each decoder block's attention (per-head
        # dim = decoder_dim // n_heads), replacing the old additive PE. Shared
        # across blocks since it is parameter-free (just cached cos/sin tables).
        # self.rope = RotaryPositionalEmbedding(decoder_dim // 8)
        self.rope = None
        self.formant_estimator = FormantShiftHeadSimple(decoder_dim, apply_norm=False)
        self.decoder_blocks = nn.ModuleList([
            # new conv_kernel_size=31
            ConformerBlock2(d_model=decoder_dim, n_heads=4, conv_kernel=127, ffn_mult=1, dropout=0.1, local_window=256, d_cond=spk_dim, causal=False, rope=self.rope),
            ConformerBlock2(d_model=decoder_dim, n_heads=4, conv_kernel=63, ffn_mult=1, dropout=0.1, local_window=64, d_cond=spk_dim, causal=False, rope=self.rope),
            ConformerBlock2(d_model=decoder_dim, n_heads=4, conv_kernel=31, ffn_mult=1, dropout=0.1, local_window=32, d_cond=spk_dim, causal=False, rope=self.rope),
            ConformerBlock2(d_model=decoder_dim, n_heads=4, conv_kernel=15, ffn_mult=1, dropout=0.1, d_cond=spk_dim, causal=False, rope=self.rope),
            # ConformerBlock(dim_model=decoder_dim, num_heads=8, conv_kernel_size=7, ff_multiplier=1, dropout=0.1, local_window=4) # try ff = 1
            # for _ in range(decoder_layers)
        ])
        self.output_proj = weight_norm(nn.Linear(decoder_dim, decoder_dim))
        # self.reflow = RectifiedFlow(LYNXNet2(in_dims=768, dim_cond=768, n_layers=4, n_chans=1024, kernel_size=7, use_wn=False, lite=False), out_dims=768, train_embed=True, loss_type='l1_lognorm')

        
    def forward(self, contentvec_feats, whisper_feats, hubert_feats, speaker_feat, mel, f0, vol, infer=False, noise_fac=1e-4, t_start=1.0, infer_step=50, alpha=0., latent_only=False):
        B, L, _ = contentvec_feats.shape
        # contentvec_copy = contentvec_feats
        # contentvec_feats = self.contentvec_projection(contentvec_feats)
        speaker_feat_orig = speaker_feat
        if len(speaker_feat_orig.shape) == 1:
            speaker_feat_orig = speaker_feat_orig.unsqueeze(0)
        speaker_feat_orig = F.normalize(speaker_feat_orig, dim=1)
        if infer:
            contentvec_feats_proj = contentvec_feats
            whisper_feats += torch.randn_like(whisper_feats) * noise_fac
            hubert_feats += torch.randn_like(hubert_feats) * 0
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = speaker_feat.unsqueeze(0)

        else:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 2.
            whisper_feats += torch.randn_like(whisper_feats)
            hubert_feats += torch.randn_like(hubert_feats) * 2.
        if not infer:
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = F.normalize(speaker_feat).unsqueeze(0).unsqueeze(0).expand(-1, L, -1)
            else:
                speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
                
            
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, f0, infer=False)
            mu_ps, logvar_ps = self.posterior_encoder(mel, speaker_feat)
            z_ps = mu_ps + torch.randn_like(mu_ps) * torch.exp(logvar_ps)
            z_fwd, logdet_fwd = self.flow(z_ps.transpose(1, 2), g=speaker_feat.transpose(1, 2))
            z_bkw, log_det_bkw = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_fwd = z_fwd.transpose(1, 2)
            z_bkw = z_bkw.transpose(1, 2)
            # log_det_bkw = None
        else:
            speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats,f0,infer=True)
            z_bkw, log_det_bkw = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_bkw = z_bkw.transpose(1, 2)
            mu_ps= None 
            logvar_ps = None
            z_ps = None
            # log_det_bkw = None
            logdet_fwd = None
            spk_pred = None
            z_fwd = None
            
        if infer:
            x = z_bkw
        else:
            x = z_ps
       
        # Position is now injected via RoPE inside each decoder block's
        # attention, so no additive positional encoding is applied here.
        x = self.decoder_input_proj(x)
        # fused = torch.zeros_like(x)  # accumulator at decoder_dim, not input_dim
        for block in self.decoder_blocks:
            x = block(x, speaker_feat_orig)
        decode_proj = self.output_proj(x)
        stylized_feats = self.refiner(decode_proj)
        if self.train_formant:
            formant = self.formant_estimator(stylized_feats, f0)
        else:
            formant = torch.tensor(0)
        # reflow_loss = torch.tensor(0)

        return stylized_feats, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, logdet_fwd, z_ps, mask
    
    def gen_noise(self, pri:torch.Tensor, pos:torch.Tensor):
        _, L, _ = pri.shape
        diff = pos.detach() - pri.detach()
        std, mean = torch.std_mean(diff, dim=1)          # each [B, C]
        mean = mean.unsqueeze(1).expand(-1, L, -1)        # [B, L, C]
        std = std.unsqueeze(1).expand(-1, L, -1)          # [B, L, C]
        return mean + torch.randn_like(pri) * std
    
    
class GlowVcStylizerWN5Mod7(nn.Module):
    def __init__(self, input_dim=2304, latent_dim=256, mel_bins=512, decoder_dim=768, num_flow_layers=4, decoder_layers=1, spk_dim=192, train_formant=True):
        super().__init__()
        # try decoder_layers = 2 next
        self.latent_dim = latent_dim
        # self.contentvec_projection = nn.Linear(768, 256)
        self.spk_dim = spk_dim
        # Prior Encoder (your stylizer)
        self.prior_encoder = ModifiedSoftVcStylizerMod6(input_dim=input_dim, content_dim=latent_dim, speaker_dim=spk_dim, n_blocks=4)
        # Posterior Encoder
        self.posterior_encoder = MelVaeEncoderWN2(mel_bins=mel_bins, latent_dim=latent_dim, useCausual=False, conv_kernel_size=5, speaker_dim=spk_dim)
        self.train_formant = train_formant
        self.flow = ResidualCouplingBlockVits2(latent_dim, latent_dim, 5, 1, num_flow_layers, gin_channels=spk_dim)
                
        self.refiner = StylizerRefiner()
        
        self.decoder_input_proj = weight_norm(nn.Linear(latent_dim, decoder_dim))
        # RoPE injects position inside each decoder block's attention (per-head
        # dim = decoder_dim // n_heads), replacing the old additive PE. Shared
        # across blocks since it is parameter-free (just cached cos/sin tables).
        # self.rope = RotaryPositionalEmbedding(decoder_dim // 8)
        self.rope = None
        self.formant_estimator = FormantShiftHeadSimple(decoder_dim, apply_norm=False)
        self.decoder_blocks = nn.ModuleList([
            # new conv_kernel_size=31
            ConformerBlock2(d_model=decoder_dim, n_heads=4, conv_kernel=15, ffn_mult=1, dropout=0.1, local_window=128, d_cond=spk_dim, causal=False, rope=self.rope),
            ConformerBlock2(d_model=decoder_dim, n_heads=4, conv_kernel=15, ffn_mult=1, dropout=0.1, local_window=64, d_cond=spk_dim, causal=False, rope=self.rope),
            ConformerBlock2(d_model=decoder_dim, n_heads=4, conv_kernel=15, ffn_mult=1, dropout=0.1, local_window=32, d_cond=spk_dim, causal=False, rope=self.rope),
            ConformerBlock2(d_model=decoder_dim, n_heads=4, conv_kernel=15, ffn_mult=1, dropout=0.1, d_cond=spk_dim, causal=False, rope=self.rope),
            # ConformerBlock(dim_model=decoder_dim, num_heads=8, conv_kernel_size=7, ff_multiplier=1, dropout=0.1, local_window=4) # try ff = 1
            # for _ in range(decoder_layers)
        ])
        self.output_proj = weight_norm(nn.Linear(decoder_dim, decoder_dim))
        # self.reflow = RectifiedFlow(LYNXNet2(in_dims=768, dim_cond=768, n_layers=4, n_chans=1024, kernel_size=7, use_wn=False, lite=False), out_dims=768, train_embed=True, loss_type='l1_lognorm')

        
    def forward(self, contentvec_feats, whisper_feats, hubert_feats, speaker_feat, mel, f0, vol, infer=False, noise_fac=1e-4, t_start=1.0, infer_step=50, alpha=0., latent_only=False):
        B, L, _ = contentvec_feats.shape
        # contentvec_copy = contentvec_feats
        # contentvec_feats = self.contentvec_projection(contentvec_feats)
        speaker_feat_orig = speaker_feat
        if len(speaker_feat_orig.shape) == 1:
            speaker_feat_orig = speaker_feat_orig.unsqueeze(0)
        speaker_feat_orig = F.normalize(speaker_feat_orig, dim=1)
        if infer:
            contentvec_feats_proj = contentvec_feats
            whisper_feats += torch.randn_like(whisper_feats) * noise_fac
            hubert_feats += torch.randn_like(hubert_feats) * 0
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = speaker_feat.unsqueeze(0)

        else:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 2.
            whisper_feats += torch.randn_like(whisper_feats)
            hubert_feats += torch.randn_like(hubert_feats) * 2.
        if not infer:
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = F.normalize(speaker_feat).unsqueeze(0).unsqueeze(0).expand(-1, L, -1)
            else:
                speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
                
            
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, f0, infer=False)
            mu_ps, logvar_ps = self.posterior_encoder(mel, speaker_feat)
            z_ps = mu_ps + torch.randn_like(mu_ps) * torch.exp(logvar_ps)
            z_fwd, logdet_fwd = self.flow(z_ps.transpose(1, 2), g=speaker_feat.transpose(1, 2))
            z_bkw, log_det_bkw = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_fwd = z_fwd.transpose(1, 2)
            z_bkw = z_bkw.transpose(1, 2)
            # log_det_bkw = None
        else:
            speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats,f0,infer=True)
            z_bkw, log_det_bkw = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_bkw = z_bkw.transpose(1, 2)
            mu_ps= None 
            logvar_ps = None
            z_ps = None
            # log_det_bkw = None
            logdet_fwd = None
            spk_pred = None
            z_fwd = None
            
        if infer:
            x = z_bkw
        else:
            # p = torch.rand((1))
            # if p.item() >= 0.2:
            #     x = z_ps
            # else:
            #     x = z_bkw
            x = z_ps
       
        # Position is now injected via RoPE inside each decoder block's
        # attention, so no additive positional encoding is applied here.
        x = self.decoder_input_proj(x)
        # fused = torch.zeros_like(x)  # accumulator at decoder_dim, not input_dim
        for block in self.decoder_blocks:
            x = block(x, speaker_feat_orig)
        decode_proj = self.output_proj(x)
        stylized_feats = self.refiner(decode_proj)
        if self.train_formant:
            formant = self.formant_estimator(stylized_feats, f0)
        else:
            formant = torch.tensor(0)
        # reflow_loss = torch.tensor(0)

        return stylized_feats, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, logdet_fwd, z_ps, mask
    
class GlowVcStylizerWN5Mod8(nn.Module):
    def __init__(self, input_dim=2304, latent_dim=256, mel_bins=512, decoder_dim=768, num_flow_layers=4, decoder_layers=1, spk_dim=192, train_formant=True):
        super().__init__()
        # try decoder_layers = 2 next
        self.latent_dim = latent_dim
        # self.contentvec_projection = nn.Linear(768, 256)
        self.spk_dim = spk_dim
        # Prior Encoder (your stylizer)
        self.prior_encoder = ModifiedSoftVcStylizerMod8(input_dim=input_dim, content_dim=latent_dim, speaker_dim=spk_dim, n_blocks=3)
        # Posterior Encoder
        self.posterior_encoder = MelVaeEncoderWN2(mel_bins=mel_bins, latent_dim=latent_dim, useCausual=False, conv_kernel_size=5, speaker_dim=spk_dim, layers=3)
        self.train_formant = train_formant
        self.flow = ResidualCouplingBlockVits2(latent_dim, latent_dim, 5, 1, num_flow_layers, gin_channels=spk_dim, n_flows=4)
                
        self.refiner = StylizerRefiner()
        
        self.decoder_input_proj = weight_norm(nn.Linear(latent_dim, decoder_dim))
        # RoPE injects position inside each decoder block's attention (per-head
        # dim = decoder_dim // n_heads), replacing the old additive PE. Shared
        # across blocks since it is parameter-free (just cached cos/sin tables).
        # self.rope = RotaryPositionalEmbedding(decoder_dim // 8)
        self.rope = None
        self.formant_estimator = FormantShiftHeadSimple(decoder_dim, apply_norm=False)
        self.film = FiLM(spk_dim, decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            # new conv_kernel_size=31
            # ConformerBlock(dim_model=decoder_dim, num_heads=4, conv_kernel_size=15, ff_multiplier=1, dropout=0.1)
            ConformerBlock2(d_model=decoder_dim, n_heads=8, conv_kernel=15, ffn_mult=1, dropout=0.1, d_cond=0, causal=False, rope=self.rope),
        ])
        self.output_proj = weight_norm(nn.Linear(decoder_dim, decoder_dim))
        # self.reflow = RectifiedFlow(LYNXNet2(in_dims=768, dim_cond=768, n_layers=4, n_chans=1024, kernel_size=7, use_wn=False, lite=False), out_dims=768, train_embed=True, loss_type='l1_lognorm')

        
    def forward(self, contentvec_feats, whisper_feats, hubert_feats, speaker_feat, mel, f0, vol, steps=0, infer=False, noise_fac=1e-4, t_start=1.0, infer_step=50, alpha=0., latent_only=False):
        B, L, _ = contentvec_feats.shape
        # contentvec_copy = contentvec_feats
        # contentvec_feats = self.contentvec_projection(contentvec_feats)
        speaker_feat_orig = speaker_feat
        if len(speaker_feat_orig.shape) == 1:
            speaker_feat_orig = speaker_feat_orig.unsqueeze(0)
        speaker_feat_orig = F.normalize(speaker_feat_orig, dim=1)
        if infer:
            contentvec_feats_proj = contentvec_feats
            whisper_feats += torch.randn_like(whisper_feats) * noise_fac
            hubert_feats += torch.randn_like(hubert_feats) * 0
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = speaker_feat.unsqueeze(0)

        else:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 2.
            whisper_feats += torch.randn_like(whisper_feats)
            hubert_feats += torch.randn_like(hubert_feats) * 2.
        if not infer:
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = F.normalize(speaker_feat).unsqueeze(0).unsqueeze(0).expand(-1, L, -1)
            else:
                speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
                
            
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, f0, infer=False)
            mu_ps, logvar_ps = self.posterior_encoder(mel, speaker_feat)
            z_ps = mu_ps + torch.randn_like(mu_ps) * torch.exp(logvar_ps)
            z_fwd, logdet_fwd = self.flow(z_ps.transpose(1, 2), g=speaker_feat.transpose(1, 2))
            z_bkw, log_det_bkw = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_fwd = z_fwd.transpose(1, 2)
            z_bkw = z_bkw.transpose(1, 2)
            # log_det_bkw = None
        else:
            speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats,f0,infer=True)
            z_bkw, log_det_bkw = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_bkw = z_bkw.transpose(1, 2)
            mu_ps= None 
            logvar_ps = None
            z_ps = None
            # log_det_bkw = None
            logdet_fwd = None
            spk_pred = None
            z_fwd = None
            
        if infer:
            x = z_bkw
        else:
            if steps >= 20000:
                p = torch.rand((1))
                if p.item() >= 0.3:
                    x = z_ps
                else:
                    x = z_bkw
            else:
                x = z_ps
       
        # Position is now injected via RoPE inside each decoder block's
        # attention, so no additive positional encoding is applied here.
        x = self.decoder_input_proj(x)
        x = self.film(x, speaker_feat_orig)
        for block in self.decoder_blocks:
            x = block(x, None)
        decode_proj = self.output_proj(x)
        stylized_feats = self.refiner(decode_proj)
        if self.train_formant:
            formant = self.formant_estimator(stylized_feats, f0)
        else:
            formant = torch.tensor(0)
        # reflow_loss = torch.tensor(0)

        return stylized_feats, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, logdet_fwd, z_ps, mask
    
class GlowVcStylizerWN5Mod9(nn.Module):
    def __init__(self, input_dim=1536, latent_dim=384, mel_bins=512, decoder_dim=768, num_flow_layers=4, decoder_layers=1, spk_dim=192, train_formant=True):
        super().__init__()
        # try decoder_layers = 2 next
        self.latent_dim = latent_dim
        # self.contentvec_projection = nn.Linear(768, 256)
        self.spk_dim = spk_dim
        # Prior Encoder (your stylizer)
        self.prior_encoder = ModifiedSoftVcStylizerMod9(input_dim=input_dim, content_dim=latent_dim, speaker_dim=spk_dim, n_blocks=3)
        # Posterior Encoder
        self.posterior_encoder = MelVaeEncoderWN2(mel_bins=mel_bins, latent_dim=latent_dim, useCausual=False, conv_kernel_size=5, speaker_dim=spk_dim, layers=3)
        self.train_formant = train_formant
        self.flow = ResidualCouplingBlockVits2(latent_dim, latent_dim, 5, 1, num_flow_layers, gin_channels=spk_dim, n_flows=4)
                
        self.refiner = StylizerRefiner()
        
        self.decoder_input_proj = weight_norm(nn.Linear(latent_dim, decoder_dim))
        # RoPE injects position inside each decoder block's attention (per-head
        # dim = decoder_dim // n_heads), replacing the old additive PE. Shared
        # across blocks since it is parameter-free (just cached cos/sin tables).
        # self.rope = RotaryPositionalEmbedding(decoder_dim // 8)
        self.rope = None
        self.formant_estimator = FormantShiftHeadSimple(decoder_dim, apply_norm=False)
        self.film = FiLM(spk_dim, decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            # new conv_kernel_size=31
            # ConformerBlock(dim_model=decoder_dim, num_heads=4, conv_kernel_size=15, ff_multiplier=1, dropout=0.1)
            ConformerBlock2(d_model=decoder_dim, n_heads=8, conv_kernel=15, ffn_mult=1, dropout=0.1, d_cond=0, causal=False, rope=self.rope),
        ])
        self.output_proj = weight_norm(nn.Linear(decoder_dim, decoder_dim))
        # self.reflow = RectifiedFlow(LYNXNet2(in_dims=768, dim_cond=768, n_layers=4, n_chans=1024, kernel_size=7, use_wn=False, lite=False), out_dims=768, train_embed=True, loss_type='l1_lognorm')

        
    def forward(self, contentvec_feats, whisper_feats, hubert_feats, speaker_feat, mel, f0, vol, steps=0, infer=False, noise_fac=1e-4, t_start=1.0, infer_step=50, alpha=0., latent_only=False):
        B, L, _ = contentvec_feats.shape
        # contentvec_copy = contentvec_feats
        # contentvec_feats = self.contentvec_projection(contentvec_feats)
        speaker_feat_orig = speaker_feat
        if len(speaker_feat_orig.shape) == 1:
            speaker_feat_orig = speaker_feat_orig.unsqueeze(0)
        speaker_feat_orig = F.normalize(speaker_feat_orig, dim=1)
        if infer:
            contentvec_feats_proj = contentvec_feats
            whisper_feats += torch.randn_like(whisper_feats) * noise_fac
            hubert_feats += torch.randn_like(hubert_feats) * 0
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = speaker_feat.unsqueeze(0)

        else:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 2.
            whisper_feats += torch.randn_like(whisper_feats) * 2.
            hubert_feats += torch.randn_like(hubert_feats) * 2.
        if not infer:
            if len(speaker_feat.shape) == 1:
                # Instead of expand():
                speaker_feat = F.normalize(speaker_feat).unsqueeze(0).unsqueeze(0).expand(-1, L, -1)
            else:
                speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
                
            
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, f0, infer=False)
            mu_ps, logvar_ps = self.posterior_encoder(mel, speaker_feat)
            z_ps = mu_ps + torch.randn_like(mu_ps) * torch.exp(logvar_ps)
            z_fwd, logdet_fwd = self.flow(z_ps.transpose(1, 2), g=speaker_feat.transpose(1, 2))
            z_bkw, log_det_bkw = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_fwd = z_fwd.transpose(1, 2)
            z_bkw = z_bkw.transpose(1, 2)
            # log_det_bkw = None
        else:
            speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
            z_pr, mu_pr, logvar_pr, spk_pred, mask = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats,f0,infer=True)
            z_bkw, log_det_bkw = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_bkw = z_bkw.transpose(1, 2)
            mu_ps= None 
            logvar_ps = None
            z_ps = None
            # log_det_bkw = None
            logdet_fwd = None
            spk_pred = None
            z_fwd = None
            
        if infer:
            x = z_bkw
        else:
            if steps >= 20000:
                p = torch.rand((1))
                if p.item() >= 0.3:
                    x = z_ps
                else:
                    x = z_bkw
            else:
                x = z_ps
       
        # Position is now injected via RoPE inside each decoder block's
        # attention, so no additive positional encoding is applied here.
        x = self.decoder_input_proj(x)
        x = self.film(x, speaker_feat_orig)
        for block in self.decoder_blocks:
            x = block(x, None)
        decode_proj = self.output_proj(x)
        stylized_feats = self.refiner(decode_proj)
        if self.train_formant:
            formant = self.formant_estimator(stylized_feats, f0)
        else:
            formant = torch.tensor(0)
        # reflow_loss = torch.tensor(0)

        return stylized_feats, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, logdet_fwd, z_ps, mask