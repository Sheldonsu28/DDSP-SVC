import torch
import torch.nn as nn
import torch.nn.functional as F

from ddsp.model_conformer_naive import ConformerNaiveEncoder
from reflow.lynxnet2 import Transpose, WnWrapper, WnWrapper1
from stylizer.stlyizer_gen2 import ConformerBlock2, FiLM, RoformerBlock, RotaryPositionalEmbedding
from stylizer.util import ConformerBlock
from stylizer.wavenet import WN, WN_mask

# from reflow.lynxnet2 import Transpose
# from stylizer.util import CausalConv1d, WaveNetBlock, WaveNetScaleTranslateNet

class ConditionalAffineCouplingLayer(nn.Module):
    """A conditional affine coupling layer for a normalizing flow."""
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int = 512, pRelu=False):
        super().__init__()
        self.latent_dim = latent_dim
                
        self.scale_translate_net = nn.Sequential(
            nn.Linear(latent_dim // 2 + condition_dim, hidden_dim),
            nn.ReLU() if not pRelu else nn.PReLU(),
            # ConformerBlock(hidden_dim, 1, 5, 2, 0., local_window=8, causal=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if not pRelu else nn.PReLU(),
            nn.Linear(hidden_dim, latent_dim - (latent_dim // 2))
        )
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        z1, z2 = z.chunk(2, dim=-1)
        st_input = torch.cat([z1, condition], dim=-1)
        s_t = self.scale_translate_net(st_input)
        scale = torch.sigmoid(s_t + 2.0) + 1e-6
        y2 = z2 * scale
        y = torch.cat([z1, y2], dim=-1)
        log_det_j = torch.sum(torch.log(scale), dim=[1, 2])
        return y, log_det_j

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        y1, y2 = y.chunk(2, dim=-1)
        st_input = torch.cat([y1, condition], dim=-1)
        s_t = self.scale_translate_net(st_input)
        scale = torch.sigmoid(s_t + 2.0) + 1e-6
        z2 = y2 / scale
        z = torch.cat([y1, z2], dim=-1)
        log_det_j = torch.sum(torch.log(scale), dim=[1, 2])
        return z, log_det_j
    
    def zero_output(self):
        pass
        # self.post.weight.data.zero_()
        # self.post.bias.data.zero_()
        
        

class ConditionalAffineCouplingLayer5(nn.Module):
    """A conditional affine coupling layer for a normalizing flow."""
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int = 512, pRelu=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.film = FiLM(condition_dim, latent_dim)
                
        self.scale_translate_net = nn.Sequential(
            nn.Linear(latent_dim // 2, hidden_dim),
            nn.ReLU() if not pRelu else nn.PReLU(),
            # ConformerBlock(hidden_dim, 1, 5, 2, 0., local_window=8, causal=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if not pRelu else nn.PReLU(),
            nn.Linear(hidden_dim, latent_dim - (latent_dim // 2))
        )
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        z1, z2 = z.chunk(2, dim=-1)
        st_input = self.film(z, condition)
        s_t = self.scale_translate_net(st_input)
        scale = torch.sigmoid(s_t + 2.0) + 1e-6
        y2 = z2 * scale
        y = torch.cat([z1, y2], dim=-1)
        log_det_j = torch.sum(torch.log(scale), dim=[1, 2])
        return y, log_det_j

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        y1, y2 = y.chunk(2, dim=-1)
        st_input = self.film(y, condition)
        s_t = self.scale_translate_net(st_input)
        scale = torch.sigmoid(s_t + 2.0) + 1e-6
        z2 = y2 / scale
        z = torch.cat([y1, z2], dim=-1)
        log_det_j = torch.sum(torch.log(scale), dim=[1, 2])
        return z, log_det_j
    
    def zero_output(self):
        pass
        # self.post.weight.data.zero_()
        # self.post.bias.data.zero_()

from torch.nn.utils import weight_norm
class ConditionalAffineCouplingLayerMod2(nn.Module):
    """A conditional affine coupling layer for a normalizing flow."""
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int = 512, pRelu=False):
        super().__init__()
        self.latent_dim = latent_dim
                
        self.scale_translate_net = nn.Sequential(
            nn.Linear(latent_dim // 2 + condition_dim, hidden_dim),
            nn.ReLU() if not pRelu else nn.PReLU(),
            # ConformerBlock(hidden_dim, 1, 5, 2, 0., local_window=8, causal=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if not pRelu else nn.PReLU(),
            nn.Linear(hidden_dim, latent_dim - (latent_dim // 2))
        )
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        z1, z2 = z.chunk(2, dim=-1)
        st_input = torch.cat([z1, condition], dim=-1)
        s_t = self.scale_translate_net(st_input)
        scale = torch.sigmoid(s_t + 2.0) + 1e-6
        y2 = z2 * scale
        y = torch.cat([z1, y2], dim=-1)
        log_det_j = torch.sum(torch.log(scale), dim=[1, 2])
        return y, log_det_j

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        y1, y2 = y.chunk(2, dim=-1)
        st_input = torch.cat([y1, condition], dim=-1)
        s_t = self.scale_translate_net(st_input)
        scale = torch.sigmoid(s_t + 2.0) + 1e-6
        z2 = y2 / scale
        z = torch.cat([y1, z2], dim=-1)
        log_det_j = torch.sum(torch.log(scale), dim=[1, 2])
        return z, log_det_j
    
    def zero_output(self):
        pass
        # self.post.weight.data.zero_()
        # self.post.bias.data.zero_()
        
        
class ConditionalAffineCouplingLayer4(nn.Module):
    """A conditional affine coupling layer for a normalizing flow."""
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
                
        self.scale_translate_net = nn.Sequential(
            nn.Linear(latent_dim // 2 + condition_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, latent_dim - (latent_dim // 2))
        )
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor, reverse=False):
        z1, z2 = z.chunk(2, dim=-1)
        st_input = torch.cat([z1, condition], dim=-1)
        s_t = self.scale_translate_net(st_input)
        scale = torch.sigmoid(s_t + 2.0) + 1e-6
        if reverse:
            z2 = z2 / scale
            z = torch.cat([z1, z2], dim=-1)
            log_det_j = torch.sum(torch.log(scale), dim=[1, 2])
            return z, log_det_j
        y2 = z2 * scale
        y = torch.cat([z1, y2], dim=-1)
        log_det_j = torch.sum(torch.log(scale), dim=[1, 2])
        return y, log_det_j

    # def inverse(self, y: torch.Tensor, condition: torch.Tensor):
    #     y1, y2 = y.chunk(2, dim=-1)
    #     st_input = torch.cat([y1, condition], dim=-1)
    #     s_t = self.scale_translate_net(st_input)
    #     scale = torch.sigmoid(s_t + 2.0) + 1e-6
    #     z2 = y2 / scale
    #     z = torch.cat([y1, z2], dim=-1)
    #     log_det_j = torch.sum(torch.log(scale), dim=[1, 2])
    #     return z, log_det_j
        
        
class ConditionalAffineCouplingLayer2(nn.Module):
    """A conditional affine coupling layer for a normalizing flow."""
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int = 512,
                 init_identity: bool = True, clamp_tanh: float = 3.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.split = latent_dim // 2
        self.out_dim = latent_dim - self.split  # size of z2

        # We output BOTH log_s and t → 2*out_dim
        self.st_net = nn.Sequential(
            nn.Linear(self.split + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.out_dim)
        )
        self.clamp_tanh = clamp_tanh

        if init_identity:
            # Initialize last layer to small values so the layer starts ~ identity
            last = self.st_net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        # z: [B, L, D], condition: [B, L, C]
        z1, z2 = z[:, :, :self.split], z[:, :, self.split:]
        st_input = torch.cat([z1, condition], dim=-1)
        st = self.st_net(st_input)                      # [B, L, 2*out_dim]
        log_s, t = st.split(self.out_dim, dim=-1)       # [B, L, out_dim] each

        # Stable log-scale: clamp via tanh
        log_s = self.clamp_tanh * torch.tanh(log_s)     # ∈ (-c, c)
        y2 = z2 * torch.exp(log_s) + t
        y = torch.cat([z1, y2], dim=-1)

        # Forward log-det is sum(log_s) over time & features
        log_det_j = torch.sum(log_s, dim=(1, 2))        # [B]
        return y, log_det_j

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        y1, y2 = y[:, :, :self.split], y[:, :, self.split:]
        st_input = torch.cat([y1, condition], dim=-1)
        st = self.st_net(st_input)
        log_s, t = st.split(self.out_dim, dim=-1)

        log_s = self.clamp_tanh * torch.tanh(log_s)
        z2 = (y2 - t) * torch.exp(-log_s)
        z = torch.cat([y1, z2], dim=-1)

        # IMPORTANT: inverse log-det is NEGATIVE of forward’s
        log_det_j = -torch.sum(log_s, dim=(1, 2))       # [B]
        return z, log_det_j
    
    def zero_output(self):
        last = self.st_net[-1]
        last.weight.zero_()
        last.bias.zero_()
        
        
class ConditionalAffineCouplingLayer3(nn.Module):
    """A conditional affine coupling layer for a normalizing flow."""
    def __init__(self, latent_dim, condition_dim, hidden_dim=512,
                 init_identity=True, clamp_tanh=2.0, center_volume=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.split = latent_dim // 2
        self.out_dim = latent_dim - self.split
        self.center_volume = center_volume
        self.clamp_tanh = clamp_tanh

        # self.input_layer = nn.Linear(latent_dim, latent_dim)
        # self.wn = WN_mask(latent_dim, 5, 1, 4, condition_dim)
        self.st_net = nn.Sequential(
            nn.Linear(self.split + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.out_dim),
        )
        if init_identity:
            last = self.st_net[-1]
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, z, condition):
        # z = self.input_layer(z)
        # z = self.wn(z.transpose(1, 2), condition.transpose(1, 2)).transpose(1, 2)
        z1, z2 = z[:, :, :self.split], z[:, :, self.split:]
        st = self.st_net(torch.cat([z1, condition], dim=-1))
        log_s, t = st.split(self.out_dim, dim=-1)

        # clamp log-scale
        log_s = torch.tanh(log_s)

        # NEW: remove per-sample mean to prevent runaway volume
        if self.center_volume:
            log_s = log_s - log_s.mean(dim=(1, 2), keepdim=True)

        y2 = z2 * torch.exp(log_s) + t
        y = torch.cat([z1, y2], dim=-1)
        log_det_j = torch.sum(log_s, dim=(1, 2))  # ≈ 0 when centered
        return y, log_det_j

    def inverse(self, y, condition):
        # y = self.input_layer(y)
        # y = self.wn(y.transpose(1, 2), condition.transpose(1, 2)).transpose(1, 2)
        y1, y2 = y[:, :, :self.split], y[:, :, self.split:]
        st = self.st_net(torch.cat([y1, condition], dim=-1))
        log_s, t = st.split(self.out_dim, dim=-1)
        log_s = torch.tanh(log_s)
        if self.center_volume:
            log_s = log_s - log_s.mean(dim=(1, 2), keepdim=True)
        z2 = (y2 - t) * torch.exp(-log_s)
        z = torch.cat([y1, z2], dim=-1)
        log_det_j = -torch.sum(log_s, dim=(1, 2))
        return z, log_det_j
        
        
class ConditionalAffineCouplingLayerWN(nn.Module):
    """A conditional affine coupling layer for a normalizing flow."""
    def __init__(self, latent_dim: int, condition_dim: int, clamp=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.wn = WnWrapper1(latent_dim//2, 5, 1, 4, gin_channels=condition_dim, p_dropout=0.1)
        self.clamp = float(clamp)
        
        self.out_proj = nn.Conv1d(latent_dim // 2, latent_dim, kernel_size=1)
    
    def _st_from_wn(self, z1: torch.Tensor, condition: torch.Tensor):
        """
        z1: (B, T, C1), condition: (B, T, Cg) or (B, 1, Cg) or (B, Cg)
        Returns log_scale, translate with shape (B, T, C2).
        """
        x = z1.transpose(1, 2)             # (B, C1, T) for WaveNet
        wn_out = self.wn(x, g=condition.transpose(1, 2))            # (B, C1, T)  -- your wrapper’s forward
        st = self.out_proj(wn_out)          # (B, 2*C2, T)
        st = st.transpose(1, 2)             # (B, T, 2*C2)
        s, t = torch.split(st, self.latent_dim//2, dim=-1)    # (B, T, C2) each
        log_scale = torch.tanh(s) * self.clamp     # keep scales bounded
        translate = t
        return log_scale, translate
    
    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        z1, z2 = z.split([self.latent_dim//2, self.latent_dim//2], dim=-1)
        log_s, t = self._st_from_wn(z1, condition)
        y1 = z1
        y2 = (z2 + t) * torch.exp(log_s)
        y = torch.cat([y1, y2], dim=-1)
        # sum over transformed dims (time and C2)
        log_det = -log_s.sum(dim=[1, 2])
        return y, log_det


    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        y1, y2 = y.split([self.latent_dim//2, self.latent_dim//2], dim=-1)
        log_s, t = self._st_from_wn(y1, condition)

        z1 = y1
        z2 = y2 * torch.exp(-log_s) - t
        z = torch.cat([z1, z2], dim=-1)

        forward_log_det = -log_s.sum(dim=[1, 2])
        return z, forward_log_det
    
    def zero_post_weights(self):
        print('zero')
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            print('zero bias')
            nn.init.zeros_(self.out_proj.bias)
        
        
# class ResidualCouplingLayer(nn.Module):
#     def __init__(
#         self,
#         channels,
#         hidden_channels,
#         kernel_size,
#         dilation_rate,
#         n_layers,
#         p_dropout=0,
#         gin_channels=0,
#         mean_only=False,
#     ):
#         assert channels % 2 == 0, "channels should be divisible by 2"
#         super().__init__()
#         self.channels = channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.dilation_rate = dilation_rate
#         self.n_layers = n_layers
#         self.half_channels = channels // 2
#         self.mean_only = mean_only

#         self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
#         # no use gin_channels
#         self.enc = WnWrapper1(
#             hidden_channels,
#             kernel_size,
#             dilation_rate,
#             n_layers,
#             p_dropout=p_dropout,
#         )
#         self.post = nn.Conv1d(
#             hidden_channels, self.half_channels * (2 - mean_only), 1)
#         self.post.weight.data.zero_()
#         self.post.bias.data.zero_()
#         # SNAC Speaker-normalized Affine Coupling Layer
#         self.snac = nn.Conv1d(gin_channels, 2 * self.half_channels, 1)

#     def forward(self, x, g=None):
#         # print(self.post.bias.data)
#         # print(g.shape)
#         speaker = self.snac(g)
#         speaker_m, speaker_v = speaker.chunk(2, dim=1)  # (B, half_channels, 1)
#         x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
#         # x0 norm
#         x0_norm = (x0 - speaker_m) * torch.exp(-speaker_v)
#         h = self.pre(x0_norm)
#         # don't use global condition
#         h = self.enc(h)
#         stats = self.post(h)
#         if not self.mean_only:
#             m, logs = torch.split(stats, [self.half_channels] * 2, 1)
#         else:
#             m = stats
#             logs = torch.zeros_like(m)
        
#         # x1 norm before affine xform
#         x1_norm = (x1 - speaker_m) * torch.exp(-speaker_v)
#         x1 = (m + x1_norm * torch.exp(logs))
#         x = torch.cat([x0, x1], 1)
#         # speaker var to logdet
#         logdet = torch.sum(logs, [1, 2]) - torch.sum(
#             speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
#         return x, logdet
        
#     def inverse(self, x, g=None):
        
#         speaker = self.snac(g)
#         speaker_m, speaker_v = speaker.chunk(2, dim=1)  # (B, half_channels, 1)
#         x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
#         # x0 norm
#         x0_norm = (x0 - speaker_m) * torch.exp(-speaker_v)
#         h = self.pre(x0_norm)
#         # don't use global condition
#         h = self.enc(h)
#         stats = self.post(h)
#         if not self.mean_only:
#             m, logs = torch.split(stats, [self.half_channels] * 2, 1)
#         else:
#             m = stats
#             logs = torch.zeros_like(m)
        
#         x1 = (x1 - m) * torch.exp(-logs)
#         # x1 denorm before output
#         x1 = (speaker_m + x1 * torch.exp(speaker_v))
#         x = torch.cat([x0, x1], 1)
#         # speaker var to logdet
#         logdet = torch.sum(-logs, [1, 2]) + torch.sum(
#             speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
#         return x, logdet

#     def zero_post_weights(self):
#         self.post.weight.data.zero_()
#         self.post.bias.data.zero_()

class ConditionalNormalizingFlow(nn.Module):
    """A stack of conditional affine coupling layers."""
    def __init__(self, latent_dim: int, condition_dim: int, num_flow: int=4, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            # RealNVPCouplingLayer(latent_dim, condition_dim, num_wavenet_blocks=num_layers) for _ in range(num_flow)
            ConditionalAffineCouplingLayer(latent_dim, condition_dim) for _ in range(num_layers)
            # ResidualCouplingLayer(latent_dim, 192, 5, 1, 4, gin_channels=condition_dim) for _ in range(num_layers)
        ])

    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in self.layers:
            z, log_det_j = layer(z, condition)
            log_det_j_total += log_det_j
        return z, log_det_j_total

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in reversed(self.layers):
            y, log_det_j = layer.inverse(y, condition)
            log_det_j_total += log_det_j
        return y, log_det_j_total
    
    def zeros_layer(self):
        for layer in self.layers:
            layer.zero_post_weights()
            
class ConditionalNormalizingFlow5(nn.Module):
    """A stack of conditional affine coupling layers."""
    def __init__(self, latent_dim: int, condition_dim: int, num_flow: int=4, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            # RealNVPCouplingLayer(latent_dim, condition_dim, num_wavenet_blocks=num_layers) for _ in range(num_flow)
            ConditionalAffineCouplingLayer5(latent_dim, condition_dim) for _ in range(num_layers)
            # ResidualCouplingLayer(latent_dim, 192, 5, 1, 4, gin_channels=condition_dim) for _ in range(num_layers)
        ])

    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in self.layers:
            z, log_det_j = layer(z, condition)
            log_det_j_total += log_det_j
        return z, log_det_j_total

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in reversed(self.layers):
            y, log_det_j = layer.inverse(y, condition)
            log_det_j_total += log_det_j
        return y, log_det_j_total
    
    def zeros_layer(self):
        for layer in self.layers:
            layer.zero_post_weights()
            
            
            
class ConditionalNormalizingFlowTest(nn.Module):
    """A stack of conditional affine coupling layers."""
    def __init__(self, latent_dim: int, condition_dim: int, num_flow: int=4, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            # RealNVPCouplingLayer(latent_dim, condition_dim, num_wavenet_blocks=num_layers) for _ in range(num_flow)
            ConditionalAffineCouplingLayer(latent_dim, condition_dim) for _ in range(num_layers)
            # ResidualCouplingLayer(latent_dim, 192, 5, 1, 4, gin_channels=condition_dim) for _ in range(num_layers)
        ])

    def forward(self, z: torch.Tensor, condition: torch.Tensor, reverse=False):
        log_det_j_total = 0
        if reverse:
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
            for layer in reversed(self.layers):
                z, log_det_j = layer.inverse(z, condition)
                log_det_j_total += log_det_j
            return z, log_det_j_total

        for layer in self.layers:
            z, log_det_j = layer(z, condition)
            log_det_j_total += log_det_j
        return z, log_det_j_total

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in reversed(self.layers):
            y, log_det_j = layer.inverse(y, condition)
            log_det_j_total += log_det_j
        return y, log_det_j_total
    
    def zeros_layer(self):
        for layer in self.layers:
            layer.zero_post_weights()
            
class ConditionalNormalizingFlow2(nn.Module):
    """A stack of conditional affine coupling layers."""
    def __init__(self, latent_dim: int, condition_dim: int, num_flow: int=4, num_layers=4):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ConditionalAffineCouplingLayer3(latent_dim, condition_dim))
            layers.append(Flip2())    
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in self.layers:
            z, log_det_j = layer(z, condition)
            log_det_j_total += log_det_j
        return z, log_det_j_total

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in reversed(self.layers):
            y, log_det_j = layer.inverse(y, condition)
            log_det_j_total += log_det_j
        return y, log_det_j_total
    
    def zeros_layer(self):
        for layer in self.layers:
            try:
                layer.zero_post_weights()
            except Exception as e:
                pass

def _cap_logdet_per_elem(log_det, num_elems, cap_per_elem=0.25):
    # log_det: [B], num_elems: int = L * (D//2) * num_layers_effective
    per_elem = log_det / max(1, num_elems)
    per_elem = per_elem.clamp(min=-cap_per_elem, max=cap_per_elem)
    return per_elem * num_elems     
            
class ConditionalNormalizingFlow3(nn.Module):
    """A stack of conditional affine coupling layers."""
    def __init__(self, latent_dim: int, condition_dim: int, num_flow: int=4, num_layers=4):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ConditionalAffineCouplingLayer3(latent_dim, condition_dim))
            layers.append(Flip2())    
        self.layers = nn.ModuleList(layers)

    def forward(self, z, condition):
        log_det_total = z.new_zeros(z.size(0))
        L, D = z.shape[1], z.shape[2]
        for layer in self.layers:
            z, log_det = layer(z, condition)
            log_det_total = log_det_total + log_det
        # Cap per-element logdet (elements counted only on transformed half)
        num_elems = L * (D // 2)
        log_det_total = _cap_logdet_per_elem(log_det_total, num_elems, cap_per_elem=0.25)
        return z, log_det_total

    def inverse(self, y, condition):
        log_det_total = y.new_zeros(y.size(0))
        L, D = y.shape[1], y.shape[2]
        for layer in reversed(self.layers):
            y, log_det = layer.inverse(y, condition)
            log_det_total = log_det_total + log_det
        num_elems = L * (D // 2)
        log_det_total = _cap_logdet_per_elem(log_det_total, num_elems, cap_per_elem=0.25)
        return y, log_det_total
            
            
            
            

class Flip2(nn.Module):
    """Dimension flip that’s API-compatible with coupling layers."""
    def forward(self, x, condition=None):
        y = torch.flip(x, dims=[-1])
        # zero logdet with shape [B], correct device/dtype
        logdet = x.new_zeros(x.size(0))
        return y, logdet

    def inverse(self, y, condition=None):
        x = torch.flip(y, dims=[-1])
        logdet = y.new_zeros(y.size(0))
        return x, logdet
    
    
class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x, torch.zeros([1], device=x.device)

class ConditionalNormalizingFlowWN(nn.Module):
    """A stack of conditional affine coupling layers."""
    def __init__(self, latent_dim: int, condition_dim: int, num_flow: int=4, num_layers=4):
        super().__init__()
        self.num_layer = num_layers
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(ResidualCouplingLayer(latent_dim, 192, 5, 1, 6, gin_channels=condition_dim, mean_only=True))
            self.layers.append(Flip())

    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        z = z.transpose(1, 2)
        condition = condition.transpose(1, 2)
        C_half = z.shape[1] // 2
        T      = z.shape[2]
        for layer in self.layers:
            z, log_det_j = layer(z, condition)
            log_det_j_total += -log_det_j / (C_half * T)
        return z.transpose(1, 2), log_det_j_total / self.num_layer

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        y = y.transpose(1, 2)
        condition = condition.transpose(1, 2)
        C_half = y.shape[1] // 2
        T      = y.shape[2]
        for layer in reversed(self.layers):
            y, log_det_j = layer.inverse(y, condition)
            log_det_j_total += -log_det_j / (C_half * T)
        return y.transpose(1, 2), log_det_j_total / self.num_layer
    
    def zeros_layer(self):
        for layer in self.layers:
            try:
                layer.zero_post_weights()
            except Exception as e:
                pass
            
            
class WNConditionalNormalizingFlow(nn.Module):
    """A stack of conditional affine coupling layers."""
    def __init__(self, latent_dim: int, condition_dim: int, num_flow: int=4, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(ResidualCouplingLayer(latent_dim, 192, 5, 1, 5, gin_channels=condition_dim, mean_only=True))
            self.layers.append(Flip())

    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        shape = z.shape
        for layer in self.layers:
            z, log_det_j = layer(z, torch.ones((shape[0], 1, shape[2],), device=z.device), condition, reverse=False)
            log_det_j_total += log_det_j
        return z, log_det_j_total

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        shape = y.shape
        for layer in reversed(self.layers):
            y, log_det_j = layer(y, torch.ones((shape[0], 1, shape[2]),  device=y.device), condition, reverse=True)
            log_det_j_total += log_det_j
        return y, log_det_j_total
            
class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        self.n_flows = n_flows
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, g=None, reverse=False):
        B, _, _  = x.shape
        total_logdet = torch.ones(B, device=x.device)
        if not reverse:
            for flow in self.flows:
                x, log_det = flow(x, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            for flow in reversed(self.flows):
                x, log_det = flow(x, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()
            
    def zero_last_layer(self):
        for i in range(self.n_flows):
            self.flows[i * 2].zero_last_layer()
               
            
            
class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # no use gin_channels
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
        )
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        # SNAC Speaker-normalized Affine Coupling Layer
        self.snac = nn.Conv1d(gin_channels, 2 * self.half_channels, 1)

    def forward(self, x, g=None, reverse=False):
        speaker = self.snac(g)
        speaker_m, speaker_v = speaker.chunk(2, dim=1)  # (B, half_channels, 1)
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        # x0 norm
        x0_norm = (x0 - speaker_m) * torch.exp(-speaker_v)
        h = self.pre(x0_norm)
        # don't use global condition
        h = self.enc(h)
        stats = self.post(h)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            # x1 norm before affine xform
            x1_norm = (x1 - speaker_m) * torch.exp(-speaker_v)
            x1 = (m + x1_norm * torch.exp(logs))
            x = torch.cat([x0, x1], 1)
            # speaker var to logdet
            logdet = torch.sum(logs, [1, 2]) - torch.sum(
                speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            # x1 denorm before output
            x1 = (speaker_m + x1 * torch.exp(speaker_v))
            x = torch.cat([x0, x1], 1)
            # speaker var to logdet
            logdet = torch.sum(-logs, [1, 2]) + torch.sum(
                speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
            return x, logdet

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
    
    def zero_last_layer(self):
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        
        
class ResidualCouplingLayer2(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # no use gin_channels
        # self.enc = WN(
        #     hidden_channels,
        #     kernel_size,
        #     dilation_rate,
        #     n_layers,
        #     p_dropout=p_dropout,
        # )
        # self.rope = RotaryPositionalEmbedding(hidden_channels // 4)
        # self.rope = None
        self.enc = nn.Sequential(
            *[
                RoformerBlock(hidden_channels, n_heads=4, ffn_mult=1, dropout=p_dropout, local_window=8, d_cond=0) for i in range(n_layers)
            ]
        )
        
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        # SNAC Speaker-normalized Affine Coupling Layer
        self.snac = nn.Conv1d(gin_channels, 2 * self.half_channels, 1)

    def forward(self, x, g=None, reverse=False):
        speaker = self.snac(g)
        speaker_m, speaker_v = speaker.chunk(2, dim=1)  # (B, half_channels, 1)
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        # x0 norm
        x0_norm = (x0 - speaker_m) * torch.exp(-speaker_v)
        h = self.pre(x0_norm)
        # don't use global condition
        h = self.enc(h.transpose(1, 2)).transpose(1, 2)
        stats = self.post(h)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            # x1 norm before affine xform
            x1_norm = (x1 - speaker_m) * torch.exp(-speaker_v)
            x1 = (m + x1_norm * torch.exp(logs))
            x = torch.cat([x0, x1], 1)
            # speaker var to logdet
            logdet = torch.sum(logs, [1, 2]) - torch.sum(
                speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            # x1 denorm before output
            x1 = (speaker_m + x1 * torch.exp(speaker_v))
            x = torch.cat([x0, x1], 1)
            # speaker var to logdet
            logdet = torch.sum(-logs, [1, 2]) + torch.sum(
                speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
            return x, logdet

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
    
    def zero_last_layer(self):
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        
        
class ResidualCouplingLayer3(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # no use gin_channels
        # self.enc = WN(
        #     hidden_channels,
        #     kernel_size,
        #     dilation_rate,
        #     n_layers,
        #     p_dropout=p_dropout,
        # )
        # self.rope = RotaryPositionalEmbedding(hidden_channels // 4)
        # self.rope = None
        self.enc = nn.Sequential(
            *[
                RoformerBlock(hidden_channels, n_heads=4, ffn_mult=1, dropout=p_dropout, local_window=8  * (2**i), d_cond=0) for i in range(n_layers)
            ]
        )
        
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        # SNAC Speaker-normalized Affine Coupling Layer
        self.snac = nn.Conv1d(gin_channels, 2 * self.half_channels, 1)

    def forward(self, x, g=None, reverse=False):
        speaker = self.snac(g)
        speaker_m, speaker_v = speaker.chunk(2, dim=1)  # (B, half_channels, 1)
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        # x0 norm
        x0_norm = (x0 - speaker_m) * torch.exp(-speaker_v)
        h = self.pre(x0_norm)
        # don't use global condition
        h = self.enc(h.transpose(1, 2)).transpose(1, 2)
        stats = self.post(h)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            # x1 norm before affine xform
            x1_norm = (x1 - speaker_m) * torch.exp(-speaker_v)
            x1 = (m + x1_norm * torch.exp(logs))
            x = torch.cat([x0, x1], 1)
            # speaker var to logdet
            logdet = torch.sum(logs, [1, 2]) - torch.sum(
                speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            # x1 denorm before output
            x1 = (speaker_m + x1 * torch.exp(speaker_v))
            x = torch.cat([x0, x1], 1)
            # speaker var to logdet
            logdet = torch.sum(-logs, [1, 2]) + torch.sum(
                speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
            return x, logdet

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
    
    def zero_last_layer(self):
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        
        

class ResidualCouplingLayerVits(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # no use gin_channels
        # self.encode = ConformerNaiveEncoder(1, 2, hidden_channels, True, False, conv_dropout=0, atten_dropout=0.0, kernel_size=1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        # SNAC Speaker-normalized Affine Coupling Layer
        # self.snac = nn.Conv1d(gin_channels, 2 * self.half_channels, 1)

    def forward(self, x, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0)
        # h = self.encode(h.transpose(1, 2)).transpose(1, 2)
        h = self.enc(h, g=g)
        stats = self.post(h)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels]*2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs)
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1,2])
            return x, logdet
        else:
            dummy = None
            x1 = (x1 - m) * torch.exp(-logs)
            x = torch.cat([x0, x1], 1)
            return x, dummy

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
    
    def zero_last_layer(self):
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

class ResidualCouplingBlockVits(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        self.n_flows = n_flows
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayerVits(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, g=None, reverse=False):
        B, _, _  = x.shape
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, g=g, reverse=reverse)
            return x, _
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, g=g, reverse=reverse)
            return x, _

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()
            
    def zero_last_layer(self):
        for i in range(self.n_flows):
            self.flows[i * 2].zero_last_layer()
            
            
class ConditionalNormalizingFlow(nn.Module):
    """A stack of conditional affine coupling layers."""
    def __init__(self, latent_dim: int, condition_dim: int, num_flow: int=4, num_layers=4, pRelu=False):
        super().__init__()
        self.layers = nn.ModuleList([
            # RealNVPCouplingLayer(latent_dim, condition_dim, num_wavenet_blocks=num_layers) for _ in range(num_flow)
            ConditionalAffineCouplingLayer(latent_dim, condition_dim, pRelu=pRelu) for _ in range(num_layers)
            # ResidualCouplingLayer(latent_dim, 192, 5, 1, 4, gin_channels=condition_dim) for _ in range(num_layers)
        ])
        
        
    # def forward(self, x, g=None, reverse=False):
    #     B, _, _  = x.shape
    #     if not reverse:
    #         for flow in self.layers:
    #             x, _ = flow(x, g)
    #         return x, _
    #     else:
    #         for flow in reversed(self.layers):
    #             x, _ = flow.inverse(x, g)
    #         return x, _

    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in self.layers:
            z, log_det_j = layer(z, condition)
            log_det_j_total += log_det_j
        return z, log_det_j_total

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in reversed(self.layers):
            y, log_det_j = layer.inverse(y, condition)
            log_det_j_total += log_det_j
        return y, log_det_j_total
    
    def zeros_layer(self):
        for layer in self.layers:
            layer.zero_post_weights()

class ConditionalNormalizingFlowMod2(nn.Module):
    """A stack of conditional affine coupling layers."""
    def __init__(self, latent_dim: int, condition_dim: int, num_flow: int=4, num_layers=4, pRelu=True):
        super().__init__()
        self.layers = nn.ModuleList([
            # RealNVPCouplingLayer(latent_dim, condition_dim, num_wavenet_blocks=num_layers) for _ in range(num_flow)
            ConditionalAffineCouplingLayerMod2(latent_dim, condition_dim, pRelu=pRelu) for _ in range(num_layers)
            # ResidualCouplingLayer(latent_dim, 192, 5, 1, 4, gin_channels=condition_dim) for _ in range(num_layers)
        ])
        
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in self.layers:
            z, log_det_j = layer(z, condition)
            log_det_j_total += log_det_j
        return z, log_det_j_total

    def inverse(self, y: torch.Tensor, condition: torch.Tensor):
        log_det_j_total = 0
        # z = z.transpose(1, 2)
        # condition = condition.transpose(1, 2)
        for layer in reversed(self.layers):
            y, log_det_j = layer.inverse(y, condition)
            log_det_j_total += log_det_j
        return y, log_det_j_total
    
    def zeros_layer(self):
        for layer in self.layers:
            layer.zero_post_weights()

class FixedLengthWN(nn.Module):
    """
    A version of the WN module for inputs with fixed, consistent lengths.
    The `x_mask` logic has been removed.
    """
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, g=None): # MODIFICATION: Removed x_mask from signature
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            
            in_act = x_in + g_l
            t_act = torch.tanh(in_act[:, :self.hidden_channels, :])
            s_act = torch.sigmoid(in_act[:, self.hidden_channels:, :])
            acts = t_act * s_act
            
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = x + res_acts # MODIFICATION: Removed * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
                
        return output
    
    
class ResidualCouplingBlockVits2(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        self.n_flows = n_flows
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())
            
    def forward(self, x, g=None, reverse=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()
            
    def zero_last_layer(self):
        for i in range(self.n_flows):
            self.flows[i * 2].zero_last_layer()
            
            
class ResidualCouplingBlockVits3(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        self.n_flows = n_flows
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer3(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())
            
    def forward(self, x, g=None, reverse=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()
            
    def zero_last_layer(self):
        for i in range(self.n_flows):
            self.flows[i * 2].zero_last_layer()
            
class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x, logdet
    
    def inverse(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x, logdet