import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from stylizer.modules_grl import GradientReversal
from stylizer.util import ConformerBlock

class CnnSpeakerClassifier2(nn.Module):
    """
    A lightweight, CNN-based speaker classifier that focuses on local features.

    Args:
        input_dim (int): The dimension of the input embeddings (e.g., 256).
        num_channels (int): The number of channels in the CNN layers.
        output_dim (int): The dimension of the final speaker embedding (e.g., 256).
    """
    def __init__(self, input_dim=256, num_channels=256, output_dim=256, lambda_reversal=1.0):
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


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        """
        Args:
            num_embeddings (int): The size of the codebook (e.g., 512, 1024).
            embedding_dim (int): The size of the embedding vectors (must match encoder output).
            commitment_cost (float): Factor to scale the commitment loss (beta). 
                                     Usually between 0.1 and 1.0.
        """
        super(VectorQuantizer, self).__init__()
        # self.pre_vq_norm = nn.LayerNorm(embedding_dim)
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._embedding.weight.requires_grad = False
        
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon
        
        # NEW: Flag to track if we have initialized with real data yet
        self.register_buffer('_inited', torch.tensor(0))
        
    def _tile(self, x):
        d, ew = x.shape
        if d < self._num_embeddings:
            n_repeats = (self._num_embeddings + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def forward(self, inputs):
        # inputs shape: [Batch, Channel, Height, Width] (if image) or [Batch, Channel, Time] (if audio)
        # We need to flatten inputs to [Batch*H*W, Channel] for distance calculation
        
        # 1. Reshape inputs to [N, Dim]
        # inputs = self.pre_vq_norm(inputs)
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        if self.training and self._inited.item() == 0:
            print(" [!] Initializing Codebook from first batch of data...")
            
            # Select random vectors from the current batch to be the initial codebook
            # If batch is smaller than codebook, we loop it
            indices = torch.randperm(flat_input.size(0))[:self._num_embeddings]
            initial_centers = flat_input[indices]
            
            # If we don't have enough data points in one batch to fill the codebook
            if initial_centers.shape[0] < self._num_embeddings:
                 # Repeat the data with slight noise to fill the codebook
                 # (Simple tiling logic)
                 repeat_factor = (self._num_embeddings // initial_centers.shape[0]) + 1
                 initial_centers = initial_centers.repeat(repeat_factor, 1)[:self._num_embeddings]
                 initial_centers += torch.randn_like(initial_centers) * 0.01 # Add jitter

            self._embedding.weight.data.copy_(initial_centers)
            self._ema_w.data.copy_(initial_centers)
            self._ema_cluster_size.fill_(1) # Reset cluster usage
            self._inited.fill_(1)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # --- EMA UPDATE STEP ---
        # This block runs automatically during your training loop
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n
            )
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            # This is the manual update of the codebook
            self._embedding.weight.data.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        # -----------------------
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encodings
    
    
    
class VQVAE(nn.Module):
    def __init__(self, input_dim=1280, content_dim=256, num_embeddings=10000, spk_dim=192):
        super(VQVAE, self).__init__()
        
        self.pre = weight_norm(nn.Conv1d(input_dim, content_dim, kernel_size=5, padding=2))
        self.pre2 = weight_norm(nn.Conv1d(768, content_dim, kernel_size=5, padding=2))
        self.pre3 = weight_norm(nn.Conv1d(256, content_dim, kernel_size=5, padding=2))
        self.vq = VectorQuantizer(num_embeddings, content_dim, commitment_cost=0.25)
        self.encoder = ConformerBlock(dim_model=content_dim, num_heads=2, conv_kernel_size=5, ff_multiplier=1, dropout=0.1)
        self.speaker_classifer = CnnSpeakerClassifier2(output_dim=spk_dim)

       

    def forward(self, content_vec, whisper, hubert, infer=False):
        # 1. Encode
        z = self.pre(whisper.transpose(1, 2)) + self.pre2(content_vec.transpose(1, 2)) + self.pre3(hubert.transpose(1, 2))
        
        z_permuted = self.encoder(z.transpose(1, 2)).contiguous()
        
        vq_loss, quantized, perplexity, _ = self.vq(z_permuted)
        
        # Permute back: [Batch, Time, Dim] -> [Batch, Dim, Time]
        quantized = quantized.permute(0, 1, 2).contiguous()
        if not infer:
            spk_pred = self.speaker_classifer(quantized)
        else:
            spk_pred = None
        
        return quantized, vq_loss, perplexity, spk_pred
    
    
class ResidualVQ(nn.Module):
    def __init__(self,input_dim=1280, content_dim=256, num_quantizers=8, spk_dim=192, num_embeddings=1024):
        super(ResidualVQ, self).__init__()
        self.encoder = ConformerBlock(dim_model=content_dim, num_heads=2, conv_kernel_size=5, ff_multiplier=1, dropout=0.1)
        self.pre = weight_norm(nn.Conv1d(input_dim, content_dim, kernel_size=5, padding=2))
        self.pre2 = weight_norm(nn.Conv1d(768, content_dim, kernel_size=5, padding=2))
        self.pre3 = weight_norm(nn.Conv1d(256, content_dim, kernel_size=5, padding=2))
        self.speaker_classifer = CnnSpeakerClassifier2(output_dim=spk_dim)
        self.num_quantizers = num_quantizers
        # Create a list of VQ modules
        self.layers = nn.ModuleList([VectorQuantizer(num_embeddings, content_dim) for _ in range(num_quantizers)])
        self.weights = [1, 1, 2, 2, 4, 4, 8, 8]

    def forward(self, content_vec, whisper, hubert, infer=False, n_quantizers=None):
        x = self.pre(whisper.transpose(1, 2)) + self.pre2(content_vec.transpose(1, 2)) + self.pre3(hubert.transpose(1, 2))
        z = self.encoder(x.transpose(1, 2))
        quantized_out = 0
        residual = z
        all_losses = 0
        perplexity = 0
        
        if self.training:
            # Drop layers with probability p (e.g., keep at least 1)
            n_quantizers = torch.randint(1, len(self.layers) + 1, (1,)).item()
        elif n_quantizers is None:
            # At inference, use all by default
            n_quantizers = len(self.layers)
        
        for i, layer in enumerate(self.layers):
            if i < n_quantizers:
            # Quantize the residual from the previous layer
                loss, quantized, perplexity_layer, _ = layer(residual)
                # Add this layer's contribution to the final output
                quantized_out += quantized
                # Calculate new residual (what's left to explain?)
                residual = residual - quantized
                all_losses += loss * self.weights[i]
                perplexity += perplexity_layer
            else:
                break
            
        if not infer:
            spk_pred = self.speaker_classifer(quantized_out)
        else:
            spk_pred = None
            
        return quantized_out, all_losses, perplexity / n_quantizers, spk_pred