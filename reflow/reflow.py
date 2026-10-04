import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import random
from torch.utils.checkpoint import checkpoint

from stylizer.stft_loss import drift_loss


def delta(x):
    return x[..., 1:] - x[..., :-1] 

def delta_loss(Chat:torch.Tensor, C, w1=1.0, w2=0.5):
    kernel = gaussian_kernel1d(6.0, Chat.device, Chat.dtype)
    d1h, d1 = delta(conv_time_1d(Chat,kernel)), delta(conv_time_1d(C, kernel))
    d2h, d2 = delta(d1h), delta(d1)
    L1 = torch.mean((d1h - d1).abs())
    L2 = (d2h - d2).abs().mean()
    return w1 * L1 + w2 * L2

def random_masking(pred, gt, num=43):
    _, _, _, L = pred.shape
    start_index = random.randint(0, max(L - num, 0))
    return pred[:, :, :, start_index: start_index + num], gt[:, :, :, start_index: start_index + num]
    
def gaussian_kernel1d(sigma: float, device=None, dtype=None):
    # kernel size: 2*ceil(truncate*sigma)+1 (odd)
    radius = max(1, int(math.ceil(3*sigma)))
    x = torch.arange(-radius, radius+1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma)**2)
    k /= k.sum()
    return k.view(1,1,-1)

def conv_time_1d(x, kernel):
    B, _, D, T = x.shape
    x4 = x.permute(0,2,1,3).reshape(B*D, 1, T)    # [B*D,1,T]
    pad = (kernel.size(-1)//2,)*2
    x4 = F.pad(x4, pad, mode='reflect')
    y = F.conv1d(x4, kernel)                      # [B*D,1,T]
    y = y.view(B, D, 1, T).permute(0,2,1,3)       # [B,1,D,T]
    return y


def focus_loss(loss_map, alphas=[1.0, 2.0, 3.0, 5.0], weights=[0.5, 1.0, 1.5, 5.0]):
    """
    Functional implementation of the Hybrid Adaptive L2 Loss.
    
    Args:
        input (Tensor): Generator predictions [B, ...].
        target (Tensor): Ground truth [B, ...].
        alpha (float): Threshold strictness (mean + alpha * std). 
                       Higher = focuses only on worst outliers.
        lambda_focus (float): Weight multiplier for the hard-mined loss.
        return_details (bool): If True, returns (total_loss, global_loss, masked_loss).
                               If False, returns total_loss (standard for optimizers).
    """
    # 1. Calculate element-wise squared error (L2)
    # This keeps the shape [B, 1, D, L]
        
    # --- Part B: Adaptive Masked Loss (Refinement) ---
    # We detach stats to ensure the threshold itself is not differentiated
    masks = []
    with torch.no_grad():
        # print(loss_map.shape)
        flat_errors = loss_map.reshape(-1)
        mu = flat_errors.mean()
        sigma = flat_errors.std()
        
        for i in range(len(alphas)):
            # Dynamic Threshold
            threshold = mu + (alphas[i] * sigma)
        
            # Create Boolean Mask
            masks.append(loss_map > threshold)
    loss = torch.tensor(0.0, device=loss_map.device, dtype=loss_map.dtype)
    for j in range(len(masks)):
        # Apply mask safely
        if masks[j].sum() > 0:
            # Mean of only the "hard" pixels
            masked_loss = loss_map[masks[j]].mean()
        else:
            # Fallback for perfect batches (avoid NaN)
            masked_loss = torch.tensor(0.0, device=loss_map.device, dtype=loss_map.dtype)
        loss += masked_loss * weights[j]
    
    return loss

class RectifiedFlow(nn.Module):
    def __init__(self, 
                velocity_fn, 
                out_dims=128,
                spec_min=-12, 
                spec_max=2,
                train_embed=False,
                loss_type='l2_lognorm',
                t_mu=0.0,
                t_sigma=1.0):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max
        self.train_embed=train_embed
        self.loss_type = loss_type
        self.t_mu = t_mu
        self.t_sigma = t_sigma
    
    def reflow_loss(self, x_1, t, cond, loss_type='l2_lognorm', gin=None):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_pred = checkpoint(self.velocity_fn, x_t, 1000 * t, cond, gin=gin, use_reentrant=False)
        
        if loss_type == 'l1':
            loss = (x_1 - x_0 - v_pred).abs().mean()
        elif loss_type == 'l1_lognorm':
            weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / ( 1 - t)) ** 2)
            gt = x_1 - x_0
            
            loss_map = weights[:, None, None, None] * F.l1_loss(gt, v_pred, reduction='none')
            # loss = torch.mean(loss_map)
           
            # loss_map = weights[:, None, None, None] * F.l1_loss(gt, v_pred, reduction='none')
            loss = torch.mean(loss_map)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_1 - x_0, v_pred)

        elif loss_type == 'l2_lognorm':
            weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / ( 1 - t)) ** 2)
            gt = x_1 - x_0
            
            loss_map = weights[:, None, None, None] * F.mse_loss(gt, v_pred, reduction='none')
            # loss = torch.mean(loss_map)
           
            # loss_map = weights[:, None, None, None] * F.l1_loss(gt, v_pred, reduction='none')
            loss = torch.mean(loss_map)
           
        if self.train_embed:
            return loss, x_0 + v_pred
        return torch.clip(loss, max=10000)
    
    def sample_euler(self, x, t, dt, cond, gin=None):
        x += self.velocity_fn(x, 1000 * t, cond, gin=gin) * dt
        t += dt
        return x, t
        
    def sample_rk4(self, x, t, dt ,cond, gin=None):
        k_1 = self.velocity_fn(x, 1000 * t, cond, gin=gin)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, 1000 * (t + 0.5 * dt), cond, gin=gin)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, 1000 * (t + 0.5 * dt), cond, gin=gin)
        k_4 = self.velocity_fn(x + k_3 * dt, 1000 * (t + dt), cond, gin=gin)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t
     
    def forward(self, 
                condition, 
                gt_spec=None, 
                infer=True,
                infer_step=10,
                method='euler',
                t_start=0.0,
                use_tqdm=True, 
                gin=None):
        cond = condition.transpose(1, 2) # [B, H, T]
        # print(cond.shape, 'after')
        b, device = condition.shape[0], condition.device
        if t_start < 0.0:
            t_start = 0.0
        if not infer:
            if not self.train_embed:
                x_1 = self.norm_spec(gt_spec)
            else:
                x_1 = gt_spec
            x_1 = x_1.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
           
            t = t_start + (1.0 - t_start) * torch.rand(b, device=device)
            t = torch.clip(t, 1e-7, 1-1e-7)
            return self.reflow_loss(x_1, t, cond=cond, gin=gin, loss_type=self.loss_type)
            # else:
            #     # sample t ~ logit-normal(t_mu, t_sigma) with unweighted l2 loss:
            #     # same objective as uniform t + 'l2_lognorm' weights, but lower gradient variance
            #     t = torch.sigmoid(self.t_mu + self.t_sigma * torch.randn(b, device=device))
            #     t = torch.clip(t, 1e-7, 1-1e-7)
            #     return self.reflow_loss(x_1, t, cond=cond, loss_type='l2')
        else:
            shape = (cond.shape[0], 1, self.out_dims, cond.shape[2]) # [B, 1, M, T]
            
            # initial condition and step size of the ODE
            if gt_spec is None:
                x = torch.randn(shape, device=device)
                t = torch.full((b,), 0.0, device=device, dtype=torch.float32)
                dt = 1.0 / infer_step
            else:
                if not self.train_embed:
                    norm_spec = self.norm_spec(gt_spec)
                else:
                    norm_spec = gt_spec
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :] # [B, 1, M, T]
                x = t_start * norm_spec + (1 - t_start) * torch.randn(shape, device=device)
                t = torch.full((b,), t_start, device=device, dtype=torch.float32)
                dt = (1.0 - t_start) / infer_step 
                  
            if method == 'euler':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_euler(x, t, dt, cond, gin=gin)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_euler(x, t, dt, cond, gin=gin)
            
            elif method == 'rk4':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond, gin=gin)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond, gin=gin)
            
            else:
                raise NotImplementedError(method)
                
            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]
            if not self.train_embed:
                return self.denorm_spec(x)
            return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
