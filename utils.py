import torch
hann_window = {}
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def compute_spec(audio, sampling_rate=44100):
    audio_norm = audio / 32768.0
    audio_norm = audio_norm.unsqueeze(0)
    n_fft = 1024
    sampling_rate = sampling_rate
    hop_size = 512
    win_size = 1024
    spec = spectrogram_torch(
        audio_norm, n_fft, sampling_rate, hop_size, win_size, center=False)
    spec = torch.squeeze(spec, 0)
    return spec


def map_normal_diagonal(x, mu_src, cov_src, mu_tgt, cov_tgt):
    """
    Maps tensor x (N, T, D) from Source Dist to Target Dist.
    Assumes Diagonal Covariance (independent features).
    
    Args:
        x (Tensor): Input tensor of shape [1, T, dim]
        mu_src (Tensor): Source mean [1, dim]
        cov_src (Tensor): Source covariance/variance [1, dim] (Not std!)
        mu_tgt (Tensor): Target mean [1, dim]
        cov_tgt (Tensor): Target covariance/variance [1, dim]
    """
    # 1. Convert Covariance (Variance) to Std Dev
    # We add a tiny epsilon (1e-6) to avoid division by zero or sqrt of negative
    sigma_src = cov_src
    sigma_tgt = cov_tgt
    
    # 2. Standardize (Whiten) -> z ~ N(0, 1)
    # PyTorch broadcasting automatically handles [1, T, dim] vs [1, dim]
    # It aligns the last dimension (dim) and broadcasts across T.
    z = (x - mu_src) / sigma_src
    
    # 3. Color (Project to Target)
    y = z * sigma_tgt + mu_tgt
    
    return y


def map_normal_cholesky_sequence(x, mu_src, cov_src, mu_tgt, cov_tgt):
    """
    Maps a sequence x [1, T, dim] from Source Multivariate Normal to Target.
    Handles correlations between dimensions using Cholesky decomposition.
    
    Args:
        x (Tensor): Input tensor of shape [1, T, dim]
        mu_src (Tensor): Source mean [1, dim]
        cov_src (Tensor): Source covariance Matrix [dim, dim] (Square Matrix!)
        mu_tgt (Tensor): Target mean [1, dim]
        cov_tgt (Tensor): Target covariance Matrix [dim, dim]
    """
    
    B, T, D = x.shape
    output_list = []
    
    # Pre-calculate Cholesky decompositions once
    jitter = 1e-6 * torch.eye(D, device=x.device)
    L_src = torch.linalg.cholesky(cov_src + jitter)
    L_tgt = torch.linalg.cholesky(cov_tgt + jitter)
    
    # Process in chunks
    for i in range(0, T, chunk_size):
        x_chunk = x[:, i : i + chunk_size, :]
        
        # --- Run the Logic on the Chunk ---
        # 1. Center & Transpose
        x_c = (x_chunk - mu_src).transpose(-1, -2)
        
        # 2. Whiten
        z = torch.linalg.solve_triangular(L_src, x_c, upper=False)
        
        # 3. Color
        y_c = torch.matmul(L_tgt, z)
        
        # 4. Transpose & Shift
        y_chunk = y_c.transpose(-1, -2) + mu_tgt
        
        output_list.append(y_chunk)
        
        # Optional: Clear cache if really tight
        # torch.cuda.empty_cache()

    return torch.cat(output_list, dim=1)