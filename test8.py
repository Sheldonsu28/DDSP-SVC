import torch
import matplotlib.pyplot as plt
import numpy as np

from reflow.solver import calculate_angle_si_snr, calculate_mel_si_snr


def visualize_embedding(tensor, title="ContentVec Visualization"):
    """
    tensor: torch.Tensor of shape [1, 768, L] or [768, L]
    """
    # 1. Handle Shape: Squeeze batch dimension if present
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0) # Becomes [768, L]
    
    # 2. Convert to Numpy
    data = tensor.detach().cpu().float().numpy()
    
    # 3. Smart Normalization for Visualization
    # Transformer embeddings often have outliers. We clip the top/bottom 1% 
    # to make the texture visible.
    vmin = np.percentile(data, 1)
    vmax = np.percentile(data, 99)
    
    # 4. Plotting
    plt.figure(figsize=(12, 6))
    
    # 'aspect=auto' allows the plot to stretch to fit the window
    # 'origin=lower' puts dimension 0 at the bottom
    # 'interpolation=nearest' ensures we see raw pixels (no blurring)
    plt.imshow(data, 
               aspect='auto', 
               origin='lower', 
               cmap='viridis', # 'magma' or 'inferno' are also good
               vmin=vmin, 
               vmax=vmax,
               interpolation='nearest')
    
    plt.colorbar(label='Activation Strength')
    plt.xlabel('Time (Frames)')
    plt.ylabel(f'Feature Dimension (0-{data.shape[1]})')
    plt.title(f"{title}\nShape: {data.shape}")
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# Create a dummy ContentVec tensor [1, 768, 200]
# We add some random "vertical stripes" to simulate phonemes


def new():
    x = np.arange(0, 1, 0.01)
    y = 0.398942 / x / (1 - x) * np.exp(-0.5 * np.log(x / ( 1 - x)) ** 2)
    plt.plot(x, y, marker='o')
    plt.show()

# Call the function
if __name__ == "__main__":
    # dummy_data = torch.randn(1, 768, 200) 
    # dummy_data[:, :, 50:70] += 2.0  # Simulated active region
    # new()
    # data = torch.from_numpy(np.expand_dims(np.load('./data/train/units/1/1.wav.npy'), axis=0))
    data_units = torch.from_numpy(np.load('./data/train/units/1/1.wav.npy')).unsqueeze(0)
    data_hu = torch.from_numpy(np.load('./data/train/hubert_units/1/1.wav.npy')).unsqueeze(0)
    data_whis = torch.from_numpy(np.load('./data/train/whisper_units/1/1.wav.npy')).unsqueeze(0)
    print('Units')
    pred_units =  data_units + torch.randn_like(data_units) * 0.42
    print('angle snr', calculate_angle_si_snr(data_units, pred_units))
    print('snr', calculate_mel_si_snr(data_units, pred_units))
    print('\n')
    
    print('Hu units')
    pred_hu_units =  data_hu + torch.randn_like(data_hu) * 0.02
    print('angle snr', calculate_angle_si_snr(data_hu, pred_hu_units))
    print('snr', calculate_mel_si_snr(data_hu, pred_hu_units))
    print('\n')
    
    print('Whis nits')
    pred_whis_units =  data_whis + torch.randn_like(data_whis) * 1
    print('angle snr', calculate_angle_si_snr(data_whis, pred_whis_units))
    print('snr', calculate_mel_si_snr(data_whis, pred_whis_units))
    
    
    
    # norm = torch.norm(data, dim= 1)
    # print(norm.mean(), norm.std())
    # visualize_embedding(data)