import shutil
import librosa
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from discriminators.discriminator import Discriminator
from logger.utils import get_network_paras_amount

def load_audio(path, sr=22050):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def compute_mel(audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db

def compute_mse_per_bin(mel1, mel2):
    assert mel1.shape == mel2.shape, "Mel spectrograms must be same shape"
    mse = np.mean((mel1 - mel2) ** 2, axis=1)  # Mean over time
    return mse

def plot_mse(mse_values):
    plt.figure(figsize=(10, 4))
    plt.plot(mse_values)
    plt.title('MSE per Mel Bin')
    plt.xlabel('Mel Bin')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def copy_missing(dir_a: str, dir_b: str, dir_c: str) -> int:
    """
    Copy every file that is present in dir_a but NOT in dir_b into dir_c.
    Sub-folders are ignored (one-level directories only).  
    Only the file contents are copied—no metadata is preserved.

    Args:
        dir_a (str): Source directory A.
        dir_b (str): Reference directory B.
        dir_c (str): Destination directory C (created if necessary).

    Returns:
        int: Number of files copied.
    """
    a = Path(dir_a).expanduser().resolve()
    b = Path(dir_b).expanduser().resolve()
    c = Path(dir_c).expanduser().resolve()

    # Basic sanity checks
    for d in (a, b):
        if not d.is_dir():
            raise NotADirectoryError(f"{d} is not a directory")

    c.mkdir(parents=True, exist_ok=True)
    files_a = {p.name for p in a.iterdir() if p.is_file()}
    files_b = {p.name for p in b.iterdir() if p.is_file()}
    missing = files_a - files_b
    for fname in sorted(missing):
        shutil.copyfile(a / fname, c / fname)   # simple copy, no metadata
        print(f"Copied {fname}")

    return len(missing)
if __name__ == '__main__':
    
    # copy_missing('F:\\dataset\\standard_ellie_dataset_v6_only_ellie', 'F:\\dataset\\standard_ellie_dataset_v7', 'F:\\dataset\\test')
    # path1 = '568.wav'
    # path2 = '568_f.wav'

    # sr = 44100
    # audio1 = load_audio(path1, sr=sr)
    # audio2 = load_audio(path2, sr=sr)

    # min_len = min(len(audio1), len(audio2))
    # audio1 = audio1[:min_len]
    # audio2 = audio2[:min_len]

    # mel1 = compute_mel(audio1, sr=sr)
    # mel2 = compute_mel(audio2, sr=sr)

    # mse = compute_mse_per_bin(mel1, mel2)
    # print(mse.mean())
    # plot_mse(mse)
    # s = torch.load('exp/reflow-test/model_2.pt')
    a = torch.load('sovits5-0.pth')
    # a = torch.load('new_gan.pt')
    # print(a.keys())
    # print(a['files']['meta.yaml'])
    b = torch.load('new_style_vc.pt')
    # b = torch.load('model_0_med.pt')
    b['model_d'] = a['model_d']
    b['global_step'] = 0
    # m = Discriminator({})
    # m.load_state_dict(a['model_d'])
    torch.save(b, 'model.pt')
    # pass
    

    
    
