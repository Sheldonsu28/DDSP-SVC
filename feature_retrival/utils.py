import os
import traceback
# import traceback
# from sklearn.cluster import MiniBatchKMeans
# import torch
import librosa
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
import torch


def train_index(spk_name, root_dir = "data", normalize=False, clustering=True, feat='units'):  #from: RVC https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    n_cpu = os.cpu_count() or 4
    print("The feature index is constructing.")
    train_dir = os.path.join(root_dir,'train', feat ,str(spk_name))
    val_dir = os.path.join(root_dir,'val', feat, str(spk_name))
    listdir_res = []
    for file in os.listdir(train_dir):
       if ".wav.npy" in file:
          listdir_res.append(os.path.join(train_dir,file))

    for file in os.listdir(val_dir):
       if ".wav.npy" in file:
          listdir_res.append(os.path.join(val_dir,file))

    if len(listdir_res) == 0:
        raise Exception("You need to run pre_process_first")
    npys = []
    for name in sorted(listdir_res):
        # phone = torch.load(name)[0].transpose(-1,-2).numpy()
        phone = np.load(name)
        # print(phone.shape)
        # print(phone.transpose(-1,-2).shape)
        npys.append(phone)
    # print(big_npy.shape)
    big_npy = np.concatenate(npys, 0)  
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    centers = 10000
    if big_npy.shape[0] > 2e5 and clustering:
        # if(1):
        info = f"Trying doing kmeans {big_npy.shape[0]} shape to {int(centers/1000)}k centers." 
        print(info)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=centers,
                    verbose=True,
                    batch_size=512 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            info = traceback.format_exc()
            print(info)
    # a = torch.ones((3, 2))
    # print(torch.nn.functional.normalize(a, dim=1))
    if normalize:
        big_npy_un_norm = torch.from_numpy(big_npy)
        big_npy = torch.nn.functional.normalize(big_npy_un_norm, dim=1, eps=1e-12).numpy()
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(big_npy.shape[1] , "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    # faiss.write_index(
    #     index,
    #     f"added_{spk_name}.index"
    # )
    print("Successfully build index")
    return index



def compute_all_energy_std_mean(spk_name, root_dir = "data"):
    train_dir = os.path.join(root_dir,'train', 'audio' ,str(spk_name))
    val_dir = os.path.join(root_dir,'val', 'audio' ,str(spk_name))
    listdir_res = []
    for file in os.listdir(train_dir):
       if ".wav" in file:
          listdir_res.append(os.path.join(train_dir,file))

    for file in os.listdir(val_dir):
       if ".wav" in file:
          listdir_res.append(os.path.join(val_dir,file))

    if len(listdir_res) == 0:
        raise Exception("You need to run pre_process_first")
    
    energies = []
    
    for name in sorted(listdir_res):
        audio, _ = librosa.load(name, sr=44100)
        energy = torch.log(extract_energy(audio) + 1e-5)
        energies.append(energy)
        
    all_energy_values = torch.cat(energies)
    energy_mean = all_energy_values.mean()
    energy_std = all_energy_values.std()
    return energy_mean, energy_std

def extract_energy(waveform, frame_length=160, hop_length=512):
    return torch.from_numpy(librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length).squeeze(0))


def compute_std_mean(spk_name, root_dir="data", folder='units'):  #from: RVC https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    n_cpu = os.cpu_count() or 4
    print("The feature index is constructing.")
    train_dir = os.path.join(root_dir,'train', folder ,str(spk_name))
    val_dir = os.path.join(root_dir,'val', folder, str(spk_name))
    listdir_res = []
    for file in os.listdir(train_dir):
       if ".wav.npy" in file:
          listdir_res.append(os.path.join(train_dir,file))

    for file in os.listdir(val_dir):
       if ".wav.npy" in file:
          listdir_res.append(os.path.join(val_dir,file))

    if len(listdir_res) == 0:
        raise Exception("You need to run pre_process_first")
    npys = []
    for name in sorted(listdir_res):
        phone = np.load(name)

        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    std, mean = torch.std_mean(big_npy, dim=0)
    return std, mean

# if __name__ == "__main__":
#     compute_all_std_mean(1)