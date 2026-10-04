import parselmouth
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from reflow.lynxnet2 import LYNXNet2
from reflow.reflow import RectifiedFlow

def extract_formants(audio_path, time_step=0.01, max_formant=5500):
    snd = parselmouth.Sound(audio_path)
    formant = snd.to_formant_burg(time_step=time_step, max_number_of_formants=5, maximum_formant=max_formant)

    duration = snd.duration
    times = np.arange(0, duration, time_step)
    f1, f2 = [], []

    for t in times:
        f1_val = formant.get_value_at_time(1, t)
        f2_val = formant.get_value_at_time(2, t)
        f1.append(f1_val if not np.isnan(f1_val) else 0)
        f2.append(f2_val if not np.isnan(f2_val) else 0)

    return times, np.array(f1), np.array(f2)

def plot_formants(audio_path):
    times, f1, f2 = extract_formants(audio_path)

    plt.figure(figsize=(12, 4))
    plt.plot(times, f1, label="F1 (vowel openness)", color="red")
    plt.plot(times, f2, label="F2 (vowel frontness)", color="blue")
    plt.title("Formant Trajectories")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 3000)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
def calculateSVD(root_dir, spk_name=1):
    # unit_train_dir = os.path.join(root_dir,'train', 'hubert_units', '1')
    # unit_val_dir = os.path.join(root_dir,'val', 'hubert_units', '1')
    
    # whisper_train_dir = os.path.join(root_dir,'train', 'whisper_units' ,str(spk_name))
    # whisper_val_dir = os.path.join(root_dir,'val', 'whisper_units' ,str(spk_name))
    
    # hubert_train_dir = os.path.join(root_dir,'train', 'hubert_units' )
    # hubert_val_dir = os.path.join(root_dir,'val', 'hubert_units' )
    
    speaker_train_dir = os.path.join(root_dir,'train', 'speaker' )
    speaker_val_dir = os.path.join(root_dir,'val', 'speaker')
    
    unit_dir_res = []
    for file in os.listdir(speaker_train_dir):
        if ".wav.npy" in file:
            unit_dir_res.append(os.path.join(speaker_train_dir,file))

    for file in os.listdir(speaker_val_dir):
        if ".wav.npy" in file:
            unit_dir_res.append(os.path.join(speaker_val_dir, file))
          
    # whisper_dir_res = []
    
    # for file in os.listdir(whisper_train_dir):
    #     if ".wav.npy" in file:
    #         whisper_dir_res.append(os.path.join(whisper_train_dir,file))

    # for file in os.listdir(whisper_val_dir):
    #     if ".wav.npy" in file:
    #         whisper_dir_res.append(os.path.join(whisper_val_dir, file))
    
    # speaker_dir_res = []
    
    # for file in os.listdir(unit_train_dir):
    #     if ".wav.npy" in file:
    #         speaker_dir_res.append(os.path.join(speaker_train_dir,file))
            
    # for file in os.listdir(speaker_val_dir):
    #     if ".wav.npy" in file:
    #         speaker_dir_res.append(os.path.join(speaker_val_dir,file))
          
    # hubert_dir_res = []
    
    # for file in os.listdir(hubert_train_dir):
    #     if ".wav.npy" in file:
    #         hubert_dir_res.append(os.path.join(hubert_train_dir,file))

    # for file in os.listdir(hubert_val_dir):
    #     if ".wav.npy" in file:
    #         hubert_dir_res.append(os.path.join(hubert_val_dir, file))
          
    

    # if len(unit_dir_res) == 0:
    #     raise Exception("You need to run pre_process_first")
    
    # unit_dir_res = [a for a in unit_dir_res if '605' in a]
    # whisper_dir_res = [a for a in whisper_dir_res if '172' in a]
    # hubert_train_dir = [a for a in hubert_train_dir if '172' in a]
    # print(len(unit_dir_res))
    
    # unit_npys = []
    # for name in sorted(unit_dir_res):
    #     phone = np.load(name)
    #     unit_npys.append(phone)
    # unit_npys = np.concatenate(unit_npys, 0)

    
    # whisper_npys = []
    # for name in sorted(whisper_dir_res):
    #     phone = np.load(name)
    #     whisper_npys.append(phone)
    # whisper_npys = np.concatenate(whisper_npys, 0)
    
    
    # hubert_npys = []
    # for name in sorted(hubert_dir_res):
    #     phone = np.load(name)
    #     hubert_npys.append(phone)
    # hubert_npys = np.concatenate(hubert_npys, 0)
    # hubert_mean = np.mean(hubert_npys, axis=0)
    
    sepaker_npys = []
    print(len(unit_dir_res))
    for name in sorted(unit_dir_res):
        phone = np.expand_dims(np.load(name), 0)
        sepaker_npys.append(phone)
    sepaker_npys = np.concatenate(sepaker_npys, 0)
    # print(sepaker_npys.shape)
    sepaker_mean = np.mean(sepaker_npys, axis=0)
    # print(sepaker_npys.shape)
    # cont = np.concatenate([unit_npys, whisper_npys, hubert_npys], 1)
    # cont_mean = cont.mean(0)
    print(sepaker_mean.shape)
    np.save('data/speaker_elysia_new.npy', sepaker_mean)
    
    # unit_mean = unit_npys.mean(0)
    # np.save('exp/unit_mean.npy', unit_mean)
    
    # centerd = cont - cont_mean
    # centerd_var = centerd.var()
    # print(centerd_var)
    # np.save('exp/var_unit.npy', unit_npys.var(0))
    # unit_centered = unit_npys - unit_mean
    # print(unit_npys.std(0))
    # y = np.linalg.svd(unit_npys, full_matrices=True, compute_uv=False).squeeze()
    # s = y.sum()
    # l = y.shape[0]
    # print(y.shape)
    # acc = 0
    # arr = []
    # for i in range(l):
    #     acc += y[i]
    #     arr.append(f"{i}, {acc / s}\r\n")
        
    # with open('./lookup.txt', 'w') as f:
    #     f.writelines(arr)
    

悲伤的爱莉 = ['942']

def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

if __name__ == "__main__":
    calculateSVD('.\\data')
    # large_ddsp = torch.load('ddsp6.3_6x512_6x1024.sf_dlc')
    # print(large_ddsp['files']['meta.yaml'])
    # large_reflow = torch.load('ddsp6.3_6x512_10x2048.sf_dlc')
    # large_reflow_ddsp_keys = [k for k in large_reflow['files']['model_0.pt']['model'].keys() if str(k).startswith('ddsp_model')]
    # large_ddsp_ddsp_keys = [k for k in large_ddsp['files']['model_0.pt']['model'].keys() if str(k).startswith('ddsp_model')]
    # for key in large_reflow_ddsp_keys:
    #     del large_reflow['files']['model_0.pt']['model'][key]
        
    # for key in large_ddsp_ddsp_keys:
    #     large_reflow['files']['model_0.pt']['model'][key] = large_ddsp['files']['model_0.pt']['model'][key]

    # ckpt = large_reflow['files']['model_0.pt']
    # large_ddsp = torch.load('ddsp6.3_6x512_10x2048.sf_dlc')
    # ckpt = large_ddsp['files']['model_0.pt']
    # large_reflow['global_step'] = 0
    # item = torch.save(large_reflow, 'model_0_ultra.pt')
    # print(large_ddsp['files']['model_0.pt']['model'].keys())
    # model = large_ddsp['files']['model_0.pt']
    
    
    
    # ckpt1 = torch.load('model_15001.pt')
    
    # pure_vc.load_state_dict(ckpt1['model'])
    
    # reflow_model = RectifiedFlow(LYNXNet2(in_dims=768, dim_cond=768, n_layers=6, n_chans=512), out_dims=768)
    # print(reflow_model.state_dict().keys())
    # reflow_model.apply(weights_init_uniform_rule)
    
    # pure_vc.reflow_diffuser = reflow_model
    
    # ckpt1['model'] = pure_vc.state_dict()
    # ckpt1['global_step'] = 0
    # print(ckpt.keys())
    # torch.save(large_reflow, 'out.ckpt')
    
    # m = GlowVcStylizer()
    # m.load_state_dict(pure_vc.state_dict())
    # ddsp = torch.load('ddsp6.3_6x512_10x2048.sf_dlc')['files']['model_0.pt']
    # ckpt = {
    #     'model': large_reflow['files']['model_0.pt']['model'],
    #     'global_step':0
    # }
    
    # torch.save(ckpt, 'model_0_high.pt')
    
    
    
    
    
    
    