import os
import random
import re
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import concurrent.futures


def get_npy_shape(file_path):
    with open(file_path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape = np.lib.format.read_array_header_1_0(f)[0]
        elif version == (2, 0):
            shape = np.lib.format.read_array_header_2_0(f)[0]
        else:
            raise ValueError("Unsupported .npy file version")
    return shape
        
def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list

def get_data_loaders(args, whole_audio=False, frontend=False, load_audio=False, train_aug=True, load_high_res_mel=False, aug_embd=False):
    data_train = AudioDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=train_aug,
        train_frontend=frontend,
        load_audio=load_audio,
        load_high_res_mel=load_high_res_mel,
        aug_embd=aug_embd
        )
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=2,
        hop_size=args.data.block_size, 
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        train_frontend=frontend,
        load_audio=True,
        load_high_res_mel=load_high_res_mel)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid


def get_base_data_loaders(args, whole_audio=False, frontend=False, load_audio=False, train_aug=True, finetune=False, load_mel=False, load_high_res_mel=False):
    data_train = BaseDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=train_aug,
        train_frontend=frontend,
        load_audio=load_audio,
        finetune=finetune,
        load_mel=load_mel,
        load_high_res_mel=load_high_res_mel
        )
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False,
        prefetch_factor=4 if args.train.cache_device=='cpu' else None,
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=2,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        train_frontend=True,
        load_audio=True,
        load_high_res_mel=load_high_res_mel)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
        train_frontend=False,
        load_audio=False,
        load_high_res_mel=False,
        aug_embd=False
    ):
        super().__init__()
        self.load_audio = load_audio
        self.load_high_res_mel = load_high_res_mel
        self.aug_embd = aug_embd
        self.train_frontend = train_frontend
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.crop_len = int(waveform_sec * sample_rate / hop_size)
        self.path_root = path_root
        self.device = device
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        self.speaker_mean = torch.from_numpy(np.load('data/speaker_elysia_new.npy')).to(device).squeeze(0)
        self.pitch_aug_dict = np.load(os.path.join(self.path_root, 'pitch_aug_dict.npy'), allow_pickle=True).item()
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)
        
        def _load_single_file(name_ext):
            name = os.path.splitext(name_ext)[0]

            path_f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
            f0 = np.load(path_f0)
            f0_len = len(f0)
            f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)

            path_volume = os.path.join(self.path_root, 'volume', name_ext) + '.npy'
            volume = np.load(path_volume)
            volume_len = len(volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)

            path_augvol = os.path.join(self.path_root, 'aug_vol', name_ext) + '.npy'
            aug_vol = np.load(path_augvol)
            aug_vol_len = len(aug_vol)
            aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)

            if n_spk is not None and n_spk > 1:
                dirname_split = re.split(r"_|\-", os.path.dirname(name_ext), 2)[0]
                spk_id = int(dirname_split) if str.isdigit(dirname_split) else 0
                if spk_id < 1 or spk_id > n_spk:
                    raise ValueError(' [x] Muiti-speaker traing error : spk_id must be a positive integer from 1 to n_spk ')
            else:
                spk_id = 1
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            path_mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
            path_augmel = os.path.join(self.path_root, 'aug_mel', name_ext) + '.npy'
            path_highresmel = os.path.join(self.path_root, 'mel_high_res', name_ext) + '.npy'
            path_aughighresmel = os.path.join(self.path_root, 'aug_mel_high_res', name_ext) + '.npy'
            path_units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
            path_unitw = os.path.join(self.path_root, 'whisper_units', name_ext) + '.npy'
            path_unith = os.path.join(self.path_root, 'hubert_units', name_ext) + '.npy'
            path_augunits = os.path.join(self.path_root, 'units_aug', name_ext) + '.npy'
            path_augunitw = os.path.join(self.path_root, 'whisper_units_aug', name_ext) + '.npy'
            path_augunith = os.path.join(self.path_root, 'hubert_units_aug', name_ext) + '.npy'
            path_speaker = os.path.join(self.path_root, 'speaker', name_ext) + '.npy'
            # path_emo = os.path.join(self.path_root, 'emo', name_ext) + '.npy'

            mel_len = get_npy_shape(path_mel)[0]
            aug_mel_len = get_npy_shape(path_augmel)[0]
            units_len = get_npy_shape(path_units)[0]
            if self.aug_embd:
                units_len = min(units_len, get_npy_shape(path_augunits)[0])

            path_audio = os.path.join(self.path_root, 'audio', name_ext)
            gt_audio = torch.from_numpy(librosa.load(path_audio, sr=44100)[0]).to(device)
            frame_len = min(mel_len, aug_mel_len, units_len, f0_len, volume_len, aug_vol_len)
            if self.load_high_res_mel:
                high_res_mel_len = get_npy_shape(path_highresmel)[0]
                aug_high_res_mel_len = get_npy_shape(path_aughighresmel)[0]
                frame_len = min(frame_len, high_res_mel_len, aug_high_res_mel_len)
            if load_all_data:
                mel = np.load(path_mel)
                mel = torch.from_numpy(mel).to(device)

                aug_mel = np.load(path_augmel)
                aug_mel = torch.from_numpy(aug_mel).to(device)

                if self.load_high_res_mel:
                    high_res_mel = np.load(path_highresmel)
                    high_res_mel = torch.from_numpy(high_res_mel).to(device)
                    aug_high_res_mel = np.load(path_aughighresmel)
                    aug_high_res_mel = torch.from_numpy(aug_high_res_mel).to(device)

                units = np.load(path_units)
                units = torch.from_numpy(units).to(device)
                # units_mean = units.mean(dim=0)
                if self.aug_embd:
                    units_aug = np.load(path_augunits)
                    units_aug = torch.from_numpy(units_aug).to(device)
                if train_frontend:
                    unitw_len = get_npy_shape(path_unitw)[0]
                    unith_len = get_npy_shape(path_unith)[0]
                    frame_len = min(frame_len, unitw_len, unith_len)
                    if self.aug_embd:
                        frame_len = min(frame_len,
                                        get_npy_shape(path_augunitw)[0],
                                        get_npy_shape(path_augunith)[0])

                    units_w = np.load(path_unitw)
                    units_w = torch.from_numpy(units_w).to(device)

                    units_h = np.load(path_unith)
                    units_h = torch.from_numpy(units_h).to(device)

                    if self.aug_embd:
                        units_w_aug = np.load(path_augunitw)
                        units_w_aug = torch.from_numpy(units_w_aug).to(device)

                        units_h_aug = np.load(path_augunith)
                        units_h_aug = torch.from_numpy(units_h_aug).to(device)
                    # h_mean = units_h.mean(dim=0)
                    # emo = np.load(path_emo)
                    # emo_h = torch.from_numpy(emo).to(device).float()
                    
                    speaker_h = np.load(path_speaker)
                    speaker_h = torch.from_numpy(speaker_h).to(device)
                    spk_embd = self.speaker_mean if torch.cosine_similarity(speaker_h, self.speaker_mean, dim=0) > 0.85 else speaker_h
                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    if self.load_high_res_mel:
                        high_res_mel = high_res_mel.half()
                        aug_high_res_mel = aug_high_res_mel.half()
                    units = units.half()
                    if self.aug_embd:
                        units_aug = units_aug.half()
                    if train_frontend:
                        units_w = units_w.half()
                        units_h = units_h.half()
                        if self.aug_embd:
                            units_w_aug = units_w_aug.half()
                            units_h_aug = units_h_aug.half()
                if train_frontend:
                    self.data_buffer[name_ext] = {
                            'frame_len': frame_len,
                            'mel': mel,
                            # 'energy': energy,
                            # 'emo': emo_h,
                            'aug_mel': aug_mel,
                            'units': units,
                            'units_w': units_w,
                            'units_h': units_h,
                            'f0': f0,
                            'volume': volume,
                            'aug_vol': aug_vol,
                            'spk_id': spk_id,
                            'spk_embd': spk_embd
                            }
                    if self.aug_embd:
                        self.data_buffer[name_ext]['units_aug'] = units_aug
                        self.data_buffer[name_ext]['units_w_aug'] = units_w_aug
                        self.data_buffer[name_ext]['units_h_aug'] = units_h_aug
                    if load_audio:
                        self.data_buffer[name_ext]['gt_audio'] = gt_audio
                    if self.load_high_res_mel:
                        self.data_buffer[name_ext]['mel_high_res'] = high_res_mel
                        self.data_buffer[name_ext]['aug_mel_high_res'] = aug_high_res_mel
                else:
                    self.data_buffer[name_ext] = {
                            'frame_len': frame_len,
                            'mel': mel,
                            # 'energy': energy,
                            'aug_mel': aug_mel,
                            'units': units,
                            # 'spk_embd': spk_embd,
                            # 'units_w': units_w,
                            # 'units_h': units_h,
                            'f0': f0,
                            'volume': volume,
                            'aug_vol': aug_vol,
                            'spk_id': spk_id
                            }
                    if self.aug_embd:
                        self.data_buffer[name_ext]['units_aug'] = units_aug
                    if load_audio:
                        self.data_buffer[name_ext]['gt_audio'] = gt_audio
                    if self.load_high_res_mel:
                        self.data_buffer[name_ext]['mel_high_res'] = high_res_mel
                        self.data_buffer[name_ext]['aug_mel_high_res'] = aug_high_res_mel

            else:
                data_dict = {
                        'frame_len': frame_len,
                        'f0': f0,
                        'volume': volume,
                        # 'energy': energy,
                        # 'spk_embd': spk_embd,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
                if load_audio:
                    self.data_buffer[name_ext]['gt_audio'] = gt_audio
           

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        if data_buffer['frame_len'] < self.crop_len:
            return self.__getitem__( (file_idx + 1) % len(self.paths))
            
        # get item
        return self.get_data(name_ext, data_buffer)
    
    def frame_idx_to_audio(self, embedding_start_idx, embedding_end_idx,
                              encoder_sr=16000, audio_sr=44100,
                              hop_size=160, block_size=512):
        """
        Given a slice of embedding indices, return the corresponding slice
        in the ground truth audio (at 44.1kHz) that matches the generated audio.
        """
        # Step 1: encoder time (in samples)
        encoder_start_sample = embedding_start_idx * hop_size

        # Step 2: map to 44.1kHz audio domain
        ratio = audio_sr / encoder_sr
        audio_start_sample = int(round(encoder_start_sample * ratio))

        # Step 3: decoder audio length (already in 44.1kHz samples)
        num_frames = embedding_end_idx - embedding_start_idx
        decoder_audio_length = num_frames * block_size

        audio_end_sample = audio_start_sample + decoder_audio_length

        return audio_start_sample, audio_end_sample

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        start_frame = 0 if self.whole_audio else random.randint(0, data_buffer['frame_len'] - self.crop_len)
        units_frame_len = data_buffer['frame_len'] if self.whole_audio else self.crop_len
        aug_flag = random.choice([True, False]) and self.use_aug

        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = os.path.join(self.path_root, mel_key, name_ext) + '.npy'
            mel = np.load(mel)
            mel = mel[start_frame : start_frame + units_frame_len]
            mel = torch.from_numpy(mel)
        else:
            mel = mel[start_frame : start_frame + units_frame_len]

        # load high-resolution mel (matching the clean/aug choice of the regular mel)
        if self.load_high_res_mel:
            hr_mel_key = 'aug_mel_high_res' if aug_flag else 'mel_high_res'
            mel_high_res = data_buffer.get(hr_mel_key)
            if mel_high_res is None:
                mel_high_res = os.path.join(self.path_root, hr_mel_key, name_ext) + '.npy'
                mel_high_res = np.load(mel_high_res)
                mel_high_res = mel_high_res[start_frame : start_frame + units_frame_len]
                mel_high_res = torch.from_numpy(mel_high_res)
            else:
                mel_high_res = mel_high_res[start_frame : start_frame + units_frame_len]

        # load units (pitch-augmented ones when the augmented mel is used)
        units_aug_flag = self.aug_embd

        units_key = 'units_aug' if units_aug_flag else 'units'
        units_dir = 'units_aug' if units_aug_flag else 'units'
        units = data_buffer.get(units_key)
        if units is None:
            units = os.path.join(self.path_root, units_dir, name_ext) + '.npy'
            units = np.load(units)
            units = units[start_frame : start_frame + units_frame_len]
            units = torch.from_numpy(units)
        else:
            units = units[start_frame : start_frame + units_frame_len]
        if self.train_frontend:
            units_w_key = 'units_w_aug' if units_aug_flag else 'units_w'
            units_w_dir = 'whisper_units_aug' if units_aug_flag else 'whisper_units'
            units_w = data_buffer.get(units_w_key)
            if units_w is None:
                units_w = os.path.join(self.path_root, units_w_dir, name_ext) + '.npy'
                units_w = np.load(units_w)
                units_w = units_w[start_frame : start_frame + units_frame_len]
                units_w = torch.from_numpy(units_w)
            else:
                units_w = units_w[start_frame : start_frame + units_frame_len]

            units_h_key = 'units_h_aug' if units_aug_flag else 'units_h'
            units_h_dir = 'hubert_units_aug' if units_aug_flag else 'hubert_units'
            units_h = data_buffer.get(units_h_key)
            if units_h is None:
                units_h = os.path.join(self.path_root, units_h_dir, name_ext) + '.npy'
                units_h = np.load(units_h)
                units_h = units_h[start_frame : start_frame + units_frame_len]
                units_h = torch.from_numpy(units_h)
            else:
                units_h = units_h[start_frame : start_frame + units_frame_len]


            # emo_h = data_buffer.get('emo')
            # if emo_h is None:
            #     emo_h = os.path.join(self.path_root, 'emo', name_ext) + '.npy'
            #     emo_h = np.load(emo_h)
            #     emo_h = emo_h[start_frame : start_frame + units_frame_len]
            #     emo_h = torch.from_numpy(emo_h)
            # else:
            #     emo_h = emo_h[start_frame : start_frame + units_frame_len]
                
            # h_mean = units_h.mean(dim=0).to(self.device)
            spk_embd = data_buffer.get('spk_embd')
            if spk_embd is None:
                path_speaker = os.path.join(self.path_root, 'speaker', name_ext) + '.npy'
                spk_embd = np.load(path_speaker)
                spk_embd = torch.from_numpy(spk_embd).to(self.device)
                
            # spk_embd = self.speaker_mean if torch.cosine_similarity(units_mean, self.speaker_mean) > 0.85 else units_mean


        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dict[name_ext]
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        # vlen = volume.shape[0]
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        
        # start_percent = start_frame /vlen
        # end_percent = (start_frame + units_frame_len)/vlen
        # energy = data_buffer.get('energy')
        # energy = energy[start_frame : start_frame + units_frame_len]
        # load spk_id
        spk_id = data_buffer.get('spk_id')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()
        if self.load_audio:
            gt_audio = data_buffer.get('gt_audio')
            # audio_len = gt_audio.shape[0]
            # start_audio = start_percent * audio_len
            # end_audio = end_percent * audio_len
            a_start, a_end = self.frame_idx_to_audio(start_frame, start_frame+units_frame_len)
            gt_audio = gt_audio[a_start: a_end]
        # spk_embd = data_buffer.get('spk_embd')
        if self.train_frontend:
            if not self.load_audio:
                d = dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, units_w=units_w, units_h=units_h, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext,spk_embd=spk_embd)
            else:
                d = dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, units_w=units_w, units_h=units_h, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext, gt_audio=gt_audio, spk_embd=spk_embd)
        else:
            if not self.load_audio:
                d = dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext)
            else:
                d = dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext, gt_audio=gt_audio)
        if self.load_high_res_mel:
            d['mel_high_res'] = mel_high_res
        return d

    def __len__(self):
        return len(self.paths)



class BaseDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
        train_frontend=False,
        load_audio=False,
        finetune=False,
        load_mel=False,
        load_high_res_mel=False
    ):
        super().__init__()
        self.load_audio = load_audio
        self.load_mel = load_mel
        self.load_high_res_mel = load_high_res_mel
        self.train_frontend = train_frontend
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.crop_len = int(waveform_sec * sample_rate / hop_size)
        self.path_root = path_root
        self.device = device
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        # self.speaker_mean = torch.from_numpy(np.load('data/speaker_elysia_old.npy')).to(device).squeeze(0)
        self.speaker_mean = torch.from_numpy(np.load('data/speaker_elysia_old.npy')).to(device).squeeze(0)
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the data from :', path_root)
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            name = os.path.splitext(name_ext)[0]

            path_units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
            path_unitw = os.path.join(self.path_root, 'whisper_units', name_ext) + '.npy'
            path_unith = os.path.join(self.path_root, 'hubert_units', name_ext) + '.npy'
            path_speaker = os.path.join(self.path_root, 'speaker', name_ext) + '.npy'
            path_mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
            path_highresmel = os.path.join(self.path_root, 'mel_high_res', name_ext) + '.npy'

            units_len = get_npy_shape(path_units)[0]
            unitw_len = get_npy_shape(path_unitw)[0]
            unith_len = get_npy_shape(path_unith)[0]
            frame_len = min(units_len, unitw_len, unith_len)
            if self.load_mel:
                mel_len = get_npy_shape(path_mel)[0]
                frame_len = min(frame_len, mel_len)
            if self.load_high_res_mel:
                high_res_mel_len = get_npy_shape(path_highresmel)[0]
                frame_len = min(frame_len, high_res_mel_len)

            if load_all_data:
                units = np.load(path_units)
                units = torch.from_numpy(units).to(device)

                units_w = np.load(path_unitw)
                units_w = torch.from_numpy(units_w).to(device)

                units_h = np.load(path_unith)
                units_h = torch.from_numpy(units_h).to(device)

                speaker_h = np.load(path_speaker)
                speaker_h = torch.from_numpy(speaker_h).to(device)
                spk_embd = self.speaker_mean if torch.cosine_similarity(speaker_h, self.speaker_mean, dim=0) > 0.85 else speaker_h

                if self.load_mel:
                    mel = np.load(path_mel)
                    mel = torch.from_numpy(mel).to(device)

                if self.load_high_res_mel:
                    high_res_mel = np.load(path_highresmel)
                    high_res_mel = torch.from_numpy(high_res_mel).to(device)

                if fp16:
                    units = units.half()
                    units_w = units_w.half()
                    units_h = units_h.half()
                    if self.load_mel:
                        mel = mel.half()
                    if self.load_high_res_mel:
                        high_res_mel = high_res_mel.half()

                self.data_buffer[name_ext] = {
                        'frame_len': frame_len,
                        'units': units,
                        'units_w': units_w,
                        'units_h': units_h,
                        'spk_embd': spk_embd
                        }
                if self.load_mel:
                    self.data_buffer[name_ext]['mel'] = mel
                if self.load_high_res_mel:
                    self.data_buffer[name_ext]['mel_high_res'] = high_res_mel
            else:
                self.data_buffer[name_ext] = {
                    'frame_len': frame_len
                }


    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        if data_buffer['frame_len'] < self.crop_len:
            return self.__getitem__( (file_idx + 1) % len(self.paths))

        # get item
        return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        start_frame = 0 if self.whole_audio else random.randint(0, data_buffer['frame_len'] - self.crop_len)
        units_frame_len = data_buffer['frame_len'] if self.whole_audio else self.crop_len

        # load units
        units = data_buffer.get('units')
        if units is None:
            units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
            units = np.load(units)
            units = units[start_frame : start_frame + units_frame_len]
            units = torch.from_numpy(units)
        else:
            units = units[start_frame : start_frame + units_frame_len]

        # load whisper units
        units_w = data_buffer.get('units_w')
        if units_w is None:
            units_w = os.path.join(self.path_root, 'whisper_units', name_ext) + '.npy'
            units_w = np.load(units_w)
            units_w = units_w[start_frame : start_frame + units_frame_len]
            units_w = torch.from_numpy(units_w)
        else:
            units_w = units_w[start_frame : start_frame + units_frame_len]

        # load hubert units
        units_h = data_buffer.get('units_h')
        if units_h is None:
            units_h = os.path.join(self.path_root, 'hubert_units', name_ext) + '.npy'
            units_h = np.load(units_h)
            units_h = units_h[start_frame : start_frame + units_frame_len]
            units_h = torch.from_numpy(units_h)
        else:
            units_h = units_h[start_frame : start_frame + units_frame_len]

        # load speaker embedding
        spk_embd = data_buffer.get('spk_embd')
        if spk_embd is None:
            path_speaker = os.path.join(self.path_root, 'speaker', name_ext) + '.npy'
            spk_embd = np.load(path_speaker)
            spk_embd = torch.from_numpy(spk_embd).to(self.device)

        d = dict(units=units, units_w=units_w, units_h=units_h, spk_embd=spk_embd, name=name, name_ext=name_ext)

        # load mel
        if self.load_mel:
            mel = data_buffer.get('mel')
            if mel is None:
                mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
                mel = np.load(mel)
                mel = mel[start_frame : start_frame + units_frame_len]
                mel = torch.from_numpy(mel)
            else:
                mel = mel[start_frame : start_frame + units_frame_len]
            d['mel'] = mel

        # load high-resolution mel
        if self.load_high_res_mel:
            mel_high_res = data_buffer.get('mel_high_res')
            if mel_high_res is None:
                mel_high_res = os.path.join(self.path_root, 'mel_high_res', name_ext) + '.npy'
                mel_high_res = np.load(mel_high_res)
                mel_high_res = mel_high_res[start_frame : start_frame + units_frame_len]
                mel_high_res = torch.from_numpy(mel_high_res)
            else:
                mel_high_res = mel_high_res[start_frame : start_frame + units_frame_len]
            d['mel_high_res'] = mel_high_res

        return d

    def __len__(self):
        return len(self.paths)