from dataclasses import dataclass
import os
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import pyworld as pw
import parselmouth
import torchcrepe
from transformers import AutoModel, HubertModel, Wav2Vec2FeatureExtractor
from fairseq import checkpoint_utils
import fairseq.utils
from encoder.whisper.inference import pred_ppg
from encoder.whisper.model import ModelDimensions, Whisper
from encoder.hubert.model import HubertSoft
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchaudio.transforms import Resample

from stylizer.stft_loss import fmt_row, stft_bandwise_ratios, stft_energy_ratio, summarize_bandwise, summarize_ratio
from .unit2control import Unit2Control
from .core import frequency_filter, upsample, remove_above_fmax, MaskedAvgPool1d, MedianPool1d
import time

CREPE_RESAMPLE_KERNEL = {}
F0_KERNEL = {}

class F0_Extractor:
    def __init__(self, f0_extractor, sample_rate = 44100, hop_size = 512, f0_min = 65, f0_max = 800):
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        if f0_extractor == 'crepe':
            key_str = str(sample_rate)
            if key_str not in CREPE_RESAMPLE_KERNEL:
                CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width = 128)
            self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
        if f0_extractor == 'rmvpe':
            if 'rmvpe' not in F0_KERNEL :
                from encoder.rmvpe import RMVPE
                F0_KERNEL['rmvpe'] = RMVPE('pretrain/rmvpe/model.pt', hop_length=160)
            self.rmvpe = F0_KERNEL['rmvpe']
        if f0_extractor == 'fcpe':
            self.device_fcpe = 'cuda' if torch.cuda.is_available() else 'cpu'
            if 'fcpe' not in F0_KERNEL :
                from torchfcpe import spawn_bundled_infer_model
                F0_KERNEL['fcpe'] = spawn_bundled_infer_model(device=self.device_fcpe)
            self.fcpe = F0_KERNEL['fcpe']
                
    def extract(self, audio, uv_interp = False, device = None, silence_front = 0): # audio: 1d numpy array
        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1
                
        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) : ]
        
        # extract f0 using parselmouth
        if self.f0_extractor == 'parselmouth':
            l_pad = int(np.ceil(1.5 / self.f0_min * self.sample_rate))
            r_pad = int(self.hop_size * ((len(audio) - 1) // self.hop_size + 1) - len(audio) + l_pad + 1)
            s = parselmouth.Sound(np.pad(audio, (l_pad, r_pad)), self.sample_rate).to_pitch_ac(
                time_step = self.hop_size / self.sample_rate, 
                voicing_threshold = 0.6,
                pitch_floor = self.f0_min, 
                pitch_ceiling = self.f0_max)
            assert np.abs(s.t1 - 1.5 / self.f0_min) < 0.001
            f0 = np.pad(s.selected_array['frequency'], (start_frame, 0))
            if len(f0) < n_frames:
                f0 = np.pad(f0, (0, n_frames - len(f0)))
            f0 = f0[: n_frames]
            
        # extract f0 using dio
        elif self.f0_extractor == 'dio':
            _f0, t = pw.dio(
                audio.astype('double'), 
                self.sample_rate, 
                f0_floor = self.f0_min, 
                f0_ceil = self.f0_max, 
                channels_in_octave=2, 
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = pw.stonemask(audio.astype('double'), _f0, t, self.sample_rate)
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))
        
        # extract f0 using harvest
        elif self.f0_extractor == 'harvest':
            f0, _ = pw.harvest(
                audio.astype('double'), 
                self.sample_rate, 
                f0_floor = self.f0_min, 
                f0_ceil = self.f0_max, 
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))
        
        # extract f0 using crepe        
        elif self.f0_extractor == 'crepe':
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            resample_kernel = self.resample_kernel.to(device)
            wav16k_torch = resample_kernel(torch.FloatTensor(audio).unsqueeze(0).to(device))
            
            f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, self.f0_min, self.f0_max, pad=True, model='full', batch_size=512, device=device, return_periodicity=True)
            pd = MedianPool1d(pd, 4)
            f0 = torchcrepe.threshold.At(0.05)(f0, pd)
            f0 = MaskedAvgPool1d(f0, 4)
            
            f0 = f0.squeeze(0).cpu().numpy()
            f0 = np.array([f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.005)), len(f0) - 1))] for n in range(n_frames - start_frame)])
            f0 = np.pad(f0, (start_frame, 0))
        
        # extract f0 using rmvpe
        elif self.f0_extractor == "rmvpe":
            f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
            
        # elif self.f0_extractor == 'rmvpe':
        #     base_step_sec = 0.01  # RMVPE's native frame step (~10 ms)
        #     offsets_sec = np.array(
        #         [0.0, base_step_sec / 4, base_step_sec / 2, 3 * base_step_sec / 4],
        #         dtype=float,
        #     )  # 0, 2.5, 5.0, 7.5 ms

        #     # Store per-pass outputs
        #     per_pass_f0 = []
        #     per_pass_uv = []
        #     per_pass_len = []

        #     # 1) RMVPE passes (no interpolation)
        #     for off in offsets_sec:
        #         pad = int(round(off * self.sample_rate))
        #         audio_shifted = np.pad(audio, (pad, 0)) if pad > 0 else audio

        #         f0_k = self.rmvpe.infer_from_audio(
        #             audio_shifted, self.sample_rate, device=device, thred=0.03, use_viterbi=False
        #         )
        #         uv_k = (f0_k == 0)

        #         per_pass_f0.append(f0_k.astype(float))
        #         per_pass_uv.append(uv_k.astype(bool))
        #         per_pass_len.append(len(f0_k))

        #     # 2) Build interleaved (time, f0) point set using only voiced frames from all passes
        #     #    Frame times for pass k are at t = i * 0.01 - offset_k
        #     times_all = []
        #     vals_all = []
        #     for k, off in enumerate(offsets_sec):
        #         f0_k = per_pass_f0[k]
        #         uv_k = per_pass_uv[k]
        #         Lk = len(f0_k)

        #         # native times then shift back by 'off'
        #         t_k = (base_step_sec * np.arange(Lk)) - off

        #         # Keep only voiced frames and non-negative times (discard < 0 to avoid extrapolation noise)
        #         voiced_idx = (~uv_k) & np.isfinite(f0_k) & (t_k >= 0.0)
        #         if np.any(voiced_idx):
        #             times_all.append(t_k[voiced_idx])
        #             vals_all.append(f0_k[voiced_idx])

        #     if len(times_all) == 0:
        #         # No voiced points at all — return all zeros on requested grid
        #         target_time = (self.hop_size / self.sample_rate) * np.arange(n_frames - start_frame)
        #         f0_interp = np.zeros_like(target_time, dtype=float)
        #         f0_out = np.pad(f0_interp, (start_frame, 0))
        #         return f0_out

        #     x = np.concatenate(times_all, axis=0)
        #     y = np.concatenate(vals_all, axis=0)

        #     # Sort by time; drop any duplicates by stable unique (optional but safer for np.interp)
        #     order = np.argsort(x)
        #     x = x[order]
        #     y = y[order]
        #     # Deduplicate identical timestamps by keeping the last one (rare but possible at boundaries)
        #     if x.size >= 2:
        #         keep = np.ones_like(x, dtype=bool)
        #         keep[1:] = x[1:] != x[:-1]
        #         x = x[keep]
        #         y = y[keep]

        #     # 3) Final target grid (your pipeline grid)
        #     target_time = (self.hop_size / self.sample_rate) * np.arange(n_frames - start_frame)

        #     # Guard: need at least 2 points for np.interp; otherwise constant fill
        #     if x.size == 1:
        #         f0_interp = np.full_like(target_time, y[0], dtype=float)
        #     else:
        #         # 4) Single final interpolation from dense interleaved set to target grid
        #         # Left/right fill with 0 to avoid extrapolated nonsense outside coverage
        #         f0_interp = np.interp(target_time, x, y, left=0.0, right=0.0)

        #     # 5) Compute UV on the target grid via nearest-frame voting (no interpolation)
        #     #    For each pass, map target_time + offset to nearest RMVPE frame index
        #     votes = np.zeros_like(target_time, dtype=int)
        #     for k, off in enumerate(offsets_sec):
        #         f0_k = per_pass_f0[k]
        #         uv_k = per_pass_uv[k]
        #         Lk = len(f0_k)

        #         # For a target time t on original timeline, the corresponding time on shifted audio is t + off.
        #         # Nearest RMVPE frame index on that pass:
        #         idx = np.rint((target_time + off) / base_step_sec).astype(int)

        #         # Mark out-of-range as unvoiced (no vote)
        #         valid = (idx >= 0) & (idx < Lk)
        #         # We count a "voiced vote" where valid and not unvoiced
        #         voiced_vote = np.zeros_like(target_time, dtype=bool)
        #         vv = valid & (~uv_k[np.clip(idx, 0, Lk - 1)])
        #         voiced_vote[vv] = True

        #         votes += voiced_vote.astype(int)

        #     uv_final = votes < 1
        #     f0_interp[uv_final] = 0.0

        #     # 6) Left pad to match your original convention
        #     f0 = np.pad(f0_interp, (start_frame, 0))
        
        # extract f0 using fcpe
        elif self.f0_extractor == "fcpe":
            _audio = torch.from_numpy(audio).to(self.device_fcpe).unsqueeze(0)
            f0 = self.fcpe(_audio, sr=self.sample_rate, decoder_mode="local_argmax", threshold=0.006)
            f0 = f0.squeeze().cpu().numpy()
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
            
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")
                    
        # interpolate the unvoiced f0 
        if uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min
        return f0


class Volume_Extractor:
    def __init__(self, hop_size = 512, win_size = 2048):
        self.hop_size = hop_size
        self.win_size = win_size
        
    def extract(self, audio): # audio: 1d numpy array
        n_frames = int(len(audio) // self.hop_size) + 1
        audio = np.pad(audio, (int(self.win_size // 2), int((self.win_size + 1) // 2)), mode = 'reflect')
        audio2 = audio ** 2
        mean = np.array([np.mean(audio[int(n * self.hop_size) : int(n * self.hop_size + self.win_size)]) for n in range(n_frames)])
        mean_square = np.array([np.mean(audio2[int(n * self.hop_size) : int(n * self.hop_size + self.win_size)]) for n in range(n_frames)])
        volume = np.sqrt(np.clip(mean_square - mean ** 2, 0, None))
        return volume
    
         
class Units_Encoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate = 16000, encoder_hop_size = 320, device = None,
                 cnhubertsoft_gate=10, grad_flow=False):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if encoder == 'hubertsoft':
            self.model = Audio2HubertSoft(encoder_ckpt, grad_flow=grad_flow).to(device)
            is_loaded_encoder = True
        if encoder == 'hubertsofttta2x':
            self.model = Audio2HubertSoftTTA2X(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertsoftOri':
            self.model = Audio2HubertSoftOriginal(encoder_ckpt, device=device, grad_flow=grad_flow)
            is_loaded_encoder = True
        if encoder == 'contentvec768l12':
            self.model = Audio2ContentVec768L12(encoder_ckpt, device=device, grad_flow=grad_flow)
            is_loaded_encoder = True
        if encoder == 'contentvec768l12tta2x':
            self.model = Audio2ContentVec768L12TTA2X(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'whisper-large-pgg-tta2x':
            self.model = Audio2WhisperPpgLargeTTA2X(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'whisper-large-pgg':
            self.model = Audio2WhisperPpgLarge(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'cnhubertsoftfish':
            self.model = CNHubertSoftFish(encoder_ckpt, device=device, gate_size=cnhubertsoft_gate)
            is_loaded_encoder = True
        if encoder == 'emotionvec':
            self.model = Audio2EmoVec(encoder_ckpt, device=device)
            is_loaded_encoder = True
            
        if not is_loaded_encoder:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")
            
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size
        
    def encode(self, 
                audio, # B, T
                sample_rate,
                hop_size): 
        
        # resample
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        
        # encode
        if audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio_res, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res)
        
        # alignment
        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
        return units_aligned
    
    def encode_emo(self, file, model, target_len): 
        # resample
        res = model.generate(file, output_dir="./outputs", granularity="frame", extract_embedding=False, disable_pbar=True)
        units = self.resize_array(res[0].get('layer_res')[4].squeeze(0).cpu(), target_len)
        return units
    
    def resize_array(self, tensor, new_length):
        N_orig = tensor.shape[0]
        x_original = (np.arange(N_orig) + 0.5) / N_orig
        x_target = (np.arange(new_length) + 0.5) / new_length

        # 3. Loop through each column, interpolate it, and stack them back together
        interpolated_array = np.column_stack([
            np.interp(x_target, x_original, tensor[:, i]) 
            for i in range(tensor.shape[1])
        ])
        
        return torch.from_numpy(interpolated_array)
        
class Audio2HubertSoft(torch.nn.Module):
    def __init__(self, path, h_sample_rate = 16000, h_hop_size = 320, grad_flow=False):
        super().__init__()
        print(' [Encoder Model] HuBERT Soft')
        self.hubert = HubertSoft()
        print(' [Loading] ' + path)
        checkpoint = torch.load(path, weights_only=False)
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.grad_flow = grad_flow
        print(self.grad_flow)
        if self.grad_flow:
            self.hubert.train()
        else:
            self.hubert.eval()
     
    def forward(self, 
                audio): # B, T
        if self.grad_flow:
            units = self.hubert.units(audio.unsqueeze(1))
            return units
        else:
            with torch.inference_mode():  
                units = self.hubert.units(audio.unsqueeze(1))
                return units
        
class Audio2HubertSoftOriginal(torch.nn.Module):
    def __init__(self, path, h_sample_rate = 16000, h_hop_size = 320, device='cpu', grad_flow=False):
        super().__init__()
        print(' [Encoder Model] HuBERT Soft')
        self.hubert = HubertSoft()
        print(' [Loading] ' + path)
        path = 'pretrain/hubert/hubert_soft.pt'
        checkpoint = torch.load(path)
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.grad_flow = grad_flow
        print(self.grad_flow)
        if self.grad_flow:
            self.hubert.train()
        else:
            self.hubert.eval()
        self.hubert = self.hubert.to(device)
        self.device = device
     
    def __call__(self, audio): # B, T
        if self.grad_flow:
            units = self.hubert.units(audio.unsqueeze(1))
            return units
        else:
            with torch.inference_mode():  
                units = self.hubert.units(audio.unsqueeze(1))
                return units

class Audio2HubertSoftTTA2X(torch.nn.Module):
    def __init__(self, path, h_sample_rate = 16000, h_hop_size = 320, device='cpu'):
        super().__init__()
        print(' [Encoder Model] HuBERT Soft')
        self.hubert = HubertSoft()
        print(' [Loading] ' + path)
        path = 'pretrain/hubert/hubert_soft.pt'
        checkpoint = torch.load(path)
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.hubert.eval()
        self.hubert = self.hubert.to(device)
        self.device = device
     
    def __call__(self, audio): # B, T
        # print(audio.get_device(), self.device)
        with torch.no_grad():
            feats = self.hubert.units(audio.unsqueeze(1))
            audio2 = F.pad(audio, (160,0))
            feats2 = self.hubert.units(audio2.unsqueeze(1))
            n = feats2.shape[1] - feats.shape[1]
            if n > 0:
                feats = F.pad(feats, (0, 0, 0, 1))
            feats_tta = torch.cat((feats2, feats), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
            feats_tta = feats_tta[:, 1:, :]
        return feats_tta

class Audio2ContentVec():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = self.hubert.final_proj(logits[0])
        units = feats  # .transpose(2, 1)
        return units


class Audio2ContentVec768():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0]
        units = feats  # .transpose(2, 1)
        return units


class Audio2ContentVec768L12():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu', grad_flow=False):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.hubert = HubertModelWithFinalProj(HubertConfig())
        checkpoint = torch.load(path)
        self.hubert.load_state_dict(checkpoint)
        self.hubert = self.hubert.to(self.device)
        self.grad_flow = grad_flow
        if self.grad_flow:
            self.hubert.train()
        else:
            self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.reshape(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 12,  # layer 12
        }
        if self.grad_flow:
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0]
        else:
            with torch.no_grad():
                logits = self.hubert.extract_features(**inputs)
                feats = logits[0]
        units = feats  # .transpose(2, 1)
        return units    


class Audio2ContentVec768L12TTA2X():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=160, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.hubert = HubertModelWithFinalProj(HubertConfig())
        checkpoint = torch.load(path)
        self.hubert.load_state_dict(checkpoint)
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            feats = self.hubert(audio)["last_hidden_state"]
            audio = F.pad(audio, (160, 0))
            feats2 = self.hubert(audio)["last_hidden_state"]
            n = feats2.shape[1] - feats.shape[1]
            if n > 0:
                feats = F.pad(feats, (0, 0, 0, 1))
            feats_tta = torch.cat((feats2, feats), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
            feats_tta = feats_tta[:, 1:, :]
            if n > 0:
                feats_tta = feats_tta[:, :-1, :]
        units = feats_tta  # .transpose(2, 1)
        return units
    
class Audio2EmoVec():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=160, device='cpu'):
        self.device = device
        path = 'pretrain/emotionvec/model.pt'
        print(' [Encoder Model] Emo Vec')
        print(' [Loading] ' + path)

        self.model = None

    def __call__(self, file):  # B, T
        return None

@dataclass
class UserDirModule:
    user_dir: str
      
    
class Audio2ContentVec768L12TTA4X():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=160, device='cpu'):
        self.device = device
        self.h_sample_rate = h_sample_rate
        self.h_hop_size = h_hop_size  # this was your "half-hop" for 2x
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="")
        self.hubert = self.models[0].to(self.device).eval()

    def _extract(self, source, output_layer=12):
        padding_mask = torch.zeros_like(source, dtype=torch.bool)
        with torch.no_grad():
            feats = self.hubert.extract_features(
                source=source, padding_mask=padding_mask, output_layer=output_layer
            )[0]
        return feats  # shape: (B, T_frames, C)

    def __call__(self, audio):  # audio: (B, T) or (T,)
        # ensure (B, T)
        if audio.dim() == 1:
            wav_tensor = audio.view(1, -1).to(self.device)
        else:
            wav_tensor = audio.to(self.device)

        # TTA-4x: shifts = 0, q, 2q, 3q where q = h_hop_size//2
        q = max(int(self.h_hop_size) // 2, 1)  # quarter-hop in samples
        shifts = [0, q, 2*q, 3*q]

        # 1) run four passes with left padding
        feats_list = []
        base_feats = None
        for s in shifts:
            if s > 0:
                src = F.pad(wav_tensor, (s, 0))
            else:
                src = wav_tensor
            feats = self._extract(src)
            feats_list.append(feats)
            if base_feats is None:
                base_feats = feats  # unshifted reference

        # 2) align lengths (some streams can be +1 frame longer due to boundary effects)
        lens = [f.shape[1] for f in feats_list]
        max_len = max(lens)
        for i, f in enumerate(feats_list):
            if f.shape[1] < max_len:
                # right-pad frames by repeating the last frame (keeps content stable)
                pad_frames = max_len - f.shape[1]
                last = f[:, -1:, :].expand(-1, pad_frames, -1)
                feats_list[i] = torch.cat([f, last], dim=1)

        # 3) interleave: concat on feature dim then reshape to time-interleaved order
        #    Stack order is [shift0, shift1, shift2, shift3] → interleave on time axis
        Fcat = torch.cat(feats_list, dim=2)                  # (B, T, C*4)
        B, T, C4 = Fcat.shape
        C = C4 // 4
        feats_tta = Fcat.reshape(B, T*4, C)                  # interleaved time: 0,1,2,3, 4,5,6,7, ...

        # 4) trim: drop first (4-1)=3 frames to align centers; drop any tail overflow vs the unshifted length
        trim_head = 3
        feats_tta = feats_tta[:, trim_head:, :]

        # If the unshifted stream had length L0, we generally want close to 4*L0 frames,
        # but boundary effects add extras. Clamp to <= 4*L0.
        L0 = base_feats.shape[1]
        target_max = 4 * L0
        if feats_tta.shape[1] > target_max:
            feats_tta = feats_tta[:, :target_max, :]

        # Also, if you want exact parity with your 2x code's "n>0 → drop last" logic,
        # you can optionally shave one more frame when max_len > L0:
        extra = max_len - L0
        if extra > 0 and feats_tta.shape[1] > 0:
            feats_tta = feats_tta[:, :-extra, :]

        return feats_tta
    
    
class Audio2WhisperPpgLargeTTA2X(torch.nn.Module):
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        super().__init__()
        # path = 'pretrain/ppg-large/large-v3.pt'
        path = 'pretrain/ppg-large/large-v2.pt'
        self.device = device
        print(' [Encoder Model] Whisper PPG Large V2 TTA2X')
        print(' [Loading] ' + path)
        
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        dims = ModelDimensions(**checkpoint["dims"])
        self.model = Whisper(dims)
        del self.model.decoder
        cut = len(self.model.encoder.blocks) // 4
        cut = -1 * cut
        del self.model.encoder.blocks[cut:]
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model = self.model.eval()
        self.model = self.model.half()
        self.model = self.model.to(device)

    def __call__(self, audio):  # B, T
        with torch.no_grad():
            feats = pred_ppg(self.model, audio, self.device)
            audio2 = F.pad(audio, (160,0))
            feats2 = pred_ppg(self.model, audio2, self.device)
            n = feats2.shape[1] - feats.shape[1]
            if n > 0:
                feats = F.pad(feats, (0, 0, 0, 1))
            feats_tta = torch.cat((feats2, feats), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
            feats_tta = feats_tta[:, 1:, :]
        return feats_tta
    
class Audio2WhisperPpgLarge(torch.nn.Module):
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu', grad_flow=False):
        super().__init__()
        path = 'pretrain/ppg-large/large-v2.pt'
        self.device = device
        print(' [Encoder Model] Whisper PPG Large V2')
        print(' [Loading] ' + path)
        
        checkpoint = torch.load(path, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        self.model = Whisper(dims)
        del self.model.decoder
        cut = len(self.model.encoder.blocks) // 4
        cut = -1 * cut
        del self.model.encoder.blocks[cut:]
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model = self.model.eval()
        self.model = self.model.half()
        self.model = self.model.to(device)

    def __call__(self, audio):  # B, T
        with torch.no_grad():
            res = pred_ppg(self.model, audio, self.device)
        return res


class CNHubertSoftFish(torch.nn.Module):
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu', gate_size=10):
        super().__init__()
        self.device = device
        self.gate_size = gate_size

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "./pretrain/TencentGameMate/chinese-hubert-base")
        self.model = HubertModel.from_pretrained("./pretrain/TencentGameMate/chinese-hubert-base")
        self.proj = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(768, 256))
        # self.label_embedding = nn.Embedding(128, 256)

        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, audio):
        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        return self._forward(input_values[0])

    @torch.no_grad()
    def _forward(self, input_values):
        features = self.model(input_values)
        features = self.proj(features.last_hidden_state)

        # Top-k gating
        topk, indices = torch.topk(features, self.gate_size, dim=2)
        features = torch.zeros_like(features).scatter(2, indices, topk)
        features = features / features.sum(2, keepdim=True)

        return features.to(self.device)  # .transpose(1, 2)

    
class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
class CombSubSuperFast(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            n_unit=256,
            n_spk=1,
            num_layers=3,
            dim_model=256,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        #Unit2Control
        split_map = {
            'harmonic_magnitude': win_length // 2 + 1, 
            'harmonic_phase': win_length // 2 + 1,
            'noise_magnitude': win_length // 2 + 1,
            'noise_phase': win_length // 2 + 1
        }
        self.unit2ctrl = Unit2Control(
                            n_unit, 
                            block_size, 
                            n_spk, 
                            split_map,
                            num_layers=num_layers,
                            dim_model=dim_model,
                            use_norm=use_norm,
                            use_attention=use_attention, 
                            use_pitch_aug=use_pitch_aug)
    
    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth
    
    def smooth(self, x):
        if x.dim() == 2:               # (B, T)  → treat as (B, T, 1)
            x = x.unsqueeze(-1)

        B, T, C = x.shape
        # (C groups) depth-wise conv expects shape (B, C, T)
        x_ = x.permute(0, 2, 1)        # (B, C, T)

        # 5-tap kernel: [1,2,2,2,1] normalised to sum=1
        kernel = torch.tensor([0.125, 0.25, 0.25, 0.25, 0.125],
                            device=x.device, dtype=x.dtype)
        kernel = kernel.view(1, 1, -1).repeat(C, 1, 1)  # (C, 1, 5)

        # Depth-wise 1-D conv (groups=C) with padding=2
        blurred = F.conv1d(x_, kernel, padding=2, groups=C)

        blurred = blurred.permute(0, 2, 1)  # back to (B, T, C)
        return blurred.squeeze(-1) if blurred.shape[-1] == 1 else blurred
    
    def gen_pink_noise(self, combtooth):
        # return pink noise of same shape, device and dtype as combtooth (B, T)
        # FFT runs in float32: torch.fft.rfft does not support bf16/fp16
        B, T = combtooth.shape
        white = torch.randn(B, T, device=combtooth.device, dtype=torch.float32)
        spectrum = torch.fft.rfft(white, dim=-1)
        n_freq = spectrum.shape[-1]
        # 1/sqrt(f) amplitude shaping; DC bin zeroed so the noise is mean-free
        freqs = torch.arange(n_freq, device=combtooth.device, dtype=torch.float32)
        filt = torch.zeros_like(freqs)
        filt[1:] = 1.0 / torch.sqrt(freqs[1:])
        spectrum = spectrum * filt
        pink = torch.fft.irfft(spectrum, n=T, dim=-1)
        # normalise to unit std so downstream filters see a consistent noise level
        pink = pink / (pink.std(dim=-1, keepdim=True) + 1e-8)
        return pink.to(combtooth.dtype)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        '''
        # combtooth exciter signal
        # combtooth = self.additive_combtooth(f0_frames)
        combtooth = self.fast_source_gen(f0_frames)
        combtooth_frames = combtooth.unfold(1, self.block_size, self.block_size)
        
        # noise exciter signal
        # noise = self.gen_pink_noise(combtooth)
        noise = torch.randn_like(combtooth)
        noise_frames = noise.unfold(1, self.block_size, self.block_size)
        
        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, combtooth_frames, noise_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift)
        
        signal = self.apply_filters(ctrls, combtooth, noise)

        return signal, hidden

    # Kept out of the compiled graph: inductor fails to lower the complex scalar
    # multiply below when it builds the backward pass.
    @torch.compiler.disable
    def apply_filters(self, ctrls, combtooth, noise):
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude'] + 1.j * np.pi * ctrls['noise_phase']) / 128.0
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = 'reflect'
        else:
            pad_mode = 'constant'
        # harmonic part filter
        combtooth_stft = torch.stft(
                            combtooth,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True,
                            pad_mode = pad_mode)
        
        # noise part filter
        noise_stft = torch.stft(
                            noise,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True,
                            pad_mode = pad_mode)
        
        # apply the filters 
        signal_stft = combtooth_stft * src_filter.permute(0, 2, 1) + noise_stft * noise_filter.permute(0, 2, 1)
        
        # take the istft to resynthesize audio.
        signal = torch.istft(
                        signal_stft,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)

        return signal