import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bigvgan.bigv import load_BigV_model
from bigvgan.utils import MAX_WAV_VALUE
from ddsp.loss import MRSTFTLoss
from nsf_hifigan.nvSTFT import STFT
from nsf_hifigan.models import load_model,load_config
from torchaudio.transforms import Resample

from stylizer.stylizer import GlowVcStylizerWN5, GlowVcStylizerWN5Mod3, GlowVcStylizerWN5Mod4, GlowVcStylizerWN5Mod6, GlowVcStylizerWN5Mod7, GlowVcStylizerWN5Mod8, GlowVcStylizerWN5Mod9
from stylizer.util import create_post_processor, create_reflow
from .reflow import RectifiedFlow, focus_loss
from .lynxnet2 import LYNXNet2
from ddsp.vocoder import CombSubSuperFast

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def load_model_vocoder(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r", encoding='utf-8') as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model
    if args.model.type == 'RectifiedFlow':
        model = Unit2Wav(
                    args.data.sampling_rate,
                    args.data.block_size,
                    args.model.win_length,
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_norm,
                    args.model.use_attention,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_aux_layers,
                    args.model.n_aux_chans,
                    args.model.n_layers,
                    args.model.n_chans)
                   
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
        
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args

def load_style_model(model_path,device='cpu'):
 
    # load model                   
    print(' [Loading] ' + model_path +f' {device}')
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    return model

def load_vc_style_model(model_path ,device='cpu', ver=0):
 
    # load model
    # model = SoftVcStylizer()
    model = GlowVcStylizerWN5() if ver == 0 else GlowVcStylizerWN5Mod6()
    if ver == 2:
        model = GlowVcStylizerWN5Mod3()
    if ver == 3:
        model =  GlowVcStylizerWN5Mod7()
    if ver == 4:
        model =  GlowVcStylizerWN5Mod8()
    if ver == 5:
        model =  GlowVcStylizerWN5Mod9()
    print(' [Loading] ' + model_path +f' {device}')
    ckpt = torch.load(model_path, map_location=torch.device(device))
    print(ckpt.keys())
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    return model

def load_vc_style_reflow_model(model_path,device='cpu'):
    model = create_reflow(version=2)
    # model = FlowEnhancer()
    print(' [Loading] ' + model_path +f' {device}')
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    return model

def load_post_processor_model(model_path,device='cpu'):
    model = create_post_processor()
    # model = FlowStyle2()
    print(' [Loading] ' + model_path +f' {device}')
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    return model

def load_flow_style_model(model_path,device='cpu'):
    model = GlowVcStylizerWN5()
    print(' [Loading] ' + model_path +f' {device}')
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    return model


class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device = device)
        elif vocoder_type == 'nsf-hifigan-log10':
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device = device)
        elif vocoder_type == 'big-vgan':
            self.vocoder = BigvGan(vocoder_ckpt, device = device)
        elif vocoder_type == 'high_res':
            self.vocoder = HghResMel(vocoder_ckpt, device=device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
            
        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()

    def extract(self, audio, sample_rate=0, keyshift=0):
                
        # resample
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)    
        
        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel
   
    def infer(self, mel, f0):
        f0 = f0[:,:mel.size(1),0] # B, n_frames
        audio = self.vocoder(mel, f0)
        return audio
    
    
class BigvGan(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        # print(self.h)
        self.stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                40, 
                16000)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return self.h.num_mels
    
    @torch.compiler.disable
    def extract(self, audio, keyshift=0):       
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel
    
    @torch.compiler.disable
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load BigV: ', self.model_path)
            self.model, self.h = load_BigV_model(self.model_path, device=self.device)
        with torch.no_grad():
            print(mel.shape)
            c = torch.sqrt(mel.pow(2) + 1e-9)
            c = torch.log(torch.clamp(c.transpose(1, 2), min=1e-5) * 1)
            # c = mel.transpose(1, 2)
            audio = self.model(c)
            return audio
        
        
class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
        self.stft_high_res = STFT(
                self.h.sampling_rate, 
                512, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return self.h.num_mels
    def extract(self, audio, keyshift=0, high_res=False):
        if high_res: 
            mel = self.stft_high_res.get_mel(audio/32768.0, keyshift=keyshift, use_complex=False).transpose(1, 2) # B, n_frames, bins
        else:
            mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel

    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio

def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            try:
                m.bias.data.fill_(0)
            except Exception as e:
                pass
            
class HghResMel(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)

        self.stft_high_res = STFT(
                self.h.sampling_rate, 
                512, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return 512
    
    def extract(self, audio, keyshift=0, high_res=False):
        # audio is expected in the [-1, 1] range (same as the regular mel path)
        mel = self.stft_high_res.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel

class NsfHifiGANLog10(NsfHifiGAN): 
    @torch.compiler.disable   
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class Unit2Wav(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False,
            out_dims=128,
            n_aux_layers=3,
            n_aux_chans=256,
            n_layers=6, 
            n_chans=512, 
            kernel_size=31,
            wn_normal=False):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.ddsp_model = CombSubSuperFast(
                            sampling_rate, 
                            block_size, 
                            win_length, 
                            n_unit, 
                            n_spk, 
                            n_aux_layers if n_aux_layers is not None else 3,
                            n_aux_chans if n_aux_chans is not None else 256,
                            use_norm,
                            use_attention, 
                            use_pitch_aug)
        self.reflow_model = RectifiedFlow(LYNXNet2(in_dims=out_dims, dim_cond=out_dims, n_layers=n_layers, n_chans=n_chans, kernel_size=kernel_size, wn_normal=wn_normal), out_dims=out_dims)
    # def band_freq_loss(self, mel_out, mel_target, start=0,end=56):
    #     return F.l1_loss(mel_out[:, :, start:end], mel_target[:, :, start:end])
    
    def re_init(self):
        self.reflow_model.apply(weights_init_uniform_rule)
        
    @torch.compiler.disable
    def extract_mel(self, ddsp_wav, vocoder):
        return vocoder.extract(ddsp_wav)
    
    @torch.compiler.disable
    def infer_audio(self, mel, f0, vocoder):
        return vocoder.infer(mel, f0)

    @torch.compiler.disable
    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, 
                silence_front=0, use_tqdm=False):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        ddsp_wav, hidden = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, infer=infer)
        start_frame = int(silence_front * self.sampling_rate / self.block_size)
        if vocoder is not None:
            ddsp_mel = self.extract_mel(ddsp_wav[:, start_frame * self.block_size:], vocoder)
        else:
            ddsp_mel = None
            
        if not infer:
            ddsp_loss_map = F.mse_loss(ddsp_mel, gt_spec, reduction='none')
            ddsp_loss = ddsp_loss_map.mean()
            # stft_loss = self.mrstft(ddsp_wav, gt_audio)
            # ddsp_loss += stft_loss
            # ddsp_band_loss = 0. * (self.band_freq_loss(ddsp_mel, gt_spec) + self.band_freq_loss(ddsp_mel, gt_spec, 110, 124))
            ddsp_band_loss = 0. * ddsp_loss
            if t_start < 1.0:
                reflow_loss = self.reflow_model(ddsp_mel, gt_spec=gt_spec, t_start=t_start, infer=False)
            else:
                reflow_loss = torch.tensor(0)
            return ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if t_start < 1.0:
                mel = self.reflow_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return self.infer_audio(mel, f0[:, -mel.shape[1]:], vocoder)
            else:
                return mel