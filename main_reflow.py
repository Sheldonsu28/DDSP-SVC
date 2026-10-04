import io
import os
import pickle
# from funasr.auto.auto_model import AutoModel
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
# import pyworld as pw
# import parselmouth
from torch.nn import functional as F
import hashlib
from ast import literal_eval
# from feature_retrival.utils import extract_energy
# from reflow.lynxnet2 import LYNXNet2
# from reflow.reflow import RectifiedFlow
from slicer import Slicer
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample
from reflow.vocoder import load_flow_style_model, load_model_vocoder, load_post_processor_model, load_style_model, load_vc_style_model, load_vc_style_reflow_model
from tqdm import tqdm
import torchaudio.transforms as T

from stylizer.range_compression import range_compression
from stylizer.span_masking import mask_keep_a_mask_b, mask_one_keep_n_safe
from stylizer.util import gaussian_blur_1d
from test10 import DiagonalLagrangian, acceleration_metrics
from utils import map_normal_diagonal
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
import faiss
import sys
sys.modules['faiss.swigfaiss_avx2'] = faiss.swigfaiss
def generate_pink_noise_mel(unit):
    """
    Generates pink noise in the Mel-frequency space.
    
    Args:
        n_mels (int): Number of Mel frequency bins.
        n_frames (int): Number of temporal frames.
        device (str): Device to run the tensor operations on ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: A tensor of shape (n_mels, n_frames) representing pink noise.
    """
    device = unit.device
    B, T, D = unit.shape
    n_mels = D
    n_frames = T
    # 1. Generate White Noise in Mel Space (Standard Normal Distribution)
    white_noise = torch.randn(n_mels, n_frames, device=device)
    
    # 2. Compute the 1/f spectral slope across the Mel bins
    # We add 1e-8 to avoid division by zero for the 0th index 
    mel_indices = torch.arange(1, n_mels + 1, device=device, dtype=torch.float32)
    slope = 1.0 / torch.sqrt(mel_indices)
    
    # 3. Apply the 1/f roll-off shape to the white noise
    # We reshape slope to (n_mels, 1) to allow broadcasting across time frames
    pink_noise = white_noise * slope.unsqueeze(1)
    
    # 4. Normalize to range [-1, 1] for stable audio operations
    pink_noise = pink_noise / (pink_noise.abs().max() + 1e-8)
    
    return pink_noise

def check_args(ddsp_args, diff_args):
    if ddsp_args.data.sampling_rate != diff_args.data.sampling_rate:
        print("Unmatch data.sampling_rate!")
        return False
    if ddsp_args.data.block_size != diff_args.data.block_size:
        print("Unmatch data.block_size!")
        return False
    if ddsp_args.data.encoder != diff_args.data.encoder:
        print("Unmatch data.encoder!")
        return False
    return True
    
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_ckpt",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    parser.add_argument(
        "-id",
        "--spk_id",
        type=str,
        required=False,
        default=1,
        help="speaker id (for multi-speaker model) | default: 1",
    )
    parser.add_argument(
        "-mix",
        "--spk_mix_dict",
        type=str,
        required=False,
        default="None",
        help="mix-speaker dictionary (for multi-speaker model) | default: None",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones) | default: 0",
    )
    parser.add_argument(
        "-f",
        "--formant_shift_key",
        type=str,
        required=False,
        default=0,
        help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
    )
    parser.add_argument(
        "-v",
        "--vocal_register_shift_key",
        type=str,
        required=False,
        default=0,
        help="vocal register changed (number of semitones) , only for pc-type vocoder| default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='rmvpe',
        help="pitch extrator type: parselmouth, dio, harvest, crepe, fcpe, rmvpe (default)",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1100,
        help="max f0 (Hz) | default: 1100",
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=str,
        required=False,
        default=-60,
        help="response threshold (dB) | default: -60",
    )
    parser.add_argument(
        "-step",
        "--infer_step",
        type=str,
        required=False,
        default='auto',
        help="sample steps | default: auto",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='auto',
        help="euler or rk4 | default: auto",
    )
    parser.add_argument(
        "-ts",
        "--t_start",
        type=str,
        required=False,
        default='auto',
        help="t_start | default: auto",
    )
    
    parser.add_argument(
        "-fr",
        "--f_retrieve",
        type=str,
        required=False,
        default=0.0,
        help="f_retrieve | default: 0",
    )
    return parser.parse_args(args=args, namespace=namespace)

def style_mix(feature, stylied_feature, ratio, dev):
    # simlarity = 1 - column_wise_cosine_similarity(feature, stylied_feature)
    # ratio = simlarity * ratio
    # print(stylied_feature.shape)
    c =  (1 - ratio) * feature + ratio * stylied_feature
    return c


def column_wise_cosine_similarity(a, b):
    result = np.empty((a.shape[0]))
    for i in range(a.shape[0]):
        result[i] = np.clip(np.dot(a[i, :], b[i, :]) / (np.linalg.norm(a[i, :]) * np.linalg.norm(b[i, :])), 0, 1)
    return result.reshape((a.shape[0], 1))
    
def split(audio, sample_rate, hop_size, db_thresh = -40, min_len = 1700):
    slicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                        start_frame, 
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result

def compile_model(model,fullgraph=False, mode="default"):
    model = torch.compile(model, fullgraph=fullgraph, mode=mode,dynamic=True)
    return model


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    #device = 'cpu' 
    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load reflow model
    model, vocoder, args = load_model_vocoder(cmd.model_ckpt, device=device)
    print('Compiling model')
    # model = compile_model(model, fullgraph=False,  mode="reduce-overhead")
    print('Compile complete')
    # raise
    
    # load input
    audio, sample_rate = librosa.load(cmd.input, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
    win_size = args.data.volume_smooth_size * sample_rate / args.data.sampling_rate

    # get MD5 hash from wav file
    md5_hash = ""
    with open(cmd.input, 'rb') as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
        print("MD5: " + md5_hash)
    
    cache_dir_path = os.path.join(os.path.dirname(__file__), "cache")
    cache_file_path = os.path.join(cache_dir_path, f"{cmd.pitch_extractor}_{hop_size}_{cmd.f0_min}_{cmd.f0_max}_{md5_hash}.npy")
    
    is_cache_available = os.path.exists(cache_file_path)
    if is_cache_available:
        # f0 cache load
        print('Loading pitch curves for input audio from cache directory...')
        f0 = np.load(cache_file_path, allow_pickle=False)
    else:
        # extract f0
        print('Pitch extractor type: ' + cmd.pitch_extractor)
        pitch_extractor = F0_Extractor(
                            cmd.pitch_extractor, 
                            sample_rate, 
                            hop_size, 
                            float(cmd.f0_min), 
                            float(cmd.f0_max))
        print('Extracting the pitch curve of the input audio...')
        f0 = pitch_extractor.extract(audio, uv_interp = True, device = device)
        
        # f0 cache save
        os.makedirs(cache_dir_path, exist_ok=True)
        np.save(cache_file_path, f0, allow_pickle=False)
    
    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
    
    # key change
    f0 = f0 * 2 ** (float(cmd.key) / 12)
    
    # formant change
    formant_shift_key = torch.from_numpy(np.array([[float(cmd.formant_shift_key)]])).float().to(device)
    
    # vocal register change
    if vocoder.vocoder.h.pc_aug:
        vocal_register_factor = 2 ** (float(cmd.vocal_register_shift_key) / 12)
    else:
        print('Vocal register shift is not supported for current vocoder!')
        vocal_register_factor = 1
    
    # extract volume 
    print('Extracting the volume envelope of the input audio...')
    volume_extractor = Volume_Extractor(hop_size, win_size)
    volume = volume_extractor.extract(audio)
    mask = (volume > 10 ** (float(cmd.threshold) / 20)).astype('float')
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    mask = upsample(mask, args.data.block_size).squeeze(-1)
    volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)
    
    # load units encoder
    if args.data.encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size,
                        cnhubertsoft_gate=cnhubertsoft_gate,
                        device = device)
    
    use_style = True
    use_style_reflow = False
    cycle_inference  = True
    whisper_mix = True
    use_post_processor = False
    z_masking = False
    mask_num = 4
    style_reflow_steps = 50
    style_reflow_start = 0.0
    post_step = 50
    post_start = 0.7
    units_encoder2 = None
    units_encoder3 = None
    style_reflow = None
    if use_style:
        units_encoder2 = Units_Encoder(
                            'whisper-large-pgg-tta2x' if args.data.encoder_hop_size == 160 else 'whisper-large-pgg', 
                            args.data.encoder_ckpt, 
                            args.data.encoder_sample_rate, 
                            args.data.encoder_hop_size,
                            cnhubertsoft_gate=cnhubertsoft_gate,
                            device = device)
        
        units_encoder3 = Units_Encoder(
                            'hubertsofttta2x' if args.data.encoder_hop_size == 160 else 'hubertsoftOri',
                            args.data.encoder_ckpt, 
                            args.data.encoder_sample_rate, 
                            args.data.encoder_hop_size,
                            cnhubertsoft_gate=cnhubertsoft_gate,
                            device = device)
        
        emo_encoder = Units_Encoder(
                            'emotionvec',
                            args.data.encoder_ckpt, 
                            args.data.encoder_sample_rate, 
                            args.data.encoder_hop_size,
                            cnhubertsoft_gate=cnhubertsoft_gate,
                            device = device)
        model_id = "iic/emotion2vec_plus_large"
    #     emo_model = AutoModel(
    #     model=model_id,
    #     hub="huggingface",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    # )
                            
    # speaker id or mix-speaker dictionary
    spk_mix_dict = literal_eval(cmd.spk_mix_dict)
    spk_id = torch.LongTensor(np.array([[int(cmd.spk_id)]])).to(device)
    if spk_mix_dict is not None:
        print('Mix-speaker mode')
    else:
        print('Speaker ID: '+ str(int(cmd.spk_id)))
    
    # sampling method    
    if cmd.method == 'auto':
        method = args.infer.method
    else:
        method = cmd.method
        
    # infer step
    if cmd.infer_step == 'auto':
        infer_step = args.infer.infer_step
    else:
        infer_step = int(cmd.infer_step)
    
    # t_start
    if cmd.t_start == 'auto':
        if args.model.t_start is not None:
            t_start = float(args.model.t_start)
        else:
            t_start = 0.0
    else:
        t_start = float(cmd.t_start)
        if args.model.t_start is not None and t_start < args.model.t_start:
            t_start = args.model.t_start
            
    if infer_step > 0:
        print('Sampling method: '+ method)
        print('infer step: '+ str(infer_step))
        print('t_start: '+ str(t_start))
    elif infer_step < 0:
        print('infer step cannot be negative!')
        exit(0)

   
    if use_style:
        style_model = load_vc_style_model('./exp/reflow-style-elysia_new/model_8000.pt', device=device, ver=0)
        # emb_path = 'data/hubert_mean_ellie.npy'
        # emb_path = 'data/speaker_cyrene.npy'
        # emb_path = 'data/speaker_curruption_spk.npy'
        # emb_path = 'data/speaker_cipher.npy'
        emb_path = 'data/speaker_elysia_new.npy'
        if os.path.exists(emb_path):
            hubert_mean = torch.from_numpy(np.load(emb_path)).to(device)
            # hubert_mean2 = torch.from_numpy(np.load(emb_path2)).to(device)
            # hubert_mean = style_mix(hubert_mean, hubert_mean2, 0.6, device)
            # print(hubert_mean.shape)
            
        # emb2_path = 'data/speaker_elysia_whisper.npy'
        # if os.path.exists(emb2_path):
        #     whisper_mean = torch.from_numpy(np.load(emb2_path)).to(device)
        # style_reflow = torch.compile(style_reflow, fullgraph=False,  mode="max-autotune")
        
   
    fr_weight = float(cmd.f_retrieve)
    best_high = 349.23  #  466.16 for elysia,  349.23 for cyrene
    best_low = 207.65 # 261.63 for elysia, 207.65 for cyrene
    # spk = 'currupt'
    # spk = 'cyrene'
    spk = 'elysia'
    cluster_model_path = f'./exp/feature_index/{spk}/feature_and_index.pkl'
    hubert_cluster_model_path = f'./exp/feature_index/{spk}/feature_and_index_hubert.pkl'
    if whisper_mix:
        whisper_cluster_model_path = f'./exp/feature_index/{spk}/feature_and_index_whisper.pkl'
    # if args.data.encoder == 'contentvec768l12tta2x':
    #     cluster_model_path = './exp/feature_index/{spk}/feature_and_index.pkl'
    if fr_weight > 0.0:
        with open(cluster_model_path,"rb") as f:
            cluster_model = pickle.load(f)
            now_spk_id = 1
            feature_index = cluster_model[now_spk_id]
            big_npy = feature_index.reconstruct_n(0, feature_index.ntotal)
            print(f'Index loaded: {cluster_model_path}')
            
        with open(hubert_cluster_model_path,"rb") as f:
            hubert_cluster_model = pickle.load(f)
            now_spk_id = 1
            hubert_feature_index = hubert_cluster_model[now_spk_id]
            hubert_big_npy = hubert_feature_index.reconstruct_n(0, hubert_feature_index.ntotal)
            print(f'Index loaded: {hubert_cluster_model_path}')
        if whisper_mix:
            with open(whisper_cluster_model_path,"rb") as f:
                whisper_cluster_model = pickle.load(f)
                now_spk_id = 1
                whisper_feature_index = whisper_cluster_model[now_spk_id]
                whisper_big_npy = whisper_feature_index.reconstruct_n(0, whisper_feature_index.ntotal)
                print(f'Index loaded: {whisper_cluster_model_path}')
            
    else:
        cluster_model = None
        big_npy = None
        now_spk_id = -1
    
    # forward and save the output
    result = np.zeros(0)
    current_length = 0
    # segments = split(audio, sample_rate, hop_size, db_thresh=cmd.threhold)
    segments:tuple[int, list[np.ndarray]] = split(audio, sample_rate, hop_size, db_thresh=-40)
    
    # sum_mean = torch.from_numpy(np.load('exp/all_mean.npy')).to(args.device)
    # unit_mean = torch.from_numpy(np.load('exp/unit_mean_brethy.npy')).to(args.device)
    print('Cut the input audio into ' + str(len(segments)) + ' slices')
   
    
    with torch.inference_mode():
    # if True:
        # mels = []
        for segment in tqdm(segments):
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
            seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
            # seg_units +=  torch.randn_like(seg_units) * 2
            seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
            vol_mask = seg_volume == 0
            # seg_units -= torch.mean(seg_units, dim=1)
            # print(segment[1].shape)
            seg16k = librosa.resample(segment[1], orig_sr=sample_rate, target_sr=16000)
            # print('input', seg_input.max(), seg_input.min())
            # print('vol', seg_volume.max(), seg_volume.min())
            # print('f0', seg_f0.max(), seg_f0.min())
            f0_adj_hz = seg_f0
            q_ori = seg_units.detach().float()
            if seg_volume.max() != 0 or seg_volume.min() != 0:
                    
                z_fwd = None
                if use_style:
                    auto_formants = 0
                    seg_units1 = units_encoder2.encode(seg_input, sample_rate, hop_size)
                    seg_units2 = units_encoder3.encode(seg_input, sample_rate, hop_size)
                    # seg_units1 += torch.randn_like(seg_units1)
                    # seg_units2 += torch.randn_like(seg_units2) * 2
                    byte_io = io.BytesIO()
                    sf.write(byte_io, seg16k, 16000, format='WAV')
                    # print(seg_units1.shape)
                    std_q, mu_q =torch.std_mean(seg_units1, dim=1)
                    # print(std_q.shape)
                    # emo_units = emo_encoder.encode_emo(byte_io.getvalue(), emo_model, seg_units1.shape[1]).cuda().float()
                    if fr_weight > 0.0 and not cycle_inference:
                        # seg_units_norm = torch.norm(seg_units, dim=2, keepdim=True)
                        # seg_units = torch.nn.functional.normalize(seg_units, dim=2, eps=1e-12)
                        _, idx = feature_index.search(seg_units.squeeze(0).cpu(), k=6)
                        npy = torch.from_numpy(big_npy[idx]).transpose(0, 1).mean(dim=0).to(device)
                        # seg_units = style_mix(seg_units, npy, fr_weight, device)
                        src_std_seg_unit, src_mean_seg_unit = torch.std_mean(seg_units, dim=1)
                        tgt_std_seg_unit, tgt_mean_seg_unit = torch.std_mean(npy, dim=0)
                        seg_units = map_normal_diagonal(seg_units, src_mean_seg_unit, src_std_seg_unit, tgt_mean_seg_unit, tgt_std_seg_unit)
                        
                        
                        # seg_units2_norm = torch.norm(seg_units2, dim=2, keepdim=True)
                        # seg_units2 = torch.nn.functional.normalize(seg_units2, dim=2, eps=1e-12)
                        _, idx = hubert_feature_index.search(seg_units2.squeeze(0).cpu(), k=6)
                        hubert_npy = torch.from_numpy(hubert_big_npy[idx]).transpose(0, 1).mean(dim=0).to(device)
                        # seg_units2 = style_mix(seg_units2, hubert_npy, fr_weight, device)
                        
                        src_std_seg_unit2, src_mean_seg_unit2 = torch.std_mean(seg_units2, dim=1)
                        tgt_std_seg_unit2, tgt_mean_seg_unit2 = torch.std_mean(hubert_npy, dim=0)
                        seg_units2 = map_normal_diagonal(seg_units2, src_mean_seg_unit2, src_std_seg_unit2, tgt_mean_seg_unit2, tgt_std_seg_unit2)
                        
                        if whisper_mix:
                            # seg_units1_norm = torch.norm(seg_units1, dim=2, keepdim=True)
                            # seg_units1 = torch.nn.functional.normalize(seg_units1, dim=2, eps=1e-12)
                            _, idx = whisper_feature_index.search(seg_units1.squeeze(0).cpu(), k=4)
                            whisper_npy = torch.from_numpy(whisper_big_npy[idx]).transpose(0, 1).mean(dim=0).to(device)
                            # seg_units1 = style_mix(seg_units1, whisper_npy, fr_weight, device) * seg_units1_norm
                            src_std_seg_unit1, src_mean_seg_unit1 = torch.std_mean(seg_units1, dim=1)
                            tgt_std_seg_unit1, tgt_mean_seg_unit1 = torch.std_mean(whisper_npy, dim=0)
                            seg_units1 = map_normal_diagonal(seg_units1, src_mean_seg_unit1, src_std_seg_unit1, tgt_mean_seg_unit1, tgt_std_seg_unit1)
                            # mean_index = whisper_npy.mean(dim=0)
                            # mean_seg = seg_units1.mean(dim=1)
                            # seg_units1 = seg_units1 - mean_seg + mean_index

                    
                    # energy = torch.log(extract_energy(seg_input.to('cpu')) + 1e-5).unsqueeze(0).to(device).transpose(1, 2)
                    # print(energy.shape)
                    # if use_style_reflow:
                    #     units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  auto_formants, z_ps, _ = style_model(seg_units, seg_units1, seg_units2, hubert_mean, None, seg_f0, None, infer=True, noise_fac=1e-4, alpha=1.0)
                    #     # units, z_f, z_r, z_pr, mu_pr, logvar_pr, z_ps, mu_ps, logvar_ps, logdet_f, logdet_r, formant, spk_pred, reflow_loss = style_model(seg_units, seg_units1, seg_units2, hubert_mean,  None, seg_f0, None, infer=True)
                    #     # units, formant, vq_loss, perplexity, spk_pred = style_model(seg_units, seg_units1,seg_units2,hubert_mean,  None, seg_f0, None, infer=True, noise_fac=0.0)
                    # else:
                    # print(seg_units.shape, seg_units1.shape, seg_units2.shape, hubert_mean.shape) 
                    # print(hubert_mean.shape)
                    units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  auto_formants, z_ps, _s = style_model(seg_units, seg_units1, seg_units2, hubert_mean, None, seg_f0, seg_volume, infer=True, noise_fac=0.0, alpha=1.0)
                        # units, z_f, z_r, z_pr, mu_pr, logvar_pr, z_ps, mu_ps, logvar_ps, logdet_f, logdet_r, formant, spk_pred, reflow_loss = style_model(seg_units, seg_units1, seg_units2, hubert_mean,  None, seg_f0, None, infer=True)
                        # units, formant, vq_loss, perplexity, spk_pred = style_model(seg_units, seg_units1,seg_units2,hubert_mean,  None, seg_f0, None, infer=True, noise_fac=0.0)
                    # units, _, _, _, _, _, _, _, _, _, _, _  = style_model(seg_units, seg_units1, seg_units2, hubert_mean, None, seg_f0, seg_volume, t_start=0.0, infer=True, infer_step=style_reflow_steps)
                    # units, _, _, _, _, _, _, _, _, _, _, _, _, _  = style_model(seg_units, seg_units1, seg_units2, spk_id, None, seg_f0, seg_volume, t_start=0.0, infer=True, infer_step=style_reflow_steps)
                    # units, z_f, z_r, z_pr, mu_pr, logvar_pr, z_ps, mu_ps, logvar_ps, logdet_f, logdet_r, _, spk_pred, _ = style_model(seg_units, seg_units1, seg_units2,spk_id, None, seg_f0, None, infer=True, noise_fac=1e-4, infer_step=style_reflow_steps)
                    # auto_formants = 0
                else:
                    units = seg_units
                    auto_formants = 0
                    
                    if fr_weight > 0.0:
                        _, idx = feature_index.search(units.squeeze(0).cpu(), k=4)
                        npy = torch.from_numpy(big_npy[idx]).transpose(0, 1).mean(dim=0).to(device)
                        # seg_units = style_mix(seg_units, npy, fr_weight, device)
                        src_std_seg_unit, src_mean_seg_unit = torch.std_mean(units, dim=1)
                        tgt_std_seg_unit, tgt_mean_seg_unit = torch.std_mean(npy, dim=0)
                        units = map_normal_diagonal(seg_units, src_mean_seg_unit, src_std_seg_unit, tgt_mean_seg_unit, tgt_std_seg_unit)
                    
                # if fr_weight > 0.0:
                #     _, idx = feature_index.search(seg_units.squeeze(0).cpu(), k=8)
                #     npy = torch.from_numpy(big_npy[idx]).transpose(0, 1).mean(dim=0).to(device)
                #     units = style_mix(units, npy, fr_weight, device)
                
                if use_style and use_style_reflow:
                    units = style_reflow(z_fwd, gt_spec=seg_units, infer=True, infer_step=style_reflow_steps, method='euler', t_start=style_reflow_start, use_tqdm=False)
                    
                if cycle_inference:
                    if fr_weight > 0.0:
                        

                        _, idx = feature_index.search(units.squeeze(0).cpu(), k=4)
                        npy = torch.from_numpy(big_npy[idx]).transpose(0, 1).mean(dim=0).to(device)

                        src_std_seg_unit, src_mean_seg_unit = torch.std_mean(units, dim=1)
                        tgt_std_seg_unit, tgt_mean_seg_unit = torch.std_mean(npy, dim=0)
                        # tgt_mean_seg_unit = torch.mean(torch.from_numpy(big_npy).transpose(0, 1), dim=1).to(device)
                        units = map_normal_diagonal(units, src_mean_seg_unit, src_std_seg_unit, tgt_mean_seg_unit, tgt_std_seg_unit)
                        

                        _, idx = hubert_feature_index.search(seg_units2.squeeze(0).cpu(), k=4)
                        hubert_npy = torch.from_numpy(hubert_big_npy[idx]).transpose(0, 1).mean(dim=0).to(device)
                        
                        src_std_seg_unit2, src_mean_seg_unit2 = torch.std_mean(seg_units2, dim=1)
                        tgt_std_seg_unit2, tgt_mean_seg_unit2 = torch.std_mean(hubert_npy, dim=0)
                        # tgt_mean_seg_unit2 = torch.mean(torch.from_numpy(hubert_big_npy).transpose(0, 1), dim=1).to(device)
                        seg_units2 = map_normal_diagonal(seg_units2, src_mean_seg_unit2, src_std_seg_unit2, tgt_mean_seg_unit2, tgt_std_seg_unit2)
                        
                        if whisper_mix:

                            _, idx = whisper_feature_index.search(seg_units1.squeeze(0).cpu(), k=4)
                            whisper_npy = torch.from_numpy(whisper_big_npy[idx]).transpose(0, 1).mean(dim=0).to(device)

                            
                            
                            src_std_seg_unit1, src_mean_seg_unit1 = torch.std_mean(seg_units1, dim=1)
                            tgt_std_seg_unit1, tgt_mean_seg_unit1 = torch.std_mean(whisper_npy, dim=0)
                            # tgt_mean_seg_unit1 = torch.mean(torch.from_numpy(whisper_big_npy).transpose(0, 1), dim=1).to(device)
                            seg_units1 = map_normal_diagonal(seg_units1, src_mean_seg_unit1, src_std_seg_unit1, tgt_mean_seg_unit1, tgt_std_seg_unit1)
                            
                            # print(whisper_npy.shape, seg_units1.shape)
                            # print(mean_index.shape, mean_seg.shape, seg_units1.shape , whisper_npy.shape)
                            # seg_units1 = seg_units1 - mean_seg + mean_index
                            
                    # units, formant, vq_loss, perplexity, spk_pred = style_model(seg_units, seg_units1,seg_units2,hubert_mean,  None, seg_f0, None, infer=True, noise_fac=0.0)
                    units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, logdet_fwd, auto_formants, z_ps, msk = style_model(units, seg_units1, seg_units2, hubert_mean, None, seg_f0, seg_volume, infer=True, noise_fac=0.0, alpha=1.0)
                    
                # if use_style_reflow:
                #     units = style_reflow(z_fwd, gt_spec=seg_units, infer=True, infer_step=style_reflow_steps, method='euler', t_start=style_reflow_start, use_tqdm=False)
                # diff = units - seg_units
                # units += diff * 0.3
                # f0_aug, _, _ = range_compression(seg_f0, best_low, best_high, vol_mask)
                f0_aug = seg_f0
                # units += torch.randn_like(units) * 0.5
                seg_mel = model(
                        units, 
                        f0_aug, 
                        seg_volume, 
                        spk_id = spk_id, 
                        spk_mix_dict = spk_mix_dict,
                        # aug_shift = formant_shift_key,
                        aug_shift = formant_shift_key + auto_formants,
                        vocoder=vocoder,
                        infer_step=infer_step, 
                        method=method,
                        t_start=t_start,
                        use_tqdm=False)
                # q = units.detach().float()

                # metrics = acceleration_metrics(mode, q, dt=1.0)
                # metrics_ori = acceleration_metrics(mode, q_ori, dt=1.0)

                # print(f"Baseline/styled:         {metrics_ori['baseline_mse'].item():.6f}/ {metrics['baseline_mse'].item():.6f}")
                # print(f"Acceleration MSE/styled: {metrics_ori['acceleration_mse'].item():.6f}/ {metrics['acceleration_mse'].item():.6f}")
                # print(f"Normalized error/styled: {metrics_ori['normalized_error'].item():.4f}/ {metrics['normalized_error'].item():.4f}")
                # print(f"Improvement/styled:      {metrics_ori['improvement_percent'].item():.1f}%/ {metrics['improvement_percent'].item():.1f}%")
                
                # if use_post_processor:
                #     seg_mel = post_reflow(seg_mel, gt_spec=seg_mel, infer=True, infer_step=30, method='euler', t_start=0.7, use_tqdm=False)
                seg_output = vocoder.infer(seg_mel, f0_adj_hz)
            
            else:
                print('\nfind slience, skipping')
                seg_output = torch.zeros((1, 1,(start_frame + seg_units.size(1)) * args.data.block_size - start_frame * args.data.block_size), device=seg_volume.device)
            # print(start_frame * args.data.block_size, (start_frame + seg_units.size(1)) * args.data.block_size, seg_units.size(1), seg_output.shape, mask.shape)
            seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]
            seg_output = seg_output.squeeze().cpu().numpy()
            
            silent_length = round(start_frame * args.data.block_size) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)
        sf.write(cmd.output, result, args.data.sampling_rate)
        # mel = torch.concatenate(mels, 1).transpose(1, 2).cpu().numpy()
        # print(mel.shape)
        # np.save(cmd.output+'.npy', mel)
