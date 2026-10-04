import os
import time
import numpy as np
import torch
import librosa
from ddsp.loss import MRSTFTLoss
from logger.saver import Saver
from logger import utils
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.profiler import profile, ProfilerActivity
import gc
import torch.nn as nn
import random
import torchaudio
from nsf_hifigan.nvSTFT import STFT
from reflow.reflow import focus_loss
from stylizer.drift import DriftingLoss
from stylizer.span_masking import mask_keep_a_mask_b, mask_one_keep_n_safe, mask_one_keep_n_safe_randoffset, span_masking
from stylizer.key_shifter import pattern_loss, residual_targets, volume_mask_from_rms
from stylizer.kl_loss import flow_kl_forward_mc, flow_kl_forward_mc_safe, flow_kl_inverse_mc, flow_kl_inverse_mc_safe, make_mask_BLD
from stylizer.loss import DDSPComboLoss, DDSPComboLossConfig
from stylizer.stft_loss import MultiResolutionSTFTLoss, harmonic_emphasis_mel_loss_logged, hnr_loss_mel, drift_loss
from stylizer.stylizer import GradientScaler
from stylizer.util import centroid_loss_from_mels, close_form_kl_loss, compute_flow_loss, gaussian_deviation_loss, kl_budget_loss, kl_div_loss, kl_loss2, kl_loss_between_gaussians, kl_loss_distributions, kl_loss_new, mod_kl_loss, mod_kl_loss1, mod_kl_loss1_stable, mod_kl_loss2, random_piecewise_time_warp, simple_kl_loss, vits_kl_loss, vits_kl_loss2, vits_kl_loss_with_free_bits
from torch.utils import checkpoint

from test10 import acceleration_metrics
def get_gradient_ratios(lossA, lossB, x_f, eps=1e-6):
    # print(lossA.shape)
    # print(lossB.shape)
    # print(x_f.shape)
    grad_lossA_xf = torch.autograd.grad(torch.sum(lossA), x_f, retain_graph=True)[0]
    grad_lossB_xf = torch.autograd.grad(torch.sum(lossB), x_f, retain_graph=True)[0]
    gamma = grad_lossA_xf / grad_lossB_xf

    return gamma

from torch.amp import autocast, GradScaler

def calculate_mel_snr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SNR
    snr = 10 * torch.log10(mean_square_reference / variance_error)
    return snr


def calculate_mel_si_snr(gt_mel, pred_mel):
    # 将测试图像按比例调整以最小化误差
    scale = torch.sum(gt_mel * pred_mel) / torch.sum(gt_mel ** 2)
    test_image_scaled = scale * gt_mel
    # 计算误差图像
    error_image = pred_mel - test_image_scaled 
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SI-SNR
    si_snr = 10 * torch.log10(mean_square_reference / variance_error)
    return si_snr

def band_mel_loss(mel_pred, mel_gt, start=16, end=54):  # For 4kHz if n_mels=128
    return F.mse_loss(mel_pred[:, start:end, :], mel_gt[:, start:end, :])

def spectral_contrast_loss(mel):
    # mel: (B, n_mels, T)
    diff = mel[:, 1:, :] - mel[:, :-1, :]  # vertical spectral contrast
    return -torch.mean(torch.abs(diff))

def mid_band_mel_loss(mel_pred, mel_gt, mel_frequencies, f_lo=700, f_hi=3000, weight=3.0):
    emphasis_mask = torch.ones_like(mel_gt)
    # mel_frequencies: (n_mels,) numpy array or torch.tensor
    for i, freq in enumerate(mel_frequencies):
        if f_lo <= freq <= f_hi:
            emphasis_mask[:, i, :] *= weight
    return (emphasis_mask * torch.abs(mel_pred - mel_gt)).mean()

def calculate_mel_psnr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算误差图像的均方误差
    mse = torch.mean(error_image ** 2)
    # 计算参考图像的最大可能功率
    max_power = torch.max(gt_mel) ** 2
    # 计算并返回PSNR
    psnr = 10 * torch.log10(max_power / mse)
    return psnr

def calculate_angle_si_snr(gt_tensor, pred_tensor, eps=1e-8):
    # gt_tensor, pred_tensor: [B, T, D]
    # 在维度 D 上把每一帧当作一个 D 维向量，计算尺度不变 SNR。
    # SI-SNR 只取决于两个向量之间的夹角 theta:
    #   SI-SNR = 10 * log10(cos^2(theta) / sin^2(theta))
    # 因此称为 "angle" SI-SNR。
    # reference = gt, estimate = pred
    # 把 pred 投影到 gt 上得到 target 分量，残差即 noise 分量。
    dot = torch.sum(pred_tensor * gt_tensor, dim=-1, keepdim=True)        # <pred, gt>
    gt_energy = torch.sum(gt_tensor ** 2, dim=-1, keepdim=True)           # <gt, gt>
    scale = dot / (gt_energy + eps)                                       # 尺度不变缩放系数
    target = scale * gt_tensor                                            # pred 在 gt 方向上的投影
    noise = pred_tensor - target                                         # 正交残差

    target_energy = torch.sum(target ** 2, dim=-1)                        # [B, T]
    noise_energy = torch.sum(noise ** 2, dim=-1)                          # [B, T]
    si_snr = 10 * torch.log10(target_energy / (noise_energy + eps) + eps)  # 逐帧 SI-SNR
    return si_snr.mean()


def test(args, model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, .5, 1), (2048, 240, 1200,  .5, 1), (4096, 480, 2400, .5, 1), (512, 50, 240, .5, 1)]).to(args.device)

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    test_hnr_loss = 0.
    test_harm_loss = 0.
    

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0
    ddsp_mel_val_sisnr_all = 0.

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -6
    spec_max = 6
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            units = data['units']
            mel= model(
                    units, 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel= model(
                data['units'], 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            
            test_harm_loss += harmonic_emphasis_mel_loss_logged(ddsp_mel, data['mel'], data['f0'])
            test_hnr_loss += hnr_loss_mel(ddsp_mel,  data['mel'], data['f0'])
            
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            ddsp_mel_norm =  torch.clip(ddsp_mel, spec_min, spec_max)
            ddsp_pre_mel_norm = ddsp_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            ddsp_mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, ddsp_pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_harm_loss /= num_batches
    test_hnr_loss /= num_batches
    test_ddsp_band_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num
    ddsp_mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    print(' DDSP Mel Val SI-SNR', ddsp_mel_val_sisnr_all)
    saver.log_value({
        'validation/ddsp_mel_val_sisnr': ddsp_mel_val_sisnr_all
    })
    saver.log_value({
        'validation/hnr_loss': test_hnr_loss
    })
    saver.log_value({
        'validation/harm_loss': test_harm_loss
    })
    return test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_ddsp_msrl_loss

def test_lagrangian(args, model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, .5, 1), (2048, 240, 1200,  .5, 1), (4096, 480, 2400, .5, 1), (512, 50, 240, .5, 1)]).to(args.device)

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    test_hnr_loss = 0.
    test_harm_loss = 0.
    test_base_acc = 0.
    test_acc_err= 0.
    

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0
    ddsp_mel_val_sisnr_all = 0.

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            units = data['units']
            loss = model.loss(units)
            
            q = units.float()
            mass = model.mass(q[:, 1:-1])     # [B, L-2, D]
            test_acc_err += (loss / mass).square().mean()
            # signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            # song_time = signal.shape[-1] / args.data.sampling_rate
            # rtf = run_time / song_time
            # print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            # rtf_all.append(rtf)
            
            # loss
            # loss = model.loss(units)
            test_ddsp_loss += loss.item()
            with torch.no_grad():
                q = units.float()  # [B, L, 768]
                a = q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]  # dt=1
                test_base_acc += a.square().mean()
                

            # print("Acceleration prediction MSE:", acceleration_error.item())
            # test_reflow_loss += loss.item()
            # test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            
            # test_harm_loss += harmonic_emphasis_mel_loss_logged(ddsp_mel, data['mel'], data['f0'])
            # test_hnr_loss += hnr_loss_mel(ddsp_mel,  data['mel'], data['f0'])
            
            # log mel
            # saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            # audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            # saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})
            q = q.detach().float()
            metrics = acceleration_metrics(model, q, dt=1.0)

            print(f"Baseline:         {metrics['baseline_mse'].item():.6f}")
            print(f"Acceleration MSE: {metrics['acceleration_mse'].item():.6f}")
            print(f"Normalized error: {metrics['normalized_error'].item():.4f}")
            print(f"Improvement:      {metrics['improvement_percent'].item():.1f}%")

            # WAV2MEL = STFT(
            #             sr=args.data.sampling_rate,
            #             n_mels=128,
            #             n_fft=2048,
            #             win_size=2048,
            #             hop_length=512,
            #             fmin=40,
            #             fmax=22050,
            #             clip_val=1e-5)
            # # audio = audio.unsqueeze(0)
            # pre_mel = WAV2MEL.get_mel(signal[0, ...])
            # pre_mel = pre_mel.transpose(-1, -2)
            # gt_mel = WAV2MEL.get_mel(audio[0, ...])
            # gt_mel = gt_mel.transpose(-1, -2)
            # # 如果形状不同,裁剪使得形状相同
            # if pre_mel.shape[1] != gt_mel.shape[1]:
            #     gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            # saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            # mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            # gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            # gt_mel_norm = gt_mel_norm / spec_range + spec_min
            # pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            # pre_mel_norm = pre_mel_norm / spec_range + spec_min
            # ddsp_mel_norm =  torch.clip(ddsp_mel, spec_min, spec_max)
            # ddsp_pre_mel_norm = ddsp_mel_norm / spec_range + spec_min
            # mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            # mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            # mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            # ddsp_mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, ddsp_pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_harm_loss /= num_batches
    test_hnr_loss /= num_batches
    test_ddsp_band_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    test_base_acc /= num_batches
    test_acc_err /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num
    ddsp_mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        'validation/test_base_acc': test_base_acc
    })
    
    saver.log_value({
        'validation/test_acc_err': test_acc_err
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    print(' DDSP Mel Val SI-SNR', ddsp_mel_val_sisnr_all)
    saver.log_value({
        'validation/ddsp_mel_val_sisnr': ddsp_mel_val_sisnr_all
    })
    saver.log_value({
        'validation/hnr_loss': test_hnr_loss
    })
    saver.log_value({
        'validation/harm_loss': test_harm_loss
    })
    return test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_ddsp_msrl_loss


def test_base(args, model, validation_model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()
    validation_model.eval()
    # mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, .5, 1), (2048, 240, 1200,  .5, 1), (4096, 480, 2400, .5, 1), (512, 50, 240, .5, 1)]).to(args.device)

    # losses
    test_angle_loss = 0.
    test_reconstruction_loss = 0.
    test_spk_loss = 0.
    test_angle_snr = 0.
    test_snr = 0.
    
    
    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    test_hnr_loss = 0.
    test_harm_loss = 0.
    

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0
    ddsp_mel_val_sisnr_all = 0.

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    spkc_criterion = nn.CosineEmbeddingLoss()
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            units = data['units']
            units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask = model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], None, None, None, infer=True)
            ed_time = time.time()
            
            
            mel= validation_model(
                    units, 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            
            
            ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel= validation_model(
                data['units'], 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
           
            test_angle_loss += (1.0 - F.cosine_similarity(data['units'], units)).mean().item()
            # test_spk_loss += 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.ones(spk_pred.size(0), device=args.device)).item()
            test_reconstruction_loss += F.l1_loss(data['units'], units).item()
            test_angle_snr += calculate_angle_si_snr(data['units'], units).item()
            test_snr += calculate_mel_si_snr(data['units'], units).item()
            
            
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})
            
            
            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            ddsp_mel_norm =  torch.clip(ddsp_mel, spec_min, spec_max)
            ddsp_pre_mel_norm = ddsp_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            ddsp_mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, ddsp_pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
           
            
    # report
    # test_spk_loss /= num_batches
    test_angle_loss /= num_batches
    test_reconstruction_loss /= num_batches
    test_angle_snr /= num_batches
    test_snr /= num_batches
    
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_harm_loss /= num_batches
    test_hnr_loss /= num_batches
    test_ddsp_band_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num
    ddsp_mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    # print(' [test_spk_loss] test_spk_loss:', test_spk_loss)
    print(' [test_angle_loss] test_angle_loss:', test_angle_loss)
    print(' [test_reconstruction_loss] test_reconstruction_loss:', test_reconstruction_loss)
   
    saver.log_value({
        'validation/test_spk_loss': test_spk_loss
    })
    saver.log_value({
        'validation/test_angle_loss': test_angle_loss
    })
    saver.log_value({
        'validation/test_reconstruction_loss': test_reconstruction_loss
    })
    
    saver.log_value({
        'validation/angle_si_snr': test_angle_snr
    })
    saver.log_value({
        'validation/si_snr': test_snr
    })
    
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    print(' DDSP Mel Val SI-SNR', ddsp_mel_val_sisnr_all)
    saver.log_value({
        'validation/ddsp_mel_val_sisnr': ddsp_mel_val_sisnr_all
    })
    saver.log_value({
        'validation/hnr_loss': test_hnr_loss
    })
    saver.log_value({
        'validation/harm_loss': test_harm_loss
    })

    return test_reconstruction_loss, test_angle_loss, test_spk_loss


def test_reflow_gan(args, model, model_d, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()
    model_d.eval()
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]).to(args.device)

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_ddsp_msrl_loss = 0.

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            units = data['units']
            mel = model(
                    units, 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model(
                data['units'], 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            test_ddsp_band_loss += ddsp_band_loss.item()
            test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches 
    test_ddsp_band_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    return test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_ddsp_msrl_loss

def test_gan(args, model, model_d, vocoder, loader_test, saver, dtype):
    print(' [*] testing...')
    model.eval()
    model_d.eval()
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]).to(args.device)
    transform = torchaudio.transforms.Resample(44100, 32000, dtype=dtype).to('cuda')
    

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    test_spk_loss = 0. 

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    print(num_batches)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            units = data['units']
            mel, _, _, _ = model(
                    units, 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            ddsp_loss, reflow_loss, _, ddsp_wav, _, stylized_feats , mu, logvar, spk_pred = model(
                data['units'], 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            ddsp_band_loss = kl_budget_loss(mu, logvar)
            test_rec_loss += nn.functional.mse_loss(stylized_feats, data['units']).item()
            test_spk_loss += 2 * spkc_criterion(data['spk_embd'], spk_pred, torch.Tensor(spk_pred.size(0)).to(args.device).fill_(1.0)).item()
            test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches 
    test_ddsp_band_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    return test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_ddsp_msrl_loss, test_rec_loss, test_spk_loss


def test_style(args, model, vocoder, loader_test, saver, style_model, sum_mean, unit_mean, unit_var):
    print(' [*] testing...')
    model.eval()
    style_model.eval()
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]).to(args.device)


    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    # test_gaussian_loss = 0.0

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            unit, _, _ = style_model(data['units'], data['units_w'], data['units_h'], sum_mean, unit_mean)
            mel = model(
                    unit,
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            unit, mu, logvar = style_model.infer(data['units'], data['units_w'], data['units_h'],  sum_mean, unit_mean)
            ddsp_loss, reflow_loss, _, ddsp_wav, ddsp_mel = model(
                unit, 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            # test_gaussian_loss += gaussian_deviation_loss(unit, unit_mean, unit_var).item()
            ddsp_band_loss = kl_budget_loss(mu, logvar)
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            test_ddsp_band_loss += ddsp_band_loss.item()
            test_rec_loss += nn.functional.mse_loss(unit, data['units']).item()
            test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    # test_gaussian_loss /=num_batches
    test_rec_loss /= num_batches
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    test_ddsp_band_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' [test_rec_loss] test_rec_loss:', test_rec_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    return test_ddsp_loss, test_reflow_loss,test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss


def test_style_gan(args, style_model, model, vocoder, loader_test, saver, dtype, sum_mean, unit_mean, unit_var):
    print(' [*] testing...')
    model.eval()
    style_model.eval()
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]).to(args.device)
    spkc_criterion = nn.CosineEmbeddingLoss()

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    # test_gaussian_loss = 0.0

    # mel mse val
    mel_val_L1_all = 0
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            unit, _, _ = style_model(data['units'], data['units_w'], data['units_h'], sum_mean, unit_mean)
            mel = model(
                    unit,
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            unit, mu, logvar = style_model.infer(data['units'], data['units_w'], data['units_h'],  sum_mean, unit_mean)
            ddsp_loss, reflow_loss, _, ddsp_wav, ddsp_mel = model(
                unit, 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            # test_gaussian_loss += gaussian_deviation_loss(unit, unit_mean, unit_var).item()
            ddsp_band_loss = kl_budget_loss(mu, logvar)
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            test_ddsp_band_loss += ddsp_band_loss.item()
            test_rec_loss += nn.functional.mse_loss(unit, data['units']).item()
            test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            mel_val_L1_all +=  torch.nn.functional.l1_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    # test_gaussian_loss /=num_batches
    test_rec_loss /= num_batches
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    test_ddsp_band_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_L1_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' [test_rec_loss] test_rec_loss:', test_rec_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    print(' Mel Val L1', mel_val_L1_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    saver.log_value({
        'validation/mel_val_L1': mel_val_L1_all
    })
    return test_ddsp_loss, test_reflow_loss,test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss


def test_style_vc_gan(args, style_model, model, vocoder, loader_test, saver, dtype, sum_mean, unit_mean, unit_var):
    print(' [*] testing...')
    model.eval()
    style_model.eval()
    # mrsl = MultiResolutionSTFTLoss('cuda',  [(512, 128, 512, 0.2, 0.05), (1024, 256, 1024, 0.25, 0.1), (2048, 512, 2048, 0.30, 0.18), (4096, 1024, 4096, 0.35, 0.12)]).to(args.device)
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])
    spkc_criterion = nn.CosineEmbeddingLoss()

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    test_spk_loss = 0. 
    test_formant = 0.
    test_reflow_style_loss = 0.
    test_smooth_loss_loss = 0.
    # test_mag_loss_reflow_loss = 0.
    # test_interval_loss_loss = 0.0

    # mel mse val
    mel_val_L1_all = 0
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    # mel_freqs_hz = torch.tensor(
    #     librosa.mel_frequencies(n_mels=128, fmin=40.0, fmax=16000.0),
    #     device=args.device
    # )
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            # units, formant, vq_loss, perplexity, spk_pred = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  data['mel'], data['f0'], data['volume'], infer=True)
            # units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  data['mel'], data['f0'], data['volume'], infer=True)
            units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant,  log_det_fwd, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  data['mel_high_res'], data['f0'], data['volume'],0 , infer=True)
            mel = model(
                    units,
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    aug_shift= formant,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            # unit, mu, logvar, spk_pred = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  data['f0'])
            ddsp_loss, reflow_loss, _, ddsp_wav, ddsp_mel = model(
                units, 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                aug_shift= formant,
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            # test_gaussian_loss += gaussian_deviation_loss(unit, unit_mean, unit_var).item()
            # norm = z_fwd.shape[1] * (z_fwd.shape[2] // 2)
            # loss_kl_f = kl_loss_new(z_f, logs_q, m_p, logs_p, logdet_f, spec_mask)
            # loss_kl_r = kl_loss_new(z_r, logs_p, m_q, logs_q, logdet_r, spec_mask)
            # ddsp_band_loss = loss_kl_f + loss_kl_r
            # ddsp_band_loss = 0.2* mod_kl_loss2(mu_ps, logvar_ps, mu_pr, logvar_pr, log_det_fwd/norm) + 0.2*mod_kl_loss2(mu_pr, logvar_pr, mu_ps, logvar_ps, log_det_bkw/norm)
            # test_spk_loss += 2 * spkc_criterion(speaker_feat.squeeze(1), spk_pred, torch.Tensor(spk_pred.size(0)).to(args.device).fill_(1.0)).item()
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()

            # test_ddsp_band_loss += ddsp_band_loss.item()
            test_rec_loss += nn.functional.mse_loss(units, data['units']).item()
            test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            # test_formant += torch.mean(formant)
            # test_reflow_style_loss += reflow_style_loss.item()
            # test_centroid_loss += centroid_loss_from_mels(ddsp_mel, data['mel'], mel_freqs_hz=mel_freqs_hz)
            # test_style_reflow_loss += style_reflow_loss.item()
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            mel_val_L1_all +=  torch.nn.functional.l1_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    # test_gaussian_loss /=num_batches
    test_rec_loss /= num_batches
    test_spk_loss /= num_batches
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    test_ddsp_band_loss /= num_batches
    test_formant /= num_batches
    test_reflow_style_loss /= num_batches
    test_smooth_loss_loss /= num_batches
    # test_mag_loss_reflow_loss /= num_batches
    # test_interval_loss_loss /= num_batches
    # test_centroid_loss /= num_batches
    # test_style_reflow_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_L1_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' [test_rec_loss] test_rec_loss:', test_rec_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    print(' Mel Val L1', mel_val_L1_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    # std_n_mean = torch.std_mean(delta_cnt)
    # print(f' shifts, max{delta_cnt.max()}, min{delta_cnt.min()}, mean: {std_n_mean[1]}, std: {std_n_mean[0]} ')
    
    # saver.log_value({
    #     'validation/test_smooth_loss_loss': test_smooth_loss_loss
    # })
    # saver.log_value({
    #     'validation/test_mag_loss_reflow_loss': test_mag_loss_reflow_loss
    # })
    # saver.log_value({
    #     'validation/test_interval_loss_loss': test_interval_loss_loss
    # })
      
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    saver.log_value({
        'validation/mel_val_L1': mel_val_L1_all
    })
    saver.log_value({
        'validation/test_reflow_style_loss': test_reflow_style_loss
    })
    return test_ddsp_loss, test_reflow_loss,test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_formant

def test_style_vc_flow(args, style_model, model, reflow_model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()
    style_model.eval()
    reflow_model.eval()
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])
    # spkc_criterion = nn.CosineEmbeddingLoss()

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    test_spk_loss = 0. 
    test_style_reflow_loss = 0.
    # test_gaussian_loss = 0.0

    # mel mse val
    mel_val_L1_all = 0
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            # _, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred = style_model(data['units'].float(), data['units_w'].float(), data['units_h'].float(), data['spk_embd'].float(), data['mel'].float())
            units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant,  log_det_fwd, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  None, data['f0'], data['volume'],0 , infer=True)
            # unit, _, _, _, _, _, z_fwd, _, _, _, _, formant, _, _ = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], None, data['f0'], None, infer=True, noise_fac=0.0)
            # latent = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], None, data['f0'], None, infer=True, noise_fac=0.0, latent_only=True)
            unit_out = reflow_model(units, gt_spec=data['units'], infer=True, infer_step=50, method='euler', t_start=0.0, use_tqdm=False, gin=None)
            style_reflow_loss, _ = reflow_model(units, gt_spec=data['units'], t_start=0.0, infer=False, gin=None)
            mel = model(
                    unit_out,
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    aug_shift=formant,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            # unit, mu, logvar, spk_pred = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  data['f0'])
            ddsp_loss, reflow_loss, _, ddsp_wav, ddsp_mel = model(
                unit_out, 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                aug_shift=formant,
                t_start=args.model.t_start)
            # test_gaussian_loss += gaussian_deviation_loss(unit, unit_mean, unit_var).item()
            # ddsp_band_loss = 0.2 * (mod_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + mod_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw))
            # test_spk_loss += 2 * spkc_criterion(data['spk_embd'], spk_pred, torch.Tensor(spk_pred.size(0)).to(args.device).fill_(1.0)).item()
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            # test_ddsp_band_loss += ddsp_band_loss.item()
            test_rec_loss += nn.functional.mse_loss(unit_out, data['units']).item()
            test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            test_style_reflow_loss += style_reflow_loss.item()
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            mel_val_L1_all +=  torch.nn.functional.l1_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    # test_gaussian_loss /=num_batches
    test_rec_loss /= num_batches
    test_spk_loss /= num_batches
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    test_ddsp_band_loss /= num_batches
    test_style_reflow_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_L1_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' [test_rec_loss] test_rec_loss:', test_rec_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    print(' Mel Val L1', mel_val_L1_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    saver.log_value({
        'validation/mel_val_L1': mel_val_L1_all
    })
    return test_ddsp_loss, test_reflow_loss,test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_style_reflow_loss




def test_post_processor(args, model, reflow_model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()
    reflow_model.eval()
    # spkc_criterion = nn.CosineEmbeddingLoss()

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    test_spk_loss = 0. 
    test_style_reflow_loss = 0.
    # test_gaussian_loss = 0.0

    # mel mse val
    mel_val_L1_all = 0
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            mel = model(
                    data['units'],
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            
            mel_out = reflow_model(mel, gt_spec=data['mel'], infer=True, infer_step=50, method='euler', t_start=0.0, use_tqdm=False)
            style_reflow_loss = reflow_model(mel, gt_spec=data['mel'], t_start=0.0, infer=False)
            
            signal = vocoder.infer(mel_out, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            test_style_reflow_loss += style_reflow_loss.item()
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            mel_val_L1_all +=  torch.nn.functional.l1_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    # test_gaussian_loss /=num_batches
    test_rec_loss /= num_batches
    test_spk_loss /= num_batches
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    test_ddsp_band_loss /= num_batches
    test_style_reflow_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_L1_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' [test_rec_loss] test_rec_loss:', test_rec_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    print(' Mel Val L1', mel_val_L1_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    saver.log_value({
        'validation/mel_val_L1': mel_val_L1_all
    })
    return test_ddsp_loss, test_reflow_loss,test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_style_reflow_loss

def test_enhance(args, model, vocoder, loader_test, saver, style_model=None):
    print(' [*] testing...')
    model.eval()

    # losses
    test_mrs_loss = 0.
    test_mel_loss = 0.
    test_mid_loss = 0.
    test_contrast_loss = 0.
    

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    mrs_loss_fun = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]).to(args.device)
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            mel = data['mel'].transpose(1, 2)
            gt_mel = data['gt_mel']
            pred_mel = model(mel).transpose(1, 2)
            pred_audio = vocoder.infer(pred_mel, data['f0']).squeeze(1)
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = pred_audio.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            # print(data['gt_audio'].shape)
            gt_audio = data['gt_audio']
            min_len = min(gt_audio.shape[1], pred_audio.shape[1])
            mrs_loss = mrs_loss_fun(pred_audio[ :, :min_len], gt_audio[ :, :min_len])
            mel_loss = F.l1_loss(pred_mel, gt_mel)
            mid_loss = band_mel_loss(pred_mel, gt_mel)
            contrast_loss = spectral_contrast_loss(pred_mel)
            
            test_mrs_loss += mrs_loss
            test_mel_loss += mel_loss
            test_mid_loss += mid_loss
            test_contrast_loss += contrast_loss
            
            # log mel
            saver.log_spec(data['name'][0], data['gt_mel'], pred_mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(pred_audio)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': pred_audio})

            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pred_audio = pred_audio.unsqueeze(0)
            # print(pred_audio[0, ...].shape, audio[0, ...].shape)
            pre_mel = WAV2MEL.get_mel(pred_audio[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            # print(pred_mel.shape, data['gt_mel'].transpose(1, 2).shape)
            mel_val_mse_all += torch.nn.functional.mse_loss(pred_mel, data['gt_mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['gt_mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(pred_mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            # print(mel_val_snr_all)
    # report
    test_mrs_loss /= num_batches
    test_mel_loss /= num_batches 
    test_mid_loss /= num_batches
    test_contrast_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_mrs_loss] test_mrs_loss:', test_mrs_loss)
    print(' [test_mel_loss] test_mel_loss:', test_mel_loss)
    print(' [test_mid_loss] test_mid_loss:', test_mid_loss)
    print(' [test_contrast_loss] test_contrast_loss:', test_contrast_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    return test_mrs_loss, test_mel_loss, test_mid_loss, test_contrast_loss


def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step, train_reflow=True)
    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    
    mrsl = MRSTFTLoss().to('cuda')
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    # for params in model.ddsp_model.parameters():
    #     params.requires_grad = False
        
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
                    
            units = data['units']
            # forward
            if dtype == torch.float32:
                ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel= model(units, data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel =model(units, data['f0'], data['volume'], data['spk_id'], 
                                    aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            # mrsl_loss = mrsl(ddsp_wav, data['gt_audio'])
            # harm_loss = harmonic_emphasis_mel_loss_logged(ddsp_mel, data['mel'], data['f0'])
            # hnr_loss = hnr_loss_mel(ddsp_mel,  data['mel'], data['f0'])
            # lcp_env_loss = lpc_envelope_loss(ddsp_wav, data['gt_audio'])
            # c_loss_obj = combo_loss(data['gt_audio'], ddsp_wav, signal_har, signal_noise)
            # c_loss = c_loss_obj['total']
            # handle nan loss
            # drift = drift_loss(ddsp_mel, data['mel'])
            loss = (ddsp_loss + ddsp_band_loss) + reflow_loss
            loss = loss.float()
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                del ddsp_loss
                del reflow_loss
                del ddsp_band_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    # 'train/L_mel': c_loss_obj['mel'],
                    # 'train/mfcc': c_loss_obj['mfcc'],
                    # 'train/mrstft': c_loss_obj['mrstft'],
                    # 'train/noise_budget': c_loss_obj['noise_budget'],
                    # 'train/harm_loss': harm_loss.item(),
                    # 'train/ddsp_drift': drift.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/ddsp_band_loss': ddsp_band_loss,
                    'train/reflow_loss': reflow_loss.item(),
                    # 'train/hnr_loss': hnr_loss.item(),
                    # 'train/ddsp_msrl_loss': mrsl_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_ddsp_msrl_loss = test(args, model, vocoder, loader_test, saver)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + test_ddsp_msrl_loss) + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                })
                
                model.train()

def train_lagrangian(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step, train_reflow=True)
    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    
    mrsl = MRSTFTLoss().to('cuda')
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    # for params in model.ddsp_model.parameters():
    #     params.requires_grad = False
        
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
                    
            units = data['units']
            # forward
            
            with torch.no_grad():
                q = units.float()  # [B, L, 768]
                a = q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]  # dt=1
                baseline = a.square().mean()

            if dtype == torch.float32:
                # ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel= model(units, data['f0'], data['volume'], data['spk_id'], 
                                # aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                loss = model.loss(units)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    loss = model.loss(units)
                    # ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel =model(units, data['f0'], data['volume'], data['spk_id'], 
                                    # aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            # mrsl_loss = mrsl(ddsp_wav, data['gt_audio'])
            # harm_loss = harmonic_emphasis_mel_loss_logged(ddsp_mel, data['mel'], data['f0'])
            # hnr_loss = hnr_loss_mel(ddsp_mel,  data['mel'], data['f0'])
            # lcp_env_loss = lpc_envelope_loss(ddsp_wav, data['gt_audio'])
            # c_loss_obj = combo_loss(data['gt_audio'], ddsp_wav, signal_har, signal_noise)
            # c_loss = c_loss_obj['total']
            # handle nan loss
            # drift = drift_loss(ddsp_mel, data['mel'])
            # loss = (ddsp_loss + ddsp_band_loss) + reflow_loss
            loss = loss.float()
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                del loss
                # del reflow_loss
                # del ddsp_band_loss
                continue
            # elif torch.isnan(reflow_loss):
            #     raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/acc_base': baseline.item(),
                    # 'train/L_mel': c_loss_obj['mel'],
                    # 'train/mfcc': c_loss_obj['mfcc'],
                    # 'train/mrstft': c_loss_obj['mrstft'],
                    # 'train/noise_budget': c_loss_obj['noise_budget'],
                    # 'train/harm_loss': harm_loss.item(),
                    # 'train/ddsp_drift': drift.item(),
                    # 'train/ddsp_loss': ddsp_loss.item(),
                    # 'train/ddsp_band_loss': ddsp_band_loss,
                    # 'train/reflow_loss': reflow_loss.item(),
                    # 'train/hnr_loss': hnr_loss.item(),
                    # 'train/ddsp_msrl_loss': mrsl_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_ddsp_msrl_loss = test_lagrangian(args, model, vocoder, loader_test, saver)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + test_ddsp_band_loss + test_ddsp_msrl_loss) + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                })
                
                model.train() 
                
def train_base(args, initial_global_step, model, optimizer, scheduler, validation_model, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step, train_reflow=False)
    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    validation_model.eval()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    # for params in model.ddsp_model.parameters():
    #     params.requires_grad = False
    spkc_criterion = nn.CosineEmbeddingLoss()
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
                    
            units = data['units']
            # forward
            if dtype == torch.float32:
                units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, reflow_loss,  log_det_fwd, z_ps, mask = model(data['units'].float(), data['units_w'].float(), data['units_h'].float(), data['spk_embd'].float(),  data['mel_high_res'].float(), None, None)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, reflow_loss,  log_det_fwd, z_ps, mask = model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  data['mel_high_res'], None, None)
            
            spk_pred = spk_pred.float()
            units = units.float()
            reconstruction_loss = F.l1_loss(data['units'].float(), units)
            angle_loss = (1.0 - F.cosine_similarity(data['units'].float(), units)).mean()
            spk_loss = 1 * spkc_criterion(data['spk_embd'].float(), spk_pred, torch.ones(spk_pred.size(0), device=args.device))
            kl_loss = vits_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + vits_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw) *0.5
            
            
            loss = reconstruction_loss + spk_loss * 2.0 + kl_loss + angle_loss * 2.0 + reflow_loss
            loss = loss.float()
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                # del ddsp_loss
                del reflow_loss
                # del ddsp_band_loss
                continue
            else:
                loss = loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.style_dir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    # 'train/L_mel': c_loss_obj['mel'],
                    # 'train/mfcc': c_loss_obj['mfcc'],
                    # 'train/mrstft': c_loss_obj['mrstft'],
                    # 'train/noise_budget': c_loss_obj['noise_budget'],
                    # 'train/harm_loss': harm_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/spk_loss': spk_loss.item(),
                    'train/angle_loss': angle_loss.item(),
                    'train/reconstruction_loss': reconstruction_loss.item(),
                    # 'train/hnr_loss': hnr_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_reconstruction_loss, test_angle_loss, test_spk_loss = test_base(args, model, validation_model, vocoder, loader_test, saver)
                test_loss = test_reconstruction_loss + test_angle_loss + test_spk_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                # saver.log_value({
                #     'validation/loss': test_loss,
                #     'validation/test_reconstruction_loss': test_reconstruction_loss,
                #     'validation/test_angle_loss': test_angle_loss,
                #     'validation/test_spk_loss': test_spk_loss,
                # })
                
                model.train()
                
                
                
def train_vc_drift(args, initial_global_step, style_model, model_g, optimizer, scheduler, vocoder, loader_train, loader_test, units_encoder):
    from torch.distributions import MultivariateNormal, kl_divergence
    saver = Saver(args, initial_global_step=initial_global_step)
    # mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    style_count = utils.get_network_paras_amount({'style':style_model})
    saver.log_info('--- model size ---')
    saver.log_info(style_count)

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model_g = model_g.to(torch.float32)
    model_g.train()
    style_model.train()

    for param in model_g.parameters():
        param.requires_grad = False
    
    for param in units_encoder.model.hubert.parameters():
        param.requires_grad = False


    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)

    spkc_criterion = nn.CosineEmbeddingLoss()
    # drift_crit = DriftingLoss(normalize_features=False, normalize_drift=False)
    accumulation_counter = 0
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()
            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward

            # units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, formant, style_flow_loss, z_mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'])
            if dtype == torch.float32:
                units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['volume'])

                ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(units, data['f0'] , data['volume'], data['spk_id'],
                                    aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)

                generated_unit = units_encoder.encode(ddsp_wav, 44100, 320)
                gt_unit = units_encoder.encode(data['gt_audio'], 44100, 320)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['volume'])

                    ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(units, data['f0'] , data['volume'], data['spk_id'],
                                        aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                    
                    generated_unit = units_encoder.encode(ddsp_wav, 44100, 320)
                    gt_unit = units_encoder.encode(data['gt_audio'], 44100, 320)
            spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.ones(spk_pred.size(0), device=args.device))
           
            reconstruction_loss = F.l1_loss(data['units'], units)

            std_q, mu_q = torch.std_mean(data['units'], dim=1)
            std_p, mu_p = torch.std_mean(units, dim=1)
            distribution_loss = kl_loss_distributions(mu_p, std_p, mu_q, std_q)
            

            kl_loss = mod_kl_loss2(mu_ps, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + 0.5 * mod_kl_loss2(mu_pr, logvar_pr, mu_ps, logvar_ps, log_det_bkw)
            # print(data['units'].shape)
            # drift = drift_crit(units.unsqueeze(1).float(), data['units'].unsqueeze(1).float())[0]
            
            drift = F.l1_loss(generated_unit,gt_unit)

            loss = (3 * ddsp_loss) + reflow_loss + reconstruction_loss + spk_loss * 2 + kl_loss + drift + distribution_loss
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                del ddsp_loss
                # del mrsl_loss
                # del kl_loss
                del reflow_loss
                del ddsp_band_loss
                # del spk_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
             

            # log loss
            if saver.global_step % args.train.interval_log == 0 and (accumulation_counter == 0):
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/ddsp_band_loss': ddsp_band_loss.item(),
                    'train/unit_rec_loss': reconstruction_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/drift': drift.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/distribution_loss': distribution_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0 and( accumulation_counter == 0):
                optimizer_save = optimizer if args.train.save_opt else None
                                
                # save latest
                saver.save_model(style_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_formant = test_style_vc_gan(args, style_model, model_g, vocoder, loader_test, saver, dtype, None, None, None)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + test_ddsp_msrl_loss) + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/rec_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                })
                
                model_g.train()
                
def train_osgan(args, initial_global_step, model_g, model_d, optimizer, optimizer_d, scheduler, scheduler_d, vocoder, loader_train, loader_test):
# saver
    saver = Saver(args, initial_global_step=initial_global_step,  train_reflow=True)
    # mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    # model_g = model_g.to(torch.float32)
    model_g.train()
    model_d.train()


    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    transform = torchaudio.transforms.Resample(44100, 32000).to('cuda')

    # spkc_criterion = nn.CosineEmbeddingLoss()
    gradient_accumulation_steps = 1
    accumulation_counter = 0
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            if accumulation_counter == 0:
                saver.global_step_increment()
                optimizer.zero_grad()
                optimizer_d.zero_grad()
            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward

            # units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, formant, style_flow_loss, z_mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'])
            if dtype == torch.float32:
                ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(data['units'], data['f0'] , data['volume'], data['spk_id'],
                                    aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(data['units'], data['f0'] , data['volume'], data['spk_id'],
                                        aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            
            # o = lambda unit_gen, formant_gen:  model_g(unit_gen, data['f0'] , data['volume'], data['spk_id'], aug_shift=data['aug_shift'] + formant_gen, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            
            # ddsp_loss, reflow_loss, ddsp_band_loss , ddsp_wav, ddsp_mel = checkpoint.checkpoint(o,units, formant, use_reentrant=False)
            transformed_fake = transform(ddsp_wav)
            transformed_real = transform(data['gt_audio'])

            disc_fake = model_d(transformed_fake)
            disc_real = model_d(transformed_real)

            
            score_loss = torch.zeros(1, device=args.device)
            for (_, score_fake) in disc_fake:
                score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
            score_loss = score_loss / len(disc_fake)

            feat_loss = torch.zeros(1, device=args.device)
            for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                for fake, real in zip(feat_fake, feat_real):
                    feat_loss += torch.mean(torch.abs(fake - real))
            feat_loss = feat_loss / len(disc_fake)
            feat_loss = feat_loss * 2
            
            # mrsl_loss = mrsl(ddsp_wav, data['gt_audio'])
            
            loss_d = torch.zeros(1, device=args.device)
            for (_, sf), (_, sr) in zip(disc_fake, disc_real):
                loss_d = loss_d + ((sr - 1.0) ** 2).mean() + (sf ** 2).mean()
            loss_d = loss_d / len(disc_fake)

            loss = (3 * ddsp_loss) + reflow_loss + score_loss + feat_loss
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                del ddsp_loss
                # del mrsl_loss
                # del kl_loss
                del reflow_loss
                del score_loss
                del feat_loss
                del ddsp_band_loss
                # del spk_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                d_params = [p for p in model_d.parameters() if p.requires_grad]
                # Keep discriminator in FP32 - no scaling
                grads_D = torch.autograd.grad(
                    loss_d / gradient_accumulation_steps,
                    d_params,
                    retain_graph=True,        # keep graph for G grad pass
                    create_graph=False,
                    allow_unused=True
                )
                # assign & step
                for p, g in zip(d_params, grads_D):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                g_params = [p for p in model_g.parameters() if p.requires_grad]
                # Scale generator loss for AMP
                scaled_loss = scaler.scale(loss / gradient_accumulation_steps) if dtype != torch.float32 else (loss / gradient_accumulation_steps)
                grads_G = torch.autograd.grad(
                    scaled_loss,
                    g_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )
                for p, g in zip(g_params, grads_G):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                accumulation_counter += 1
                if accumulation_counter >= gradient_accumulation_steps:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model_d.parameters(), max_norm=1.0)
                    optimizer_d.step()
                    scheduler_d.step()

                    torch.nn.utils.clip_grad.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
                    if dtype != torch.float32:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()

                    # (optional) early free grads
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_d.zero_grad(set_to_none=True)
                    accumulation_counter = 0
                # gc.collect()
                # torch.cuda.empty_cache()

            # Profiler CUDA memory usage for each batch
            # prof_mem = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=5)
            # saver.log_info(f'Batch {batch_idx} Profiler CUDA Memory:\n{prof_mem}')

            # log loss
            if saver.global_step % args.train.interval_log == 0 and (accumulation_counter == 0):
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/ddsp_band_loss': ddsp_band_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                 
                    'train/score_loss': score_loss.item(),
                    'train/d_loss': loss_d.item(),
                  
                    'train/feat_loss': feat_loss,
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0 and( accumulation_counter == 0):
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_gan_model(model_g, model_d, optimizer_save, optimizer_d, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_ddsp_msrl_loss = test(args, model_g, vocoder, loader_test, saver)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + test_ddsp_msrl_loss) + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                })
                
                model_g.train()
                model_d.train()
                
                
def train_reflow_gan(args, initial_global_step, model, model_d, optimizer, optimizer_d, scheduler, scheduler_d, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step, train_reflow=True)
    
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)])
    
    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    transform = torchaudio.transforms.Resample(44100, 32000, dtype=dtype).to('cuda')
    for params in model.ddsp_model.parameters():
        params.requires_grad = False
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
                    
            units = data['units']
            # forward
            if dtype == torch.float32:
                ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model(units.float(), data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                
                transformed_fake = transform(ddsp_wav)
                disc_fake = model_d(transformed_fake)
                score_loss = 0.0
                for (_, score_fake) in disc_fake:
                    score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
                score_loss = score_loss / len(disc_fake)
                score_loss = score_loss / 6 

                
                transformed_real = transform(data['gt_audio'])
                disc_real = model_d(transformed_real)
                feat_loss = 0.0
                for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                    for fake, real in zip(feat_fake, feat_real):
                        feat_loss += torch.mean(torch.abs(fake - real))
                feat_loss = feat_loss / len(disc_fake)
                feat_loss = feat_loss * 2
                feat_loss = feat_loss / 6

                
                disc_fake1 = model_d(transformed_fake.detach())
                disc_real1 = model_d(transformed_real)
                loss_d = 0.0
                for (_, score_fake1), (_, score_real1) in zip(disc_fake1, disc_real1):
                    loss_d += torch.mean(torch.pow(score_real1 - 1.0, 2))
                    loss_d += torch.mean(torch.pow(score_fake1, 2))
                loss_d = loss_d / len(disc_fake)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model(units, data['f0'], data['volume'], data['spk_id'], 
                                    aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                    transformed_fake = transform(ddsp_wav)
                    
                    disc_fake = model_d(transformed_fake)
                    score_loss = 0.0
                    for (_, score_fake) in disc_fake:
                        score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
                    score_loss = score_loss / len(disc_fake)
                    score_loss = score_loss / 6 
                    transformed_real = transform(data['gt_audio'])
                    
                    disc_real = model_d(transformed_real)
                    feat_loss = 0.0
                    for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                        for fake, real in zip(feat_fake, feat_real):
                            feat_loss += torch.mean(torch.abs(fake - real))
                    feat_loss = feat_loss / len(disc_fake)
                    feat_loss = feat_loss * 2
                    feat_loss = feat_loss / 6
                    
                    
                    disc_fake1 = model_d(transformed_fake.detach())
                    disc_real1 = model_d(transformed_real)
                    loss_d = 0.0
                    for (_, score_fake1), (_, score_real1) in zip(disc_fake1, disc_real1):
                        loss_d += torch.mean(torch.pow(score_real1 - 1.0, 2))
                        loss_d += torch.mean(torch.pow(score_fake1, 2))
                    loss_d = loss_d / len(disc_fake)
            mrsl_loss = mrsl(ddsp_wav, data['gt_audio']) / 7
            # gt_mel_norm = torch.clip(data['mel'], -2, 10)
            # gt_mel_norm = gt_mel_norm / 12 -2
            # ddsp_mel_norm = torch.clip(ddsp_mel, -2, 10)
            # ddsp_mel_norm = ddsp_mel_norm / 12 - 2
            # si_snr_loss = ((47.5 - calculate_mel_si_snr(gt_mel_norm, ddsp_mel_norm))**2)/5
            # handle nan loss
            if torch.isnan( (ddsp_loss + ddsp_band_loss + mrsl_loss + score_loss + feat_loss) + reflow_loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                del ddsp_loss
                del reflow_loss
                del score_loss
                del feat_loss
                del ddsp_band_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * (ddsp_loss + ddsp_band_loss + mrsl_loss + score_loss + feat_loss) + reflow_loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
                    optimizer.step()
                    
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_d.parameters(), max_norm=1)
                    optimizer_d.step()
                else:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    
                    optimizer_d.zero_grad()              
                    scaler.scale(loss_d).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_d.parameters(), max_norm=1)
                    scaler.step(optimizer_d)
                    scaler.update()
                scheduler.step()
                scheduler_d.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/g_loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/ddsp_band_loss': ddsp_band_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/score_loss': score_loss,
                    # 'train/si_snr_loss': si_snr_loss.item(),
                    'train/d_loss': loss_d,
                    'train/feat_loss': feat_loss,
                    'train/ddsp_msrl_loss': mrsl_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_gan_model(model, model_d, optimizer_save, optimizer_d, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_ddsp_msrl_loss = test_reflow_gan(args, model, model_d, vocoder, loader_test, saver)
                test_loss =args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + test_ddsp_msrl_loss) + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    # 'validation/test_score_loss': test_score_loss,
                    # 'validation/test_feat_loss': test_feat_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                })
                
                model.train()
                model_d.train()
                
                
                
def train_frontend_gan(args, initial_global_step, model_g, model_d, optimizer, optimizer_d, scheduler, scheduler_d, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step, train_reflow=True)
    
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)])
    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    params_count_d = utils.get_network_paras_amount({'model_d': model_d})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    saver.log_info(params_count_d)
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model_g.train()
    model_d.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    transform = torchaudio.transforms.Resample(44100, 32000, dtype=dtype).to('cuda')
    
    # for params in model.ddsp_model.parameters():
    #     params.requires_grad = False
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer_d.zero_grad()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
                    
            units = data['units']
            # forward
            if dtype == torch.float32:
                ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(units.float(), data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                transformed_fake = transform(ddsp_wav)
                disc_fake = model_d(transformed_fake)
                score_loss = 0.0
                for (_, score_fake) in disc_fake:
                    score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
                score_loss = score_loss / len(disc_fake)
                
                transformed_real = transform(data['gt_audio'])
                disc_real = model_d(transformed_real)
                feat_loss = 0.0
                for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                    for fake, real in zip(feat_fake, feat_real):
                        feat_loss += torch.mean(torch.abs(fake - real))
                feat_loss = feat_loss / len(disc_fake)
                feat_loss = feat_loss * 2
                
                disc_fake1 = model_d(transformed_fake.detach())
                disc_real1 = model_d(transformed_real)
                loss_d = 0.0
                for (_, score_fake1), (_, score_real1) in zip(disc_fake1, disc_real1):
                    loss_d += torch.mean(torch.pow(score_real1 - 1.0, 2))
                    loss_d += torch.mean(torch.pow(score_fake1, 2))
                loss_d = loss_d / len(disc_fake)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(units, data['f0'], data['volume'], data['spk_id'], 
                                    aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                    transformed_fake = transform(ddsp_wav)
                    
                    disc_fake = model_d(transformed_fake)
                    score_loss = 0.0
                    for (_, score_fake) in disc_fake:
                        score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
                    score_loss = score_loss / len(disc_fake)
                    transformed_real = transform(data['gt_audio'])
                    
                    disc_real = model_d(transformed_real)
                    feat_loss = 0.0
                    for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                        for fake, real in zip(feat_fake, feat_real):
                            feat_loss += torch.mean(torch.abs(fake - real))
                    feat_loss = feat_loss / len(disc_fake)
                    feat_loss = feat_loss * 2
                    
                    disc_fake1 = model_d(transformed_fake.detach())
                    disc_real1 = model_d(transformed_real)
                    loss_d = 0.0
                    for (_, score_fake1), (_, score_real1) in zip(disc_fake1, disc_real1):
                        loss_d += torch.mean(torch.pow(score_real1 - 1.0, 2))
                        loss_d += torch.mean(torch.pow(score_fake1, 2))
                    loss_d = loss_d / len(disc_fake)
                    
            mel_loss = torch.nn.functional.l1_loss(ddsp_mel, data['mel'])
            mrsl_loss = mrsl(ddsp_wav, data['gt_audio'])
            # handle nan loss
            if torch.isnan(ddsp_loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                del ddsp_loss
                del reflow_loss
                del score_loss
                del feat_loss
                del ddsp_band_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * (ddsp_loss + ddsp_band_loss + mrsl_loss + mel_loss) + reflow_loss + score_loss + feat_loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_g.parameters(), max_norm=1)
                    optimizer.step()
                    
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_d.parameters(), max_norm=1)
                    optimizer_d.step()
                else:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_g.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    
                    optimizer_d.zero_grad()              
                    scaler.scale(loss_d).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_d.parameters(), max_norm=1)
                    scaler.step(optimizer_d)
                    scaler.update()
                scheduler.step()
                scheduler_d.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/g_loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/ddsp_band_loss': ddsp_band_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/score_loss': score_loss,
                    'train/d_loss': loss_d,
                    'train/feat_loss': feat_loss,
                    'train/ddsp_msrl_loss': mrsl_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_gan_model(model_g, model_d, optimizer_save, optimizer_d, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_ddsp_msrl_loss, test_score_loss, test_feat_loss = test_gan(args, model_g, model_d, vocoder, loader_test, saver, dtype)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + test_ddsp_msrl_loss) + test_reflow_loss + test_score_loss + test_feat_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_score_loss': test_score_loss,
                    'validation/test_feat_loss': test_feat_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                })
                
                model_g.train()
                model_d.train()
                
                
                
def train_stlye(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test, style_model):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    style_count = utils.get_network_paras_amount({'style':style_model})
    saver.log_info('--- model size ---')
    saver.log_info(style_count)
    
    # sum_mean = torch.from_numpy(np.load('exp/all_mean.npy')).to(args.device)
    # unit_mean = torch.from_numpy(np.load('exp/unit_mean.npy')).to(args.device)
    # unit_var = torch.from_numpy(np.load('exp/var_unit.npy')).to(args.device)
    
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    style_model.train()
    
    for param in model.parameters():
        param.requires_grad = False

    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    # for params in model.ddsp_model.parameters():
    #     params.requires_grad = False
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward
            if dtype == torch.float32:
                units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['volume'])
                ddsp_loss, reflow_loss, _ , ddsp_wav, ddsp_mel = model(units, data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['volume'])
                    ddsp_loss, reflow_loss, _, ddsp_wav, ddsp_mel =model(units, data['f0'], data['volume'], data['spk_id'], 
                                    aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            
            # kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
            mrsl_loss = mrsl(ddsp_wav, data['gt_audio'])
            reconstruction_loss = F.l1_loss(data['units'], units)
            loss_kl_r = vits_kl_loss2(z_bkw, logvar_ps, mu_pr, logvar_pr, None, useTopK=False)
            loss_kl_f = vits_kl_loss2(z_fwd, logvar_pr, mu_ps, logvar_ps, None, useTopK=False)
            kl_loss = loss_kl_f + loss_kl_r * 0.5
            
            # ddsp_band_loss = kl_budget_loss(mu, logvar)
            # gaussian_loss = gaussian_deviation_loss(units, unit_mean, unit_var)
            # handle nan loss
            if torch.isnan(ddsp_loss + kl_loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                del ddsp_loss
                del reflow_loss
                del kl_loss
                del reconstruction_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * (ddsp_loss + kl_loss + mrsl_loss) + reflow_loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=style_model.parameters(), max_norm=1)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=style_model.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step(loss)
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/unit_rec_loss': reconstruction_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/ddsp_msrl_loss': mrsl_loss.item(),
                    # 'train/gau_loss': gaussian_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(style_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss = test_style(args, model, vocoder, loader_test, saver, style_model, None, None, None)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + mrsl_loss) + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    'validation/test_rec_loss': test_rec_loss
                })
                
                model.train()
                style_model.train()
                
                
def train_style_gan(args, initial_global_step, style_model, model_g, model_d, optimizer, optimizer_d, scheduler, scheduler_d, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    style_count = utils.get_network_paras_amount({'style':style_model})
    saver.log_info('--- model size ---')
    saver.log_info(style_count)
    
    # sum_mean = torch.from_numpy(np.load('exp/all_mean.npy')).to(args.device)
    # unit_mean = torch.from_numpy(np.load('exp/unit_mean.npy')).to(args.device)
    # unit_var = torch.from_numpy(np.load('exp/var_unit.npy')).to(args.device)
    
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model_g.train()
    style_model.train()
    model_d.train()
    
    for param in model_g.parameters():
        param.requires_grad = False

    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    transform = torchaudio.transforms.Resample(44100, 32000, dtype=dtype).to('cuda')
    spkc_criterion = nn.CosineEmbeddingLoss()
    
    # for params in model.ddsp_model.parameters():
    #     params.requires_grad = False
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward
            if dtype == torch.float32:
                # units, mu, logvar = style_model(data['units'].float(), data['units_w'].float(), data['units_h'].float(), None, None)
                ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel, stylized_feats , mu, logvar, spk_pred = model_g(data['units'].float(), data['units_w'].float(), data['units_h'].float(),data['spk_embd'].float(), data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                spk_loss = 2 * spkc_criterion(data['spk_embd'], spk_pred, torch.Tensor(spk_pred.size(0)).to(args.device).fill_(1.0))
                transformed_fake = transform(ddsp_wav)
                disc_fake = model_d(transformed_fake)
                score_loss = 0.0
                for (_, score_fake) in disc_fake:
                    score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
                score_loss = score_loss / len(disc_fake)
                
                transformed_real = transform(data['gt_audio'])
                disc_real = model_d(transformed_real)
                feat_loss = 0.0
                for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                    for fake, real in zip(feat_fake, feat_real):
                        feat_loss += torch.mean(torch.abs(fake - real))
                feat_loss = feat_loss / len(disc_fake)
                feat_loss = feat_loss * 2
                
                disc_fake1 = model_d(transformed_fake.detach())
                disc_real1 = model_d(transformed_real)
                loss_d = 0.0
                for (_, score_fake1), (_, score_real1) in zip(disc_fake1, disc_real1):
                    loss_d += torch.mean(torch.pow(score_real1 - 1.0, 2))
                    loss_d += torch.mean(torch.pow(score_fake1, 2))
                loss_d = loss_d / len(disc_fake)
            else:
                # with autocast(device_type=args.device, dtype=dtype):
                #     units, mu, logvar = style_model(data['units'], data['units_w'], data['units_h'], None, None)
                #     ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel =model_g(units, data['f0'], data['volume'], data['spk_id'], 
                #                     aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                    
                #     transformed_fake = transform(ddsp_wav)
                    
                #     disc_fake = model_d(transformed_fake)
                #     score_loss = 0.0
                #     for (_, score_fake) in disc_fake:
                #         score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
                #     score_loss = score_loss / len(disc_fake)
                #     transformed_real = transform(data['gt_audio'])
                    
                #     disc_real = model_d(transformed_real)
                #     feat_loss = 0.0
                #     for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                #         for fake, real in zip(feat_fake, feat_real):
                #             feat_loss += torch.mean(torch.abs(fake - real))
                #     feat_loss = feat_loss / len(disc_fake)
                #     feat_loss = feat_loss * 2
                    
                #     disc_fake1 = model_d(transformed_fake.detach())
                #     disc_real1 = model_d(transformed_real)
                #     loss_d = 0.0
                #     for (_, score_fake1), (_, score_real1) in zip(disc_fake1, disc_real1):
                #         loss_d += torch.mean(torch.pow(score_real1 - 1.0, 2))
                #         loss_d += torch.mean(torch.pow(score_fake1, 2))
                #     loss_d = loss_d / len(disc_fake)
                
                raise
            
            # kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
            # mel_loss = 5 * torch.nn.functional.l1_loss(ddsp_mel, data['mel'])
            mrsl_loss = mrsl(ddsp_wav, data['gt_audio'])
            reconstruction_loss = nn.functional.mse_loss(stylized_feats, data['units'].float())
            
            kl_loss = kl_budget_loss(mu, logvar)
            # gaussian_loss = gaussian_deviation_loss(units, unit_mean, unit_var)
            # handle nan loss
            if torch.isnan(ddsp_loss + ddsp_band_loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                del ddsp_loss
                del reflow_loss
                del ddsp_band_loss
                del reconstruction_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * (ddsp_loss + mrsl_loss + kl_loss) + reflow_loss + score_loss + feat_loss + reconstruction_loss + spk_loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=style_model.parameters(), max_norm=1)
                    optimizer.step()
                    
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_d.parameters(), max_norm=1)
                    optimizer_d.step()
                else:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=style_model.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    
                    optimizer_d.zero_grad()              
                    scaler.scale(loss_d).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_d.parameters(), max_norm=1)
                    scaler.step(optimizer_d)
                    scaler.update()
                scheduler.step()
                scheduler_d.step()
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/ddsp_band_loss': ddsp_band_loss.item(),
                    'train/unit_rec_loss': reconstruction_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/ddsp_msrl_loss': mrsl_loss.item(),
                    'train/score_loss': score_loss,
                    'train/d_loss': loss_d,
                    'train/feat_loss': feat_loss,
                    # 'train/gau_loss': gaussian_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_gan_model(style_model, model_d, optimizer_save, optimizer_d, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss = test_gan(args, style_model, model_g, vocoder, loader_test, saver, dtype, None, None, None)
                test_loss = test_loss = args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + mrsl_loss)+test_spk_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    'validation/test_rec_loss': test_rec_loss,
                    'validation/test_spk_loss': test_spk_loss
                })
                
                model_g.train()
                model_d.train()
                style_model.train()
                
                
def train_vc_gan(args, initial_global_step, style_model, model_g, model_d, optimizer, optimizer_d, scheduler, scheduler_d, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    style_count = utils.get_network_paras_amount({'style':style_model})
    saver.log_info('--- model size ---')
    saver.log_info(style_count)
    
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model_g.train()
    style_model.train()
    model_d.train()
    
    for param in model_g.parameters():
        param.requires_grad = False
    

    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    transform = torchaudio.transforms.Resample(44100, 32000).to('cuda')
    spkc_criterion = nn.CosineEmbeddingLoss()
    
    # for params in model_g.parameters():
    #     params.requires_grad = False
        
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()
            optimizer_d.zero_grad()
            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward
            if dtype == torch.float32:
                units, spk_pred, style_flow_loss, formant = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['f0'])
                ddsp_loss, reflow_loss, ddsp_band_loss , ddsp_wav, ddsp_mel = model_g(units, data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                
                spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.Tensor(spk_pred.size(0)).to(args.device).fill_(1.0))
                transformed_fake = transform(ddsp_wav.float())
                transformed_real = transform(data['gt_audio'].float())
                                
                disc_fake = model_d(transformed_fake)
                disc_real = model_d(transformed_real)
                
                score_loss = 0.0
                for (_, score_fake) in disc_fake:
                    score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
                score_loss = score_loss / len(disc_fake)

                feat_loss = 0.0
                for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                    for fake, real in zip(feat_fake, feat_real):
                        feat_loss += torch.mean(torch.abs(fake - real))
                feat_loss = feat_loss / len(disc_fake)
                feat_loss = feat_loss * 2
                
                disc_fake1 = model_d(transformed_fake.detach())
                mrsl_loss = mrsl(ddsp_wav, data['gt_audio'])
                reconstruction_loss = torch.clamp(nn.functional.mse_loss(units, data['units']),min=0., max=20)
                # if reconstruction_loss == 80:
                #     print(data['name_ext'])
                
                # kl_loss = 0.2 * compute_flow_loss( mu_pr, logvar_pr,    # → prior params p(z|c)
                #                             mu_ps, logvar_ps,    # → posterior params q(z|x)
                #                             z_fwd, log_det_fwd,  # → forward flow outputs
                #                             z_bkw, log_det_bkw)
                # norm = z_fwd.shape[1] * (z_fwd.shape[2] // 2)
                # centroid_loss = centroid_loss_from_mels(ddsp_mel, data['mel'], mel_freqs_hz=mel_freqs_hz)
                # kl_loss = mod_kl_loss2(mu_ps, logvar_ps, mu_pr, logvar_pr, log_det_fwd/norm) + 0.5 * mod_kl_loss2(mu_pr, logvar_pr, mu_ps, logvar_ps, log_det_bkw/norm)
                # kl_loss = mod_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + 0.5 * mod_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw)
                # handle nan loss
                # loss = (3 * ddsp_loss + kl_loss) + reflow_loss + reconstruction_loss + 2*spk_loss + mrsl_loss
                loss = (3 * ddsp_loss) + reflow_loss + score_loss + feat_loss + reconstruction_loss + 2*spk_loss + mrsl_loss + style_flow_loss
            # else:
            #     with autocast(device_type=args.device, dtype=dtype):
            #         units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, formant, style_flow_loss = style_model(data['units'], data['units_w'], data['units_h'],  data['spk_embd'],  data['mel'], data['f0'])
            #         ddsp_loss, reflow_loss, ddsp_band_loss , ddsp_wav, ddsp_mel =model_g(units, data['f0'], data['volume'], data['spk_id'], 
            #                         aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
                    
            #         spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.ones(spk_pred.size(0), device=args.device))
                    
                spk_loss = spk_loss.float()
                ddsp_loss = ddsp_loss.float()
                ddsp_wav = ddsp_wav.float()
                units = units.float()
                # mu_pr = mu_pr.float()
                # logvar_pr =logvar_pr.float()
                # mu_ps =mu_ps.float()
                # logvar_ps = logvar_ps.float()
                # z_fwd = z_fwd.float()
                # log_det_fwd = log_det_fwd.float()
                # z_bkw = z_bkw.float()
                # log_det_bkw = log_det_bkw.float()
                reflow_loss = reflow_loss.float()
                
                transformed_fake = transform(ddsp_wav)
                transformed_real = transform(data['gt_audio'])
                                
                disc_fake = model_d(transformed_fake)
                disc_real = model_d(transformed_real)
                
                score_loss = torch.zeros(1, device=args.device)
                for (_, score_fake) in disc_fake:
                    score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
                score_loss = score_loss / len(disc_fake)
                            
                feat_loss = torch.zeros(1, device=args.device)
                for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                    for fake, real in zip(feat_fake, feat_real):
                        feat_loss += torch.mean(torch.abs(fake - real))
                feat_loss = feat_loss / len(disc_fake)
                feat_loss = feat_loss * 2
                
                
                mrsl_loss = mrsl(ddsp_wav, data['gt_audio'])
                reconstruction_loss = torch.clamp(nn.functional.mse_loss(units, data['units']),min=0., max=20)
                # kl_loss = mod_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + 0.5 * mod_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw)
                loss = (3 * ddsp_loss) + reflow_loss + score_loss + feat_loss + reconstruction_loss + 2*spk_loss + mrsl_loss + style_flow_loss
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                del ddsp_loss
                del mrsl_loss
                # del kl_loss
                del reflow_loss
                del score_loss
                del feat_loss
                del ddsp_band_loss
                del reconstruction_loss
                del style_flow_loss
                del spk_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=style_model.parameters(), max_norm=1)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=style_model.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                
            disc_fake1 = model_d(transformed_fake.detach())
            disc_real1 = model_d(transformed_real)
            loss_d = torch.zeros(1, device=args.device)
            for (_, score_fake1), (_, score_real1) in zip(disc_fake1, disc_real1):
                loss_d += torch.mean(torch.pow(score_real1 - 1.0, 2))
                loss_d += torch.mean(torch.pow(score_fake1, 2))
            loss_d = loss_d / len(disc_fake)
            
            if dtype == torch.float32:  
                loss_d.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_d.parameters(), max_norm=1)
                optimizer_d.step()
            else:
                scaler.scale(loss_d).backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(parameters=model_d.parameters(), max_norm=1)
                scaler.step(optimizer_d)
                scaler.update()
            scheduler_d.step()
            optimizer_d.zero_grad()

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/ddsp_band_loss': ddsp_band_loss.item(),
                    'train/style_flow_loss': style_flow_loss.item(),
                    'train/unit_rec_loss': reconstruction_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    # 'train/kl_loss': kl_loss.item(),
                    # 'train/centroid_loss': centroid_loss.item(),
                    'train/score_loss': score_loss,
                    'train/d_loss': loss_d,
                    'train/spk_loss': spk_loss,
                    'train/feat_loss': feat_loss,
                    'train/mrsl_loss': mrsl_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_gan_model(style_model, model_d, optimizer_save, optimizer_d, postfix=f'{saver.global_step}')
                # saver.save_model(style_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_formant = test_style_vc_gan(args, style_model, model_g, vocoder, loader_test, saver, dtype, None, None, None)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + test_ddsp_msrl_loss)+test_spk_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    # 'validation/test_style_reflow_loss': test_style_reflow_loss,
                    'validation/test_rec_loss': test_rec_loss,
                    'validation/avg_formant': test_formant,
                    'validation/test_spk_loss': test_spk_loss
                })
                
                model_g.train()
                model_d.train()
                style_model.train()
                
                
def train_vc_osgan(args, initial_global_step, style_model, model_g, model_d, optimizer, optimizer_d, scheduler, scheduler_d, vocoder, loader_train, loader_test):
# saver
    saver = Saver(args, initial_global_step=initial_global_step)
    # mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    style_count = utils.get_network_paras_amount({'style':style_model})
    saver.log_info('--- model size ---')
    saver.log_info(style_count)

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model_g = model_g.to(torch.float32)
    model_g.train()
    style_model.train()
    model_d.train()

    for param in model_g.parameters():
        param.requires_grad = False


    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    transform = torchaudio.transforms.Resample(44100, 32000).to('cuda')

    spkc_criterion = nn.CosineEmbeddingLoss()
    gradient_accumulation_steps = 1
    accumulation_counter = 0
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            if accumulation_counter == 0:
                saver.global_step_increment()
                optimizer.zero_grad()
                optimizer_d.zero_grad()
            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward

            # units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, formant, style_flow_loss, z_mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'])
            if dtype == torch.float32:
                # units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['emo'])
                units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, log_det_fwd, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel_high_res'], data['f0'],  None, saver.global_step)

               
                ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(units, data['f0'] , data['volume'], data['spk_id'],
                                    aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    # units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['emo'])
                    units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, log_det_fwd, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel_high_res'], data['f0'], None, saver.global_step)
                    ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(units, data['f0'] , data['volume'], data['spk_id'],
                                        aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)

            # o = lambda unit_gen, formant_gen:  model_g(unit_gen, data['f0'] , data['volume'], data['spk_id'], aug_shift=data['aug_shift'] + formant_gen, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)

            # ddsp_loss, reflow_loss, ddsp_band_loss , ddsp_wav, ddsp_mel = checkpoint.checkpoint(o,units, formant, use_reentrant=False)
            spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.ones(spk_pred.size(0), device=args.device))
            transformed_fake = transform(ddsp_wav)
            transformed_real = transform(data['gt_audio'])

            # Run discriminator in fp16 to halve MRD STFT / conv tensor sizes.
            # Losses are cast to fp32 after the block so backward is stable.
            # Losses are computed inside the checkpointed region (per
            # sub-discriminator, batched fake+real pass) so the fmap
            # activations are actually freed instead of being kept alive as
            # checkpoint outputs.
            if dtype == torch.float32:
                score_loss, feat_loss, loss_d = model_d.compute_losses(transformed_fake, transformed_real)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    score_loss, feat_loss, loss_d = model_d.compute_losses(transformed_fake, transformed_real)

            # Cast to fp32 before backward — D params are fp32, no scaler needed.
            score_loss = score_loss.float()
            feat_loss = feat_loss.float()
            loss_d = loss_d.float()
            units = units.float()
            # generated_angle = units.reshape(-1, units.size(2))
            # gt_angle = data['units'].reshape(-1, units.size(2))
            # reconstruction_loss = reconsturct_criterion(generated_angle, gt_angle, torch.ones(gt_angle.size(0), device=args.device))
            reconstruction_loss = F.mse_loss(data['units'], units)
            angle_loss = (1.0 - F.cosine_similarity(data['units'].float(), units)).mean()
          
            # loss_kl_r = vits_kl_loss2(z_bkw, logvar_ps, mu_pr, logvar_pr, None, useTopK=False)
            # loss_kl_f = vits_kl_loss2(z_fwd, logvar_pr, mu_ps, logvar_ps, None, useTopK=False)
            # kl_loss = kl_loss_between_gaussians(mu_pr, logvar_pr, mu_ps, logvar_ps) + kl_loss_between_gaussians(mu_ps, logvar_ps, mu_pr, logvar_pr)
            # kl_loss = loss_kl_f + loss_kl_r * 0.5
            # z_ps = z_ps.transpose(1, 2)
            # latent_norm_loss = F.l1_loss(torch.norm(z_r, dim=1), torch.norm(z_ps, dim=1))
            # generated_latent = z_r.reshape(-1, z_r.size(1))
            # gt_latent = z_ps.reshape(-1, z_ps.size(1))
            # latent_criterion = reconsturct_criterion(generated_latent, gt_latent, torch.ones(gt_latent.size(0), device=args.device))
            # norm = z_fwd.shape[1] * (z_fwd.shape[2] // 2)
            # kl_loss = mod_kl_loss2(mu_ps, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + 0.5 * mod_kl_loss2(mu_pr, logvar_pr, mu_ps, logvar_ps, log_det_bkw)
            kl_loss = vits_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + vits_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw) *0.5
            # noise_loss = noise_filter.abs().mean()
            # print(kl_loss.item())
            
            # loss = (3 * ddsp_loss) + reflow_loss + score_loss + feat_loss + reconstruction_loss + kl_loss + 2 * spk_loss

            loss = (3.0 * ddsp_loss) + reflow_loss + score_loss + feat_loss + reconstruction_loss + spk_loss * 2.0 + kl_loss + angle_loss * 2.0
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                del ddsp_loss
                # del mrsl_loss
                # del kl_loss
                del reflow_loss
                del score_loss
                del feat_loss
                del ddsp_band_loss
                del reconstruction_loss
                # del spk_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                d_params = [p for p in model_d.parameters() if p.requires_grad]
                # Keep discriminator in FP32 - no scaling
                grads_D = torch.autograd.grad(
                    loss_d / gradient_accumulation_steps,
                    d_params,
                    retain_graph=True,        # keep graph for G grad pass
                    create_graph=False,
                    allow_unused=True
                )
                # assign & step
                for p, g in zip(d_params, grads_D):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                g_params = [p for p in style_model.parameters() if p.requires_grad]
                # Scale generator loss for AMP
                scaled_loss = scaler.scale(loss / gradient_accumulation_steps) if dtype != torch.float32 else (loss / gradient_accumulation_steps)
                grads_G = torch.autograd.grad(
                    scaled_loss,
                    g_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )
                for p, g in zip(g_params, grads_G):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                accumulation_counter += 1
                if accumulation_counter >= gradient_accumulation_steps:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model_d.parameters(), max_norm=1.0)
                    optimizer_d.step()
                    scheduler_d.step()

                    torch.nn.utils.clip_grad.clip_grad_norm_(style_model.parameters(), max_norm=1.0)
                    if dtype != torch.float32:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()

                    # (optional) early free grads
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_d.zero_grad(set_to_none=True)
                    accumulation_counter = 0
                # gc.collect()
                # torch.cuda.empty_cache()

            # Profiler CUDA memory usage for each batch
            # prof_mem = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=5)
            # saver.log_info(f'Batch {batch_idx} Profiler CUDA Memory:\n{prof_mem}')

            # log loss
            if saver.global_step % args.train.interval_log == 0 and (accumulation_counter == 0):
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/ddsp_band_loss': ddsp_band_loss.item(),
                    # 'train/noise_loss': noise_loss.item(),
                    'train/unit_rec_loss': reconstruction_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    # 'train/norm_loss': norm_loss.item(),
                    'train/latent_angle_loss': angle_loss.item(),
                    # 'train/perplexity': perplexity.item(),
                    # 'train/vq_loss': vq_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/score_loss': score_loss.item(),
                    'train/d_loss': loss_d.item(),
                    # 'train/loss_kl_f': loss_kl_f.item(),
                    # 'train/loss_kl_r': loss_kl_r.item(),
                    # 'train/transfer_reflow_loss': transfer_reflow_loss.item(),
                    # 'train/spk_loss': spk_loss,
                    'train/feat_loss': feat_loss,
                    # 'train/mrsl_loss': mrsl_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0 and( accumulation_counter == 0):
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_gan_model(style_model, model_d, optimizer_save, optimizer_d, postfix=f'{saver.global_step}')
                # saver.save_model(style_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_formant = test_style_vc_gan(args, style_model, model_g, vocoder, loader_test, saver, dtype, None, None, None)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + test_ddsp_msrl_loss)+test_spk_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    # 'validation/test_style_reflow_loss': test_style_reflow_loss,
                    'validation/test_rec_loss': test_rec_loss,
                    'validation/avg_formant': test_formant,
                    'validation/test_spk_loss': test_spk_loss
                })
                
                model_g.train()
                model_d.train()
                style_model.train()
                
                
                
def train_vc_osgan2(args, initial_global_step, style_model, model_g, model_d, optimizer, optimizer_d, scheduler, scheduler_d, vocoder, loader_train, loader_test):
# saver
    saver = Saver(args, initial_global_step=initial_global_step)
    # mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    style_count = utils.get_network_paras_amount({'style':style_model})
    saver.log_info('--- model size ---')
    saver.log_info(style_count)

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model_g = model_g.to(torch.float32)
    model_g.train()
    style_model.train()
    model_d.train()

    for param in model_g.parameters():
        param.requires_grad = False


    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    transform = torchaudio.transforms.Resample(44100, 32000).to('cuda')

    spkc_criterion = nn.CosineEmbeddingLoss()
    gradient_accumulation_steps = 1
    accumulation_counter = 0
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            if accumulation_counter == 0:
                saver.global_step_increment()
                optimizer.zero_grad()
                optimizer_d.zero_grad()
            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward

            if dtype == torch.float32:
                units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, f0_preded, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, vol_preded = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['volume'])

                ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(units, data['f0'] , data['volume'], data['spk_id'],
                                    aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, f0_preded, log_det_bkw, spk_pred, reflow_loss,  formant, z_ps, vol_preded = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['volume'])
                    ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel = model_g(units, data['f0'] , data['volume'], data['spk_id'],
                                        aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)


            spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.ones(spk_pred.size(0), device=args.device))
            transformed_fake = ddsp_wav
            transformed_real = data['gt_audio']

            # Run discriminator in fp16 to halve MRD STFT / conv tensor sizes.
            # Losses are cast to fp32 after the block so backward is stable.
            if dtype == torch.float32:
                disc_fake = model_d(transformed_fake)
                disc_real = model_d(transformed_real)

                score_loss = torch.zeros(1, device=args.device)
                for (_, score_fake) in disc_fake:
                    score_loss = score_loss + torch.mean(torch.pow(score_fake - 1.0, 2))
                score_loss = score_loss / len(disc_fake)

                feat_loss = torch.zeros(1, device=args.device)
                for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                    for fake, real in zip(feat_fake, feat_real):
                        feat_loss = feat_loss + torch.mean(torch.abs(fake - real.detach()))
                feat_loss = (feat_loss / len(disc_fake)) * 2


                loss_d = torch.zeros(1, device=args.device)
                for (_, sf), (_, sr) in zip(disc_fake, disc_real):
                    loss_d = loss_d + ((sr - 1.0) ** 2).mean() + (sf ** 2).mean()
                loss_d = loss_d / len(disc_fake)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    disc_fake = model_d(transformed_fake)
                    disc_real = model_d(transformed_real)

                    score_loss = torch.zeros(1, device=args.device)
                    for (_, score_fake) in disc_fake:
                        score_loss = score_loss + torch.mean(torch.pow(score_fake - 1.0, 2))
                    score_loss = score_loss / len(disc_fake)

                    feat_loss = torch.zeros(1, device=args.device)
                    for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                        for fake, real in zip(feat_fake, feat_real):
                            feat_loss = feat_loss + torch.mean(torch.abs(fake - real.detach()))
                    feat_loss = (feat_loss / len(disc_fake)) * 2


                    loss_d = torch.zeros(1, device=args.device)
                    for (_, sf), (_, sr) in zip(disc_fake, disc_real):
                        loss_d = loss_d + ((sr - 1.0) ** 2).mean() + (sf ** 2).mean()
                    loss_d = loss_d / len(disc_fake)

            # Cast to fp32 before backward — D params are fp32, no scaler needed.
            score_loss = score_loss.float()
            feat_loss = feat_loss.float()
            loss_d = loss_d.float()

            reconstruction_loss = F.l1_loss(data['units'], units)
          
            kl_loss = close_form_kl_loss(mu_pr, logvar_pr)
            # kl_loss = mod_kl_loss2(mu_ps, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + 0.5 * mod_kl_loss2(mu_pr, logvar_pr, mu_ps, logvar_ps, log_det_bkw)

            vol_target = data['volume']                    # (B, T) or (B, T, 1)
            if vol_target.dim() == 3:
                vol_target = vol_target.squeeze(-1)
            
            vol_pred = vol_preded.squeeze(-1)          # (B, T), > 0
            T_v = min(vol_pred.size(-1), vol_target.size(-1))
            vol_pred = vol_pred[..., :T_v]
            vol_target = vol_target[..., :T_v]
            eps = 1e-3
            log_vol_pred = torch.log(vol_pred + eps)
            log_vol_target = torch.log(torch.clamp(vol_target, min=0.0) + eps)
            loss_vol = (log_vol_pred - log_vol_target).abs().mean()
            f0_linear = data['f0'] 
            if f0_linear.dim() == 3:
                f0_linear = f0_linear.squeeze(-1)
            f0_pred = f0_preded.squeeze(-1)            # (B, T) in Hz, > 0
            voiced_mask = (vol_target > 0.0).float()        # voiced if > 10 Hz
            # Match lengths defensively
            T_f0 = min(f0_pred.size(-1), f0_linear.size(-1))
            f0_pred = f0_pred[..., :T_f0]
            f0_linear = f0_linear[..., :T_f0]
            voiced_mask = voiced_mask[..., :T_f0]
            log_f0_target = torch.log(torch.clamp(f0_linear, min=10.0))
            log_f0_pred = torch.log(torch.clamp(f0_pred, min=10.0))
            # Masked L1 in log space
            loss_f0 = ((log_f0_pred - log_f0_target).abs() * voiced_mask).sum() \
                    / (voiced_mask.sum() + 1e-6)

            loss = (3 * ddsp_loss) + reflow_loss + score_loss + feat_loss + reconstruction_loss + spk_loss * 2 + kl_loss + 0.5 * loss_f0 + 0.3 * loss_vol
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                del ddsp_loss
                # del mrsl_loss
                del kl_loss
                del reflow_loss
                del score_loss
                del feat_loss
                del ddsp_band_loss
                del reconstruction_loss
                # del spk_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                d_params = [p for p in model_d.parameters() if p.requires_grad]
                # Keep discriminator in FP32 - no scaling
                grads_D = torch.autograd.grad(
                    loss_d / gradient_accumulation_steps,
                    d_params,
                    retain_graph=True,        # keep graph for G grad pass
                    create_graph=False,
                    allow_unused=True
                )
                # assign & step
                for p, g in zip(d_params, grads_D):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                g_params = [p for p in style_model.parameters() if p.requires_grad]
                # Scale generator loss for AMP
                scaled_loss = scaler.scale(loss / gradient_accumulation_steps) if dtype != torch.float32 else (loss / gradient_accumulation_steps)
                grads_G = torch.autograd.grad(
                    scaled_loss,
                    g_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )
                for p, g in zip(g_params, grads_G):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                accumulation_counter += 1
                if accumulation_counter >= gradient_accumulation_steps:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model_d.parameters(), max_norm=1.0)
                    optimizer_d.step()
                    scheduler_d.step()

                    torch.nn.utils.clip_grad.clip_grad_norm_(style_model.parameters(), max_norm=1.0)
                    if dtype != torch.float32:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()

                    # (optional) early free grads
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_d.zero_grad(set_to_none=True)
                    accumulation_counter = 0
                # gc.collect()
                # torch.cuda.empty_cache()

            # Profiler CUDA memory usage for each batch
            # prof_mem = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=5)
            # saver.log_info(f'Batch {batch_idx} Profiler CUDA Memory:\n{prof_mem}')

            # log loss
            if saver.global_step % args.train.interval_log == 0 and (accumulation_counter == 0):
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/ddsp_band_loss': ddsp_band_loss.item(),
                    'train/loss_vol': loss_vol.item(),
                    'train/unit_rec_loss': reconstruction_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/loss_f0': loss_f0.item(),
                    # 'train/latent_angle_loss': latent_angle_loss.item(),
                    # 'train/perplexity': perplexity.item(),
                    # 'train/vq_loss': vq_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/score_loss': score_loss.item(),
                    'train/d_loss': loss_d.item(),
                    # 'train/loss_kl_f': loss_kl_f.item(),
                    # 'train/loss_kl_r': loss_kl_r.item(),
                    # 'train/transfer_reflow_loss': transfer_reflow_loss.item(),
                    # 'train/spk_loss': spk_loss,
                    'train/feat_loss': feat_loss,
                    # 'train/mrsl_loss': mrsl_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0 and( accumulation_counter == 0):
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_gan_model(style_model, model_d, optimizer_save, optimizer_d, postfix=f'{saver.global_step}')
                # saver.save_model(style_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_formant = test_style_vc_gan(args, style_model, model_g, vocoder, loader_test, saver, dtype, None, None, None)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + ddsp_band_loss + test_ddsp_msrl_loss)+test_spk_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    # 'validation/test_style_reflow_loss': test_style_reflow_loss,
                    'validation/test_rec_loss': test_rec_loss,
                    'validation/avg_formant': test_formant,
                    'validation/test_spk_loss': test_spk_loss
                })
                
                model_g.train()
                model_d.train()
                style_model.train()
                
                
def train_vc_osgan3(args, initial_global_step, style_model, model_g, model_d, optimizer, optimizer_d, scheduler, scheduler_d, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    style_count = utils.get_network_paras_amount({'style':style_model})
    saver.log_info('--- model size ---')
    saver.log_info(style_count)
    
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model_g = model_g.to(torch.float32)
    model_g.train()
    style_model.train()
    model_d.train()
    
    for param in model_g.parameters():
        param.requires_grad = False
    

    saver.log_info('======= start training =======')
    # scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    # transform = torchaudio.transforms.Resample(44100, 32000).to('cuda')
    spkc_criterion = nn.CosineEmbeddingLoss()
    spkc_latent = nn.CosineEmbeddingLoss()
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()
            optimizer_d.zero_grad()
            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward
            
            # units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, formant, style_flow_loss, z_mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['volume'])
            units, z_f, z_r, z_pr, mu_pr, logvar_pr, z_ps, mu_ps, logvar_ps, logdet_f, logdet_r, formant, spk_pred = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'],  data['volume'])
            ddsp_loss, reflow_loss, ddsp_band_loss , ddsp_wav, ddsp_mel = model_g(units, data['f0'] , data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            
            spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.ones(spk_pred.size(0), device=args.device))

            latent_loss =  1 * spkc_criterion(z_r, z_ps, torch.ones(spk_pred.size(0), device=args.device))     
            disc_fake = model_d(units.transpose(1, 2))
            disc_real = model_d(data['units'].transpose(1, 2))
            
            score_loss = torch.zeros(1, device=args.device)
            for (_, score_fake) in disc_fake:
                score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
            score_loss = score_loss / len(disc_fake)

            feat_loss = torch.zeros(1, device=args.device)
            for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                for fake, real in zip(feat_fake, feat_real):
                    # print(torch.mean(torch.abs(real)))
                    feat_loss += torch.mean(torch.abs(fake - real))
            feat_loss = feat_loss / len(disc_fake)
            feat_loss = feat_loss * 2
                        
            loss_d = torch.zeros(1, device=args.device)
            for (_, sf), (_, sr) in zip(disc_fake, disc_real):
                loss_d = loss_d + ((sr - 1.0) ** 2).mean() + (sf ** 2).mean()
            loss_d = loss_d / len(disc_fake)
            reconstruction_loss_map = nn.functional.l1_loss(units, data['units'], reduction='none')
            reconstruction_loss = reconstruction_loss_map.mean() + focus_loss(reconstruction_loss_map, [1.0], [1.0])
            loss_kl_f = vits_kl_loss2(z_f.transpose(1, 2), logvar_ps, mu_pr, logvar_pr, logdet_f)
            loss_kl_r = vits_kl_loss2(z_r.transpose(1, 2), logvar_pr, mu_ps, logvar_ps, logdet_r)
            kl_loss = loss_kl_f + loss_kl_r * 0.5
            # kl_loss = mod_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + 0.5 * mod_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw)
            loss = (3 * ddsp_loss) + reflow_loss + score_loss + reconstruction_loss + feat_loss  + kl_loss + spk_loss + latent_loss
           
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                del ddsp_loss
                del kl_loss
                del reflow_loss
                del score_loss
                del feat_loss
                del ddsp_band_loss
                del reconstruction_loss
                del spk_loss
                continue
            # elif torch.isnan(reflow_loss):
            #     raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                optimizer_d.zero_grad(set_to_none=True)
                d_params = [p for p in model_d.parameters() if p.requires_grad]
                grads_D = torch.autograd.grad(
                    loss_d,
                    d_params,
                    retain_graph=True,        # keep graph for G grad pass
                    create_graph=False,
                    allow_unused=True
                )
                # assign & step
                for p, g in zip(d_params, grads_D):
                    if g is not None:
                        p.grad = g
                torch.nn.utils.clip_grad.clip_grad_norm_(model_d.parameters(), max_norm=1.0)
                optimizer_d.step()
                scheduler_d.step()
                
                optimizer.zero_grad(set_to_none=True)
                g_params = list(style_model.parameters())
                grads_G = torch.autograd.grad(
                    loss,
                    g_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )
                for p, g in zip(g_params, grads_G):
                    if g is not None:
                        p.grad = g
                torch.nn.utils.clip_grad.clip_grad_norm_(style_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # (optional) early free grads
                optimizer.zero_grad(set_to_none=True)
                optimizer_d.zero_grad(set_to_none=True)
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    # 'train/ddsp_band_loss': ddsp_band_loss.item(),
                    # 'train/style_flow_loss': style_flow_loss.item(),
                    'train/unit_rec_loss': reconstruction_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    # 'train/loss_pattern': loss_pattern.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/score_loss': score_loss,
                    'train/d_loss': loss_d,
                    # 'train/spk_loss': spk_loss,
                    'train/feat_loss': feat_loss,
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_gan_model(style_model, model_d, optimizer_save, optimizer_d, postfix=f'{saver.global_step}')
                # saver.save_model(style_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_formant = test_style_vc_gan(args, style_model, model_g, vocoder, loader_test, saver, dtype, None, None, None)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + test_ddsp_msrl_loss)+test_spk_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    # 'validation/test_style_reflow_loss': test_style_reflow_loss,
                    'validation/test_rec_loss': test_rec_loss,
                    'validation/avg_formant': test_formant,
                    'validation/test_spk_loss': test_spk_loss
                })
                
                model_g.train()
                model_d.train()
                style_model.train()
                
                
def train_vc_no_gan(args, initial_global_step, lag_model, style_model, model_g, optimizer, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])


    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    style_count = utils.get_network_paras_amount({'style':style_model})
    saver.log_info('--- model size ---')
    saver.log_info(style_count)
    
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model_g.train()
    style_model.train()
    
    for param in model_g.parameters():
        param.requires_grad = False
        
    for param in lag_model.parameters():
        param.requires_grad = False
    

    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    # transform = torchaudio.transforms.Resample(44100, 32000, dtype=dtype).to('cuda')
    spkc_criterion = nn.CosineEmbeddingLoss()

    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward
            if dtype == torch.float32:
                units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, log_det_fwd, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel_high_res'], data['f0'], None, saver.global_step)
            with autocast(device_type=args.device, dtype=dtype):
                units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, log_det_fwd, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel_high_res'], data['f0'], None, saver.global_step)
            lag_loss = lag_model.loss(units.float())
            spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.Tensor(spk_pred.size(0)).to(args.device).fill_(1.0))
            
            reconstruction_loss = torch.clamp(nn.functional.l1_loss(units, data['units']),min=0., max=80)
            
            kl_loss = vits_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + vits_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw) *0.5
            loss = (0.2 * kl_loss)  + 2*spk_loss + reconstruction_loss + lag_loss
          
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                del kl_loss
                del reconstruction_loss
                del spk_loss
                continue
            # elif torch.isnan(reflow_loss):
            #     raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=style_model.parameters(), max_norm=1)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=style_model.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    # 'train/ddsp_loss': ddsp_loss.item(),
                    # 'train/ddsp_band_loss': ddsp_band_loss.item(),
                    # 'train/reflow_style_loss': reflow_style_loss.item(),
                    'train/unit_rec_loss': reconstruction_loss.item(),
                    # 'train/reflow_loss': reflow_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    # 'train/centroid_loss': centroid_loss.item(),
                    'train/spk_loss': spk_loss,
                    # 'train/mrsl_loss': mrsl_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(style_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_formant= test_style_vc_gan(args, style_model, model_g, vocoder, loader_test, saver, dtype, None, None, None)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + test_ddsp_msrl_loss)+test_spk_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    # 'validation/test_style_reflow_loss': test_style_reflow_loss,
                    'validation/test_rec_loss': test_rec_loss,
                    'validation/avg_formant': test_formant,
                    'validation/test_spk_loss': test_spk_loss
                })
                
                model_g.train()
                # model_d.train()
                style_model.train()
                
                
                
                
def train_vc_reflow(args, initial_global_step, style_model, reflow_model, model_g, optimizer_reflow, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step, train_style_reflow=True)
    # mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- G model size ---')
    saver.log_info(params_count)
    
    style_count = utils.get_network_paras_amount({'style':style_model})
    saver.log_info('--- style model size ---')
    saver.log_info(style_count)
    
    flow_count = utils.get_network_paras_amount({'style':reflow_model})
    saver.log_info('--- Flow model size ---')
    saver.log_info(flow_count)

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    reflow_model.train()
    style_model.eval()

    reconsturct_criterion = nn.CosineEmbeddingLoss()

    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)

    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer_reflow.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward
            if dtype == torch.float32:
                with torch.no_grad():
                    units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant,  log_det_fwd, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  None, data['f0'], data['volume'],0 , infer=True)
                    # latent, _ = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], None, data['f0'], None, infer=True, noise_fac=0.0)
                # masked_z, _ = mask_keep_a_mask_b(z_fwd, random.randint(5, 20), random.randint(0, 15))
                
                style_reflow_loss, reconstruction = reflow_model(random_piecewise_time_warp(units.detach()), gt_spec=data['units'], t_start=0.0, infer=False)
                # generated_angle = reconstruction.squeeze(1).reshape(-1, reconstruction.size(2))
                # gt_angle = data['units'].reshape(-1, reconstruction.size(2))
                    
                # reconstruction_loss = reconsturct_criterion(generated_angle, gt_angle, torch.ones(generated_angle.size(0), device=args.device))
            else:
                with torch.no_grad():
                    # units, _, _, _, _, _, z_fwd, _, _, _, formant, _, residual_cnt, f0_hat_hz, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  data['mel'], data['f0'], data['volume'], infer=True, noise_fac=0.0)
                    units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant,  log_det_fwd, z_ps, mask = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], None, data['f0'], data['volume'],0 , infer=True)
                    # latent = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  data['mel'], data['f0'], data['volume'], infer=True, noise_fac=0.0, result_only=True)
                with autocast(device_type=args.device, dtype=dtype):
                    # masked_z, _ = mask_keep_a_mask_b(z_fwd, random.randint(5, 20), random.randint(0, 15))
                    style_reflow_loss, reconstruction = reflow_model(random_piecewise_time_warp(units.detach()), gt_spec=data['units'], t_start=0.0, infer=False)
                    
                # generated_angle = reconstruction.squeeze(1).reshape(-1, reconstruction.size(2)).float()
                # gt_angle = data['units'].reshape(-1, reconstruction.size(2)).float()
                # print(generated_angle.shape, gt_angle.shape)
                # reconstruction_loss = reconsturct_criterion(generated_angle, gt_angle, torch.ones(generated_angle.size(0), device=args.device))
                # norm_loss = F.l1_loss(torch.norm(reconstruction.squeeze(1), dim=2), torch.norm( data['units'].transpose(1, 2), dim=2))
            
            loss = (style_reflow_loss).float()
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer_reflow.zero_grad()
                del style_reflow_loss
                continue
            elif torch.isnan(style_reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=reflow_model.parameters(), max_norm=1)
                    optimizer_reflow.step()
                else:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=reflow_model.parameters(), max_norm=1)
                    scaler.step(optimizer_reflow)
                    scaler.update()
                    
                scheduler.step()
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer_reflow.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/style_reflow_loss': style_reflow_loss.item(),
                    # 'train/recon': reconstruction_loss.mean().item(),
                    # 'train/norm': norm_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer_reflow if args.train.save_opt else None
                
                # save latest
                saver.save_model(reflow_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_style_reflow_loss = test_style_vc_flow(args, style_model, model_g, reflow_model, vocoder, loader_test, saver)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + test_ddsp_band_loss + test_ddsp_msrl_loss)+test_spk_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    'validation/test_style_reflow_loss': test_style_reflow_loss,
                    'validation/test_rec_loss': test_rec_loss,
                    'validation/test_spk_loss': test_spk_loss
                })
                
                reflow_model.train()


def train_vc_reflow2(args, initial_global_step, reflow_model, model_g, optimizer_reflow, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step, train_style_reflow=True)
    # mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- G model size ---')
    saver.log_info(params_count)
    spkc_criterion = nn.CosineEmbeddingLoss()
    
    # style_count = utils.get_network_paras_amount({'style':style_model})
    # saver.log_info('--- style model size ---')
    # saver.log_info(style_count)
    
    flow_count = utils.get_network_paras_amount({'style':reflow_model})
    saver.log_info('--- Flow model size ---')
    saver.log_info(flow_count)

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    reflow_model.train()

    

    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)

    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer_reflow.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # with torch.no_grad():
            #     mask = (data['volume'] > 10 ** (float(-60) / 20))
            #     mask = mask.float().unsqueeze(-1).unsqueeze(0)
            #     mask = upsample(mask, args.data.block_size).squeeze(-1)
            # forward
            # if dtype == torch.float32:
            #     _, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss= reflow_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], t_start=0.0, infer=False)
            # else:
            #     with autocast(device_type=args.device, dtype=dtype):
            #        _, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, reflow_loss = reflow_model(data['units'], data['units_w'],  data['units_h'], data['spk_embd'], data['mel'], t_start=0.0, infer=False)
            if dtype == torch.float32:
                unit, z_f, z_r, z_pr, mu_pr, logvar_pr, z_ps, mu_ps, logvar_ps, logdet_f, logdet_r, spk_pred, reflow_loss, speaker_feat = reflow_model(data['units'], data['units_w'], data['units_h'], data['spk_id'],  data['mel'], data['f0'], data['volume'])                
                # reconstruction_loss = torch.clamp(nn.functional.l1_loss(units, data['units']),min=0., max=80)
                # spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.ones(spk_pred.size(0), device=args.device))
                # loss_kl_f = vits_kl_loss2(z_f.transpose(1, 2), logvar_ps, mu_pr, logvar_pr, logdet_f)
                # loss_kl_f = vits_kl_loss(z_f.transpose(1, 2), logvar_ps, mu_pr, logvar_pr, logdet_f)
                # loss_kl_r = vits_kl_loss2(z_r.transpose(1, 2), logvar_pr, mu_ps, logvar_ps, logdet_r)
                # loss_kl_r = vits_kl_loss(z_r.transpose(1, 2), logvar_pr, mu_ps, logvar_ps, logdet_r)

                # kl_loss = loss_kl_f + loss_kl_r * 0.5

                # kl_loss = mod_kl_loss1_stable(z_fwd.transpose(1, 2) , logvar_pr.transpose(1, 2), mu_pr.transpose(1, 2),  logvar_ps.transpose(1, 2), log_det_fwd, mask.expand_as(z_fwd.transpose(1, 2))) + 0.5 * mod_kl_loss1_stable(z_bkw.transpose(1, 2), logvar_ps.transpose(1, 2), mu_ps.transpose(1, 2), logvar_pr.transpose(1, 2), log_det_bkw, mask.expand_as(z_fwd.transpose(1, 2)), False)
        
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    unit, z_f, z_r, z_pr, mu_pr, logvar_pr, z_ps, mu_ps, logvar_ps, logdet_f, logdet_r, spk_pred, reflow_loss, speaker_feat = reflow_model(data['units'], data['units_w'], data['units_h'], data['spk_id'],  data['mel'], data['f0'], data['volume'])
                    # reconstruction_loss = torch.clamp(nn.functional.l1_loss(units, data['units']),min=0., max=80)
                    # print(speaker_feat.shape, spk_pred.shape)
                    # print(speaker_feat.mean(), spk_pred.mean())
                    # spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.ones(spk_pred.size(0), device=args.device))
                    # loss_kl_f = vits_kl_loss2(z_f.transpose(1, 2), logvar_ps, mu_pr, logvar_pr, logdet_f)
                    # loss_kl_r = vits_kl_loss2(z_r.transpose(1, 2), logvar_pr, mu_ps, logvar_ps, logdet_r)
                    
                    # loss_kl_f = vits_kl_loss(z_f.transpose(1, 2), logvar_ps, mu_pr, logvar_pr, logdet_f)
                    # loss_kl_r = vits_kl_loss(z_r.transpose(1, 2), logvar_pr, mu_ps, logvar_ps, logdet_r)
                    # kl_loss = loss_kl_f + loss_kl_r * 0.5
                    # kl_loss = mod_kl_loss1_stable(z_fwd.transpose(1, 2) , logvar_pr.transpose(1, 2), mu_pr.transpose(1, 2),  logvar_ps.transpose(1, 2), log_det_fwd, mask.expand_as(z_fwd.transpose(1, 2))) + 0.5 * mod_kl_loss1_stable(z_bkw.transpose(1, 2), logvar_ps.transpose(1, 2), mu_ps.transpose(1, 2), logvar_pr.transpose(1, 2), log_det_bkw, mask.expand_as(z_fwd.transpose(1, 2)), False)
            # kl_loss = mod_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + 0.5 * mod_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw)
            # loss = 10 * reflow_loss + spk_loss + 0.1 * kl_loss
            # loss = 10 * reflow_loss + spk_loss
            # print(logdet_f.mean().item(), logdet_r.mean().item())
            # print(kl_loss)
            loss = (reflow_loss).float()
            # loss = loss.float()
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer_reflow.zero_grad(set_to_none=True)
                del loss
                del reflow_loss
                # del spk_loss
                del kl_loss
                del spk_pred
                del mu_pr
                del logvar_pr
                del mu_ps
                del logvar_ps
                # del z_fwd
                # del log_det_fwd
                # del z_bkw
                # del log_det_bkw
                del unit
                torch.cuda.empty_cache()
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=reflow_model.parameters(), max_norm=1)
                    optimizer_reflow.step()
                else:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=reflow_model.parameters(), max_norm=1)
                    scaler.step(optimizer_reflow)
                    scaler.update()
                    
                scheduler.step()
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer_reflow.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    # 'train/kl_loss': kl_loss.item(),
                    'train/style_reflow_loss': reflow_loss.item(),
                    # 'train/reconstruction_loss': reconstruction_loss.item(),
                    # 'train/spk_loss': spk_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer_reflow if args.train.save_opt else None
                
                # save latest
                saver.save_model(reflow_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_style_reflow_loss = test_style_vc_flow2(args, model_g, reflow_model, vocoder, loader_test, saver)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + test_ddsp_band_loss + test_ddsp_msrl_loss)+test_spk_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    'validation/test_style_reflow_loss': test_style_reflow_loss,
                    'validation/test_rec_loss': test_rec_loss,
                    'validation/test_spk_loss': test_spk_loss
                })
                
                reflow_model.train()


def train_post_processor(args, initial_global_step, reflow_model, model_g, optimizer_reflow, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step, train_enhance=True)
    # mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)])

    # model size
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- G model size ---')
    saver.log_info(params_count)
    
    flow_count = utils.get_network_paras_amount({'style':reflow_model})
    saver.log_info('--- Flow model size ---')
    saver.log_info(flow_count)

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    reflow_model.train()

    

    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)

    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer_reflow.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            # forward
            if dtype == torch.float32:
                with torch.no_grad():
                    mel = model_g(
                    data['units'], 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    use_tqdm=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start
                    )
                style_reflow_loss = reflow_model(mel, gt_spec=data['mel'], t_start=0.0, infer=False)
            else:
                with torch.no_grad():
                    mel = model_g(
                        data['units'],
                        data['f0'], 
                        data['volume'], 
                        data['spk_id'],
                        vocoder=vocoder,
                        infer=True,
                        return_wav=False,
                        use_tqdm=False,
                        infer_step=args.infer.infer_step, 
                        method=args.infer.method,
                        t_start=args.model.t_start
                    )
                with autocast(device_type=args.device, dtype=dtype):
                    style_reflow_loss = reflow_model(mel, gt_spec=data['mel'], t_start=0.0, infer=False)
            loss = style_reflow_loss.float()
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer_reflow.zero_grad()
                del style_reflow_loss
                continue
            elif torch.isnan(style_reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=reflow_model.parameters(), max_norm=1)
                    optimizer_reflow.step()
                else:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(parameters=reflow_model.parameters(), max_norm=1)
                    scaler.step(optimizer_reflow)
                    scaler.update()
                    
                scheduler.step()
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer_reflow.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer_reflow if args.train.save_opt else None
                
                # save latest
                saver.save_model(reflow_model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_style_reflow_loss = test_post_processor(args, model_g, reflow_model, vocoder, loader_test, saver)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + test_ddsp_band_loss + test_ddsp_msrl_loss)+test_spk_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss,
                    'validation/test_ddsp_band_loss': test_ddsp_band_loss,
                    'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss,
                    'validation/test_style_reflow_loss': test_style_reflow_loss,
                    'validation/test_rec_loss': test_rec_loss,
                    'validation/test_spk_loss': test_spk_loss
                })
                
                reflow_model.train()



def test_style_vc_flow2(args, model, reflow_model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()
    # style_model.eval()
    reflow_model.eval()
    mrsl = MultiResolutionSTFTLoss('cuda', [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])
    spkc_criterion = nn.CosineEmbeddingLoss()

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_ddsp_band_loss = 0.
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.
    test_spk_loss = 0. 
    test_style_reflow_loss = 0.
    # test_gaussian_loss = 0.0

    # mel mse val
    mel_val_L1_all = 0
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            # _, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred = style_model(data['units'].float(), data['units_w'].float(), data['units_h'].float(), data['spk_embd'].float(), data['mel'].float())
            # unit_out, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, log_det_fwd, z_bkw, log_det_bkw, spk_pred, _ = reflow_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], t_start=0.0, infer=True)
            _, _, _, _, _, _, _, _, _, _, _, spk_pred, reflow_loss, speaker_feat = reflow_model(data['units'], data['units_w'], data['units_h'], data['spk_id'],  data['mel'], data['f0'], data['volume'])
            unit_out, _, _, _, _, _, _, _, _, _, _, _, _, _ = reflow_model(data['units'], data['units_w'], data['units_h'], data['spk_id'],  data['mel'], data['f0'], data['volume'], infer=True)
            mel = model(
                    unit_out,
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
            
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            # unit, mu, logvar, spk_pred = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'],  data['f0'])
            ddsp_loss, reflow_loss, _, ddsp_wav, ddsp_mel = model(
                unit_out, 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            # test_gaussian_loss += gaussian_deviation_loss(unit, unit_mean, unit_var).item()
            # loss_kl_f = kl_loss_new(z_f, logs_q, m_p, logs_p, logdet_f, spec_mask)
            # loss_kl_r = kl_loss_new(z_r, logs_p, m_q, logs_q, logdet_r, spec_mask)
            # ddsp_band_loss =  mod_kl_loss1_stable(z_fwd.transpose(1, 2) , 0.5*logvar_pr.transpose(1, 2), mu_pr.transpose(1, 2),  0.5*logvar_ps.transpose(1, 2), log_det_fwd, mask.expand_as(z_fwd.transpose(1, 2))) + 0.5 * mod_kl_loss1_stable(z_bkw.transpose(1, 2),  0.5*logvar_ps.transpose(1, 2), mu_ps.transpose(1, 2),  0.5*logvar_pr.transpose(1, 2), log_det_bkw, mask.expand_as(z_fwd.transpose(1, 2)), False)
            # ddsp_band_loss = mod_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + 0.5 * mod_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw)
            # print(speaker_feat.shape, spk_pred.shape)
            # test_spk_loss += 2 * spkc_criterion(speaker_feat.squeeze(1), spk_pred, torch.Tensor(spk_pred.size(0)).to(args.device).fill_(1.0)).item()
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            # test_ddsp_band_loss += 0.2* ddsp_band_loss.item()
            test_rec_loss += nn.functional.mse_loss(unit_out, data['units']).item()
            test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            # test_style_reflow_loss += reflow_loss.item()
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            WAV2MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
                        fmax=22050,
                        clip_val=1e-5)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            # 如果形状不同,裁剪使得形状相同
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            mel_val_L1_all +=  torch.nn.functional.l1_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    # test_gaussian_loss /=num_batches
    test_rec_loss /= num_batches
    test_spk_loss /= num_batches
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_ddsp_msrl_loss /=num_batches
    test_ddsp_band_loss /= num_batches
    test_style_reflow_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_L1_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' [test_rec_loss] test_rec_loss:', test_rec_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    print(' Mel Val L1', mel_val_L1_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    saver.log_value({
        'validation/mel_val_L1': mel_val_L1_all
    })
    return test_ddsp_loss, test_reflow_loss,test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_style_reflow_loss
                
