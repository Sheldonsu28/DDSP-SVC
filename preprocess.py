import os
import numpy as np
import random
import librosa
import torch
import argparse
import shutil
from speechbrain.inference.speaker import EncoderClassifier
from funasr import AutoModel
from logger import utils
from tqdm import tqdm
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from reflow.vocoder import HghResMel, Vocoder, load_style_model
from logger.utils import traverse_dir
from utils import compute_spec, spectrogram_torch

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    return parser.parse_args(args=args, namespace=namespace)
    
def preprocess(path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, high_res, device = 'cuda', use_pitch_aug = False, extensions = ['wav'], encoder2=None, encoder3=None, emo_encoder=None, emo_model=None):
    
    path_srcdir  = os.path.join(path, 'audio')
    path_unitsdir  = os.path.join(path, 'units')
    path_whisper_unit_dir = os.path.join(path, 'whisper_units')
    path_hubert_unit_dir = os.path.join(path, 'hubert_units')
    path_augunitsdir  = os.path.join(path, 'units_aug')
    path_aug_whisper_unit_dir = os.path.join(path, 'whisper_units_aug')
    path_aug_hubert_unit_dir = os.path.join(path, 'hubert_units_aug')
    path_f0dir  = os.path.join(path, 'f0')
    path_volumedir  = os.path.join(path, 'volume')
    path_augvoldir  = os.path.join(path, 'aug_vol')
    path_meldir  = os.path.join(path, 'mel')
    path_augmeldir  = os.path.join(path, 'aug_mel')
    path_highresmeldir  = os.path.join(path, 'mel_high_res')
    path_aughighresmeldir  = os.path.join(path, 'aug_mel_high_res')
    path_skipdir = os.path.join(path, 'skip')
    path_speaker = os.path.join(path, 'speaker')
    path_emo = os.path.join(path, 'emo')
    
    # list files
    filelist =  traverse_dir(
        path_srcdir,
        extensions=extensions,
        is_pure=True,
        is_sort=True,
        is_ext=True)
    
    # pitch augmentation dictionary
    pitch_aug_dict = {}
    
    # def compute_energy(file_list):
    #     for file in file_list:
    # e_mean, e_std = compute_all_energy_std_mean(1)
    
    stylize_seg = False
    style_model = None
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
    # run  
    def process(file):
        binfile = file+'.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_wunitsfil = os.path.join(path_whisper_unit_dir, binfile)
        path_hunitsfil = os.path.join(path_hubert_unit_dir, binfile)
        path_augunitsfile = os.path.join(path_augunitsdir, binfile)
        path_augwunitsfil = os.path.join(path_aug_whisper_unit_dir, binfile)
        path_aughunitsfil = os.path.join(path_aug_hubert_unit_dir, binfile)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)
        path_augvolfile = os.path.join(path_augvoldir, binfile)
        path_melfile = os.path.join(path_meldir, binfile)
        path_augmelfile = os.path.join(path_augmeldir, binfile)
        path_highresmelfile = os.path.join(path_highresmeldir, binfile)
        path_aughighresmelfile = os.path.join(path_aughighresmeldir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)
        path_speakerfile = os.path.join(path_speaker, file)
        path_emofile = os.path.join(path_emo, file)
        
        # load audio
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)
        
        embeddings:torch.Tensor = classifier.encode_batch(torch.from_numpy(audio)).squeeze().to('cpu').numpy()
        
        # energy_t = (torch.log(extract_energy(audio) + 1e-5) - e_mean) / (e_std + 1e-6)
        # energy = energy_t.to('cpu').numpy()
        
        # extract volume
        volume = volume_extractor.extract(audio)

        # draw the augmentation parameters (shared by mel and unit augmentation)
        max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        if use_pitch_aug:
            keyshift = random.uniform(-5, 5)
        else:
            keyshift = 0

        # extract mel and volume augmentaion
        if mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to('cpu').numpy()

            aug_mel_t = mel_extractor.extract(audio_t * (10 ** log10_vol_shift), sample_rate, keyshift = keyshift)
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
            aug_vol = volume_extractor.extract(audio * (10 ** log10_vol_shift))

            # extract high-resolution mel (clean + volume/pitch augmented)
            high_res_mel_t = high_res.extract(audio_t, keyshift = 0)
            high_res_mel = high_res_mel_t.squeeze().to('cpu').numpy()
            aug_high_res_mel_t = high_res.extract(audio_t * (10 ** log10_vol_shift), keyshift = keyshift)
            aug_high_res_mel = aug_high_res_mel_t.squeeze().to('cpu').numpy()
            
        # units encode
        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to('cpu').numpy()

        units2_t = None
        units2 = None
        if encoder2 is not None:
            units2_t = encoder2.encode(audio_t, sample_rate, hop_size)
            units2 = units2_t.squeeze().to('cpu').numpy()


        units3_t = None
        units3 = None
        if encoder3 is not None:
            units3_t = encoder3.encode(audio_t, sample_rate, hop_size)
            units3 = units3_t.squeeze().to('cpu').numpy()

        # pitch-augmented units encode
        # the pitch is shifted by `keyshift` semitones while the number of samples
        # (and therefore the speech rate / frame alignment) is left untouched, so the
        # augmented units stay frame-aligned with the clean ones.
        if keyshift != 0:
            audio_aug = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=keyshift)
            audio_aug = audio_aug[:len(audio)]
            if len(audio_aug) < len(audio):
                audio_aug = np.pad(audio_aug, (0, len(audio) - len(audio_aug)))
            audio_aug_t = torch.from_numpy(audio_aug).float().to(device).unsqueeze(0)

            units_aug = units_encoder.encode(audio_aug_t, sample_rate, hop_size).squeeze().to('cpu').numpy()

            units2_aug = None
            if encoder2 is not None:
                units2_aug = encoder2.encode(audio_aug_t, sample_rate, hop_size).squeeze().to('cpu').numpy()

            units3_aug = None
            if encoder3 is not None:
                units3_aug = encoder3.encode(audio_aug_t, sample_rate, hop_size).squeeze().to('cpu').numpy()
        else:
            # no pitch augmentation: the augmented units are identical to the clean ones
            units_aug = units
            units2_aug = units2
            units3_aug = units3


        emo_t = None
        emo = None
        if emo_encoder is not None:
            emo_t = emo_encoder.encode_emo(path_srcfile, emo_model, units3.shape[0])
            emo = emo_t.squeeze().to('cpu').numpy()      
       
        # if stylize_seg:
        #     units_t = style_model.infer(units_t, units2_t, units3_t)
        #     units = units_t.squeeze().to('cpu').numpy()
        #     print(units.shape, units2.shape)
            
        
        # extract f0
        f0 = f0_extractor.extract(audio, uv_interp = False)
        
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            
            # save npy     
            # os.makedirs(os.path.dirname(path_energyfile), exist_ok=True)
            # np.save(path_energyfile, energy)

            # save npy     
            os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
            np.save(path_unitsfile, units)
            
            os.makedirs(os.path.dirname(path_wunitsfil), exist_ok=True)
            np.save(path_wunitsfil, units2)
            
            os.makedirs(os.path.dirname(path_hunitsfil), exist_ok=True)
            np.save(path_hunitsfil, units3)

            os.makedirs(os.path.dirname(path_augunitsfile), exist_ok=True)
            np.save(path_augunitsfile, units_aug)

            os.makedirs(os.path.dirname(path_augwunitsfil), exist_ok=True)
            np.save(path_augwunitsfil, units2_aug)

            os.makedirs(os.path.dirname(path_aughunitsfil), exist_ok=True)
            np.save(path_aughunitsfil, units3_aug)

            os.makedirs(os.path.dirname(path_emofile), exist_ok=True)
            np.save(path_emofile, emo)
            
            os.makedirs(os.path.dirname(path_speakerfile), exist_ok=True)
            np.save(path_speakerfile, embeddings)
            
            os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
            np.save(path_f0file, f0)
            os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
            np.save(path_volumefile, volume)
            if mel_extractor is not None:
                pitch_aug_dict[file] = keyshift
                os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
                np.save(path_melfile, mel)
                os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
                np.save(path_augmelfile, aug_mel)
                os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
                np.save(path_augvolfile, aug_vol)
                os.makedirs(os.path.dirname(path_highresmelfile), exist_ok=True)
                np.save(path_highresmelfile, high_res_mel)
                os.makedirs(os.path.dirname(path_aughighresmelfile), exist_ok=True)
                np.save(path_aughighresmelfile, aug_high_res_mel)
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            print('This file has been moved to ' + path_skipfile)
    print('Preprocess the audio clips in :', path_srcdir)
    
    # single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
    
    if mel_extractor is not None:
        path_pitchaugdict = os.path.join(path, 'pitch_aug_dict.npy')
        np.save(path_pitchaugdict, pitch_aug_dict)
    
    # multi-process (have bugs)
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    '''
if __name__ == '__main__':
    model_id = "iic/emotion2vec_plus_large"
    model = AutoModel(
        model=model_id,
        hub="huggingface",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    )
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load config
    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    
    extensions = args.data.extensions
    
    # initialize f0 extractor
    f0_extractor = F0_Extractor(
                        args.data.f0_extractor, 
                        args.data.sampling_rate, 
                        args.data.block_size, 
                        args.data.f0_min, 
                        args.data.f0_max)
    
    # initialize volume extractor
    volume_extractor = Volume_Extractor(args.data.block_size, args.data.volume_smooth_size)
    
    # initialize mel extractor
    mel_extractor = None
    use_pitch_aug = False
    mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device = device)
    mel_extractor_high_res = Vocoder('high_res', args.vocoder.ckpt, device = device)
    if mel_extractor.vocoder_sample_rate != sample_rate or mel_extractor.vocoder_hop_size != hop_size:
        mel_extractor = None
        print('Unmatch vocoder parameters, mel extraction is ignored!')
    elif args.model.use_pitch_aug:
        use_pitch_aug = True
    
    # initialize units encoder
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

    units_encoder2 = Units_Encoder(
                    'whisper-large-pgg-tta2x', 
                    args.data.encoder_ckpt, 
                    args.data.encoder_sample_rate, 
                    args.data.encoder_hop_size,
                    cnhubertsoft_gate=cnhubertsoft_gate,
                    device = device)
    
    units_encoder3 = Units_Encoder(
                    'hubertsofttta2x', 
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
    
    # preprocess training set
    preprocess(args.data.train_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, mel_extractor_high_res, device = device, use_pitch_aug = use_pitch_aug, extensions = extensions, encoder2=units_encoder2, encoder3=units_encoder3, emo_encoder=None, emo_model= model)
    
    # preprocess validation set
    preprocess(args.data.valid_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, mel_extractor_high_res,  device = device, use_pitch_aug = False, extensions = extensions, encoder2=units_encoder2, encoder3=units_encoder3, emo_encoder=None, emo_model= model)
    
