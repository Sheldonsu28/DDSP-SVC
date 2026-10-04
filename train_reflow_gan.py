import os
import argparse
import torch
from torch.optim import lr_scheduler
from discriminators.discriminator import Discriminator
from optimizer.muon import Muon_AdamW
from logger import utils
from reflow.data_loaders import get_data_loaders
from reflow.solver import train_reflow_gan
from reflow.vocoder import Vocoder, Unit2Wav
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # load model
    if args.model.type == 'RectifiedFlow':
        model_g = Unit2Wav(
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
        
        model_d = Discriminator({})
                    
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model_g.to(args.device)
    
    # load parameters
    optimizer = Muon_AdamW(model_g, 
                    muon_args={'weight_decay': args.train.weight_decay}, 
                    adamw_args={'weight_decay': 0})
    optimizer_d = Muon_AdamW(model_d, 
                    muon_args={'weight_decay': args.train.weight_decay}, 
                    adamw_args={'weight_decay': 0})
    initial_global_step, model_g, optimizer, model_d, optimizer_d = utils.load_gan_model(args.env.expdir, model_g, optimizer, model_d, optimizer_d, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        
    for param_group in optimizer_d.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        
    scheduler_g = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    scheduler_d = lr_scheduler.StepLR(optimizer_d, step_size=args.train.decay_step, gamma=args.train.gamma,  last_epoch=initial_global_step-2)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hp.train.lr_decay, last_epoch=init_epoch-2)
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False, load_audio=True)
    
    # run
    train_reflow_gan(args, initial_global_step, model_g, model_d, optimizer, optimizer_d, scheduler_g, scheduler_d, vocoder, loader_train, loader_valid)
    
