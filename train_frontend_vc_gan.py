import os
import argparse
import random
import numpy as np
import torch
from torch.optim import lr_scheduler
from discriminators.discriminator import Discriminator
from logger import utils
from optimizer.muon import Muon_AdamW
from reflow.data_loaders import get_data_loaders
from reflow.solver import train_vc_gan, train_vc_no_gan, train_vc_osgan, train_vc_osgan2
from reflow.vocoder import Vocoder, Unit2Wav
from stylizer.stlyizer_gen2 import LeakageReductionFrontend
from stylizer.stylizer import  GlowVcStylizerWN5, GlowVcStylizerWN5Mod3, GlowVcStylizerWN5Mod4, GlowVcStylizerWN5Mod6, GlowVcStylizerWN5Mod7, GlowVcStylizerWN5Mod8, GlowVcStylizerWN5Mod9
from test10 import DiagonalLagrangian
# from stylizer.enhancers import MelPostNetSnake
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


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

# def set_seed(seed=0):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     # Recommended for true determinism
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == '__main__':
    # set_seed()
    
    # model = MelPostNetSnake()
    # print(utils.get_network_paras_amount({'model':model}))
    # model = Stylizer()
    # print(utils.get_network_paras_amount({'model':model}))
    # raise
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
        # from reflow.solver import train
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
        
        model_d = Discriminator({})
        # model_d = None
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    

    style_model = GlowVcStylizerWN5Mod8()
    
    optimizer = Muon_AdamW(style_model, 
                    muon_args={'weight_decay': args.train.weight_decay}, 
                    adamw_args={'weight_decay': 0})
    optimizer_d = Muon_AdamW(model_d, 
                    muon_args={'weight_decay': args.train.weight_decay}, 
                    adamw_args={'weight_decay': 0})

    # optimizer = torch.optim.AdamW(style_model.parameters())
    # optimizer_d = torch.optim.AdamW(model_d.parameters())
    
    # optimizer_d = None
    _, model_g, _ = utils.load_model(args.env.expdir, model,  torch.optim.AdamW(model.parameters(), weight_decay=0), device=args.device)
    initial_global_step, style_model, optimizer, model_d, optimizer_d = utils.load_gan_model(args.env.style_dir, style_model, optimizer, model_d, optimizer_d, device=args.device)
    # initial_global_step, style_model, optimizer= utils.load_model(args.env.style_dir, style_model, optimizer, device=args.device)
    
    lag_model = DiagonalLagrangian(768)
    _, lag_model, _ = utils.load_model('exp/reflow-test-lag/model_5500.pt', lag_model,  torch.optim.AdamW(model.parameters(), weight_decay=0), device=args.device)
     
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
        
    for param_group in optimizer_d.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler_g = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    scheduler_d = lr_scheduler.StepLR(optimizer_d, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    # scheduler_d = None
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    style_model.to(args.device)
    lag_model.to(args.device)
    # style_model.float()
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False, frontend=True, load_audio=False, train_aug=False, load_high_res_mel=True, aug_embd=False)
    # run
    # model_d.float()
    # model = torch.compile(model, mode='max-autotune', fullgraph=True, backend="cudagraphs")
    # style_model = torch.compile(style_model, mode='max-autotune', fullgraph=True, backend="cudagraphs")
    # del model_d
    # del optimizer_d
    # del scheduler_d
    # train_vc_gan(args, initial_global_step, style_model, model, optimizer, scheduler_g, vocoder, loader_train, loader_valid)train_vc_osgan2
    train_vc_osgan(args, initial_global_step, style_model, model, model_d, optimizer, optimizer_d, scheduler_g, scheduler_d, vocoder, loader_train, loader_valid)
    # train_vc_osgan2(args, initial_global_step, style_model, model, model_d, optimizer, optimizer_d, scheduler_g, scheduler_d, vocoder, loader_train, loader_valid)
    # train_vc_no_gan(args, initial_global_step, lag_model,style_model, model, optimizer, scheduler_g, vocoder, loader_train, loader_valid)

    
