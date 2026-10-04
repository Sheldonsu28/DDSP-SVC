import os
import argparse
import torch
from torch.optim import lr_scheduler
from logger import utils
from reflow.data_loaders import get_data_loaders
from reflow.solver import train_stlye
from reflow.vocoder import Vocoder, Unit2Wav
from stylizer.stylizer import GlowVcStylizerWN5Mod6, ModifiedSoftVcStylizerMod8
# from stylizer.enhancers import MelPostNetSnake
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('highest')
# torch.backends.cudnn.benchmark = True


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
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    

    style_model = ModifiedSoftVcStylizerMod8()
    _, model, _ = utils.load_model(args.env.expdir, model, torch.optim.AdamW(model.parameters(), weight_decay=0), device=args.device)
    
    style_optimizer = torch.optim.AdamW(style_model.parameters())
    initial_global_step, style_model, optimizer = utils.load_model(args.env.style_dir, style_model, style_optimizer, device=args.device)
        

    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=1250, factor=args.train.gamma)
    
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    style_model.to(args.device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False, frontend=True, load_audio=True, train_aug=False, load_high_res_mel=True, aug_embd=True)
    # run
    # model = torch.compile(model, mode='max-autotune', fullgraph=True, backend="cudagraphs")
    # style_model = torch.compile(style_model, mode='max-autotune', fullgraph=True, backend="cudagraphs")
    train_stlye(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, style_model)

    
