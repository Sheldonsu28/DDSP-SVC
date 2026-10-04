import os
import argparse
import torch
from torch.optim import lr_scheduler
from discriminators.discriminator import Discriminator
from optimizer.muon import Muon_AdamW
from logger import utils
from reflow.data_loaders import get_base_data_loaders, get_data_loaders
from reflow.solver import train_base
from reflow.vocoder import Vocoder, Unit2Wav, load_model_vocoder
from stylizer.stylizer import GlowVcStylizerWN5Mod4, GlowVcStylizerWN5Mod6, GlowVcStylizerWN5Mod7, GlowVcStylizerWN5Mod8
torch.backends.cudnn.benchmark = True
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
    vocoder = None
    
    # load model
  
    model = GlowVcStylizerWN5Mod8(train_formant=False)
    validation_model, vocoder, _ = load_model_vocoder('./exp/reflow-test/model_5000.pt', device=args.device)
                    
  
        
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    validation_model.to(args.device)
    
    # load parameters
    optimizer = Muon_AdamW(model, 
                    muon_args={'weight_decay': args.train.weight_decay}, 
                    adamw_args={'weight_decay': 0})

    initial_global_step, model, optimizer = utils.load_model(args.env.style_dir, model, optimizer, device=args.device)
    # initial_global_step = 0
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    # datas
    loader_train, loader_valid = get_base_data_loaders(args, whole_audio=False, load_audio=False, finetune=True, load_high_res_mel=True, train_aug=False)
    
    # run
    train_base(args, initial_global_step, model, optimizer, scheduler, validation_model, vocoder, loader_train, loader_valid)
    
