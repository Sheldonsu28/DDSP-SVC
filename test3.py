
import torch
import numpy as np
from discriminators.discriminator import Discriminator
from logger import utils
from reflow.lynxnet2 import LYNXNet2
from reflow.reflow import RectifiedFlow
# from stylizer.resblocks import PosteriorEncoder, TextEncoder
from reflow.vocoder import Unit2Wav, Vocoder
from stylizer.stylizer import GlowVcStylizerWN5, GlowVcStylizerWN5Mod3, GlowVcStylizerWN5Mod4, GlowVcStylizerWN5Mod6, GlowVcStylizerWN5Mod7, GlowVcStylizerWN5Mod8, GlowVcStylizerWN5Mod9
from stylizer.util import create_post_processor, create_reflow
from safetensors import safe_open
# from stylizer.util import CausalConv1d, create_reflow


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

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length)
    return x.unsqueeze(0) < length.unsqueeze(1)


def convert_safetensors_to_pt(safetensors_path, pt_path):
    """
    Converts a .safetensors file to a PyTorch .pt (or .pth) checkpoint.

    Args:
        safetensors_path (str): The path to the input .safetensors file.
        pt_path (str): The path where the output .pt (or .pth) file will be saved.
    """
    try:
        # Load the state_dict from the .safetensors file
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        # Save the state_dict to a PyTorch checkpoint file
        torch.save(state_dict, pt_path)
        print(f"Successfully converted '{safetensors_path}' to '{pt_path}'")

    except Exception as e:
        print(f"Error during conversion: {e}")
        
def wavenet_receptive_field_calculator(kernal_size, layers, dilation=1):
    if dilation == 1:
        dilation_sum = layers
    else:
        dilation_sum = (dilation ** layers - 1) // (dilation - 1)
    return 1 + (kernal_size - 1) * dilation_sum

if __name__ == '__main__':
    # import torch.nn as nn
    model = GlowVcStylizerWN5Mod9()
    # model = Discriminator2({'input_channel':768})
    # model = FlowEnhancer()
    # model = create_reflow(version=3)
    # model = create_post_processor()
    # model.vae_encoder.apply(weights_init_uniform_rule)
    # model.apply(weights_init_uniform_rule)
    # model.flow.zero_last_layer()
    # model.flow.zero_last_layer()
    # path = 'pretrain/ppg-large/large-crisper-v3.pt'
    # convert_safetensors_to_pt(path, 'large-crisper-v3.pt')
    # model.flow.zeros_layer()
    # args = utils.load_config('./configs/elysia-reflow.yaml')
    # model_d = Discriminator({})
    # _, model, _, _, _ = utils.load_gan_model(args.env.style_dir, model, {}, model_d, {}, device=args.device)

    
    # vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    # mode_ddsp =  Unit2Wav(
    #                 args.data.sampling_rate,
    #                 args.data.block_size,
    #                 args.model.win_length,
    #                 args.data.encoder_out_channels, 
    #                 args.model.n_spk,
    #                 args.model.use_norm,
    #                 args.model.use_attention,
    #                 args.model.use_pitch_aug,
    #                 vocoder.dimension,
    #                 args.model.n_aux_layers,
    #                 args.model.n_aux_chans,
    #                 args.model.n_layers,
    #                 args.model.n_chans)

    # _, model_ddsp, _ = utils.load_model(args.env.expdir, mode_ddsp, {}, device=args.device)
    # mode_ddsp.reflow_model.velocity_fn.swap_conv()
    # # print(args)
    # enhanced = EnhancedDDSP( 
    #                 args.data.sampling_rate,
    #                 args.data.block_size,
    #                 args.model.win_length,
    #                 args.data.encoder_out_channels, 
    #                 args.model.n_spk,
    #                 args.model.use_norm,
    #                 args.model.use_attention,
    #                 args.model.use_pitch_aug,
    #                 vocoder.dimension,
    #                 args.model.n_aux_layers,
    #                 args.model.n_aux_chans,
    #                 args.model.n_layers,
    #                 args.model.n_chans)
    # enhanced.stylizer = model
    # enhanced.unit2wav = model_ddsp
    torch.save({
        'model':model.state_dict(),
        'global_step': 0
    }, 'new_style_vc.pt')
    
    # # # final_wavenet_layer = model.flow.wavenet.post2
    # # # final_wavenet_layer.weight.data.zero_()
    # # # final_wavenet_layer.bias.data.zero_()
    # # # s =  nn.Conv1d(768, 768, kernel_size=5, padding=5 // 2, groups=768)
    # # # c = CausalConv1d(768, 768, 5,  groups=768, dilation=2)
    B = 1
    unit = torch.zeros((B, 100, 768))
    cond = torch.zeros((B, 100, 256))
    hu = torch.zeros((B, 100, 256))
    whis =  torch.zeros((B, 100, 1280))
    spk = torch.ones((B, 192))
    mel = torch.zeros((B, 100, 128))
    f0 = torch.zeros((B, 100, 1))
    vol = torch.zeros((B, 100, 1))
    # model(cond, gt_spec=unit, infer=False)
    # model(unit, whis, hu, spk, mel, f0, vol, infer=False)
    
    
    
    # k = 5
    # s = 1
    # l = 3
    # print(1+(k-1)*s*(2**l-1))

    # from math import floor
    # L_in = 43 * 12
    # kernel_size = 31
    # padding = 0
    # dilation = 1
    # stride = 1
    # L_out = floor(((L_in + 2 * padding - dilation * (kernel_size - 1) - 1)/ stride )+1)
    # print(L_in, L_out)
    
    # a = torch.ones((1, 1, 101))
    
    # s = torch.nn.functional.interpolate(a, scale_factor=0.5)
    # b = torch.nn.functional.interpolate(s, scale_factor=101/50)
    # print(a.shape, b.shape)