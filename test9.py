import io

from funasr import AutoModel
import librosa
import torch
import numpy as np
import soundfile as sf

from ddsp.vocoder import Units_Encoder
from reflow.vocoder import load_model_vocoder
from test8 import visualize_embedding

def resize_array(arr, new_length):
    x_original = np.linspace(0, 1, arr.shape[0])
    x_target = np.linspace(0, 1, new_length)

# 3. Loop through each column, interpolate it, and stack them back together
    interpolated_array = np.column_stack([
        np.interp(x_target, x_original, arr[:, i]) 
        for i in range(arr.shape[1])
    ])
    
    return interpolated_array

if __name__ == "__main__":
    # torch._logging.set_logs(output_code=True, graph_breaks=True, recompiles=True)
    # torch._dynamo.config.verbose = True
    # torch._inductor.config.verbose = True
    
    model, vocoder, args = load_model_vocoder('./exp/reflow-test-new/model_4500.pt', device='cuda')
    # # model_id = "iic/emotion2vec_plus_large"

    # # model = AutoModel(
    # #     model=model_id,
    # #     hub="huggingface",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    # # )
    # # # print(model.model_path)
    # # wav_file = f"./data/train/audio/1/1.wav"
    # # a = torch.from_numpy(np.load('./data/train/units/1/1.wav.npy'))
    # # b = torch.from_numpy(np.load('./data/train/emo/1/1.wav.npy'))
    # # rec_result = model.generate(wav_file, output_dir="./outputs", granularity="frame", extract_embedding=True, disable_pbar=True)
    # # # res = resize_array(rec_result[0].get('feats'),  a.shape[0])
    # # # res = torch.from_numpy(rec_result[0].get('feats'))
    # # print('start')
    # # for item in rec_result[0]['layer_res']:
    # #     print(item.shape)
    # # byte_io = io.BytesIO()
    # # audio, sample_rate = librosa.load(wav_file, sr=16000)
    # # if len(audio.shape) > 1:
    # #         audio = librosa.to_mono(audio)
    
    
    # # # audio_t = torch.from_numpy(audio).float()
    # # # audio_t = audio_t.unsqueeze(0)
    # # # encoder = Units_Encoder('emotionvec', '', 16000, 320)
    # # # sf.write(byte_io, audio, 16000, format='WAV')
    
    # # # res2 = encoder.encode_emo(byte_io.getvalue(), model, res.shape[0])
    # # # res2 = torch.from_numpy(resize_array(res, a.shape[0]))
    # # print(a.shape, b.shape)
    # # visualize_embedding(rec_result[0])
    # # # print(res.shape, res2.shape)
    # # # print(((res2- res)**2).sum())
    # # # print(res.shape)
    
    # a = [1, 2, 3, 4, 5, 6]
    # for i in a:
    #     if i > 3:
    #         a.remove(i)
    # print(a)
    
    
   
    