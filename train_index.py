import argparse
import os
import pickle

from feature_retrival import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, default="data", help="path to root dir"
    )
    # parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
    #                 help='JSON file for configuration')
    parser.add_argument(
        "--output_dir", type=str, default="exp/feature_index/elysia", help="path to output dir"
    )

    args = parser.parse_args()

    # hps = utils.get_hparams_from_file(args.config)
    spk_dic = {'Elysia': 1 }
    result = {}
    
    empty_dir = True
    
    for k,v in spk_dic.items():
        print(f"Processing contentvec now, index {k} feature...")
        
        index = utils.train_index(v if not empty_dir else '',args.root_dir, feat='units')
        result[v] = index

    with open(os.path.join(args.output_dir,"feature_and_index.pkl"),"wb") as f:
        pickle.dump(result,f)
        
    for k,v in spk_dic.items():
            print(f"Processing hubert now, index {k} feature...")
            index = utils.train_index(v if not empty_dir else '',args.root_dir,feat='hubert_units')
            result[v] = index
    
    with open(os.path.join(args.output_dir,"feature_and_index_hubert.pkl"),"wb") as f:
        pickle.dump(result,f)
        
    for k,v in spk_dic.items():
            print(f"Processing whisper now, index {k} feature...")
            index = utils.train_index(v if not empty_dir else '',args.root_dir,feat='whisper_units')
            result[v] = index
    
    with open(os.path.join(args.output_dir,"feature_and_index_whisper.pkl"),"wb") as f:
        pickle.dump(result,f)
        
        
    # std, mean = utils.compute_std_mean(v,args.root_dir) 
    