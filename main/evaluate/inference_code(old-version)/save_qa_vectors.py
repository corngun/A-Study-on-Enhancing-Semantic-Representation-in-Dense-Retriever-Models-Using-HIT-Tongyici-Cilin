from inference_encoder import InferenceEncoder
import json
from argparse import ArgumentParser
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from dataset import QA_Pairs_Dataset
from torch.utils.data import DataLoader
from collate_fn import Collate_fn

def Parser():
    parser = ArgumentParser()
    parser.add_argument('--granularity',type=int,default=1)
    parser.add_argument('--seg_tool',type=str,default='Jieba')
    args = parser.parse_args()
    
    return args

def main():
    args = Parser()
    granularity = args.granularity
    seg_tool = args.seg_tool
    
    test_data_path = str(Path(__file__).parent.parent.joinpath(
                                                    'data/QA/Tiny',
                                                    seg_tool,
                                                    f'granularity_{granularity}'
                                                    ,'test'
                                                    ,'data.json'))
    with open(test_data_path,'r',encoding='utf-8') as f:
        text_pairs = json.load(f)


    
    config_path = Path('./config.json')
    checkpoint_path = str(Path(__file__).parent.parent.joinpath('checkpoint',f'granularity_{granularity}','encoder.pt'))
    semanspec_model = InferenceEncoder(
                                config_path = config_path,
                                checkpoint_path = checkpoint_path,
                                seg_type=seg_tool,
                                device='cuda')
    
    save_dir = f'./evaluation_data/semanspec_contrastive_granularity_{granularity}'

    
    questions = [pair['question']['text'] for pair in text_pairs]
    answers = [pair['answer']['text'] for pair in text_pairs]
    
    question_ebds = []
    answer_ebds = []
    batch_size = 64
    
        
    for i in tqdm(range(0,len(questions),batch_size),desc='Encoding Question'):
        with torch.inference_mode():
            batch = questions[i:i+batch_size]
            batch_ebds = semanspec_model.encode(batch,batch_size)
            question_ebds.extend(batch_ebds)


    for i in tqdm(range(0,len(answers),batch_size),desc='Encoding Answer'):

        with torch.inference_mode():
            batch = answers[i:i+batch_size]
            batch_ebds = semanspec_model.encode(batch)
            answer_ebds.extend(batch_ebds)
            
    question_ebds = np.asarray(question_ebds)
    answer_ebds = np.asarray(answer_ebds)
    
    
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True,exist_ok=True)
    with open(Path(save_dir,'question_vectors.npy'),'wb') as f:
        np.save(f,question_ebds)
    with open(Path(save_dir,'answer_vectors.npy'),'wb') as f:
        np.save(f,answer_ebds)
    
if __name__ == '__main__':
    main()