from transformers import AutoModel
from tqdm import tqdm
import json
from pathlib import Path
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent.joinpath('Models')))
sys.path.append(str(Path(__file__).parent.parent.joinpath('Semanticspectrum')))
from ebd_ini import ContextualEmbeddingExtractor 
from semantic_tree import SemanticSpectrum

def load_data(dataset_path):

    if not Path(dataset_path).exists():
        raise FileNotFoundError('此資料夾路徑不存在!!')
    
    with open(dataset_path,'r',encoding='utf-8') as f:
        return json.load(f)
    
def Process_scope_labels(spectrum,granularity,scope_datas:dict[str,list]):

    scopes:list = spectrum.Target_level_datas(granularity,scope_only=True)
    scope_to_ids = {scope:i for i,scope in enumerate(scopes,start=3)}
    scope_to_ids.update({
        '<unk>': -100,
        'Punc_1': 0,
        'Punc_2': 1,
        'Punc_3': 2,
    })
        
    scopes:list = scope_datas.get(f'granularity_{granularity}')
    ids = torch.tensor([scope_to_ids[scope] for scope in scopes],dtype=torch.int16)
    
    return ids
    

def main():
    

    spectrum = SemanticSpectrum()

    dataset_root_path = Path(__file__).parent.parent.joinpath('data/QA/Tiny/Jieba')

    dataset_types = ['train','valid','test']

    for dir in dataset_types:
        src_file_name = f'{dir}_data(include_neg).json'
        store_file_name = f'{dir}_data(only_neg_scope_labels).pt'

        datas = load_data(Path(dataset_root_path,dir,src_file_name))
        
        negative_scope_labels = {}

        for granularity in range(5,6):
            scopes:list = spectrum.Target_level_datas(granularity,scope_only=True)
            scope_to_ids = {scope:i for i,scope in enumerate(scopes,start=3)}
            scope_to_ids.update({
                '<unk>': -100,
                'Punc_1': 0,
                'Punc_2': 1,
                'Punc_3': 2,
            })
            
            all_neg_scope_list = []

            
            for i,item in tqdm(enumerate(datas),total=len(datas),desc=f'Process granularity_{granularity} labeling'):
                neg_scope_list = []
                candidates = item['candidates']
                positive_label = item['label']
                negs = [item for i,item in enumerate(candidates) if i != positive_label]
                for neg_item in negs:
                    text = neg_item.get('text')
                    neg_scopes:list = neg_item['scopes'].get(f'granularity_{granularity}')
                    neg_ids = torch.tensor([scope_to_ids[scope] for scope in neg_scopes],dtype=torch.int16)
                    
                    neg_scope_list.append((text,neg_ids))

                all_neg_scope_list.append(neg_scope_list)
            
            negative_scope_labels.update({f'granularity_{granularity}':all_neg_scope_list})
            
        save_path = Path(dataset_root_path,dir,store_file_name)
        torch.save({
                    'negative_scope_labels':negative_scope_labels,
                    },
                save_path)

    
    
if __name__ == '__main__':
    main()