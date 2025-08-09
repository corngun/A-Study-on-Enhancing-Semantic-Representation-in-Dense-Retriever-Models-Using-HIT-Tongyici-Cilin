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
    

    jina_ebd_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3",trust_remote_code=True)
    extractor = ContextualEmbeddingExtractor(jina_ebd_model,device = 'cuda')
    spectrum = SemanticSpectrum()


    dataset_root_path = Path(__file__).parent.parent.joinpath('data/QA/Tiny/Ckip')

    dataset_types = ['train','valid','test']

    for dir in dataset_types:
        src_file_name = f'{dir}_qa_hard_negatives_labeled.json'
        store_file_name = f'{dir}_data.pt'
        
        question_ebds = []
        question_mapping_tables = []
        positive_ebds = []
        positive_mapping_tables = []
        negative_ebds = []
        negative_mapping_tables = []

        datas = load_data(Path(dataset_root_path,dir,src_file_name))
        
        for i,item in tqdm(enumerate(datas),total=len(datas),desc='Process Encoding'):
            
            q_item,candidates = item['question'],item['candidates']
            
            positive_label = item['label']
            positive = candidates[positive_label]
            negatives = [c for idx, c in enumerate(candidates) if idx != positive_label]
            
            with torch.inference_mode():
                # q_ebd = extractor(q_item['input_ids'],q_item['mapping_table']).type(torch.float16)
                # positive_ebd = extractor(positive['input_ids'],positive['mapping_table']).type(torch.float16)
                # neg_ebds = [extractor(n['input_ids'], n['mapping_table']).type(torch.float16) for n in negatives]
                q_ebd = extractor(q_item['input_ids']).type(torch.float16)
                positive_ebd = extractor(positive['input_ids']).type(torch.float16)
                neg_ebds = [extractor(n['input_ids']).type(torch.float16) for n in negatives]
            
            neg_mapping_tables = [torch.tensor(n['mapping_table']).type(torch.int16) for n in negatives]
            
            question_ebds.append(q_ebd)
            positive_ebds.append(positive_ebd)
            negative_ebds.append(neg_ebds)
            question_mapping_tables.append(torch.tensor(q_item['mapping_table']).type(torch.int16))
            positive_mapping_tables.append(torch.tensor(positive['mapping_table']).type(torch.int16))
            negative_mapping_tables.append(neg_mapping_tables)
            
        question_scope_labels = {}
        positive_scope_labels = {}
        for granularity in range(4,5):
            scopes:list = spectrum.Target_level_datas(granularity,scope_only=True)
            scope_to_ids = {scope:i for i,scope in enumerate(scopes,start=3)}
            scope_to_ids.update({
                '<unk>': -100,
                'Punc_1': 0,
                'Punc_2': 1,
                'Punc_3': 2,
            })
            
            q_scope_list = []
            positive_scope_list = []
            
            for i,item in tqdm(enumerate(datas),total=len(datas),desc=f'Process granularity_{granularity} labeling'):
                
                q_item,candidates = item['question'],item['candidates']
                positive_label = item['label']
                positive = candidates[positive_label]
                        
                q_scopes:list = q_item['scopes'].get(f'granularity_{granularity}')
                q_ids = torch.tensor([scope_to_ids[scope] for scope in q_scopes],dtype=torch.int16)
                
                positive_scopes:list = positive['scopes'].get(f'granularity_{granularity}')
                positive_ids = torch.tensor([scope_to_ids[scope] for scope in positive_scopes],dtype=torch.int16)

                q_scope_list.append(q_ids)
                positive_scope_list.append(positive_ids)
            
            question_scope_labels.update({f'granularity_{granularity}':q_scope_list})
            positive_scope_labels.update({f'granularity_{granularity}':positive_scope_list})
            
        save_path = Path(dataset_root_path,dir,store_file_name)
        torch.save({
                    'question_ebds':question_ebds,
                    'positive_ebds':positive_ebds,
                    'negative_ebds':negative_ebds,
                    'question_scope_labels':question_scope_labels,
                    'positive_scope_labels':positive_scope_labels,
                    'question_mapping_tables':question_mapping_tables,
                    'positive_mapping_tables':positive_mapping_tables,
                    'negative_mapping_tables':negative_mapping_tables
                    },
                save_path)

    
    
if __name__ == '__main__':
    main()