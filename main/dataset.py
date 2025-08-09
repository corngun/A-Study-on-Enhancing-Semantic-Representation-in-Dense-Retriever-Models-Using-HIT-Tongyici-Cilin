from torch.utils.data import Dataset
from pathlib import Path
import torch
import gc
import sys
sys.path.append(str(Path(__file__).parent.joinpath('Semanticspectrum')))
from semantic_tree import SemanticSpectrum


class QA_Pairs_Dataset(Dataset):
    """
    Args:
        data_root_path: 資料集所屬的根目錄路徑
        seg_type: 使用哪種斷詞得到的資料
        dataset_type: train or valid or test
        granularity: 範疇的細膩度
    """
    def __init__(self,data_root_path:str,seg_type:str,dataset_type:str,granularity:int):

        super(QA_Pairs_Dataset,self).__init__()

        self.granularity = granularity
        # self.with_negative_sample = with_negative_sample
        
        spectrum = SemanticSpectrum()
        scopes:list = spectrum.Target_level_datas(granularity,scope_only=True)
        self.scope_to_ids = {scope:i for i,scope in enumerate(scopes,start=3)}
        self.scope_to_ids.update({
            '<unk>': -100,
            'Punc_1': 0,
            'Punc_2': 1,
            'Punc_3': 2,
        })
        
        self.datas:dict = self.load_data(data_root_path,seg_type,dataset_type)
        
        # if with_negative_sample == False:
        #     del self.datas['negative_ebds']
        #     gc.collect()
        
    def load_data(self, data_root_path,seg_type,dataset_type):
        file_name = f'{dataset_type}_data.pt'
        dataset_path = Path(data_root_path).joinpath(seg_type,dataset_type,file_name)
        if not Path(dataset_path).exists():
            raise FileNotFoundError('此檔案路徑不存在!!')
        
        return torch.load(dataset_path)
            
            
    def __len__(self):
        return len(self.datas.get('question_ebds'))
    
    def __getitem__(self, index):
        
        return {
            'question_ebd':self.datas.get('question_ebds')[index],
            'positive_ebd':self.datas.get('positive_ebds')[index],
            'negative_ebds':self.datas.get('negative_ebds')[index],
            'question_scope_labels':self.datas.get('question_scope_labels').get(f'granularity_{self.granularity}')[index],
            'positive_scope_labels':self.datas.get('positive_scope_labels').get(f'granularity_{self.granularity}')[index],
            'question_mapping_tables':self.datas.get('question_mapping_tables')[index],
            'positive_mapping_tables':self.datas.get('positive_mapping_tables')[index],
            'negative_mapping_tables':self.datas.get('negative_mapping_tables')[index]
        }
        # else:
        #     return {
        #         'question_ebd':self.datas.get('question_ebds')[index],
        #         'positive_ebd':self.datas.get('positive_ebds')[index],
        #         'question_scope_labels':self.datas.get('question_scope_labels').get(f'granularity_{self.granularity}')[index],
        #         'positive_scope_labels':self.datas.get('positive_scope_labels').get(f'granularity_{self.granularity}')[index],
        #         'question_mapping_tables':self.datas.get('question_mapping_tables')[index],
        #         'positive_mapping_tables':self.datas.get('positive_mapping_tables')[index]
        #         }
        
    

        
