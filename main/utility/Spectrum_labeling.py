import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.joinpath('Semanticspectrum')))
from semantic_tree import SemanticSpectrum
from transformers import AutoTokenizer,AutoModel
import numpy as np
from util import read_file,batch_similarity_score,Is_not_punctuation,Is_Engtext
import torch
from word_seg import Segmenter



punctuations = {'Punc_1':['，',',',';','；'],'Punc_2':['！','!','。','.'],'Punc_3':['？','?']}

class Spectrum_label:
    def __init__(self,granularity:int,seg_tool:str,device='cuda'):
        self.granularity = granularity
        self.embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v3',trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3',trust_remote_code=True)
        # self.embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh',trust_remote_code=True,torch_dtype=torch.bfloat16).to(device)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #                                         'jinaai/jina-embeddings-v2-base-zh',
        #                                         trust_remote_code=True,
        #                                         add_prefix_space=True
        #                                         )
        spectrum = SemanticSpectrum()
        self.scopes_vocabs:dict = spectrum.Target_level_datas(granularity,scope_only = False)
        
        self.segmenter = Segmenter(self.tokenizer,seg_tool)
        
        BASE_DIR = Path(__file__).resolve().parent.parent
        file_path = BASE_DIR / 'Semanticspectrum' / 'vocab_to_scope_table' / f'granularity_{granularity}' / 'result_new.json'
        self.word2scope:dict = read_file(
            file_path
            ,type='json'
        )
        
        
        
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state['embedding_model'] = None
    #     state['tokenizer'] = None  
        
    #     return state
    
    # def __setstate__(self,state):
    #     self.__dict__.update(state)
    #     self.embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v3',trust_remote_code=True).to('cuda')
    #     self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3',trust_remote_code=True)
        
    def Encodeing(self,text:str|list[str]):
        with torch.inference_mode():
            encodes = self.embedding_model.encode(text)
        return encodes
    
        
    def Get_labels(self,post_word_list:list[str],pre_word_scopes:list[list[str]]):
        """
        Args:
        post_word_list: jieba 分詞後並對 unkword 處理的 詞彙 list
        pre_word_scopes: pre_word_list 中每個詞彙可能的所有範疇

        Return: labels(list[str]) -> 每一個 word 確定的範疇，「不包含標點符號」
        """
        
        string_text = ''.join(post_word_list)
        pivot_embedding = self.Encodeing(string_text)
        labels = []
        
        for index,(word,scopes) in enumerate(zip(post_word_list,pre_word_scopes)):
            
            if len(scopes) > 1:
                replacement_texts = []

                for scope in scopes:
                    # 該 word 在該範疇中的位置
                    word_idx = self.scopes_vocabs[scope].index(word)
                    # 選擇欲替換詞彙的 index 
                    replace_word_index = word_idx + 1 if word_idx + 1 < len(self.scopes_vocabs[scope]) else word_idx - 1
                    # 將 word 替換成該範疇中的其他詞彙
                    replace_word = self.scopes_vocabs[scope][replace_word_index]
                    replaced_list = post_word_list.copy()
                    replaced_list[index] = replace_word
                    replacement_texts.append(''.join(replaced_list))
                    # replacement_texts.append(''.join(post_word_list[:index] + [replace_word] + post_word_list[index+1:]))

                replacement_embeddings = self.Encodeing(replacement_texts)
                scores = batch_similarity_score(pivot_embedding,replacement_embeddings)
                # 選擇替換後的詞彙與原句相似度最高的範疇
                best_scope_idx = np.argmax(scores)
                labels.append(scopes[best_scope_idx])
            else:
                # if scopes[0] == '<punctuation>':
                #     pass
                # # 只有一個範疇、標示為 '<unk>'、標示為 <eng> 的情況
                # else:
                    labels.append(scopes[0])
                
        return labels
        
    # 匹配 Jieba 斷詞後的詞單轉到對應的範疇(會有一詞多範疇)
    def Word_scopes(self,word_list:list[str]) -> list[list[str]]:
        """
        Args:
            word_list(list[str]): 斷詞後的詞單
        Returns:   
            word_scopes(list[list[str]]): 斷詞後的詞單對應可能的範疇
        """
        
        word_scopes = []
        for index,word in enumerate(word_list):
            scopes = []
            # 判斷是否屬於自定義範疇標點符號
            if not Is_not_punctuation(word):
                flag = False
                for key,val in punctuations.items():
                    if word in val:
                        flag = True
                        scopes.append(key)
                        break
                # 不屬於自定義範疇標點符號，視為 unk
                if not flag:
                    scopes.append(self.tokenizer.unk_token)
                        
            elif Is_Engtext(word):
                scopes.append(self.tokenizer.unk_token)
            else:
                scopes = self.word2scope.get(word,[])
                if not scopes:
                    scopes.append(self.tokenizer.unk_token)

            word_scopes.append(scopes)  
            
        return word_scopes
    
    
    
    def __call__(self,text:str,need_scopes:False):
        pre_word_list = self.segmenter.Word_Seg(text)
        # print(f'pre_word_list: {pre_word_list}')
        processed_word_list = self.segmenter.Process_UnkWord(pre_word_list)
        input_ids,mapping_table = self.segmenter.Tokenization(processed_word_list)
        
        if need_scopes:
            pre_word_scopes = self.Word_scopes(processed_word_list)
            post_word_scopes = self.Get_labels(processed_word_list, pre_word_scopes)

            return {
                'input_ids':input_ids,
                'mapping_table':mapping_table,
                'word_scopes':post_word_scopes
            }
            
        else:
        
            return {
                'input_ids':input_ids,
                'mapping_table':mapping_table,
            }
    


# texts = '你好啊，我喜歡你!'
# qq = Spectrum_label(1,'Jieba','cuda')
# out = qq(texts,need_scopes=False)
# print(out)
# print(len(out['mapping_table']),out.get('word_scopes'))