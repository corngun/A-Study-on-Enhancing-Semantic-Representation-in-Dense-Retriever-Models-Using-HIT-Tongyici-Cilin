from transformers import AutoTokenizer
from util import read_file,Is_Engtext,Is_not_punctuation,text_clean
import jieba
import numpy as np
from ckip_transformers.nlp import CkipWordSegmenter
from pathlib import Path



class Segmenter:
    def __init__(self,tokenizer:AutoTokenizer,seg_tool:str):
        self.seg_tool = seg_tool
        BASE_DIR = Path(__file__).resolve().parent.parent  # 回到專案根目錄
        vocab_path = BASE_DIR / 'data' / 'vocabulary.txt'
        self.vocabs = read_file(vocab_path,type='txt')
        if seg_tool == 'Ckip':
            self.ws_driver = CkipWordSegmenter(model="bert-base",device = 0)
        elif seg_tool == 'Jieba':
            jieba.load_userdict(str(vocab_path))
            
        self.tokenizer = tokenizer
        self.space_token_id = self.tokenizer.convert_tokens_to_ids('▁')
        

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get('ws_driver') is not None:
            state['ws_driver'] = None
        
        return state
    
    def __setstate__(self,state):
        self.__dict__.update(state)

        if 'ws_driver' in state and state['ws_driver'] is None:
            self.ws_driver = CkipWordSegmenter(model="bert-base", device=0)
        
    
    def Word_Seg(self,text:str):
        text = text_clean(text,self.seg_tool)
        if self.seg_tool == 'Jieba':
            raw_word_lists = list(jieba.cut(text))
            # 去掉斷詞後是 空白、換行 的 word
            raw_word_lists = list(filter(lambda x:x!=' ' and x!='\n',raw_word_lists))
            
        elif self.seg_tool == 'Ckip':
            raw_word_lists = self.ws_driver([text])[0]
            # 去掉斷詞後是 空白、換行 的 word
            raw_word_lists = list(filter(lambda x:x!=' ' and x!='\n',raw_word_lists))
            # ckip 分詞在年份的詞彙前會有一個空格需要將其 replace 掉
            raw_word_lists = list(map(lambda x:x.replace(' ',''),raw_word_lists))
        return raw_word_lists
    
    def match(self,word:str,memo=None):
        """
        Args:
        word(str): 搜尋的詞彙
        memo(dict): 記憶先前計算結果的快取

        Return([] | list[list[str]]) -> 詞彙的組合

        """
        if memo is None:
            memo = {}
        
        if word in memo:
            return memo[word]
        
        res = []
        if word in self.vocabs: return [[word, ], ]
        for idx in range(len(word),-1,-1):
            if word[:idx] in self.vocabs:
                pieces = self.match(word[idx:])
                if pieces:
                    for p in pieces:
                        res.append([word[:idx], ] + p)
        
        memo[word] = res
        
        return res
    
    def Process_UnkWord(self,word_list:list[str]):
        """
        比對 Jieba 分詞後的詞彙是否有在哈大同義詞林中，若沒有則嘗試看能否用其他詞彙組合
        Args
        word_list: Jieba 分詞後的詞彙 list
        Return
        word_list: 回傳處理後的 詞彙 list
        """
        pos_offset = 0

        for pos,word in enumerate(word_list):
            # 當詞彙不屬於英文且不在詞彙表中，再去詞彙表中確認是否有可以組合(可以拆解)的詞彙
            if word not in self.vocabs and not Is_Engtext(word):
                pos = pos + pos_offset
                combine_words = self.match(word_list[pos])
                if combine_words:
                    min_combines_pos = np.argmin(list(map(len,combine_words)))
                    combination:list[str] = combine_words[min_combines_pos]
                    word_list = word_list[:pos] + combination + word_list[pos+1:] 
                    pos_offset += len(combination) - 1
        
        return word_list
        
    def Tokenization(self,word_list:list[str]):
        # encoding = self.tokenizer(word_list,is_split_into_words=True,add_special_tokens=True)
        encoding = self.tokenizer(word_list,is_split_into_words=True,add_special_tokens=True)
        input_ids = encoding.input_ids 
        # print(input_ids)

        # 使用 tokenizer 的 word_ids 功能獲取映射
        # print(input_ids)
        word_ids = encoding.word_ids()  # 去掉最後的 sep token
        # print(self.tokenizer.convert_ids_to_tokens(input_ids))
        # print(word_ids)
        # 因為 cls token  word id 是 None 因此預先先給一個對應的 先給一個對應的 placehold_len
        mapping_table = [[1]]
        prev_word_id = None
        current_count = 0

        for word_id,input_id in zip(word_ids,input_ids):
            if word_id is None:
                continue
                
            if word_id != prev_word_id:
                if prev_word_id is not None:
                    mapping_table.append([current_count])
                prev_word_id = word_id
                # 空白 token 不計算
                current_count = 1 if input_id != self.space_token_id else 0
            else:
                current_count += 1
        # 添加最後一個詞的映射
        if current_count > 0:
            mapping_table.append([current_count])
        
        mapping_table.append([1]) # </s> token
        # print(mapping_table)

        # 過濾空白 token
        filtered_input_ids = []
        
        for i, id_val in enumerate(input_ids):
            if id_val != self.space_token_id:
                filtered_input_ids.append(id_val)

        # print(f'filtered_input_ids: {filtered_input_ids}')
        return filtered_input_ids, mapping_table
    
    def __call__(self,src_text:str):
        raw_word_list = self.Word_Seg(src_text)
        processed_word_list = self.Process_UnkWord(raw_word_list)
        # final_word_list = list(filter(Is_not_punctuation,filt_1_word_list))
        # print(final_word_list)
        input_ids,mapping_table = self.Tokenization(processed_word_list)

        return {
            'input_ids':input_ids,
            'mapping_table':mapping_table
        }
    
    
# texts = 'Niess 在 2005 年有哪些建議？'
# tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3',trust_remote_code=True)
# segmenter = Segmenter(tokenizer,'Ckip')
# result = segmenter(texts)
# print(len(result.get('input_ids')),int(np.sum(result.get('mapping_table'))))


