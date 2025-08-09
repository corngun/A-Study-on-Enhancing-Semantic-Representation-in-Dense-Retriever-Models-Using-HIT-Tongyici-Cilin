from pathlib import Path
import numpy as np
import re
import json


def read_file(file_path,type:str):
    if Path(file_path).exists():
        if type == 'txt':
            with open(file_path, encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
            return lines
        elif type == 'json':
            with open(file_path, encoding='utf-8') as f:
                datas = json.load(f)
            return datas
    raise FileNotFoundError(f"File {file_path} not found.")


def batch_similarity_score(pivot_embedding:np.array,replacement_embeddings:np.array):
    
    pivot_embedding = pivot_embedding / np.linalg.norm(pivot_embedding)
    pivot_embedding = np.expand_dims(pivot_embedding,axis=0)
    replacement_embeddings = replacement_embeddings / np.linalg.norm(replacement_embeddings, axis=1, keepdims=True)
    
    similarity_scores = np.dot(replacement_embeddings, pivot_embedding.T).squeeze()
    
    return similarity_scores

    
def Is_not_punctuation(text:str):
    # 這個正則表達式會匹配標點符號，但不匹配中文字、英文字母或數字，因此標點符號會回傳一個空字串
    return re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)


def Is_Engtext(text:str):
    if re.fullmatch(r'[A-Za-z]+', text):
        return True
    return False

def text_clean(text:str,seg_type):
    if seg_type == 'Ckip':
        # 先去除條列序號（數字加點），可能前面還有換行符
        text = re.sub(r'(?m)^\s*\d+\.\s*', '', text)
        # 將多個 \n 變成一個 \n
        text = re.sub(r'\n+', '\n', text)

    text = re.sub(r'[\x00-\x1F\x7F-\x9F\u200b\u200c\u200d\ufeff]', '', text)
    text = text.replace('\n','')
    
    return text


def Print_metrics(**kwargs):
    for key,val in kwargs.items():
        print(f'{key}: {val:.6f}',end=' | ')
    print('')








