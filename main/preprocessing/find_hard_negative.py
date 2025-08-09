from langchain_chroma import Chroma
from transformers import AutoModel
from langchain_core.embeddings import Embeddings
import json
from tqdm import tqdm
from pathlib import Path
import os
import random


class Embedding_model(Embeddings):
    def __init__(self,model_name:str,device:str,remote_code=True):
        self.model = AutoModel.from_pretrained(model_name,trust_remote_code=remote_code)
        if device:
            self.model.to(device)
    def embed_query(self,text:str) -> list[float]:
        return self.model.encode(text).tolist()
        
    def embed_documents(self,texts:list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]
    
    def __repr__(self):
        return f'{type(self.model)}'
    
    def __call__(self, *args, **kwds):
        return self
    
    
def save(data,save_path):
    if not save_path.exists():
        with open(save_path,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
    else:
        with open(save_path,'r',encoding='utf-8') as f:
            tmp:list[dict] = json.load(f)
            tmp.extend(data)
            
        with open(save_path,'w',encoding='utf-8') as f:
            json.dump(tmp,f,ensure_ascii=False,indent=2)
    
    # print(f'Save!!')
    
def main():
    ebd_model = Embedding_model('jinaai/jina-embeddings-v3',device='cuda')
    vector_store = Chroma(persist_directory="../DB_data/knowledge_base(QA)/", embedding_function=ebd_model)
    batch_size = 1000
    src_root = str(Path(__file__).parent.parent.joinpath('main/data/QA/Tiny/raw'))
    file_name = 'data.json'
    save_file_name = 'qa_hard_negatives_labeled(new).json'
    
    threshhold = 0.25
    num_hard_negs = 3
    fetch_num = 500 # 確保有樣本會高於 0.25 分數的資料
    for dir in os.listdir(src_root):
        result = []
        with open(Path(src_root,dir,file_name),'r',encoding='utf-8') as f:
            datas = json.load(f)
        for item in tqdm(datas,desc='similarity_search'):
            
            query = item['question']
            positive = item['answer']
            
            # docs = vector_store.similarity_search(query,k=20)
            # contents = set([doc.page_content for doc in docs])
            # candidates = list(contents - set([positive]))
            # chooses = random.sample(candidates,k=3)
            # insert_idx = random.randint(0,len(chooses))
            # chooses.insert(insert_idx, positive)
            # result.append(
            #     {
            #         'question':query,
            #         'candidates':chooses,
            #         'label':insert_idx
            #     }
            # )
            docs_with_scores = vector_store.similarity_search_with_score(query, k=fetch_num)
            
            match_candidate_negatives = [] 
            unmatch_candidate_negatives = [] 
            for doc, score in docs_with_scores:
                content = doc.page_content
                if content == positive:
                    continue
                elif score <= threshhold :
                    unmatch_candidate_negatives.append({'text': content, 'score': score})
                else:
                    match_candidate_negatives.append({'text': content, 'score': score})


            match_candidate_negatives.sort(key=lambda x:x['score'])

            selected_hard_negatives_texts = [neg['text'] for neg in match_candidate_negatives[:num_hard_negs]]
            
            if len(selected_hard_negatives_texts) < num_hard_negs :
                unmatch_candidate_negatives.sort(key=lambda x:x['score'],reverse=True)
                # 選擇最不相似的前3個資料
                selected_hard_negatives_texts = [neg['text'] for neg in unmatch_candidate_negatives[:num_hard_negs]]

            candidates = list(selected_hard_negatives_texts)
            
            insert_idx = random.randint(0, len(candidates))
            candidates.insert(insert_idx, positive)
            result.append(
                {
                    'question':query,
                    'candidates':candidates,
                    'label':insert_idx
                }
            )
            
            if len(result) >= batch_size :
                save(result,Path(src_root,dir,save_file_name))
                result = []
        if result:
            save(result,Path(src_root,dir,save_file_name))
                        
if __name__ == '__main__':
    main()