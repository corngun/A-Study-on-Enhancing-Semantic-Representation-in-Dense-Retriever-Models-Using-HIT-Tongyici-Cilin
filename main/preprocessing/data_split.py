from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import random
from argparse import ArgumentParser
import os

def Parser():
    parser = ArgumentParser()
    parser.add_argument('--data_path',type=str,default='../../rag-lanchain/data/Chinese_Language_essay_dataset.json')
    parser.add_argument('--save_dir',type=str,default='../data/QA/')
    parser.add_argument('--sample_nums',type=int,default=100000)
    args = parser.parse_args()
    
    return args

def main():
    args = Parser()
    data_path = args.data_path

    with open(data_path,'r',encoding='utf-8') as f:
        datas = json.load(f)
        raw_qa_pairs = [ {'question': item['instruction'],'answer':item['output']} for item in datas]
    

    qa_pairs = random.sample(raw_qa_pairs,k=args.sample_nums)
    
    x_tmp,x_test = train_test_split(qa_pairs,test_size=0.1,random_state=42)
    x_train,x_val = train_test_split(x_tmp,test_size=0.2,random_state=42)
    
    all_datas = [x_train,x_val,x_test]
    dirs = ['train','valid','test']
    
    for dir,datas in zip(dirs,all_datas):
        table = []
        dir_path = Path(args.save_dir,'raw',dir)
        if not Path(dir_path).exists():
            Path(dir_path).mkdir(parents=True,exist_ok=True)
            
        for id,enrty in enumerate(datas):
            new_entry = {}
            new_entry['id'] = id
            new_entry['question'] = enrty['question']
            new_entry['answer'] = enrty['answer']

            table.append(new_entry)
        
        with open(Path(dir_path,'Qa_pairs.json'),'w',encoding='utf-8') as f:
            json.dump(table,f,ensure_ascii=False,indent=2)
    
if __name__ == '__main__':
    main()
