from pathlib import Path
import os
import sys
sys.path.append(str(Path(__file__).parent.parent.joinpath('utility')))
from Spectrum_labeling import Spectrum_label
import argparse
import json
import multiprocessing 
import torch
from tqdm import tqdm



def Parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--granularity',type=int,default=5)
    parser.add_argument('--seg_tool',type=str,default='Jieba')
    parser.add_argument('--src_data_path',type=str,default='../data/QA/Tiny/')
    args = parser.parse_args()
    
    return args


def init_work(device, granularity,seg_tool):
    """
    初始化工作進程，載入模型
    
    :param model_path: 模型路徑
    :param device: 運算設備
    :return: 初始化的 Spectrum_label 實例
    """
    global labeler
    
    labeler = Spectrum_label(granularity,seg_tool, device=device)
    
    
def process_item(data_item):

    candidates:list[str] = data_item['candidates']
    label = data_item['label']
    
    negs = [item for idx,item in enumerate(candidates) if idx != label]
    for neg_item in negs:
        neg_result = labeler(neg_item.get('text'),need_scopes=True)
        neg_item.setdefault('scopes', {})
        granularity_key = f'granularity_{labeler.granularity}'

        # 補進 granularity 範疇，如果還沒建立
        if granularity_key not in neg_item['scopes']:
            neg_item['scopes'][granularity_key] = neg_result.get('word_scopes')


    return data_item

    


            
def save_data(file_path,data:list):
    
    if not Path(file_path).exists():
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)
            
    with open(file_path,'r',encoding='utf-8') as f:
        prev_datas = json.load(f)
    
    prev_datas.extend(data)
    with open(file_path,'w',encoding='utf-8') as f:
        json.dump(prev_datas,f,ensure_ascii=False)
        
        

def main():
    args = Parser()
    granularity = args.granularity
    data_root_path = args.src_data_path
    seg_tool = args.seg_tool
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 1000
    
    # for granularity in range(1,4):
    print(granularity)
    for dir in os.listdir(Path(data_root_path,seg_tool)): 
        batch_results = []
        total_item = 0

        
        raw_data_path = Path(data_root_path,seg_tool,dir,f'{dir}_data.json')
        save_file_path = Path(data_root_path,seg_tool,dir,f'{dir}_data(include_neg).json')
        
        with open(raw_data_path,'r',encoding='utf-8') as f:
            datas:list[dict] = json.load(f)

        with multiprocessing.Pool(processes = 4,initializer = init_work,initargs=(device,granularity,args.seg_tool)) as pool:
            for idx, result in enumerate(pool.imap_unordered(process_item, datas), 1):
                print(f"Labeling... id: {idx}")
                batch_results.append(result)
                print(f"id: {idx} Done!!\n")
                # result:dict = pool.map(process_per_data,datas)
                if idx % batch_size == 0:
                    total_item += len(batch_results)
                    save_data(save_file_path,batch_results)
                    print(f"已儲存 {idx} 筆資料!")
                    batch_results = []
            if batch_results:
                total_item += len(batch_results)
                save_data(save_file_path,batch_results)
                print(f'全部完成')
                print(f"{dir} 共儲存 {total_item} 筆資料")

# 使用範例
if __name__ == '__main__':
    # 設定資料路徑和模型路徑
    multiprocessing.set_start_method("spawn",force=True)
    main()



