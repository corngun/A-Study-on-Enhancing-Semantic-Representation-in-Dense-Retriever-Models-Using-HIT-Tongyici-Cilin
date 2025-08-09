import json 
from argparse import ArgumentParser
from pathlib import Path


def Parser():
    parser = ArgumentParser()
    parser.add_argument('--pred_result_path',type=str)
    args = parser.parse_args()
    
    return args

def index_to_sample(index, total_len=10000, neg_per_query=3):
    if index < total_len:
        # return f"正樣本, query {index}"
        return {'query_id':index}
    else:
        neg_idx = index - total_len
        query_id = neg_idx // neg_per_query
        neg_in_query = neg_idx % neg_per_query
        # return f"負樣本, query {query_id}, 第{neg_in_query}個負樣本"
        
        # return query_id,neg_in_query
        return {
            'query_id': query_id,
            'neg_in_query': neg_in_query
                }




def main():
    
    args = Parser()
    
    with open(args.pred_result_path,'r') as f:
        result_dic = json.load(f)

    with open('./test_data_pairs.json','r',encoding='utf-8') as f:
        test_data_table = json.load(f)

    correct_indices = result_dic.get('correct_indices')
    # incorrect_indices = result_dic.get('incorrect_indices')
    
    result_tables = []
    for i,(scores,indices) in enumerate(zip(result_dic.get('all_topk_scores'),result_dic.get('all_topk_indices'))) :
        pred_texts = []

        for index in indices:

            result = index_to_sample(index)
            pos = result.get('query_id')

            if result.get('neg_in_query') != None  :
                neg_pos = result.get('neg_in_query')

                pred = test_data_table[pos].get('negatives')[neg_pos]
        
            else:
                pred = test_data_table[pos].get('positive')

            pred_texts.append(pred)
            
        result_tables.append(
            {
                'question':test_data_table[i].get('question'),
                'answer':test_data_table[i].get('positive'),
                'prediction':pred_texts,
                'scores':scores,
                'correct': True if i in correct_indices else False
            }
        )
    
    store_path = Path(args.pred_result_path).parent.joinpath('result_tables.json')
    with open(store_path,'w',encoding='utf-8') as f:
        json.dump(result_tables,f,ensure_ascii=False,indent=2)
            


if __name__ == '__main__':
    main()