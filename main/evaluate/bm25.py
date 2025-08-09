from rank_bm25 import BM25Okapi
import jieba  # 若是中文建議使用 jieba 分詞
from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent.parent  # 回到專案根目錄
vocab_path = BASE_DIR / 'data' / 'vocabulary.txt'
jieba.load_userdict(str(vocab_path))
# 假設你有一組 QA pairs


def main():

    with open('./test_data_pairs.json','r',encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [item["question"] for item in data]
    answers = [item["positive"] for item in data]
    negatives = [ neg for item in data for neg in item["negatives"]]
    
    candidates = answers + negatives
    # 步驟1：對答案集做斷詞
    
    tokenized_corpus = [list(jieba.cut(candidate)) for candidate in candidates]

    # 步驟2：建立 BM25 檢索庫
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = [list(jieba.cut(q)) for q in questions]

    top_k_accuracy = 0.0
    mrr_at_k = 0.0
    k = 3
    # 步驟4：取得排序結果（分數高的表示更相關）
    for i,query in enumerate(tokenized_query):
        answer = answers[i]
        scores = bm25.get_scores(query)
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        retrievals = [candidates[idx] for idx in top_k_idx]
        if answer in retrievals:
            top_k_accuracy += 1
            pos = retrievals.index(answer)
            mrr_at_k += 1 / (pos + 1)

    top_k_accuracy /= len(tokenized_query)
    mrr_at_k /= len(tokenized_query)
    
    print(f'top_k_accuracy: {top_k_accuracy}')
    print(f'mrr_at_k: {mrr_at_k}')
    
    result = {
        'top_k_accuracy':top_k_accuracy,
        'mrr_at_k':mrr_at_k
    }
    
    with open('./dual-encoder/BM25/result.json','w') as f:
        json.dump(result,f)
    

if __name__ == '__main__':
    main()
