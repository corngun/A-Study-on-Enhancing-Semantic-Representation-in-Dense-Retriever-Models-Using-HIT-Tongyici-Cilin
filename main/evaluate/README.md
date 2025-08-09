# dual-encoder 目錄:
    1. 不同模型設定在 testing set 上，每筆 Q&A 模型檢索得 top-3 data 是什麼，以及對應的 similarity score，儲存的檔案名稱為 result_dic.json
    2. 其中 BM25 目錄為使用 BM25 在 testing set 上的 Top-3 Accuracy 與 MRR@3 分數

# inference_code(old-version) 目錄:
    過去寫的 inference 架構，但是後面沒有更新

# wandb_data_point 目錄:
    儲存不同模型設定每一次跑得訓練數據，主要用於重新繪圖（loss、top-k、mrr@k）

# bm25.py:
    利用 BM25 計算 testing set top-k、mrr@k 的程式碼

# compare.ipynb:
    用於觀察與比較 「不同模型設定在 testing set 上，每筆 Q&A 模型檢索得 top-3 data 是什麼」

# create_testing_result_table.py:
    將 result_dic.json 的內容 convert to result_tables.json，轉換後才可明確得知模型所檢所到的 text 是什麼

# test_data_pairs.json:
    testing set 中的資料

# plot_graph.ipynb:
    用於重繪訓練曲線的 code