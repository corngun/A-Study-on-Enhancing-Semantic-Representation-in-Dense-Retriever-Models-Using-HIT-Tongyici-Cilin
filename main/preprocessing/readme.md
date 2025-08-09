# Step1.
執行 data_split 先從所有的 Q&A pairs 隨機選取 10W 筆資料，並分為 train、valid、test set
# Step2.
執行 find_hard_negative 選擇 Step1. 中所有 Q 的 hard negative 
# Step3.
執行 pretokenize 先將資料預先斷詞，獲得對應的 embedding model 的 token_id 
# Step4. 
執行 prelabel 對 Q&A 標上對應「解析度」的詞義範疇，有5個解析度，要跑5次 (最花時間)
# Step5.
執行 data_preencoding 先將資料轉成 tensor 並存成 .pt file 加速後面訓練


# 訓練時，僅有針對 Query 與對應 Answer 的 text ，做詞彙範疇標籤的標注，並沒有將負樣本的 text 也做範疇標注，若需要可以
1. 將 neg_prelabel.py 整合進 prelabel.py 中
2. 將 neg_preencoding.py 整合進 data_preencoding.py 中

