import torch.nn.functional as F
import torch
from tqdm import tqdm

def Cls_Acc(y_pred,y_true,ignore_index = -100):
    # 只考慮非 -100 標籤(padding、eng_text) 的位置
    
    mask = y_true != ignore_index # (batch,seq_len)
    correct = (y_pred == y_true) & mask
    return  correct.sum().item() / mask.sum().item() # 除上所有詞彙的數量 (不包含 padding token)

def Cal_recall_at_k(
                    q_ebds:list[torch.Tensor],
                    ans_ebds:list[torch.Tensor],
                    neg_ebds:list[torch.Tensor],
                    k=5,
                    batch_size:int=32,
                    device='cuda',
                    ):
    """
    Args:
        q_ebds: List[ Tensor[batch,d_ebd] ]
        ans_ebds: List[ Tensor[batch,d_ebd] ]
        neg_ebds: List[ Tensor[batch,N,d_ebd] ]
    """
    
    dim = q_ebds[0].size(1)
    q_ebds = torch.cat(q_ebds,dim=0)
    ans_ebds = torch.cat(ans_ebds,dim=0).to(device)
    #(total_data_len, N, d_ebd) -> #(total_data_len*N, d_ebd)
    neg_ebds = torch.cat(neg_ebds,dim=0).view(-1,dim).to(device) 
    # (total_data_len,d_ebd) -> (total_data_len,d_ebd)
    ans_ebds = F.normalize(ans_ebds,p=2,dim=1)
    neg_ebds = F.normalize(neg_ebds,p=2,dim=-1)
    #(total_data_len + total_data_len*N, d_ebd)
    candidate_ebds = torch.cat([ans_ebds,neg_ebds],dim=0)
    
    total_len = q_ebds.size(0)
    total_correct = 0.0
    
    mrr_at_k = 0.0

    # (total_data_len,total_data_len)
    for i in tqdm(range(0,total_len,batch_size),desc=f'Cal recall at {k}'):
        # (batch_size,total_data_len)
        q_batch = q_ebds[i:i+batch_size].to(device)
        q_batch = F.normalize(q_batch, p=2, dim=1)
        
        # (total_data_len, total_data_len + total_data_len *N)
        batch_similarity_matrix = torch.matmul(q_batch,candidate_ebds.T)
    
        # (batch_size,k)
        _,batch_topk_indices = torch.topk(batch_similarity_matrix,k=k,dim=1)
    
        # (batch_size,)
        batch_gold_indices = torch.arange(i,min(i+batch_size,total_len), device=device)
    
        correct = (batch_topk_indices == batch_gold_indices.unsqueeze(1)).any(dim=1).float()
        total_correct += correct.sum().item()
        
        eq_matrix = (batch_topk_indices == batch_gold_indices.unsqueeze(1)) #(batch,k)
        reciprocal_ranks = torch.zeros(q_batch.size(0), device=device)
        for idx in range(q_batch.size(0)):
            # 每個 query 的 topk 中是否有對應的 positive index
            pos = torch.nonzero(eq_matrix[idx], as_tuple=False)
            # 如果 pos 的數量大於 0，也就是 topk 中有對應的 positive
            if pos.numel() > 0:
                # pos is [[rank_idx]], rank is zero-based, so add 1
                reciprocal_ranks[idx] = 1.0 / (pos[0].item() + 1)
            # else, remains 0.0

        mrr_at_k += reciprocal_ranks.sum().item()

        
    recall_at_k = total_correct / total_len
    mrr_at_k = mrr_at_k / total_len

    return recall_at_k,mrr_at_k



