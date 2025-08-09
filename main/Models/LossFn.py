import torch.nn as nn
import torch.nn.functional as F
import torch


class InfoNCELoss(nn.Module):
    def __init__(self, hardneg_num,temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8 
        self.hardnegs = hardneg_num

    def forward(self, q_emb, pos_emb, neg_embs=None):
        """
        Args:
            q_emb: (B, D)
            pos_emb: (B, D)
            neg_embs: optional (B, N, D) - per-query negatives
        Returns:
            Scalar loss
        """
        B, D = q_emb.shape

        # Normalize embeddings
        # Adding eps for stability if inputs can be zero vectors
        q_emb = F.normalize(q_emb, p=2, dim=1, eps=self.eps)
        pos_emb = F.normalize(pos_emb, p=2, dim=1, eps=self.eps)

        if neg_embs is not None:
            # 只選擇第一筆負樣本進去計算 loss
            use_neg_emb = neg_embs[:, :self.hardnegs, :] 

            # Normalize external negatives
            # neg_embs = F.normalize(first_neg_emb, p=2, dim=-1, eps=self.eps) # (B, N, D)
            neg_embs = F.normalize(use_neg_emb, p=2, dim=-1, eps=self.eps)

            # Positive similarity
            pos_sim = torch.sum(q_emb * pos_emb, dim=1) / self.temperature # (B,)

            # In-batch similarity matrix (q_i vs pos_j for all i, j)
            sim_matrix = torch.matmul(q_emb, pos_emb.T) / self.temperature # (B, B)

            # External negative similarity (q_i vs neg_k for query i)
            # Use einsum for potentially clearer batch dot product
            ext_neg_sim = torch.einsum('bnd,bd->bn', neg_embs, q_emb) / self.temperature # (B, N)

            diag_mask = torch.eye(B, dtype=torch.bool, device=q_emb.device)

            # Select off-diagonal similarities
            # sim_matrix[~diag_mask] flattens the off-diagonal elements
            # .view(B, B - 1) reshapes them correctly
            in_batch_neg_sim = sim_matrix[~diag_mask].view(B, B - 1) # (B, B-1)

            # Combine positive, in-batch negatives, and external negatives
            # Positive sim needs unsqueezing to become (B, 1)
            logits = torch.cat([pos_sim.unsqueeze(1), in_batch_neg_sim, ext_neg_sim], dim=1) # (B, 1 + (B-1) + N) = (B, B+N)

            # Labels are all 0 because positive is always at index 0
            labels = torch.zeros(B, dtype=torch.long, device=q_emb.device)

        else:
            # --- In-batch Negatives Only ---
            # Compute all pairwise similarities (q_i vs pos_j)
            logits = torch.matmul(q_emb, pos_emb.T) / self.temperature # (B, B)
            # The positive pair (q_i, pos_i) is on the diagonal (index i for row i)
            labels = torch.arange(B, device=q_emb.device) # Labels are 0, 1, ..., B-1

        return F.cross_entropy(logits, labels)
    



# class margin_rank_loss(nn.Module):
#     def __init__(self, margin=0.2, temperature=0.1):
#         super().__init__()
#         self.margin = margin

#     def forward(self, q_ebd, pos_ebd, neg_ebds: torch.Tensor):
        
#         """
#         Args:
#             q_ebd:      (batch, d_ebd)
#             pos_ebd:    (batch, d_ebd)
#             neg_ebds:   (batch, 3, d_ebd) 
#         Returns:
#             Scalar loss
#         """
        
#         B, N, D = neg_ebds.shape
#         q_ebd = F.normalize(q_ebd, dim=-1)
#         pos_ebd = F.normalize(pos_ebd, dim=-1)
#         neg_ebds = F.normalize(neg_ebds, dim=-1)

#         sim_pos = F.cosine_similarity(q_ebd, pos_ebd, dim=-1)  # (batch,)
        
#         # (batch, 3)
#         sim_neg = F.cosine_similarity(
#             q_ebd.unsqueeze(1).expand(-1, N, -1),  # (batch, N, d_ebd)
#             neg_ebds,
#             dim=-1
#         ) 

#         target = torch.ones_like(sim_neg) 


#         loss = F.margin_ranking_loss(
#                         sim_pos.unsqueeze(1).expand_as(sim_neg),  # (B, N)
#                         sim_neg, 
#                         target, 
#                         margin=self.margin,
#                         reduction='mean'
#                         )

#         return loss
    
# class MarginRankLossWithProvidedAndTopKInBatchNegatives(nn.Module):
#     def __init__(self, margin=0.2, top_k_in_batch=3):
#         super().__init__()

#         self.margin = margin
#         self.top_k_in_batch = top_k_in_batch

#     def forward(self, q_ebd, pos_ebd, neg_ebds):
#         """
#         Args:
#             q_ebd:      (batch, d_ebd) 
#             pos_ebd:    (batch, d_ebd) 
#             neg_ebds:   (batch, N_provided, d_ebd)

#         Returns:
#             Scalar loss
#         """
#         B, N_provided, D = neg_ebds.shape # 獲取 batch size, 提供的負樣本數, embedding 維度


#         # 歸一化 embeddings 以計算 cosine similarity
#         q_ebd_norm = F.normalize(q_ebd, dim=-1)
#         pos_ebd_norm = F.normalize(pos_ebd, dim=-1)
#         neg_ebds_norm = F.normalize(neg_ebds, dim=-1)

#         # 1. 計算正樣本相似度 (每個 query 對應其 positive answer)
#         sim_pos = F.cosine_similarity(q_ebd_norm, pos_ebd_norm, dim=-1)  # Shape: (B,)

#         # 2. 計算與提供的負樣本的相似度
#         q_ebd_expanded_for_provided = q_ebd_norm.unsqueeze(1).expand(-1, N_provided, -1) # (B,1,D) -> (B, N_provided, D)
#         sim_provided_neg = F.cosine_similarity(
#             q_ebd_expanded_for_provided,
#             neg_ebds_norm,
#             dim=-1
#         ) # Shape: (B, N_provided)


#         # 計算所有 query 與所有 positive answer 之間的相似度矩陣
#         all_sim = q_ebd_norm @ pos_ebd_norm.T  # Shape: (B, B)

#         # 將對角線相似度設置為負無窮，以排除自身
#         mask_value = -torch.inf
#         all_sim.fill_diagonal_(mask_value)


#         # 找出每一行 (每個 query) 的 Top-K 最大相似度值
#         sim_topk_in_batch_neg, _ = torch.topk(
#             all_sim, k=self.top_k_in_batch, dim=1, largest=True, sorted=False # sorted=False 可能稍快
#         ) # Shape: (B, actual_top_k)


#         # 4. 合併所有負樣本相似度

#             # 將提供的負樣本相似度和 Top-K In-Batch 負樣本相似度合併
#         sim_neg_combined = torch.cat(
#             (sim_provided_neg, sim_topk_in_batch_neg),
#             dim=1
#         ) # Shape: (B, N_provided + actual_top_k)


#         # 5. 計算 Margin Ranking Loss
#         N_total_neg = sim_neg_combined.shape[1] # 總的負樣本數
        
#         # N_total_neg = sim_topk_in_batch_neg.shape[1] # 總的負樣本數


#         # 將 sim_pos 擴展以匹配 sim_neg_combined 的形狀
#         sim_pos_expanded = sim_pos.unsqueeze(1).expand(-1, N_total_neg) # Shape: (B, N_total_neg)

#         # target 通常是全 1 的 tensor
#         target = torch.ones_like(sim_neg_combined) # Shape: (B, N_total_neg)
        
#         # target = torch.ones_like(sim_topk_in_batch_neg) # Shape: (B, in_batch_neg)

#         loss = F.margin_ranking_loss(
#             sim_pos_expanded,      
#             sim_neg_combined,      # (B, N_total_neg)
#             # sim_topk_in_batch_neg, # (B, in_batch_neg)
#             target,                # (B, N_total_neg)
#             margin=self.margin,
#             reduction='mean'
#         )

#         return loss
