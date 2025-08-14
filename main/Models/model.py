import torch
import torch.nn as nn
from Models.Encoder_Component import Encoder
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

@dataclass
class ScopeEnhancedEncoder_Out:
    q_sentence_ebds:torch.Tensor
    positive_sentence_ebds:torch.Tensor
    q_scope_logits:torch.Tensor = None
    positive_scope_logits:torch.Tensor = None
    neg_sentence_ebds:torch.Tensor = None



            
class ScopeEnhancedDualEncoder(nn.Module):
    def __init__(self,config,**kwargs):
        super(ScopeEnhancedDualEncoder,self).__init__()
        d_in,n_head,d_ff,dropout,number_of_layers = (
            config['d_in'],
            config['n_head'],
            config['d_ff'],
            config['dropout'],
            config['number_of_layers'],
        )
        
        self.train_type = kwargs.get('train_type')
        num_scopes = kwargs.get('num_scopes')
        self.pooling_method = kwargs.get('pooling_method')
        
        if self.train_type == 'scope_enhanced':
            self.scope_proj_head = nn.Sequential(
                nn.Linear(d_in,num_scopes),
                nn.Dropout(0.1)
            )
            
            if self.pooling_method == 'scope_aware':
                self.scope_importance = nn.Parameter(torch.rand(num_scopes))

        
        self.question_encoder = Encoder(d_in,n_head,d_ff,dropout,number_of_layers)
        self.answer_encoder = Encoder(d_in,n_head,d_ff,dropout,number_of_layers)
    
    def average_by_spans(self,seq_output: torch.Tensor, spans: list[torch.Tensor])-> torch.Tensor:
        """
        seq_output: [seq_len, d_model] - 一個樣本的 encoder 輸出
        spans: List[ Tensor ] - 該樣本的詞對應的 subword index，例如 [ Tensor[ [0], [1], [2,3], [4], ...], ... ]
        return: Tensor [word_len, d_model]
        """
        word_embeds = []
        for span in spans:
            if span == [-1] :  # 跳過 padding
                continue
            span_tensor = seq_output[span, :]  # [num_subwords, d_model]
            word_embed = span_tensor.mean(dim=0)  # [d_model]
            word_embeds.append(word_embed)
            
        return torch.stack(word_embeds)
        

    def forward(
            self,
            q_seq_ebd:torch.Tensor,
            q_atten_mask:torch.Tensor,
            q_span_tables:list[list[int]],
            postive_seq_ebd:torch.Tensor,
            postive_atten_mask:torch.Tensor,
            positive_span_tables:list[list[int]],
            negative_ebds:torch.Tensor,
            negative_atten_mask:torch.Tensor,
            neg_span_tables
            ):
        

        q_out = self.question_encoder(q_seq_ebd,q_atten_mask) 
        postive_out = self.answer_encoder(postive_seq_ebd,postive_atten_mask) 
        
        
        # if negative_ebds != None:
        B, N, L, D = negative_ebds.shape
        flattened = negative_ebds.view(B * N, L, D)
        flattened_mask = negative_atten_mask.view(B * N, L)

        neg_out = self.answer_encoder(flattened, flattened_mask) 
        # neg_out_last_hidden_state = neg_out.last_hidden_state # (batch×3, seq_len, d_ebd)
        # neg_setence_ebds = self.mean_pooling(neg_out_last_hidden_state,flattened_mask)  # (batch*3, d_ebd)

        
        
        if self.train_type == 'scope_enhanced' :
            batch_size = q_out.last_hidden_state.size(0)
            
            q_word_reprs = [
                        self.average_by_spans(q_out.last_hidden_state[i], q_span_tables[i]) 
                        for i in range(batch_size)
                        ]
            
            positive_word_reprs = [
                self.average_by_spans(postive_out.last_hidden_state[i], positive_span_tables[i])
                for i in range(batch_size)
            ]
            

            q_scope_logits = [self.scope_proj_head(x) for x in q_word_reprs] 
            positive_scope_logits = [self.scope_proj_head(x) for x in positive_word_reprs] 
            q_scope_logits = pad_sequence(q_scope_logits,batch_first=True) # (batch,global_max_len,num_scopes)
            positive_scope_logits = pad_sequence(positive_scope_logits,batch_first=True)
            
            if self.pooling_method == 'scope_aware':
                q_sentence_ebds = self.scope_aware_pooling(q_word_reprs,q_scope_logits,q_span_tables)
                positive_sentence_ebds = self.scope_aware_pooling(positive_word_reprs,positive_scope_logits,positive_span_tables)
                
                neg_out_last_hidden_state = neg_out.last_hidden_state # (batch×3, seq_len, d_ebd)
                neg_out_last_hidden_state = neg_out_last_hidden_state.view(B, N, L, D) # (batch, 3, seq_len, d_ebd)
                neg_word_reprs = [
                    [
                        self.average_by_spans(neg_out_last_hidden_state[i, j], neg_span_tables[i][j])
                        for j in range(N)
                    ]
                    for i in range(batch_size)
                ] # [B][N][word_len, d_ebd]
                
                
                negative_scope_logits = [[self.scope_proj_head(x) for x in neg] for neg in neg_word_reprs] 
                # negative_scope_logits: [B][N, max_word_len, num_scopes]
                negative_scope_logits = [pad_sequence(neg_logits,batch_first=True) for neg_logits in negative_scope_logits]
                
                
                neg_sentence_ebds = [
                    torch.stack([
                        self.scope_aware_pooling(
                            neg_word_reprs[i][j], 
                            negative_scope_logits[i][j], 
                            neg_span_tables[i][j],
                            is_negative=True
                        )
                        for j in range(N)
                    ], dim=0) # (N,d_ebd)
                    for i in range(B)
                ] 
                
                # 最終堆疊所有 batch
                neg_sentence_ebds = torch.stack(neg_sentence_ebds, dim=0)  #(batch,n,d_ebd)
            
            elif self.pooling_method == 'cls':
                q_sentence_ebds = q_out.cls_out
                positive_sentence_ebds = postive_out.cls_out
                neg_setence_ebds = neg_out.cls_out
                neg_sentence_ebds = neg_setence_ebds.view(B, N, -1) # (batch, 3, d_ebd)
                
                
                
        else:   
            # (batch,d_ebd)
            q_sentence_ebds = q_out.cls_out
            positive_sentence_ebds = postive_out.cls_out
            neg_setence_ebds = neg_out.cls_out
            neg_sentence_ebds = neg_setence_ebds.view(B, N, -1) # (batch, 3, d_ebd)
        

        if self.train_type == 'scope_enhanced':
            return ScopeEnhancedEncoder_Out(
                        q_sentence_ebds,
                        positive_sentence_ebds,
                        q_scope_logits,
                        positive_scope_logits,
                        neg_sentence_ebds
                    )

                
        elif self.train_type == 'base':
            return ScopeEnhancedEncoder_Out(
                    q_sentence_ebds,
                    positive_sentence_ebds,
                    neg_sentence_ebds = neg_sentence_ebds
                )

            

    
    def scope_aware_pooling(
                            self, 
                            word_reprs: torch.Tensor, 
                            scope_logits: torch.Tensor, 
                            span_tables,
                            is_negative=False
                            ) -> torch.Tensor:
        
        if is_negative:
            # 處理單個負樣本
            valid_indices = [i for i, span in enumerate(span_tables) if span != [-1]]
            valid_word_repr = word_reprs[valid_indices]  # [valid_len, d_ebd]
            valid_scope_logit = scope_logits[valid_indices]  # [valid_len, num_scopes]
            scope_probs = torch.softmax(valid_scope_logit, dim=-1)
            importance_scores = torch.matmul(scope_probs, self.scope_importance)  # [valid_len]
    
            scope_weights = torch.softmax(importance_scores, dim=-1)  # [valid_len]
            pooled = torch.sum(scope_weights.unsqueeze(-1) * valid_word_repr, dim=0) # (d_ebd)
            
        else: 
            # 處理批次樣本（正樣本和問題）
            word_reprs = pad_sequence(word_reprs, batch_first=True) # (batch,global_max_len,d_ebd)
            # note: padding logits 因為全是 0 value，因此做 softmax 會是一個均勻分布的 value
            scope_probs = torch.softmax(scope_logits, dim=-1)    # (batch,global_max_len,num_scopes)                
            
            mask = torch.tensor([
                    [0 if span == [-1] else 1 for span in spans] 
                    for spans in span_tables
                ], device=word_reprs.device)  # (batch, global_max_len)
            
            importance_scores = torch.matmul(scope_probs, self.scope_importance) # (batch, global_max_len)
            # 把 padding 的 位置 mask 掉
            masked_scores = importance_scores.masked_fill(mask == 0, float('-1e4'))  # (batch,global_max_len) 
            scope_weights = torch.softmax(masked_scores, dim=-1)              # (batch,global_max_len)

            
            pooled = torch.sum(scope_weights.unsqueeze(-1) * word_reprs, dim=1)      # [batch, d_ebd]
            
        return pooled



    









