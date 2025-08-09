import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

@dataclass
class QA_batch:
    question_ebds: torch.Tensor
    q_scope_labels: torch.Tensor
    q_atten_masks: torch.Tensor
    q_span_tables:list[list[int]]
    positive_ebds: torch.Tensor
    positive_scope_labels: torch.Tensor
    positive_atten_masks: torch.Tensor
    positive_span_tables: list[list[int]]
    negative_ebds: torch.Tensor 
    negative_atten_masks: torch.Tensor
    negative_span_tables:list[list[int]]
    
    
    def to(self, device):
        for field in self.__dataclass_fields__:
            tensor = getattr(self, field)
            if isinstance(tensor,torch.Tensor) :
                setattr(self, field, tensor.to(device))
        return self


class Collate_fn:
    def __init__(self,padding_label:int=-100):
        # self.with_negative_sample = with_negative_sample
        self.padding_label = padding_label
        
    def pad_label(self,batch_items:list[torch.Tensor]):
        max_length = max(map(len,batch_items)) 
        batch_labels = []
        
        for items in batch_items:
            padding_len = max_length - len(items)
            pad_labels = items.tolist() + [self.padding_label] * padding_len
            batch_labels.append(pad_labels)
        
        # (batch,seq_len),(batch,seq_len)
        return torch.tensor(batch_labels)
    
    def pad_negative_ebds(self,negative_ebds:list[list[torch.Tensor]]):
        # List[Tensor[n, max_len_i, d]]
        # 先將每筆資料中的負樣本(n個) padding
        neg_ebds = [pad_sequence(negs, batch_first=True).type(torch.float32) for negs in negative_ebds]
        # 選擇所有 batch 中負樣本最長的
        neg_max_len = max(t.size(1) for t in neg_ebds)
        neg_dim = neg_ebds[0].size(-1)   
        # 每個 sample 負樣本數量是固定的
        number_neg = neg_ebds[0].size(0)   
        neg_padded = []
        neg_masks = []
        
        for neg in neg_ebds:
            pad_len = neg_max_len - neg.size(1)
            if pad_len > 0:
                pad = torch.zeros((number_neg, pad_len, neg_dim), dtype=neg.dtype)
                # pad = torch.zeros((1, pad_len, neg_dim), dtype=neg.dtype)
                neg = torch.cat([neg, pad], dim=1)  # Tensor[3,neg_max_len,d_ebd]
            mask = (neg.abs().sum(dim=-1) > 0).type(torch.int16) # Tensor[3,neg_max_len]
            neg_padded.append(neg)  
            neg_masks.append(mask)  
            
        neg_ebds = torch.stack(neg_padded, dim=0) # Tensor[batch_size,3,neg_max_len,d_ebd]
        negative_atten_masks = torch.stack(neg_masks, dim=0) # Tensor[batch_size, 3, neg_max_len]
        
        return neg_ebds,negative_atten_masks
    
    def convert_mapping_lengths_to_spans(self,mappings:list[torch.Tensor]) -> list[list[int]]:
        """
            Args:
                mappings : [ Tensor[[1],[1],[2],,,] , Tensor[[1],[2],[2],,,] ]
                將原本的 mapping_table 例如 [[1], [1], [2], [1]] → [[0], [1], [2,3], [4]]
        """
        span_tables = []
        
        for mapping_table in mappings:
            spans = []
            idx = 0
            for l in mapping_table:
                
                count = l[0]  # 因為是 [[1], [1], [2], ...] 這種格式
                spans.append(list(range(idx, idx + count)))
                idx += count
                
            span_tables.append(spans)
        return span_tables
        
    
    def pad_mapping_table(self,mappings:list[torch.Tensor]) -> list[list[int]]:
        """
        Args:
            mappings : [ Tensor[[1],[1],[2],,,] , Tensor[[1],[2],[2],,,] ]
        
        Return: 
            list[list[list[int]]] : [ [[0], [1], [2,3], [4]] , [[0], [1], [2], [3,4]] ] #[B][global_seq_len,word_spans]
        """
        span_tables = self.convert_mapping_lengths_to_spans(mappings)
        
        padded_span_tables = []
        max_length = max(map(len,span_tables)) 
        for spans in span_tables:
            pad_len = max_length - len(spans)
            if pad_len > 0:
                spans += [[-1]] * pad_len
            padded_span_tables.append(spans)
        
        return padded_span_tables
    
    def pad_negative_span_tables(self, negative_span_tables):
        """
        Args:
            negative_span_tables: List[batch][local_max_neg_num][span_list]
        Returns:
            padded_neg_span_tables: List[batch][global_max_neg_num][max_word_len]
        """
        # 先找 batch 內所有負樣本的最大 span 數
        max_len = 0
        for batch_spans in negative_span_tables:
            for neg_spans in batch_spans:
                max_len = max(max_len, len(neg_spans))
                
        # 開始補 pad
        padded_batch = []
        for batch_spans in negative_span_tables:
            padded_negs = []
            for neg_spans in batch_spans:
                pad_len = max_len - len(neg_spans)
                padded_neg = neg_spans + [[-1]] * pad_len
                padded_negs.append(padded_neg)
            padded_batch.append(padded_negs)
        return padded_batch

    def __call__(self,batch_items) -> QA_batch:

        question_ebds = [item['question_ebd'][1:-1] for item in batch_items]
        positive_ebds = [item['positive_ebd'][1:-1] for item in batch_items]
        question_scope_labels = [item['question_scope_labels'] for item in batch_items]
        positive_scope_labels = [item['positive_scope_labels'] for item in batch_items]

        negative_ebds = [ [ neg_ebd[1:-1] for neg_ebd in item['negative_ebds']] for item in batch_items ]
        # negative_ebds = [ [ neg[1:-1] for idx,neg in enumerate(item['negative_ebds']) if idx != len(item['negative_ebds']) - 1 ] for item in batch_items ]
        
        # negative_ebds = [ [item['negative_ebds'][0][1:-1]] for item in batch_items ] # 改用成一個 hard negative
        negative_ebds,negative_atten_masks = self.pad_negative_ebds(negative_ebds)
        
        # 當初做資料前處理時，也有將 <s>、</s> token 算進去
        question_mapping_tables = [item['question_mapping_tables'][1:-1] for item in batch_items]
        positive_mapping_tables = [item['positive_mapping_tables'][1:-1] for item in batch_items]
        negative_mapping_tables = [ [ neg[1:-1] for neg in item['negative_mapping_tables'] ] for item in batch_items]
            

        q_scope_labels = self.pad_label(question_scope_labels)
        positive_scope_labels = self.pad_label(positive_scope_labels)
        
        q_span_tables = self.pad_mapping_table(question_mapping_tables)
        positive_span_tables = self.pad_mapping_table(positive_mapping_tables)
        
        # 與 pad_neg_ebd 一樣，先對每筆資料中的負樣本 padding
        raw_negative_span_tables = [self.pad_mapping_table(neg) for neg in negative_mapping_tables]
        negative_span_tables = self.pad_negative_span_tables(raw_negative_span_tables) 
        
        question_ebds = pad_sequence(question_ebds, batch_first=True).type(torch.float32) # (batch,max_seq,d_ebd)
        positive_ebds = pad_sequence(positive_ebds, batch_first=True).type(torch.float32) # (batch,max_seq,d_ebd)
        
        
        q_atten_masks = (question_ebds.abs().sum(dim=-1) > 0).type(torch.int16)  # (batch,max_seq)
        positive_atten_masks = (positive_ebds.abs().sum(dim=-1) > 0).type(torch.int16) # (batch,max_seq)
        
        batchs = {
            'question_ebds':question_ebds,
            'q_scope_labels':q_scope_labels,
            'q_atten_masks':q_atten_masks,
            'q_span_tables':q_span_tables,
            'positive_ebds':positive_ebds,
            'positive_scope_labels':positive_scope_labels,
            'positive_atten_masks':positive_atten_masks,
            'positive_span_tables':positive_span_tables,
            'negative_ebds':negative_ebds,
            'negative_atten_masks':negative_atten_masks,
            'negative_span_tables':negative_span_tables
        }
            

        
        return QA_batch(**batchs)



# class Collate_fn:
#     def __init__(self,with_negative_sample,padding_label:int=-100):
#         self.with_negative_sample = with_negative_sample
#         self.padding_label = padding_label
        
#     def pad_label(self,batch_items:list[torch.Tensor]):
#         max_length = max(map(len,batch_items)) 
#         batch_labels = []
        
#         for items in batch_items:
#             padding_len = max_length - len(items)
#             pad_labels = items.tolist() + [self.padding_label] * padding_len
#             batch_labels.append(pad_labels)
        
#         # (batch,seq_len),(batch,seq_len)
#         return torch.tensor(batch_labels)
    
#     def pad_negtive_ebds(self,negative_ebds:list[list[torch.Tensor]]):
#         # List[Tensor[n, max_len_i, d]]
#         neg_ebds = [pad_sequence(negs, batch_first=True).type(torch.float32) for negs in negative_ebds]
#         neg_max_len = max(t.size(1) for t in neg_ebds)
#         neg_dim = neg_ebds[0].size(-1)   
#         # 每個 sample 負樣本數量是固定的
#         number_neg = neg_ebds[0].size(0)   
#         neg_padded = []
#         neg_masks = []
        
#         for neg in neg_ebds:
#             pad_len = neg_max_len - neg.size(1)
#             if pad_len > 0:
#                 pad = torch.zeros((number_neg, pad_len, neg_dim), dtype=neg.dtype)
#                 # pad = torch.zeros((1, pad_len, neg_dim), dtype=neg.dtype)
#                 neg = torch.cat([neg, pad], dim=1)  # Tensor[3,neg_max_len,d_ebd]
#             mask = (neg.abs().sum(dim=-1) > 0).type(torch.int16) # Tensor[3,neg_max_len]
#             neg_padded.append(neg)  
#             neg_masks.append(mask)  
            
#         neg_ebds = torch.stack(neg_padded, dim=0) # Tensor[batch_size,3,neg_max_len,d_ebd]
#         negative_atten_masks = torch.stack(neg_masks, dim=0) # Tensor[batch_size, 3, neg_max_len]
        
#         return neg_ebds,negative_atten_masks

#     def __call__(self,batch_items) -> QA_batch:

#         question_ebds = [item['question_ebd'][1:-1] for item in batch_items]
#         positive_ebds = [item['positive_ebd'][1:-1] for item in batch_items]
#         question_scope_labels = [item['question_scope_labels'] for item in batch_items]
#         positive_scope_labels = [item['positive_scope_labels'] for item in batch_items]

#         if self.with_negative_sample:
#             negative_ebds = [ [ neg[1:-1] for neg in item['negative_ebds']] for item in batch_items ]
#             # negative_ebds = [ [item['negative_ebds'][0][1:-1]] for item in batch_items ]
#             negative_ebds,negative_atten_masks = self.pad_negtive_ebds(negative_ebds)
            
#             # negative_ebds =  pad_sequence(negative_ebds, batch_first=True).type(torch.float32)
#             # negative_atten_masks = (negative_ebds.abs().sum(dim=-1) > 0).type(torch.int16)

#         q_scope_labels = self.pad_label(question_scope_labels)
#         positive_scope_labels = self.pad_label(positive_scope_labels)

#         question_ebds = pad_sequence(question_ebds, batch_first=True).type(torch.float32) # (batch,max_seq,d_ebd)
#         positive_ebds = pad_sequence(positive_ebds, batch_first=True).type(torch.float32) # (batch,max_seq,d_ebd)
        
#         q_atten_masks = (question_ebds.abs().sum(dim=-1) > 0).type(torch.int16)  # (batch,max_seq)
#         positive_atten_masks = (positive_ebds.abs().sum(dim=-1) > 0).type(torch.int16) # (batch,max_seq)
        
#         if self.with_negative_sample:
#             batchs = {
#                 'question_ebds':question_ebds,
#                 'q_scope_labels':q_scope_labels,
#                 'q_atten_masks':q_atten_masks,
#                 'positive_ebds':positive_ebds,
#                 'positive_scope_labels':positive_scope_labels,
#                 'positive_atten_masks':positive_atten_masks,
#                 'negative_ebds':negative_ebds,
#                 'negative_atten_masks':negative_atten_masks
#             }
#         else:
#             batchs = {
#                 'question_ebds':question_ebds,
#                 'q_scope_labels':q_scope_labels,
#                 'q_atten_masks':q_atten_masks,
#                 'positive_ebds':positive_ebds,
#                 'positive_scope_labels':positive_scope_labels,
#                 'positive_atten_masks':positive_atten_masks
#             }
        
#         return QA_batch(**batchs)
    
    
    