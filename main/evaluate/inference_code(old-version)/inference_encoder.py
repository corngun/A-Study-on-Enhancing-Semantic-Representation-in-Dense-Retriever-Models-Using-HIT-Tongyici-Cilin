import torch.nn as nn
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.joinpath('Models')))
from Encoder_Component import Encoder
import json
from dataclasses import dataclass
import numpy as np
from textembedder import TextEmbedder

@dataclass 
class ScopeEnhanced_embedding:
    last_hidden_state:torch.Tensor
    mean_pooled:torch.Tensor

class InferenceEncoder(nn.Module):
    def __init__(self, config_path,checkpoint_path,seg_type,device = 'cuda'):
        super(InferenceEncoder,self).__init__()
        self.embedder = TextEmbedder(seg_type,device)
        self._encoder = self.Load_encoder(config_path,checkpoint_path)
        if device:
            self._encoder.to(device)
            
        self._encoder.eval()

    def forward(self, token_ebds: torch.Tensor, atten_masks: torch.Tensor):
        encoderout = self.encoder(token_ebds, atten_masks)
        # (batch,seq_len,d_ebd) 
        last_hidden_state = encoderout.last_hidden_state
        # 不要包括 cls token
        mean_pooled = self.mean_pooling(last_hidden_state, atten_masks[:,1:])

        return ScopeEnhanced_embedding(
            last_hidden_state,
            mean_pooled
        )

    
    @torch.inference_mode()
    def encode(
        self,
        sentences:str|list[str],
        batch_size: int = 64,
        ):
        
        input_is_string = False
        if isinstance(sentences,str) or not hasattr(sentences,"__len__"):
            sentences = [sentences]
            input_is_string = True
        
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            batch_ini_embedding:dict = self.embedder(batch_sentences)
            outputs:ScopeEnhanced_embedding = self.forward(**batch_ini_embedding) # (batch,seq_len,d_ebd),(batch,d_ebd)
            
            # (batch,d_ebd)
            sentence_ebds = outputs.mean_pooled
            # embeddings = self.mean_pooling(outputs.last_hidden_state,results['atten_masks'])
            
            # extend 會將 2D tensor 拆成 list of 1D tensors，也就是會將 batch 的 dim 攤平變成 len(batch) 的 vectors
            all_embeddings.extend(sentence_ebds)

        # 對每個句子做 normalize ; len(sentences) * d_ebd -> len(sentences) * d_ebd
        all_embeddings = [nn.functional.normalize(embedding,dim=0) for embedding in all_embeddings ]

        # (len(sentences),d_ebd)
        all_embeddings = np.asarray([emb.cpu().numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]
        
        return all_embeddings
    
    
    def Load_encoder(self,config_path,checkpoint_path):
        with open(config_path,'r') as f:
            configs = json.load(f)
        
        model = Encoder(**configs)
        model.load_state_dict(torch.load(checkpoint_path,weights_only=True))
        
        return model
        
    @property
    def encoder(self):
        return self._encoder
    
    @staticmethod
    def mean_pooling(ebds: torch.Tensor, attention_mask: torch.Tensor):
        expand_attention_mask = attention_mask.unsqueeze(dim=-1).expand(ebds.size())
        return torch.sum(expand_attention_mask * ebds, dim=1) / torch.clamp(expand_attention_mask.sum(dim=1), min=1e-4)
