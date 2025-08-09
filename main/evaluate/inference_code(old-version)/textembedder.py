from transformers import AutoTokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.joinpath('utility')))
from word_seg import Segmenter
from ebd_ini import ContextualEmbeddingExtractor
import torch
from torch.nn.utils.rnn import pad_sequence


class TextEmbedder:
    def __init__(self,seg_type,device='cuda'):
        self.device = device
        tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3',trust_remote_code=True)
        self.segmenter = Segmenter(tokenizer,seg_tool = seg_type)
        self.ctt_ebd = ContextualEmbeddingExtractor(device)
    
    def collat_fn(self,ebds:list[torch.tensor]):
        max_len = max(map(len,ebds))
        batch_atten_mask = []
        for ebd in ebds:
            padding_len = max_len - len(ebd)
            atten_mask = [1] * len(ebd) + [0] * padding_len
            batch_atten_mask.append(atten_mask)
        
        # (batch,seq_len,d_ebd)
        batch_ebds = pad_sequence(ebds,batch_first=True)
        # (batch,seq_len)
        batch_atten_mask = torch.tensor(batch_atten_mask)
        
        return batch_ebds,batch_atten_mask
    
    def __call__(self,texts:str|list[str]):
        if isinstance(texts,str) or not hasattr(texts,"__len__"):
            texts = [texts]
            
        token_ebds = []

        for text in texts:
            inputs = self.segmenter(text)
            token_ebds.append(self.ctt_ebd(**inputs))
    
        token_ebds,atten_masks = self.collat_fn(token_ebds)

        if self.device:
            token_ebds = token_ebds.to(self.device)
            atten_masks = atten_masks.to(self.device)
        # token_ebds(batch,seq_len,d_ebd) | atten_masks(batch,seq_len)
        return {
                'token_ebds':token_ebds,
                'atten_masks':atten_masks
                }
