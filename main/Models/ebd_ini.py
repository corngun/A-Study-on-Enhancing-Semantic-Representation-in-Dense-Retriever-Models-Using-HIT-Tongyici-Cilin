import torch

class ContextualEmbeddingExtractor :
    def __init__(self, pre_ebd_model,device='cuda'):
        self.device = device
        self.model = pre_ebd_model.to(device)
        self.model.eval()
        
    def __call__(self,input_ids:list):
        """Using last hidden state"""
        inputs = {
            'input_ids': torch.tensor(input_ids).unsqueeze(0).to(self.device),
            'attention_mask': torch.ones(len(input_ids)).unsqueeze(0).to(self.device),
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)
        # (batch,seq_len,n_dim) -> (seq_len,n_dim)
            last_hidden_state = (outputs.last_hidden_state).squeeze(dim=0).cpu().float()
        
        # (seq_len,n_dim)
        return last_hidden_state
        
