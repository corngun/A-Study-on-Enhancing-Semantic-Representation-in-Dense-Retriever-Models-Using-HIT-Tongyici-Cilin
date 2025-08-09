import numpy as np
import torch
import os
from Models.model import ScopeEnhancedDualEncoder,ScopeEnhancedSiameseEncoder
from pathlib import Path

class EarlyStopping:
    """
    Args:
        patience(int):允許連續幾個 epoch 沒進步。
        delta(float): 允許的最小進步幅度（防止小幅波動也被視為進步）
    """
    def __init__(self, model_arch,save_path:str,patience=3,verbose=True, delta=0.01,k=3):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        self.counter = 0
        self.val_acc_min = -np.inf
        self.save_path = save_path
        self.model_architecture = model_arch
        self.k = k

    def __call__(self, val_recall_acc, model:ScopeEnhancedDualEncoder):
        # val_loss 越小越好，轉成負值樂成「越大越好」，以符合 score 的含意
        # score = -val_loss
        score = val_recall_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_recall_acc,model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_recall_acc,model)
            self.counter = 0
            
    def save_checkpoint(self,val_recall_acc,model:ScopeEnhancedDualEncoder | ScopeEnhancedSiameseEncoder):
        if self.verbose:
            print(f'Validation recall@{self.k} acc increased ({self.val_acc_min:.6f} --> {val_recall_acc:.6f}).  Saving model ...')
        
        if not Path(self.save_path).exists():
            Path(self.save_path).mkdir(parents=True,exist_ok=True)
        
        full_model_path = os.path.join(self.save_path,'best_checkpoint.pt')
        torch.save(model.state_dict(),full_model_path)
        if self.model_architecture == 'dual-encoder':
            question_encoder_path = os.path.join(self.save_path,'question_encoder.pt')
            answer_encoder_path = os.path.join(self.save_path,'answer_encoder.pt')
            
            torch.save(model.question_encoder.state_dict(),question_encoder_path)
            torch.save(model.answer_encoder.state_dict(),answer_encoder_path)
        else:
            encoder_path = os.path.join(self.save_path,'encoder.pt')
            torch.save(model.encoder.state_dict(),encoder_path)
        
        self.val_acc_min = val_recall_acc
