from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from metrics import Cls_Acc,Cal_recall_at_k
from early_stopping import EarlyStopping
import wandb
from utility.util import Print_metrics
from Models.model import ScopeEnhancedEncoder_Out,ScopeEnhancedDualEncoder,ScopeEnhancedSiameseEncoder
from Models.LossFn import InfoNCELoss


class Trainer:
    def __init__(
        self,
        model:nn.Module,
        accum_iter:int, 
        scope_pred_criterion:torch.nn,
        cl_criterion:InfoNCELoss,
        loss_weight:dict,
        optimizer:torch.optim.Optimizer,
        lr_scheduler:torch.optim.lr_scheduler,
        epochs:int,
        num_scopes:int,
        k:int,
        with_negative_sample:bool,
        train_type:str,
        device:str='cuda'
        ):
        
        self.model:ScopeEnhancedDualEncoder = model.to(device)
        # self.model:ScopeEnhancedSiameseEncoder = model.to(device)
        self.accum_iter = accum_iter
        self.scope_pred_loss_fn = scope_pred_criterion
        self.cl_loss_fn = cl_criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_scopes = num_scopes
        self.k = k
        self.device = device
        self.epochs = epochs
        self.loss_weight = loss_weight
        self.lambda_q = loss_weight['q']
        self.lambda_positive = loss_weight['positive']
        self.scalar = torch.amp.GradScaler()
        self.withnegative_sample = with_negative_sample
        self.train_type = train_type
    
    @staticmethod
    def compute_loss(
                model_outputs:ScopeEnhancedEncoder_Out,
                loss_weight:dict,
                cl_loss_fn,
                scope_pred_loss_fn,
                train_type,
                with_negative_sample,
                num_scopes,
                q_scope_labels,
                positive_labels,
                ):

        """
        logits (batch,seq_len,num_of_labels) -> (batch*seq_len,num_of_labels)
        labels (batch,seq_len) -> (batch*seq_len,)
        """
        lambda_q,lambda_positive = loss_weight['q'],loss_weight['positive']

        if with_negative_sample :
            cl_loss = cl_loss_fn(
                                model_outputs.q_sentence_ebds,
                                model_outputs.positive_sentence_ebds,
                                model_outputs.neg_sentence_ebds
                                )

        else:
            cl_loss = cl_loss_fn(model_outputs.q_sentence_ebds,model_outputs.positive_sentence_ebds)

        
        if train_type == 'scope_enhanced':
        
            q_scope_loss = scope_pred_loss_fn(model_outputs.q_scope_logits.view(-1,num_scopes),q_scope_labels.view(-1)) 
            positive_scope_loss = scope_pred_loss_fn(model_outputs.positive_scope_logits.view(-1,num_scopes),positive_labels.view(-1))
            scope_pred_loss = (q_scope_loss + positive_scope_loss) / 2.

            total_loss = cl_loss +\
                         lambda_q * q_scope_loss +\
                         lambda_positive * positive_scope_loss 
                    
            return {
                'scope_pred_loss':scope_pred_loss,
                'cl_loss':cl_loss,
                'total_loss':total_loss
                }
            
        elif train_type == 'base':
            total_loss = cl_loss
            
            return {
                    'total_loss':total_loss
                    }
        

    @staticmethod
    def compute_scope_acc(q_logits,positive_logits,q_labels,positive_labels):
        q_scope_pred = torch.softmax(q_logits,dim=-1).argmax(dim=-1)
        positive_scope_pred = torch.softmax(positive_logits,dim=-1).argmax(dim=-1)

        q_scope_acc = Cls_Acc(q_scope_pred.cpu(),q_labels.cpu())
        positive_scope_acc = Cls_Acc(positive_scope_pred.cpu(),positive_labels.cpu())
        scope_pred_acc = (q_scope_acc + positive_scope_acc) / 2

        return scope_pred_acc
        
        
    def training_one_epoch(self,epoch,train_dataloader:DataLoader):
        
        cur_epoch = epoch
        train_scope_pred_loss = 0.0
        train_cl_loss = 0.0
        train_total_loss = 0.0
        train_scope_pred_acc = 0.0
        train_q_sen_ebds = []
        train_positive_sen_ebds = []
        train_negative_sen_ebds = []
        
        
        self.model.train()
        
        for step,batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader),desc='Traning'):
            batch = batch.to(self.device)
            q_ebds,q_scope_labels,q_atten_masks = batch.question_ebds,batch.q_scope_labels,batch.q_atten_masks
            positive_ebds,positive_scope_labels,positive_atten_masks = batch.positive_ebds,batch.positive_scope_labels,batch.positive_atten_masks
            q_span_tables,positive_span_tables,neg_span_tables = batch.q_span_tables,batch.positive_span_tables,batch.negative_span_tables
            

            negative_ebds,negative_atten_masks = batch.negative_ebds,batch.negative_atten_masks
            
            
            with torch.amp.autocast(device_type='cuda'):
                model_out:ScopeEnhancedEncoder_Out = self.model(
                                                                q_ebds,
                                                                q_atten_masks,
                                                                q_span_tables,
                                                                positive_ebds,
                                                                positive_atten_masks,
                                                                positive_span_tables,
                                                                negative_ebds,
                                                                negative_atten_masks,
                                                                neg_span_tables
                                                            )

                
            train_q_sen_ebds.append(model_out.q_sentence_ebds.detach().cpu())
            train_positive_sen_ebds.append(model_out.positive_sentence_ebds.detach().cpu())
            train_negative_sen_ebds.append(model_out.neg_sentence_ebds.detach().cpu())
            



            loss_dict = self.compute_loss(
                                        model_out,
                                        self.loss_weight,
                                        self.cl_loss_fn,
                                        self.scope_pred_loss_fn,
                                        self.train_type,
                                        self.withnegative_sample,
                                        self.num_scopes,
                                        q_scope_labels,
                                        positive_scope_labels,
                                    )
            

            if self.train_type == 'scope_enhanced':

                train_scope_pred_loss += loss_dict['scope_pred_loss'].item() 
                train_cl_loss += loss_dict['cl_loss'].item()
                
                scope_pred_acc = self.compute_scope_acc(
                                                model_out.q_scope_logits,
                                                model_out.positive_scope_logits,
                                                q_scope_labels,
                                                positive_scope_labels
                                            )
                train_scope_pred_acc += scope_pred_acc


            total_loss = loss_dict['total_loss']
            train_total_loss += total_loss.item()
            
            
            if self.accum_iter != None and self.accum_iter > 1:
                # 1. 需要除上 accum_iter 
                total_loss = total_loss / self.accum_iter
                # 放大 loss 並 backward
                self.scalar.scale(total_loss).backward()
                
                if (step + 1) % self.accum_iter == 0 or (step + 1) == len(train_dataloader):
                    self.scalar.unscale_(self.optimizer)  # 先還原梯度 unscale 回去(若要結合 clip_norm 才需要)
                    clip_grad_norm_(self.model.parameters(),max_norm=3.0)
                    # update optimizer params
                    self.scalar.step(self.optimizer)
                    # update scaler factor
                    self.scalar.update()
                    # 更新參數後，將梯度歸零
                    self.optimizer.zero_grad()
                    # 在一個有效的 global-batch 更新後，再更新 lr
                    self.lr_scheduler.step()
            else:
                # 不進行梯度累加
                self.scalar.scale(total_loss).backward()
                self.scalar.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                self.scalar.step(self.optimizer)
                self.scalar.update()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
            

        lr_ = self.lr_scheduler.get_last_lr()[0]
        train_retrieval_acc,train_mrr_at_k = Cal_recall_at_k(
                                                train_q_sen_ebds,
                                                train_positive_sen_ebds,
                                                train_negative_sen_ebds,
                                                k=self.k,
                                                batch_size=16,
                                                device = self.device)


        
        train_cl_loss /= len(train_dataloader)
        train_total_loss /= len(train_dataloader)
        train_scope_pred_loss /= len(train_dataloader)
        train_scope_pred_acc /= len(train_dataloader)
        
        
        if self.train_type == 'scope_enhanced':
            train_metrics = {
                'train_scope_pred_loss':train_scope_pred_loss,
                'train_cl_loss':train_cl_loss,
                'train_total_loss':train_total_loss,
                'train_scope_pred_acc':train_scope_pred_acc,
                'train_retrieval_acc':train_retrieval_acc,
                'train_mrr_at_k':train_mrr_at_k,
                'lr':lr_
            }

            
        elif self.train_type == 'base':
            train_metrics = {
                'train_total_loss':train_total_loss,
                'train_retrieval_acc':train_retrieval_acc,
                'train_mrr_at_k':train_mrr_at_k,
                'lr':lr_
            }


            
        return train_metrics
    
    def evaluate_one_epoch(self,epoch,val_dataloader:DataLoader):
        cur_epoch = epoch
        valid_scope_pred_loss = 0.0
        valid_cl_loss = 0.0
        valid_total_loss = 0.0
        valid_scope_pred_acc = 0.0
        
        valid_q_sen_ebds = []
        valid_positive_sen_ebds = []
        valid_negative_sen_ebds = []
        
        self.model.eval()
        with torch.inference_mode(),torch.amp.autocast(device_type='cuda'):
            for batch in tqdm(val_dataloader,desc='Validation'):
                batch = batch.to(self.device)
                q_ebds,q_scope_labels,q_atten_masks = batch.question_ebds,batch.q_scope_labels,batch.q_atten_masks
                positive_ebds,positive_scope_labels,positive_atten_masks = batch.positive_ebds,batch.positive_scope_labels,batch.positive_atten_masks
                q_span_tables,positive_span_tables,neg_span_tables = batch.q_span_tables,batch.positive_span_tables,batch.negative_span_tables
                

                negative_ebds,negative_atten_masks = batch.negative_ebds,batch.negative_atten_masks
                
                with torch.amp.autocast(device_type='cuda'):
                    model_out:ScopeEnhancedEncoder_Out = self.model(
                                                                    q_ebds,
                                                                    q_atten_masks,
                                                                    q_span_tables,
                                                                    positive_ebds,
                                                                    positive_atten_masks,
                                                                    positive_span_tables,
                                                                    negative_ebds,
                                                                    negative_atten_masks,
                                                                    neg_span_tables
                                                                )

                valid_q_sen_ebds.append(model_out.q_sentence_ebds.detach().cpu())
                valid_positive_sen_ebds.append(model_out.positive_sentence_ebds.detach().cpu())
                valid_negative_sen_ebds.append(model_out.neg_sentence_ebds.detach().cpu())

                
                loss_dict = self.compute_loss(
                                            model_out,
                                            self.loss_weight,
                                            self.cl_loss_fn,
                                            self.scope_pred_loss_fn,
                                            self.train_type,
                                            self.withnegative_sample,
                                            self.num_scopes,
                                            q_scope_labels,
                                            positive_scope_labels,
                                        )
                


                if self.train_type == 'scope_enhanced' :

                    valid_scope_pred_loss += loss_dict['scope_pred_loss'].item() 
                    valid_cl_loss += loss_dict['cl_loss'].item()

                    scope_pred_acc = self.compute_scope_acc(
                                                    model_out.q_scope_logits,
                                                    model_out.positive_scope_logits,
                                                    q_scope_labels,
                                                    positive_scope_labels
                                                )
                    valid_scope_pred_acc += scope_pred_acc



                    
                valid_total_loss += loss_dict['total_loss'].item()
    
                
                
            valid_retrieval_acc,valid_mrr_at_k = Cal_recall_at_k(
                                                    valid_q_sen_ebds,
                                                    valid_positive_sen_ebds,
                                                    valid_negative_sen_ebds,
                                                    k=self.k,
                                                    batch_size=16,
                                                    device=self.device)

            
            valid_scope_pred_loss /= len(val_dataloader)
            valid_cl_loss /= len(val_dataloader)
            valid_total_loss /= len(val_dataloader)
            valid_scope_pred_acc /= len(val_dataloader)
            
            
        if self.train_type == 'scope_enhanced':    
            valid_metrics = {
                'valid_scope_pred_loss':valid_scope_pred_loss,
                'valid_cl_loss':valid_cl_loss,
                'valid_total_loss':valid_total_loss,
                'valid_scope_pred_acc':valid_scope_pred_acc,
                'valid_retrieval_acc':valid_retrieval_acc,
                'valid_mrr_at_k':valid_mrr_at_k
            }
        

        elif self.train_type == 'base': 
            valid_metrics = {
                'valid_total_loss':valid_total_loss,
                'valid_retrieval_acc':valid_retrieval_acc,
                'valid_mrr_at_k':valid_mrr_at_k
            }

        return valid_metrics
    
    
    def train(self,train_dataloader:DataLoader,val_dataloader:DataLoader,early_stopping: EarlyStopping):
        
        best_val_loss = float('inf')
        best_recall_val_acc = 0.0
        best_epoch_occur = 0
        best_scope_pred_val_acc = 0.0
        
        for epoch in range(1,self.epochs+1):
            
            
            train_metrics = self.training_one_epoch(epoch,train_dataloader)
            valid_metrics = self.evaluate_one_epoch(epoch,val_dataloader)
            
            merge_metric = train_metrics | valid_metrics
            

            if epoch == 1 :
                self.log_table:wandb.Table = wandb.Table(columns=list(merge_metric.keys()))


            wandb.log(merge_metric,step=epoch)
            self.log_table.add_data(
                *list(merge_metric.values())
            )
            
            print('epoch: {}'.format(epoch))
            Print_metrics(**train_metrics)
            Print_metrics(**valid_metrics)

            if valid_metrics['valid_retrieval_acc'] > best_recall_val_acc:
                best_val_loss = valid_metrics['valid_total_loss']
                best_recall_val_acc = valid_metrics['valid_retrieval_acc']
                best_epoch_occur = epoch
                if self.train_type == 'scope_enhanced':
                    best_scope_pred_val_acc = valid_metrics['valid_scope_pred_acc']

            early_stopping(valid_metrics['valid_retrieval_acc'],self.model)
            
            
            if early_stopping.early_stop:
                print(f'Earling stop at Epoch: {epoch}!!!')
                break
            
        wandb.log({'training_metrics':self.log_table})
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.summary['best_recall_val_acc'] = best_recall_val_acc
        wandb.summary['best_epoch_occur'] = best_epoch_occur
        
        if self.train_type == 'scope_enhanced':
            wandb.summary['best_scope_pred_val_acc'] = best_scope_pred_val_acc

        
            
            
            
