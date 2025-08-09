from torch.utils.data import DataLoader
from dataset import QA_Pairs_Dataset
from Models.model import ScopeEnhancedDualEncoder,ScopeEnhancedSiameseEncoder,ScopeEnhancedEncoder_Out
from Models.LossFn import InfoNCELoss
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collate_fn import Collate_fn
import argparse
import yaml
import wandb
from pathlib import Path
from utility.util import Print_metrics
import torch.nn as nn
from trainer import Trainer
from train import Store_name
import json

def Parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log',action='store_true',help='是否需要 log')
    parser.add_argument('--K',default=3,type = int,help='Top-K 的數量')
    args = parser.parse_args()
    
    return args


def Cal_recall_at_k(
                    q_ebds:list[torch.Tensor],
                    ans_ebds:list[torch.Tensor],
                    neg_ebds:list[torch.Tensor],
                    k=3,
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
    
    correct_indices = []  
    incorrect_indices = []
    
    all_topk_scores = []
    all_topk_indices = []
    # all_topk_is_positive = []
    
    # (total_data_len,total_data_len)
    for i in tqdm(range(0,total_len,batch_size),desc=f'Cal recall at {k}'):
        # (batch_size,total_data_len)
        q_batch = q_ebds[i:i+batch_size].to(device)
        q_batch = F.normalize(q_batch, p=2, dim=1)
        
        # (total_data_len, total_data_len + total_data_len *N)
        batch_similarity_matrix = torch.matmul(q_batch,candidate_ebds.T)
    
        # (batch_size,k)
        batch_topk_scores,batch_topk_indices = torch.topk(batch_similarity_matrix,k=k,dim=1)
    
        # (batch_size,)
        batch_gold_indices = torch.arange(i,min(i+batch_size,total_len), device=device)
    
        # recall@k
        correct = (batch_topk_indices == batch_gold_indices.unsqueeze(1)).any(dim=1).float()
        total_correct += correct.sum().item()
        
        
        # mrr@k
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
        
        
        batch_indices = batch_gold_indices.cpu()
        for j, is_correct in enumerate(correct):
            if is_correct:
                correct_indices.append(batch_indices[j].item())
            else:
                incorrect_indices.append(batch_indices[j].item())

        
        # batch_is_positive = batch_topk_indices < total_len
        
        all_topk_scores.extend(batch_topk_scores.cpu().tolist())
        all_topk_indices.extend(batch_topk_indices.cpu().tolist())
        # all_topk_is_positive.extend(batch_is_positive.cpu().tolist())
        
    recall_at_k = total_correct / total_len
    mrr_at_k = mrr_at_k / total_len
    
    result_dic = {
        'all_topk_scores':all_topk_scores,
        'all_topk_indices':all_topk_indices,
        'correct_indices': correct_indices,
        'incorrect_indices': incorrect_indices
    }

    return recall_at_k,mrr_at_k,result_dic







def test(
        model,
        test_dataloader:DataLoader,
        num_scopes,
        scope_pred_loss_fn:torch.nn,
        cl_loss_fn:InfoNCELoss,
        loss_weight:dict,
        withnegative_sample,
        train_type:str,
        k:int,
        device
        ):
    

    test_scope_pred_loss = 0.0
    test_cl_loss = 0.0
    test_total_loss = 0.0
    test_scope_pred_acc = 0.0
    test_retrieval_acc = 0.0
    
    test_q_sen_ebds = []
    test_ans_sen_ebds = []
    test_negative_sen_ebds = []
    
    
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(test_dataloader,desc='Testing'):
            batch = batch.to(device)
            q_ebds,q_scope_labels,q_atten_masks = batch.question_ebds,batch.q_scope_labels,batch.q_atten_masks
            positive_ebds,positive_scope_labels,positive_atten_masks = batch.positive_ebds,batch.positive_scope_labels,batch.positive_atten_masks
            q_span_tables,positive_span_tables,neg_span_tables = batch.q_span_tables,batch.positive_span_tables,batch.negative_span_tables
            
            negative_ebds,negative_atten_masks = batch.negative_ebds,batch.negative_atten_masks
            
            model_out:ScopeEnhancedEncoder_Out = model(
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


            test_q_sen_ebds.append(model_out.q_sentence_ebds.detach().cpu())
            test_ans_sen_ebds.append(model_out.positive_sentence_ebds.detach().cpu())
            test_negative_sen_ebds.append(model_out.neg_sentence_ebds.detach().cpu())

            loss_dict = Trainer.compute_loss(
                                        model_out,
                                        loss_weight,
                                        cl_loss_fn,
                                        scope_pred_loss_fn,
                                        train_type,
                                        withnegative_sample,
                                        num_scopes,
                                        q_scope_labels,
                                        positive_scope_labels
                                    )

            
            if train_type == 'scope_enhanced':
                test_scope_pred_loss += loss_dict['scope_pred_loss'].item() 
                test_cl_loss += loss_dict['cl_loss'].item()
                
                scope_pred_acc = Trainer.compute_scope_acc(
                                                model_out.q_scope_logits,
                                                model_out.positive_scope_logits,
                                                q_scope_labels,
                                                positive_scope_labels
                                            )
                test_scope_pred_acc += scope_pred_acc
                


                
            test_total_loss += loss_dict['total_loss'].item()

            
            
        test_retrieval_acc,test_mrr_at_k,result_dic = Cal_recall_at_k(
                                    test_q_sen_ebds,
                                    test_ans_sen_ebds,
                                    test_negative_sen_ebds,
                                    k=k,
                                    batch_size=32,
                                    device=device
                                    )
        
        test_scope_pred_loss /= len(test_dataloader)
        test_cl_loss /= len(test_dataloader)
        test_total_loss /= len(test_dataloader)
        test_scope_pred_acc /= len(test_dataloader)
        

                    
        if train_type == 'scope_enhanced':   
            test_metrics = {
                    'test_scope_pred_loss':test_scope_pred_loss,
                    'test_cl_loss':test_cl_loss,
                    'test_total_loss':test_total_loss,
                    'test_scope_pred_acc':test_scope_pred_acc,
                    'test_retrieval_acc':test_retrieval_acc,
                    'test_mrr_at_k':test_mrr_at_k
                } 

            
        elif train_type == 'base': 
            test_metrics = {
                    'test_total_loss':test_total_loss,
                    'test_retrieval_acc':test_retrieval_acc,
                    'test_mrr_at_k':test_mrr_at_k
                } 

        return test_metrics,result_dic



def main():
    args = Parser()
    config_path = Path(__file__).parent.joinpath('configs/Config.yaml')
    with open(config_path,'r') as f:
        configs = yaml.safe_load(f)
    
    granularity = configs['data'].get('granularity')
    train_type = configs['strategy'].get('train_type')
    pooling_method = configs['strategy'].get('pooling_method')
    with_negative_sample = configs['data'].get('with_negative_sample')
    hardneg_num = configs['data'].get('hardneg_num')
    seg_type = configs['data']['seg_type']
    batch_size = configs['hyperparam']['batch_size']
    temp = configs['hyperparam']['temp']
    
    print(f'granularity: {granularity} | \
            with_negative_sample: {with_negative_sample} | \
            hardneg_num: {hardneg_num} | \
            train_type: {train_type} |\
            pooling_method: {pooling_method}')
    
    
    names = Store_name(with_negative_sample,hardneg_num,train_type,pooling_method,granularity,configs['store'])

    if args.log:
        run_id_path = Path(names.get('run_id_save_path'),'id.txt')

        with open(run_id_path,'r') as f:
            run_id = f.read().strip()
        wandb.init(project='Semantic_Spectrum_encoder',id=run_id,resume='must')
    

    collate_fn = Collate_fn(padding_label=-100)

    data_root_path = Path(__file__).parent.joinpath('data/QA/Tiny')
    test_dataset = QA_Pairs_Dataset(
                    data_root_path=data_root_path,
                    seg_type=seg_type,
                    dataset_type='test',
                    granularity=granularity
                    )
    
    test_dataloder = DataLoader(test_dataset,collate_fn=collate_fn,batch_size=batch_size,shuffle=False)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_scopes = len(test_dataset.scope_to_ids) - 1 # 扣除 <unk> label
    
    model_path = Path(names.get('model_save_path'),'best_checkpoint.pt')
    print(model_path)
    model = ScopeEnhancedDualEncoder(
                                    configs['encoder'],
                                    num_scopes = num_scopes,
                                    train_type=train_type,
                                    pooling_method=pooling_method
                                    )

    
    model.load_state_dict(torch.load(model_path,weights_only=True))

    scope_pred_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    cl_criterion = InfoNCELoss(hardneg_num,temp)
    loss_weight:dict = configs['loss_weight']
    
    test_metrics,result_dic = test(
                        model,
                        test_dataloder,
                        num_scopes,
                        scope_pred_criterion,
                        cl_criterion,
                        loss_weight,
                        with_negative_sample,
                        train_type,
                        args.K,
                        device
                    )
    
    
    Print_metrics(**test_metrics)
    
    save_result_dic_parts = ['evaluate' if part == 'checkpoint' else part for part in Path(names.get('model_save_path')).parts]
    save_result_dic_path = Path(*save_result_dic_parts)
    

    
    if args.log:
        if not save_result_dic_path.exists():
            save_result_dic_path.mkdir(exist_ok=True,parents=True)
        
        with open(save_result_dic_path.joinpath('result_dic.json'),'w') as f:
            json.dump(result_dic,f)
            
        columns = list(test_metrics.keys())
        testing_result_table = wandb.Table(columns=columns,data=[list(test_metrics.values())])
        wandb.log({'testing_metrics':testing_result_table})
        wandb.finish()
    
if __name__ == '__main__':
    main()