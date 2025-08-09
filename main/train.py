from dataset import QA_Pairs_Dataset
from collate_fn import Collate_fn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from Models.model import ScopeEnhancedDualEncoder,ScopeEnhancedSiameseEncoder
from Models.LossFn import InfoNCELoss
from early_stopping import EarlyStopping
from lr_scheduler import Warmup_Cosine_scheduler
from trainer import Trainer
import yaml
from pathlib import Path


def load_config():
    with open('./configs/Config.yaml','r') as f:
        return yaml.safe_load(f)
    
def save_run_id(run:wandb,path,name:str):
    if not Path(path).exists():
        Path(path).mkdir(exist_ok=True,parents=True) 
        
    with open(Path(path,f'{name}.txt'),'w') as f:
        f.write(run.id)
        

def Store_name(
                with_negative_sample,
                hardneg_num,
                train_type,
                pooling_method,
                granularity,
                store_configs
                ):
    
    if with_negative_sample:
        append_dir = 'with_hard_neg'
    else:
        append_dir = 'without_hard_neg'
        
    model_save_dir = Path(__file__).parent.joinpath(store_configs.get('model_save_dir'),
                                                    store_configs.get('model_arch'),
                                                    append_dir)
    run_id_save_dir = Path(__file__).parent.joinpath(
                                            store_configs.get('run_id_save_dir'),
                                            store_configs.get('model_arch'),
                                            append_dir
                                            )
    if append_dir == 'with_hard_neg':
        if train_type == 'scope_enhanced':
            run_name = f'{store_configs.get("exp_name")}-{append_dir}_granularity{granularity}_{pooling_method}_{hardneg_num}hardneg'
            model_save_path = Path(model_save_dir,f'granularity_{granularity}_{pooling_method}_{hardneg_num}hardneg')
            run_id_save_path = Path(run_id_save_dir,f'granularity_{granularity}_{pooling_method}_{hardneg_num}hardneg')

        elif train_type == 'base':
            run_name = f'{store_configs.get("exp_name")}-{append_dir}_base_{hardneg_num}hardneg'
            model_save_path = Path(model_save_dir,f'base_{hardneg_num}hardneg')
            run_id_save_path = Path(run_id_save_dir,f'base_{hardneg_num}hardneg')
    else:
        if train_type == 'scope_enhanced':
            run_name = f'{store_configs.get("exp_name")}-{append_dir}_granularity{granularity}_{pooling_method}'
            model_save_path = Path(model_save_dir,f'granularity_{granularity}_{pooling_method}')
            run_id_save_path = Path(run_id_save_dir,f'granularity_{granularity}_{pooling_method}')

        elif train_type == 'base':
            run_name = f'{store_configs.get("exp_name")}-{append_dir}_base'
            model_save_path = Path(model_save_dir,f'base')
            run_id_save_path = Path(run_id_save_dir,f'base')

    return {
        'run_name':run_name,
        'model_save_path':model_save_path,
        'run_id_save_path':run_id_save_path
    }
    


def main():

    wandb.login()
    configs = load_config()
    
    train_type = configs['strategy'].get('train_type')
    pooling_method = configs['strategy'].get('pooling_method')
    seg_type = configs['data'].get('seg_type')
    batch_size,accum_iter,lr,min_lr,epochs,temp,k = (
                configs['hyperparam'].get('batch_size'),
                configs['hyperparam'].get('accum_iter'),
                configs['hyperparam'].get('learning_rate'),
                configs['hyperparam'].get('min_learning_rate'),
                configs['hyperparam'].get('epochs'),
                configs['hyperparam'].get('temp'),
                configs['hyperparam'].get('k')
    )
    
    loss_weight:dict = configs['loss_weight']
    
    with_negative_sample = configs['data'].get('with_negative_sample')
    hardneg_num = configs['data'].get('hardneg_num')
    granularity = configs['data'].get('granularity')
    
    print(f'granularity: {granularity} | \
            seg_type:{seg_type} |\
            with_negative_sample: {with_negative_sample} | \
            hardneg_num: {hardneg_num} | \
            train_type: {train_type} |\
            pooling_method: {pooling_method}')
    
    model_arch = configs['store'].get('model_arch')
    names = Store_name(with_negative_sample,hardneg_num,train_type,pooling_method,granularity,configs['store'])

    run = wandb.init(
        project='Semantic_Spectrum_encoder',
        notes='First Experiment',
        config=configs,
        name= names.get('run_name')
    ) 

    # log_table = Create_log_table(train_type)

    data_root_path = Path(Path(__file__).parent).joinpath('data/QA/Tiny')
    collate_fn = Collate_fn(padding_label= -100)
    train_dataset = QA_Pairs_Dataset(
                    data_root_path=data_root_path,
                    seg_type=seg_type,
                    dataset_type='train',
                    granularity=granularity
                    )
    
    valid_dataset = QA_Pairs_Dataset(
                    data_root_path=data_root_path,
                    seg_type=seg_type,
                    dataset_type='valid',
                    granularity=granularity
                    )
    
    
    train_dataloader = DataLoader(train_dataset,collate_fn=collate_fn,batch_size=batch_size,shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,collate_fn=collate_fn,batch_size=batch_size,shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 扣除 <unk> label
    num_scopes = len(train_dataset.scope_to_ids) - 1 
    
    if model_arch == 'dual-encoder':

        model = ScopeEnhancedDualEncoder(
                                        configs['encoder'],
                                        num_scopes = num_scopes,
                                        train_type=train_type,
                                        pooling_method = pooling_method
                                        )
    else:
        model = ScopeEnhancedSiameseEncoder(
                                    configs['encoder'],
                                    num_scopes = num_scopes,
                                    train_type=train_type,
                                    pooling_method = pooling_method
                                    )
    


    scope_pred_criterion = nn.CrossEntropyLoss(ignore_index = -100)
    cl_criterion = InfoNCELoss(hardneg_num=hardneg_num,temperature = temp)


    optimizer = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=1e-4)
    lr_scheduler = Warmup_Cosine_scheduler(
                                            optimizer,
                                            epochs,
                                            len(train_dataset) // batch_size,
                                            min_lr=min_lr,
                                            accum_iter=accum_iter
                                            )
    
    
    trainer = Trainer(
                    model,
                    accum_iter,
                    scope_pred_criterion,
                    cl_criterion,
                    loss_weight,
                    optimizer,
                    lr_scheduler,
                    epochs,
                    num_scopes,
                    k,
                    with_negative_sample,
                    train_type,
                    device
                    )
    

    early_stopping = EarlyStopping(model_arch=model_arch,save_path=names.get('model_save_path'),patience = 7,delta=5e-3,k=k)
    trainer.train(train_dataloader,valid_dataloader,early_stopping)

    save_run_id(run,names.get('run_id_save_path'),'id')
    wandb.finish()

if __name__ == '__main__':
    main()
    