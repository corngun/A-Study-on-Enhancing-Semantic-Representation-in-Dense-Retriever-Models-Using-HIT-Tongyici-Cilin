from torch.optim.lr_scheduler import CosineAnnealingLR,SequentialLR,LinearLR

def Warmup_Cosine_scheduler(
        optimizer, 
        epochs,
        steps_per_epoch,
        min_lr,
        accum_iter = 1
        ):
    """
    Args:
    optimizer: 優化器
    warmup_steps: 預熱步數
    total_steps: 總訓練步數
    min_lr: CosineAnnealing 的最小學習率
    warmup_start_factor: 預熱開始時的學習率因子
    
    Returns:
        組合了 LinearLR(預熱) 和 CosineAnnealingLR 的調度器
    """
    # 預熱階段使用 LinearLR
    warmup_steps = steps_per_epoch * 5
    total_steps = steps_per_epoch * epochs
    
    total_global_steps = total_steps // accum_iter
    warmup_global_steps = warmup_steps // accum_iter
    cosine_global_steps = total_global_steps - warmup_global_steps
    

    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor = 0.1,  # 初始學習率因子
        end_factor = 1.0,                    # 預熱結束時達到基礎學習率
        total_iters = warmup_global_steps           # 預熱總步數，即要多少個 step 回到所設定的 lr 
    )
    
    
    # 主要訓練階段使用 CosineAnnealingLR
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        # T_max = total_steps - warmup_steps ,  # 餘弦週期
        T_max = cosine_global_steps  ,
        eta_min = min_lr, # 最小學習率
        last_epoch = -1
    )
    
#     # 使用 SequentialLR 組合使用的 3 個 scheduler
    combined_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        # 經過 warmup_steps 後切換到 cosine_scheduler 最後切到 cosine_scheduler
        milestones=[warmup_global_steps - 1]  
    )
    
    return combined_scheduler

