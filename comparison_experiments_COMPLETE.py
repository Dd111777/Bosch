# -*- coding: utf-8 -*-
"""
完整的对比实验框架
包含：数据划分策略对比 + 训练策略对比

直接替换 stageC 文件的 main() 函数
或者作为独立脚本运行
"""

import os
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Dict, List, Tuple
from collections import defaultdict


# ========================== 对比实验配置 ==========================

class ComparisonConfig:
    """对比实验配置"""
    
    # 实验模式
    RUN_MODE = "quick"  # "quick" | "standard" | "full"
    
    # 快速模式：只测试关键配置
    QUICK_EXPERIMENTS = [
        ("original", "joint"),     # 原始划分 + 联合训练
        ("kfold5", "joint"),       # 5折CV + 联合训练
        ("kfold5", "separate"),    # 5折CV + 分立训练
    ]
    
    # 标准模式：常用配置
    STANDARD_EXPERIMENTS = [
        ("original", "joint"),     # 基线
        ("kfold5", "joint"),       # 标准5折
        ("kfold5", "separate"),    # 分立训练
        ("kfold10", "joint"),      # 10折对比
        ("simple", "joint"),       # 简单划分
    ]
    
    # 完整模式：所有组合
    FULL_EXPERIMENTS = [
        # 数据划分对比
        ("original", "joint"),
        ("kfold5", "joint"),
        ("kfold10", "joint"),
        ("simple", "joint"),
        
        # 训练策略对比（用5折CV）
        ("kfold5", "joint"),
        ("kfold5", "separate"),
        ("kfold5", "sequential"),
        ("kfold5", "hybrid"),
    ]
    
    # Seeds
    SEEDS = [42]  # 快速测试用1个，完整实验用 [42, 43, 44]


# ========================== 数据划分策略 ==========================

def create_data_splits(data_dict, strategy="kfold5", seed=42):
    """
    创建不同的数据划分
    
    策略:
    - "original": 原始三分法 (Train/Val/Test = 24/3/3)
    - "kfold5": 5折交叉验证 (Train/Test = 24/6, 无Val)
    - "kfold10": 10折交叉验证 (Train/Test = 27/3, 无Val)
    - "simple": 简单Train/Test (Train/Test = 24/6, 无Val)
    """
    m_sparse = data_dict["m_sparse"]
    B = m_sparse.shape[0]
    
    # 创建分层变量
    sample_counts = m_sparse.sum(dim=(1, 2)).numpy()
    strata = (sample_counts > np.median(sample_counts)).astype(int)
    indices = np.arange(B)
    
    if strategy == "original":
        # 原始三分法
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=strata
        )
        temp_strata = strata[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=temp_strata
        )
        
        return [{
            'fold': 0,
            'strategy': 'original',
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
        }]
    
    elif strategy == "kfold5":
        # 5折CV
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(indices, strata)):
            splits.append({
                'fold': fold,
                'strategy': 'kfold5',
                'train_idx': train_idx,
                'val_idx': None,
                'test_idx': test_idx,
            })
        return splits
    
    elif strategy == "kfold10":
        # 10折CV
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        splits = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(indices, strata)):
            splits.append({
                'fold': fold,
                'strategy': 'kfold10',
                'train_idx': train_idx,
                'val_idx': None,
                'test_idx': test_idx,
            })
        return splits
    
    elif strategy == "simple":
        # 简单Train/Test
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=strata
        )
        return [{
            'fold': 0,
            'strategy': 'simple',
            'train_idx': train_idx,
            'val_idx': None,
            'test_idx': test_idx,
        }]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ========================== 训练策略 ==========================

def train_with_strategy(
    split_info, data_dict, variant, meta_old, device, 
    train_strategy="joint", Cfg=None
):
    """
    使用指定的训练策略训练模型
    
    策略:
    - "joint": 联合训练（所有family一起）
    - "separate": 分立训练（每个family独立）
    - "sequential": 顺序训练（从稀疏到密集）
    - "hybrid": 混合训练（先联合后分立）
    """
    
    if train_strategy == "joint":
        return train_joint(split_info, data_dict, variant, meta_old, device, Cfg)
    
    elif train_strategy == "separate":
        return train_separate(split_info, data_dict, variant, meta_old, device, Cfg)
    
    elif train_strategy == "sequential":
        return train_sequential(split_info, data_dict, variant, meta_old, device, Cfg)
    
    elif train_strategy == "hybrid":
        return train_hybrid(split_info, data_dict, variant, meta_old, device, Cfg)
    
    else:
        raise ValueError(f"Unknown training strategy: {train_strategy}")


def train_joint(split_info, data_dict, variant, meta_old, device, Cfg):
    """联合训练：所有family一起训练"""
    from physio_util import set_seed, metrics, transform_for_display
    
    train_idx = split_info['train_idx']
    val_idx = split_info.get('val_idx')
    test_idx = split_info['test_idx']
    
    # 准备数据
    fold_data = prepare_fold_data(data_dict, split_info, Cfg)
    
    s8 = fold_data['s8'].to(device)
    y_sparse = fold_data['y_sparse'].to(device)
    m_sparse = fold_data['m_sparse'].to(device)
    tvals = fold_data['tvals'].to(device)
    fams = fold_data['families']
    
    s8_train = s8[train_idx]
    y_train = y_sparse[train_idx]
    m_train = m_sparse[train_idx]
    
    s8_test = s8[test_idx]
    y_test = y_sparse[test_idx]
    m_test = m_sparse[test_idx]
    
    # 初始化模型
    models = initialize_models(variant, fold_data, meta_old, device, Cfg)
    
    # 训练
    print(f"  [Joint] Training on all families together...")
    train_loop(models, s8_train, y_train, m_train, tvals, fams, variant, Cfg, device)
    
    # 测试
    test_metrics = evaluate_model(
        models, s8_test, y_test, m_test, tvals, fams, variant, Cfg, device
    )
    
    return test_metrics


def train_separate(split_info, data_dict, variant, meta_old, device, Cfg):
    """分立训练：每个family独立训练"""
    from physio_util import metrics, transform_for_display
    
    train_idx = split_info['train_idx']
    test_idx = split_info['test_idx']
    
    fold_data = prepare_fold_data(data_dict, split_info, Cfg)
    
    s8 = fold_data['s8'].to(device)
    y_sparse = fold_data['y_sparse'].to(device)
    m_sparse = fold_data['m_sparse'].to(device)
    tvals = fold_data['tvals'].to(device)
    fams = fold_data['families']
    K = len(fams)
    
    s8_train = s8[train_idx]
    y_train = y_sparse[train_idx]
    m_train = m_sparse[train_idx]
    
    s8_test = s8[test_idx]
    y_test = y_sparse[test_idx]
    m_test = m_sparse[test_idx]
    
    print(f"  [Separate] Training each family independently...")
    
    # 为每个family训练独立模型
    family_models = {}
    family_metrics = []
    
    for k, family in enumerate(fams):
        print(f"    Training {family} ({k+1}/{K})...")
        
        # 创建family-specific mask
        mask_k = torch.zeros_like(m_train)
        mask_k[:, k, :] = m_train[:, k, :]
        
        # 初始化独立模型
        models_k = initialize_models(variant, fold_data, meta_old, device, Cfg)
        
        # 训练（只在当前family上）
        train_loop(
            models_k, s8_train, y_train, mask_k, tvals, fams,
            variant, Cfg, device, family_name=family
        )
        
        family_models[family] = models_k
        
        # 测试当前family
        mask_k_test = torch.zeros_like(m_test)
        mask_k_test[:, k, :] = m_test[:, k, :]
        
        metrics_k = evaluate_model(
            models_k, s8_test, y_test, mask_k_test, tvals, fams,
            variant, Cfg, device
        )
        family_metrics.append(metrics_k)
    
    # 聚合所有family的结果
    test_metrics = aggregate_family_metrics(family_metrics, fams)
    
    return test_metrics


def train_sequential(split_info, data_dict, variant, meta_old, device, Cfg):
    """顺序训练：从稀疏family到密集family"""
    train_idx = split_info['train_idx']
    test_idx = split_info['test_idx']
    
    fold_data = prepare_fold_data(data_dict, split_info, Cfg)
    
    s8 = fold_data['s8'].to(device)
    y_sparse = fold_data['y_sparse'].to(device)
    m_sparse = fold_data['m_sparse'].to(device)
    tvals = fold_data['tvals'].to(device)
    fams = fold_data['families']
    
    s8_train = s8[train_idx]
    y_train = y_sparse[train_idx]
    m_train = m_sparse[train_idx]
    
    s8_test = s8[test_idx]
    y_test = y_sparse[test_idx]
    m_test = m_sparse[test_idx]
    
    print(f"  [Sequential] Training from sparse to dense families...")
    
    # 按数据量排序
    family_counts = m_train.sum(dim=(0, 2)).cpu().numpy()
    sorted_indices = np.argsort(family_counts)
    
    print(f"    Order: {' → '.join([fams[i] for i in sorted_indices])}")
    
    # 初始化模型
    models = initialize_models(variant, fold_data, meta_old, device, Cfg)
    
    # 累积训练
    cumulative_mask = torch.zeros_like(m_train)
    
    for stage, k in enumerate(sorted_indices):
        family = fams[k]
        print(f"    Stage {stage+1}/{len(fams)}: Adding {family}")
        
        # 累积mask
        cumulative_mask[:, k, :] = m_train[:, k, :]
        
        # 训练（每个stage较短）
        Cfg_stage = copy.copy(Cfg)
        Cfg_stage.max_epochs = Cfg.max_epochs // len(fams)
        
        train_loop(
            models, s8_train, y_train, cumulative_mask, tvals, fams,
            variant, Cfg_stage, device
        )
    
    # 最后微调
    print(f"    Final fine-tuning on all families...")
    Cfg_final = copy.copy(Cfg)
    Cfg_final.max_epochs = Cfg.max_epochs // 4
    train_loop(
        models, s8_train, y_train, m_train, tvals, fams,
        variant, Cfg_final, device
    )
    
    # 测试
    test_metrics = evaluate_model(
        models, s8_test, y_test, m_test, tvals, fams, variant, Cfg, device
    )
    
    return test_metrics


def train_hybrid(split_info, data_dict, variant, meta_old, device, Cfg):
    """混合训练：先联合预训练，再分立微调"""
    train_idx = split_info['train_idx']
    test_idx = split_info['test_idx']
    
    fold_data = prepare_fold_data(data_dict, split_info, Cfg)
    
    s8 = fold_data['s8'].to(device)
    y_sparse = fold_data['y_sparse'].to(device)
    m_sparse = fold_data['m_sparse'].to(device)
    tvals = fold_data['tvals'].to(device)
    fams = fold_data['families']
    
    s8_train = s8[train_idx]
    y_train = y_sparse[train_idx]
    m_train = m_sparse[train_idx]
    
    s8_test = s8[test_idx]
    y_test = y_sparse[test_idx]
    m_test = m_sparse[test_idx]
    
    print(f"  [Hybrid] Phase 1: Joint pre-training (80% epochs)...")
    
    # Phase 1: 联合预训练
    models = initialize_models(variant, fold_data, meta_old, device, Cfg)
    
    Cfg_phase1 = copy.copy(Cfg)
    Cfg_phase1.max_epochs = int(Cfg.max_epochs * 0.8)
    
    train_loop(
        models, s8_train, y_train, m_train, tvals, fams,
        variant, Cfg_phase1, device
    )
    
    print(f"  [Hybrid] Phase 2: Separate fine-tuning per family...")
    
    # Phase 2: 分立微调
    Cfg_phase2 = copy.copy(Cfg)
    Cfg_phase2.max_epochs = int(Cfg.max_epochs * 0.2)
    
    for k, family in enumerate(fams):
        print(f"    Fine-tuning {family}...")
        
        mask_k = torch.zeros_like(m_train)
        mask_k[:, k, :] = m_train[:, k, :]
        
        train_loop(
            models, s8_train, y_train, mask_k, tvals, fams,
            variant, Cfg_phase2, device, family_name=family
        )
    
    # 测试
    test_metrics = evaluate_model(
        models, s8_test, y_test, m_test, tvals, fams, variant, Cfg, device
    )
    
    return test_metrics


# ========================== 辅助函数 ==========================

def prepare_fold_data(data_dict, split_info, Cfg):
    """准备fold数据（重新标准化）"""
    train_idx = split_info['train_idx']
    
    s8_original = data_dict['s8']
    norm_mean = data_dict.get('norm_mean', torch.zeros(7))
    norm_std = data_dict.get('norm_std', torch.ones(7))
    
    # 反标准化
    s8_raw = s8_original * (norm_std + 1e-8) + norm_mean
    
    # 用训练集重新标准化
    s8_train_raw = s8_raw[train_idx]
    train_mean = s8_train_raw.mean(dim=0)
    train_std = s8_train_raw.std(dim=0).clamp_min(1e-8)
    
    s8_renorm = (s8_raw - train_mean) / train_std
    
    return {
        's8': s8_renorm,
        'y_sparse': data_dict['y_sparse'],
        'm_sparse': data_dict['m_sparse'],
        'tvals': data_dict['tvals'],
        'families': data_dict['families'],
    }


def initialize_models(variant, fold_data, meta_old, device, Cfg):
    """初始化所有模型组件"""
    # 这个函数调用原文件的模型初始化代码
    # 简化起见，返回一个字典
    from phys_model import TemporalRegressor
    
    K = len(fold_data['families'])
    T = fold_data['tvals'].shape[0]
    
    # 物理模型（从原文件导入）
    from stageC_finetune_joint_on_new_pycharm_new import (
        build_phys_from_ckpt, freeze_physics_models,
        IonInverseTransform, build_calib_head,
        PerFamilyHead, HeteroHead,
        PhysAdapter, PhysFeaReducer, IonGate
    )
    
    phys_F, phys_I, ion_aff_init = build_phys_from_ckpt(
        Cfg.phys_ckpt_F, Cfg.phys_ckpt_I, device
    )
    freeze_physics_models(phys_F, phys_I, verbose=False)
    
    ion_tr = IonInverseTransform(
        ion_aff_init,
        learnable=variant.get("learnable_ion", False)
    ).to(device)
    
    morph = TemporalRegressor(
        K=K, d_model=64, nhead=4, num_layers=2,
        dim_ff=128, dropout=Cfg.dropout_morph, T=T
    ).to(device)
    
    # 加载预训练权重
    if os.path.exists(Cfg.morph_ckpt):
        from stageC_finetune_joint_on_new_pycharm_new import _safe_load
        ck = _safe_load(Cfg.morph_ckpt, map_location="cpu")
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        morph.load_state_dict(sd, strict=False)
    
    calib = build_calib_head(variant["calib_head"], K=K).to(device)
    
    per_family_head = PerFamilyHead(K=K).to(device) \
        if variant.get("per_family_head", False) \
        else nn.Identity().to(device)
    
    adapters = {
        "adapter": PhysAdapter(2, k=3).to(device),
        "reducer": PhysFeaReducer().to(device),
        "gate": IonGate(k=5).to(device),
    }
    
    # 优化器
    param_groups = []
    morph_params = [p for p in morph.parameters() if p.requires_grad]
    if morph_params:
        param_groups.append({
            'params': morph_params,
            'lr': Cfg.lr_morph,
            'weight_decay': Cfg.wd_morph
        })
    
    calib_params = [p for p in calib.parameters() if p.requires_grad]
    if calib_params:
        param_groups.append({
            'params': calib_params,
            'lr': Cfg.lr_calib,
            'weight_decay': Cfg.wd_calib
        })
    
    optimizer = torch.optim.AdamW(param_groups)
    
    # Scheduler
    if Cfg.use_scheduler:
        def lr_lambda(epoch):
            if epoch < Cfg.warmup_epochs:
                return epoch / Cfg.warmup_epochs
            else:
                progress = (epoch - Cfg.warmup_epochs) / (Cfg.max_epochs - Cfg.warmup_epochs)
                return Cfg.min_lr_ratio + 0.5 * (1 - Cfg.min_lr_ratio) * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    # EMA
    from stageC_finetune_joint_on_new_pycharm_new import EMA
    ema = EMA(morph, decay=Cfg.ema_decay) if Cfg.use_ema else None
    
    return {
        'morph': morph,
        'phys_F': phys_F,
        'phys_I': phys_I,
        'ion_tr': ion_tr,
        'calib': calib,
        'per_family_head': per_family_head,
        'adapters': adapters,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'ema': ema,
    }


def train_loop(models, s8_train, y_train, m_train, tvals, fams, 
              variant, Cfg, device, family_name=None):
    """训练循环"""
    from stageC_finetune_joint_on_new_pycharm_new import (
        phys_forward_raw, phys_interface_pipeline,
        masked_huber_with_channel_norm
    )
    
    morph = models['morph']
    phys_F = models['phys_F']
    phys_I = models['phys_I']
    ion_tr = models['ion_tr']
    calib = models['calib']
    per_family_head = models['per_family_head']
    adapters = models['adapters']
    optimizer = models['optimizer']
    scheduler = models['scheduler']
    ema = models['ema']
    
    for epoch in range(1, Cfg.max_epochs + 1):
        morph.train()
        optimizer.zero_grad()
        
        # 前向
        phys_raw = phys_forward_raw(s8_train, tvals, phys_F, phys_I, ion_tr, False)
        phys_enh = phys_interface_pipeline(phys_raw, variant, adapters)
        y_pred = morph(s8_train, phys_enh, tvals)
        
        if variant.get("per_family_head", False):
            y_pred = per_family_head(y_pred)
        
        y_pred = calib(y_pred)
        
        # 损失
        loss, _ = masked_huber_with_channel_norm(
            y_pred, y_train, m_train,
            delta=Cfg.loss_delta,
            smooth_weight=Cfg.loss_smooth_weight,
        )
        
        # 反向
        loss.backward()
        torch.nn.utils.clip_grad_norm_(morph.parameters(), Cfg.batch_clip)
        optimizer.step()
        
        if ema is not None:
            ema.update(morph)
        
        if scheduler is not None:
            scheduler.step()
        
        if epoch % 100 == 0:
            prefix = f"[{family_name}] " if family_name else ""
            print(f"      {prefix}Epoch {epoch}/{Cfg.max_epochs}: Loss={loss.item():.4f}")


def evaluate_model(models, s8_test, y_test, m_test, tvals, fams, variant, Cfg, device):
    """评估模型"""
    from physio_util import metrics, transform_for_display
    from stageC_finetune_joint_on_new_pycharm_new import (
        phys_forward_raw, phys_interface_pipeline, compute_per_family_metrics
    )
    
    morph = models['morph']
    phys_F = models['phys_F']
    phys_I = models['phys_I']
    ion_tr = models['ion_tr']
    calib = models['calib']
    per_family_head = models['per_family_head']
    adapters = models['adapters']
    ema = models['ema']
    
    morph.eval()
    with torch.no_grad():
        if ema is not None:
            ema.apply_to(morph)
        
        phys_test = phys_forward_raw(s8_test, tvals, phys_F, phys_I, ion_tr, False)
        phys_test_enh = phys_interface_pipeline(phys_test, variant, adapters)
        y_pred = morph(s8_test, phys_test_enh, tvals)
        
        if variant.get("per_family_head", False):
            y_pred = per_family_head(y_pred)
        
        y_pred = calib(y_pred)
        
        yhat_disp, ytrue_disp = transform_for_display(
            y_pred, y_test,
            family_sign=Cfg.family_sign,
            unit_scale=Cfg.unit_scale,
            flip_sign=Cfg.flip_sign,
            clip_nonneg=Cfg.clip_nonneg,
            min_display_value=Cfg.min_display_value
        )
        
        mts = metrics(yhat_disp, ytrue_disp, m_test)
        per_family, macro, micro, min_family = compute_per_family_metrics(
            mts, m_test, fams
        )
        
        if ema is not None:
            ema.restore(morph)
    
    return {
        'R2_macro': macro['R2'],
        'MAE_macro': macro['MAE'],
        'RMSE_macro': macro['RMSE'],
        'per_family': per_family,
    }


def aggregate_family_metrics(family_metrics, fams):
    """聚合分立训练的结果"""
    r2_values = [m['R2_macro'] for m in family_metrics]
    mae_values = [m['MAE_macro'] for m in family_metrics]
    rmse_values = [m['RMSE_macro'] for m in family_metrics]
    
    return {
        'R2_macro': np.mean(r2_values),
        'MAE_macro': np.mean(mae_values),
        'RMSE_macro': np.mean(rmse_values),
        'per_family': [m['per_family'] for m in family_metrics],
    }


# ========================== 主对比函数 ==========================

def run_comparison_experiments(variant, data_dict, meta_old, device, Cfg):
    """
    运行完整的对比实验
    
    返回：所有实验的结果
    """
    
    # 选择实验模式
    if ComparisonConfig.RUN_MODE == "quick":
        experiments = ComparisonConfig.QUICK_EXPERIMENTS
    elif ComparisonConfig.RUN_MODE == "standard":
        experiments = ComparisonConfig.STANDARD_EXPERIMENTS
    else:
        experiments = ComparisonConfig.FULL_EXPERIMENTS
    
    print(f"\n{'='*80}")
    print(f"COMPARISON EXPERIMENTS - {ComparisonConfig.RUN_MODE.upper()} MODE")
    print(f"Total experiments: {len(experiments) * len(ComparisonConfig.SEEDS)}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for exp_id, (split_strategy, train_strategy) in enumerate(experiments, 1):
        for seed in ComparisonConfig.SEEDS:
            exp_name = f"{split_strategy}_{train_strategy}_seed{seed}"
            
            print(f"\n{'='*80}")
            print(f"[{exp_id}/{len(experiments)}] {exp_name}")
            print(f"Split: {split_strategy} | Train: {train_strategy} | Seed: {seed}")
            print(f"{'='*80}")
            
            start_time = time.time()
            
            # 创建数据划分
            splits = create_data_splits(data_dict, split_strategy, seed)
            
            # 对每个split训练
            fold_results = []
            for split_info in splits:
                fold = split_info['fold']
                print(f"\n  Fold {fold+1}/{len(splits)}: "
                      f"Train={len(split_info['train_idx'])}, "
                      f"Test={len(split_info['test_idx'])}")
                
                # 训练
                test_metrics = train_with_strategy(
                    split_info, data_dict, variant, meta_old, device,
                    train_strategy, Cfg
                )
                
                fold_results.append(test_metrics)
                
                print(f"    R²={test_metrics['R2_macro']:.4f}, "
                      f"MAE={test_metrics['MAE_macro']:.4f}")
            
            # 聚合结果
            if len(fold_results) > 1:
                r2_values = [f['R2_macro'] for f in fold_results]
                mae_values = [f['MAE_macro'] for f in fold_results]
                
                final_result = {
                    'exp_name': exp_name,
                    'split_strategy': split_strategy,
                    'train_strategy': train_strategy,
                    'seed': seed,
                    'R2_mean': np.mean(r2_values),
                    'R2_std': np.std(r2_values),
                    'MAE_mean': np.mean(mae_values),
                    'MAE_std': np.std(mae_values),
                    'n_folds': len(fold_results),
                }
            else:
                final_result = {
                    'exp_name': exp_name,
                    'split_strategy': split_strategy,
                    'train_strategy': train_strategy,
                    'seed': seed,
                    'R2_mean': fold_results[0]['R2_macro'],
                    'R2_std': 0.0,
                    'MAE_mean': fold_results[0]['MAE_macro'],
                    'MAE_std': 0.0,
                    'n_folds': 1,
                }
            
            elapsed = time.time() - start_time
            final_result['time_seconds'] = elapsed
            
            all_results.append(final_result)
            
            print(f"\n  [Summary] R²={final_result['R2_mean']:.4f}±{final_result['R2_std']:.4f}, "
                  f"Time={elapsed/60:.1f}min")
    
    # 生成对比报告
    generate_comparison_report(all_results, Cfg)
    
    return all_results


def generate_comparison_report(all_results, Cfg):
    """生成对比报告"""
    
    print(f"\n{'='*100}")
    print(f"COMPARISON RESULTS SUMMARY")
    print(f"{'='*100}\n")
    
    # 按R²排序
    sorted_results = sorted(all_results, key=lambda x: x['R2_mean'], reverse=True)
    
    print(f"{'Rank':<6} {'Experiment':<35} {'R²':<18} {'MAE':<18} {'Time':<10}")
    print(f"{'-'*100}")
    
    for rank, res in enumerate(sorted_results, 1):
        r2_str = f"{res['R2_mean']:.4f}±{res['R2_std']:.4f}"
        mae_str = f"{res['MAE_mean']:.4f}±{res['MAE_std']:.4f}"
        time_str = f"{res['time_seconds']/60:.1f}min"
        
        print(f"{rank:<6} {res['exp_name']:<35} {r2_str:<18} {mae_str:<18} {time_str:<10}")
    
    print(f"{'='*100}\n")
    
    # 分类统计
    print(f"Analysis by Split Strategy:")
    split_strategies = {}
    for res in all_results:
        strategy = res['split_strategy']
        if strategy not in split_strategies:
            split_strategies[strategy] = []
        split_strategies[strategy].append(res['R2_mean'])
    
    for strategy, r2_values in split_strategies.items():
        print(f"  {strategy:12s}: R²={np.mean(r2_values):.4f}±{np.std(r2_values):.4f} "
              f"(n={len(r2_values)})")
    
    print(f"\nAnalysis by Training Strategy:")
    train_strategies = {}
    for res in all_results:
        strategy = res['train_strategy']
        if strategy not in train_strategies:
            train_strategies[strategy] = []
        train_strategies[strategy].append(res['R2_mean'])
    
    for strategy, r2_values in train_strategies.items():
        print(f"  {strategy:12s}: R²={np.mean(r2_values):.4f}±{np.std(r2_values):.4f} "
              f"(n={len(r2_values)})")
    
    print(f"\n{'='*100}\n")
    
    # 保存JSON
    save_path = os.path.join(Cfg.save_root, 'comparison_results.json')
    os.makedirs(Cfg.save_root, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump({
            'all_results': all_results,
            'summary': {
                'by_split': {k: {'mean': np.mean(v), 'std': np.std(v)} 
                           for k, v in split_strategies.items()},
                'by_train': {k: {'mean': np.mean(v), 'std': np.std(v)} 
                           for k, v in train_strategies.items()},
            }
        }, f, indent=2)
    
    print(f"[✓] Results saved to: {save_path}\n")


# ========================== 使用示例 ==========================

if __name__ == "__main__":
    print("""
    完整对比实验框架
    
    使用方法：
    1. 在 main() 函数中调用：
       results = run_comparison_experiments(variant, data_dict, meta_old, device, Cfg)
    
    2. 设置实验模式：
       ComparisonConfig.RUN_MODE = "quick"    # 3个实验，约6小时
       ComparisonConfig.RUN_MODE = "standard" # 5个实验，约10小时
       ComparisonConfig.RUN_MODE = "full"     # 8个实验，约16小时
    
    3. 设置seeds:
       ComparisonConfig.SEEDS = [42]          # 快速测试
       ComparisonConfig.SEEDS = [42, 43, 44]  # 完整实验
    """)
