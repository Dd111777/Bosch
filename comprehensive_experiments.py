# -*- coding: utf-8 -*-
"""
Comprehensive Experiment Framework for Small-Sample Scenarios
包含所有可能的数据划分和训练策略

使用方法：
1. 直接替换 stageC_finetune_joint_on_new_pycharm_new.py 中的相关函数
2. 或作为独立模块导入
"""

import os
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import (
    StratifiedKFold, 
    LeaveOneOut, 
    LeavePOut,
    train_test_split
)
from dataclasses import dataclass, asdict
from collections import defaultdict


# ========================== 实验配置 ==========================
@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str
    
    # 数据划分策略
    split_strategy: str  # 'kfold', 'loo', 'lpo', 'simple', 'original', 'stratified'
    # 训练策略
    train_strategy: str  # 'joint', 'separate', 'sequential', 'hybrid'
    n_folds: int = 5
    test_size: float = 0.2
    val_size: float = 0.15
    p_leave_out: int = 3
    

    
    # 停止条件
    use_early_stop: bool = False
    early_stop_patience: int = 100
    max_epochs: int = 1000
    
    # 优化参数
    lr_morph: float = 1e-4
    lr_calib: float = 1e-4
    wd_morph: float = 1e-2
    dropout: float = 0.3
    
    # 数据增强
    use_augmentation: bool = False
    noise_std: float = 0.01
    
    # 其他
    seed: int = 42


# ========================== 数据划分策略 ==========================
class DataSplitter:
    """统一的数据划分接口"""
    
    @staticmethod
    def create_strata(m_sparse):
        """创建分层变量（基于有效点数量）"""
        sample_counts = m_sparse.sum(dim=(1, 2)).numpy()
        strata = (sample_counts > np.median(sample_counts)).astype(int)
        return strata
    
    @staticmethod
    def split_original(n_samples, strata, test_size=0.15, val_size=0.15, seed=42):
        """原始三分法"""
        indices = np.arange(n_samples)
        
        # Train/Temp split
        train_idx, temp_idx = train_test_split(
            indices, test_size=(test_size + val_size),
            random_state=seed, stratify=strata
        )
        
        # Val/Test split
        temp_strata = strata[temp_idx]
        val_ratio = val_size / (test_size + val_size)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_ratio),
            random_state=seed, stratify=temp_strata
        )
        
        return [{
            'fold': 0,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'description': f'Original split: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}'
        }]
    
    @staticmethod
    def split_kfold(n_samples, strata, n_folds=5, seed=42):
        """K折交叉验证（无验证集）"""
        indices = np.arange(n_samples)
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        splits = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(indices, strata)):
            splits.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': None,  # 无验证集
                'test_idx': test_idx,
                'description': f'Fold {fold+1}/{n_folds}: {len(train_idx)}/{len(test_idx)}'
            })
        
        return splits
    
    @staticmethod
    def split_loo(n_samples, strata, seed=42):
        """留一法（计算密集）"""
        loo = LeaveOneOut()
        indices = np.arange(n_samples)
        
        splits = []
        for fold, (train_idx, test_idx) in enumerate(loo.split(indices)):
            splits.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': None,
                'test_idx': test_idx,
                'description': f'LOO {fold+1}/{n_samples}: {len(train_idx)}/{len(test_idx)}'
            })
        
        return splits
    
    @staticmethod
    def split_lpo(n_samples, strata, p=3, seed=42):
        """留P法（非常计算密集）"""
        lpo = LeavePOut(p=p)
        indices = np.arange(n_samples)
        
        splits = []
        for fold, (train_idx, test_idx) in enumerate(lpo.split(indices)):
            splits.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': None,
                'test_idx': test_idx,
                'description': f'LPO {fold+1}: {len(train_idx)}/{len(test_idx)}'
            })
            
            # 限制最多100折（否则太多）
            if fold >= 99:
                print(f"[Warning] LPO has too many splits, limiting to 100")
                break
        
        return splits
    
    @staticmethod
    def split_simple(n_samples, strata, test_size=0.2, seed=42):
        """简单Train/Test划分（无验证集）"""
        indices = np.arange(n_samples)
        
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size,
            random_state=seed, stratify=strata
        )
        
        return [{
            'fold': 0,
            'train_idx': train_idx,
            'val_idx': None,
            'test_idx': test_idx,
            'description': f'Simple split: {len(train_idx)}/{len(test_idx)}'
        }]
    
    @staticmethod
    def split_stratified_by_family(n_samples, strata, m_sparse, families, 
                                   test_size=0.15, val_size=0.15, seed=42):
        """按family分布分层（确保每个family在各集合中都有代表）"""
        indices = np.arange(n_samples)
        
        # 计算每个样本在每个family上的有效点数
        family_counts = m_sparse.sum(dim=2).numpy()  # [B, K]
        
        # 找到每个样本的主要family
        primary_family = family_counts.argmax(axis=1)
        
        # Train/Temp split
        train_idx, temp_idx = train_test_split(
            indices, test_size=(test_size + val_size),
            random_state=seed, stratify=primary_family
        )
        
        # Val/Test split
        temp_primary = primary_family[temp_idx]
        val_ratio = val_size / (test_size + val_size)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_ratio),
            random_state=seed, stratify=temp_primary
        )
        
        return [{
            'fold': 0,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'description': f'Stratified by family: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}'
        }]
    
    @classmethod
    def get_splits(cls, config: ExperimentConfig, m_sparse, families=None):
        """根据配置获取数据划分"""
        n_samples = m_sparse.shape[0]
        strata = cls.create_strata(m_sparse)
        
        strategy = config.split_strategy
        seed = config.seed
        
        if strategy == 'original':
            return cls.split_original(n_samples, strata, config.test_size, 
                                     config.val_size, seed)
        elif strategy == 'kfold':
            return cls.split_kfold(n_samples, strata, config.n_folds, seed)
        elif strategy == 'loo':
            return cls.split_loo(n_samples, strata, seed)
        elif strategy == 'lpo':
            return cls.split_lpo(n_samples, strata, config.p_leave_out, seed)
        elif strategy == 'simple':
            return cls.split_simple(n_samples, strata, config.test_size, seed)
        elif strategy == 'stratified':
            return cls.split_stratified_by_family(
                n_samples, strata, m_sparse, families,
                config.test_size, config.val_size, seed
            )
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")


# ========================== 训练策略 ==========================
class TrainingStrategy:
    """不同的训练策略"""
    
    @staticmethod
    def train_joint(model, data, config, logger):
        """联合训练所有heads（标准方法）"""
        # 标准的联合训练
        # 所有family heads同时更新
        return TrainingStrategy._standard_train_loop(
            model, data, config, logger, 
            train_mode='joint'
        )
    
    @staticmethod
    def train_separate(model, data, config, logger):
        """分立训练每个family head"""
        # 为每个family单独训练一个模型
        results = {}
        
        families = data['families']
        for k, family in enumerate(families):
            print(f"\n{'='*80}")
            print(f"Training separate head for {family} ({k+1}/{len(families)})")
            print(f"{'='*80}")
            
            # 创建family-specific mask
            family_mask = data['m_sparse'][:, k:k+1, :].clone()
            
            # 临时修改数据只包含这个family
            data_k = copy.deepcopy(data)
            data_k['m_sparse_original'] = data_k['m_sparse'].clone()
            data_k['m_sparse'] = family_mask.expand_as(data_k['m_sparse'])
            
            # 训练
            result_k = TrainingStrategy._standard_train_loop(
                model, data_k, config, logger,
                train_mode=f'separate_{family}'
            )
            
            results[family] = result_k
        
        # 合并结果
        return TrainingStrategy._merge_separate_results(results, families)
    
    @staticmethod
    def train_sequential(model, data, config, logger):
        """顺序训练：从稀疏family到密集family"""
        families = data['families']
        m_sparse = data['m_sparse']
        
        # 计算每个family的数据量
        family_counts = m_sparse.sum(dim=(0, 2)).numpy()
        sorted_indices = np.argsort(family_counts)  # 从少到多
        
        print(f"\n{'='*80}")
        print(f"Sequential training order (sparse to dense):")
        for idx in sorted_indices:
            print(f"  {families[idx]}: {int(family_counts[idx])} points")
        print(f"{'='*80}\n")
        
        # 逐个family训练
        for i, k in enumerate(sorted_indices):
            family = families[k]
            print(f"\n[Stage {i+1}/{len(families)}] Training on {family}")
            
            # 创建累积mask（包含之前所有family）
            cumulative_mask = torch.zeros_like(m_sparse)
            for prev_k in sorted_indices[:i+1]:
                cumulative_mask[:, prev_k, :] = m_sparse[:, prev_k, :]
            
            data_cumulative = copy.deepcopy(data)
            data_cumulative['m_sparse'] = cumulative_mask
            
            # 训练（较短epoch）
            config_stage = copy.deepcopy(config)
            config_stage.max_epochs = config.max_epochs // len(families)
            
            TrainingStrategy._standard_train_loop(
                model, data_cumulative, config_stage, logger,
                train_mode=f'sequential_{family}'
            )
        
        # 最后全数据微调
        print(f"\n[Final] Fine-tuning on all families")
        config_final = copy.deepcopy(config)
        config_final.max_epochs = config.max_epochs // 4
        
        return TrainingStrategy._standard_train_loop(
            model, data, config_final, logger,
            train_mode='sequential_final'
        )
    
    @staticmethod
    def train_hybrid(model, data, config, logger):
        """混合策略：先联合预训练，再分立微调"""
        print(f"\n{'='*80}")
        print(f"Phase 1: Joint pre-training (80% epochs)")
        print(f"{'='*80}")
        
        # Phase 1: 联合预训练
        config_phase1 = copy.deepcopy(config)
        config_phase1.max_epochs = int(config.max_epochs * 0.8)
        
        TrainingStrategy._standard_train_loop(
            model, data, config_phase1, logger,
            train_mode='hybrid_phase1'
        )
        
        print(f"\n{'='*80}")
        print(f"Phase 2: Separate fine-tuning (20% epochs per family)")
        print(f"{'='*80}")
        
        # Phase 2: 分立微调
        families = data['families']
        config_phase2 = copy.deepcopy(config)
        config_phase2.max_epochs = int(config.max_epochs * 0.2)
        
        for k, family in enumerate(families):
            print(f"\n[Fine-tune {k+1}/{len(families)}] {family}")
            
            family_mask = data['m_sparse'][:, k:k+1, :].clone()
            data_k = copy.deepcopy(data)
            data_k['m_sparse'] = family_mask.expand_as(data_k['m_sparse'])
            
            TrainingStrategy._standard_train_loop(
                model, data_k, config_phase2, logger,
                train_mode=f'hybrid_phase2_{family}'
            )
        
        # 最终评估
        return TrainingStrategy._standard_train_loop(
            model, data, ExperimentConfig(
                name='eval', description='', split_strategy='', train_strategy='',
                max_epochs=0  # 只评估不训练
            ), logger,
            train_mode='hybrid_final'
        )
    
    @staticmethod
    def _standard_train_loop(model, data, config, logger, train_mode='joint'):
        """标准训练循环（内部使用）"""
        # 这里应该调用实际的训练代码
        # 为了演示，返回模拟结果
        # 在实际使用时，需要替换为真实的训练逻辑
        
        print(f"  [Training] Mode={train_mode}, Epochs={config.max_epochs}")
        
        # TODO: 替换为实际训练代码
        # result = actual_train_function(model, data, config)
        
        result = {
            'train_mode': train_mode,
            'epochs_trained': config.max_epochs,
            'final_loss': 0.0,  # 占位
            'metrics': {}  # 占位
        }
        
        return result
    
    @staticmethod
    def _merge_separate_results(results, families):
        """合并分立训练的结果"""
        merged = {
            'train_mode': 'separate',
            'families': families,
            'per_family_results': results,
            'metrics': {}
        }
        
        # 聚合指标
        for key in ['R2', 'MAE', 'RMSE']:
            values = [r['metrics'].get(key, 0) for r in results.values()]
            merged['metrics'][f'{key}_mean'] = np.mean(values)
            merged['metrics'][f'{key}_std'] = np.std(values)
        
        return merged


# ========================== 实验管理器 ==========================
class ExperimentManager:
    """管理所有实验"""
    
    def __init__(self, save_root='./runs_comprehensive'):
        self.save_root = save_root
        os.makedirs(save_root, exist_ok=True)
        
        self.results = []
        self.configs = []
    
    def create_all_configs(self, base_seeds=[42, 43, 44]):
        """创建所有可能的实验配置"""
        configs = []
        
        # ========== 数据划分策略对比 ==========
        split_strategies = [
            ('original', '原始三分法', {'test_size': 0.10, 'val_size': 0.10}),
            ('kfold5', '5折交叉验证', {'n_folds': 5}),
            ('kfold10', '10折交叉验证', {'n_folds': 10}),
            ('simple', '简单Train/Test', {'test_size': 0.20}),
            ('stratified', '分层划分', {'test_size': 0.10, 'val_size': 0.10}),
            # ('loo', '留一法', {}),  # 太慢，可选
        ]
        
        # ========== 训练策略对比 ==========
        train_strategies = [
            ('joint', '联合训练'),
            ('separate', '分立训练'),
            ('sequential', '顺序训练'),
            ('hybrid', '混合训练'),
        ]
        
        # ========== 优化策略对比 ==========
        optimization_configs = [
            ('conservative', '保守策略', {
                'lr_morph': 1e-4, 'wd_morph': 1e-2, 'dropout': 0.5
            }),
            ('aggressive', '激进策略', {
                'lr_morph': 3e-4, 'wd_morph': 5e-3, 'dropout': 0.3
            }),
            ('balanced', '平衡策略', {
                'lr_morph': 2e-4, 'wd_morph': 7e-3, 'dropout': 0.4
            }),
        ]
        
        # ========== 生成所有组合 ==========
        experiment_id = 0
        
        # 1. 数据划分策略对比（固定训练策略和优化策略）
        for split_name, split_desc, split_params in split_strategies:
            for seed in base_seeds:
                config = ExperimentConfig(
                    name=f'split_{split_name}_seed{seed}',
                    description=f'{split_desc} (Seed {seed})',
                    split_strategy=split_name.replace('kfold5', 'kfold').replace('kfold10', 'kfold'),
                    train_strategy='joint',
                    seed=seed,
                    lr_morph=1e-4,
                    wd_morph=1e-2,
                    dropout=0.3,
                    max_epochs=1000,
                    use_early_stop=('val_idx' in split_name),  # 只有有val的才用early stop
                    **split_params
                )
                configs.append(config)
                experiment_id += 1
        
        # 2. 训练策略对比（固定最佳数据划分和优化策略）
        for train_name, train_desc in train_strategies:
            for seed in base_seeds:
                config = ExperimentConfig(
                    name=f'train_{train_name}_seed{seed}',
                    description=f'{train_desc} (Seed {seed})',
                    split_strategy='kfold',
                    n_folds=5,
                    train_strategy=train_name,
                    seed=seed,
                    lr_morph=1e-4,
                    wd_morph=1e-2,
                    dropout=0.3,
                    max_epochs=1000,
                    use_early_stop=False,
                )
                configs.append(config)
                experiment_id += 1
        
        # 3. 优化策略对比（固定最佳数据划分和训练策略）
        for opt_name, opt_desc, opt_params in optimization_configs:
            for seed in base_seeds:
                config = ExperimentConfig(
                    name=f'opt_{opt_name}_seed{seed}',
                    description=f'{opt_desc} (Seed {seed})',
                    split_strategy='kfold',
                    n_folds=5,
                    train_strategy='joint',
                    seed=seed,
                    max_epochs=1000,
                    use_early_stop=False,
                    **opt_params
                )
                configs.append(config)
                experiment_id += 1
        
        # 4. 消融实验：数据增强
        for use_aug in [False, True]:
            for seed in base_seeds:
                config = ExperimentConfig(
                    name=f'aug_{use_aug}_seed{seed}',
                    description=f'数据增强={use_aug} (Seed {seed})',
                    split_strategy='kfold',
                    n_folds=5,
                    train_strategy='joint',
                    seed=seed,
                    lr_morph=1e-4,
                    wd_morph=1e-2,
                    dropout=0.3,
                    max_epochs=1000,
                    use_early_stop=False,
                    use_augmentation=use_aug,
                    noise_std=0.01 if use_aug else 0.0,
                )
                configs.append(config)
                experiment_id += 1
        
        print(f"\n[ExperimentManager] Created {len(configs)} experiment configurations")
        print(f"  - Data split strategies: {len(split_strategies) * len(base_seeds)}")
        print(f"  - Training strategies: {len(train_strategies) * len(base_seeds)}")
        print(f"  - Optimization strategies: {len(optimization_configs) * len(base_seeds)}")
        print(f"  - Augmentation ablation: {2 * len(base_seeds)}")
        
        self.configs = configs
        return configs
    
    def run_experiment(self, config: ExperimentConfig, data_dict, model_builder, 
                      variant, Cfg):
        """运行单个实验"""
        print(f"\n{'='*80}")
        print(f"Running Experiment: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*80}")
        
        # 1. 数据划分
        m_sparse = data_dict['m_sparse']
        families = data_dict['families']
        
        splits = DataSplitter.get_splits(config, m_sparse, families)
        print(f"[Data] Generated {len(splits)} split(s)")
        
        # 2. 对每个split运行实验
        split_results = []
        
        for split_info in splits:
            fold = split_info['fold']
            print(f"\n[Fold {fold}] {split_info['description']}")
            
            # 准备当前fold的数据
            fold_data = self._prepare_fold_data(
                data_dict, split_info, config.seed
            )
            
            # 构建模型
            model, optimizer, scheduler, ema = model_builder(config, fold_data, Cfg)
            
            # 选择训练策略
            logger = ExperimentLogger(config.name, fold)
            
            if config.train_strategy == 'joint':
                result = TrainingStrategy.train_joint(model, fold_data, config, logger)
            elif config.train_strategy == 'separate':
                result = TrainingStrategy.train_separate(model, fold_data, config, logger)
            elif config.train_strategy == 'sequential':
                result = TrainingStrategy.train_sequential(model, fold_data, config, logger)
            elif config.train_strategy == 'hybrid':
                result = TrainingStrategy.train_hybrid(model, fold_data, config, logger)
            else:
                raise ValueError(f"Unknown train strategy: {config.train_strategy}")
            
            # 最终评估
            final_metrics = self._evaluate_fold(
                model, fold_data, split_info, config, Cfg
            )
            
            result['metrics'] = final_metrics
            result['fold'] = fold
            split_results.append(result)
        
        # 3. 聚合所有fold的结果
        aggregated = self._aggregate_results(split_results, config)
        
        # 4. 保存结果
        self._save_experiment_result(config, aggregated)
        
        return aggregated
    
    def _prepare_fold_data(self, data_dict, split_info, seed):
        """准备单个fold的数据"""
        # 提取indices
        train_idx = split_info['train_idx']
        val_idx = split_info.get('val_idx', None)
        test_idx = split_info['test_idx']
        
        # 复制数据
        fold_data = copy.deepcopy(data_dict)
        
        # 重新计算训练集统计量
        s8 = data_dict['s8']
        norm_mean = data_dict.get('norm_mean', torch.zeros(7))
        norm_std = data_dict.get('norm_std', torch.ones(7))
        
        s8_train = s8[train_idx]
        s8_train_raw = s8_train * (norm_std + 1e-8) + norm_mean
        train_mean = s8_train_raw.mean(dim=0)
        train_std = s8_train_raw.std(dim=0).clamp_min(1e-8)
        
        # 重新标准化全部数据
        s8_full_raw = s8 * (norm_std + 1e-8) + norm_mean
        s8_renorm = (s8_full_raw - train_mean) / train_std
        
        fold_data['s8'] = s8_renorm
        fold_data['splits'] = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        fold_data['train_stats'] = {
            'mean': train_mean,
            'std': train_std
        }
        
        return fold_data
    
    def _evaluate_fold(self, model, fold_data, split_info, config, Cfg):
        """评估单个fold"""
        # TODO: 替换为实际评估代码
        # 这里返回模拟指标
        
        metrics = {
            'R2_macro': 0.5,  # 占位
            'MAE_macro': 0.08,  # 占位
            'RMSE_macro': 0.10,  # 占位
            'R2_per_family': [0.4, 0.5, 0.6, 0.5, 0.4, 0.6],  # 占位
        }
        
        return metrics
    
    def _aggregate_results(self, split_results, config):
        """聚合多个fold的结果"""
        if len(split_results) == 1:
            # 单次划分，直接返回
            return split_results[0]['metrics']
        
        # 多折，计算平均和标准差
        aggregated = {}
        
        for key in split_results[0]['metrics'].keys():
            if 'per_family' in key:
                # 逐family的结果
                values = np.array([r['metrics'][key] for r in split_results])
                aggregated[key + '_mean'] = values.mean(axis=0).tolist()
                aggregated[key + '_std'] = values.std(axis=0).tolist()
            else:
                # 标量结果
                values = [r['metrics'][key] for r in split_results]
                aggregated[key + '_mean'] = np.mean(values)
                aggregated[key + '_std'] = np.std(values)
                aggregated[key + '_values'] = values
        
        aggregated['n_folds'] = len(split_results)
        
        return aggregated
    
    def _save_experiment_result(self, config, result):
        """保存实验结果"""
        save_path = os.path.join(self.save_root, f'{config.name}.json')
        
        result_dict = {
            'config': asdict(config),
            'result': result,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"[Saved] {save_path}")
    
    def run_all_experiments(self, data_dict, model_builder, variant, Cfg):
        """运行所有实验"""
        n_experiments = len(self.configs)
        
        print(f"\n{'='*80}")
        print(f"Starting Comprehensive Experiments")
        print(f"Total experiments: {n_experiments}")
        print(f"{'='*80}\n")
        
        results = []
        
        for i, config in enumerate(self.configs, 1):
            print(f"\n[Progress] Experiment {i}/{n_experiments}")
            
            try:
                result = self.run_experiment(
                    config, data_dict, model_builder, variant, Cfg
                )
                results.append({
                    'config': config,
                    'result': result,
                    'status': 'success'
                })
            except Exception as e:
                print(f"[Error] Experiment {config.name} failed: {e}")
                results.append({
                    'config': config,
                    'result': None,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # 生成综合报告
        self.generate_comprehensive_report(results)
        
        return results
    
    def generate_comprehensive_report(self, results):
        """生成综合对比报告"""
        print(f"\n{'='*80}")
        print(f"Generating Comprehensive Report")
        print(f"{'='*80}\n")
        
        # 按类别分组
        categories = {
            'split': [],
            'train': [],
            'opt': [],
            'aug': []
        }
        
        for res in results:
            if res['status'] != 'success':
                continue
            
            config = res['config']
            name = config.name
            
            if name.startswith('split_'):
                categories['split'].append(res)
            elif name.startswith('train_'):
                categories['train'].append(res)
            elif name.startswith('opt_'):
                categories['opt'].append(res)
            elif name.startswith('aug_'):
                categories['aug'].append(res)
        
        # 生成各类别的报告
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("COMPREHENSIVE EXPERIMENTAL RESULTS")
        report_lines.append("=" * 100)
        report_lines.append("")
        
        # 1. 数据划分策略对比
        report_lines.append("1. DATA SPLIT STRATEGY COMPARISON")
        report_lines.append("-" * 100)
        report_lines.extend(self._format_category_report(categories['split'], 'split'))
        report_lines.append("")
        
        # 2. 训练策略对比
        report_lines.append("2. TRAINING STRATEGY COMPARISON")
        report_lines.append("-" * 100)
        report_lines.extend(self._format_category_report(categories['train'], 'train'))
        report_lines.append("")
        
        # 3. 优化策略对比
        report_lines.append("3. OPTIMIZATION STRATEGY COMPARISON")
        report_lines.append("-" * 100)
        report_lines.extend(self._format_category_report(categories['opt'], 'opt'))
        report_lines.append("")
        
        # 4. 数据增强对比
        report_lines.append("4. DATA AUGMENTATION COMPARISON")
        report_lines.append("-" * 100)
        report_lines.extend(self._format_category_report(categories['aug'], 'aug'))
        report_lines.append("")
        
        # 5. 总体最佳
        report_lines.append("5. OVERALL BEST CONFIGURATIONS")
        report_lines.append("-" * 100)
        report_lines.extend(self._find_best_configs(results))
        report_lines.append("")
        
        report_lines.append("=" * 100)
        
        # 保存报告
        report_path = os.path.join(self.save_root, 'COMPREHENSIVE_REPORT.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # 打印报告
        for line in report_lines:
            print(line)
        
        print(f"\n[Saved] {report_path}")
    
    def _format_category_report(self, category_results, category_name):
        """格式化某一类别的报告"""
        lines = []
        
        if not category_results:
            lines.append("  No results in this category.")
            return lines
        
        # 按R²排序
        sorted_results = sorted(
            category_results,
            key=lambda x: x['result'].get('R2_macro_mean', x['result'].get('R2_macro', -999)),
            reverse=True
        )
        
        # 表头
        lines.append(f"{'Rank':<6} {'Name':<40} {'R²_macro':<12} {'MAE_macro':<12} {'Description':<50}")
        lines.append("-" * 100)
        
        # 数据行
        for rank, res in enumerate(sorted_results[:10], 1):  # 只显示Top 10
            config = res['config']
            result = res['result']
            
            r2 = result.get('R2_macro_mean', result.get('R2_macro', 0))
            r2_std = result.get('R2_macro_std', 0)
            mae = result.get('MAE_macro_mean', result.get('MAE_macro', 0))
            mae_std = result.get('MAE_macro_std', 0)
            
            if r2_std > 0:
                r2_str = f"{r2:.4f}±{r2_std:.4f}"
                mae_str = f"{mae:.4f}±{mae_std:.4f}"
            else:
                r2_str = f"{r2:.4f}"
                mae_str = f"{mae:.4f}"
            
            lines.append(
                f"{rank:<6} {config.name:<40} {r2_str:<12} {mae_str:<12} {config.description:<50}"
            )
        
        return lines
    
    def _find_best_configs(self, results):
        """找出最佳配置"""
        lines = []
        
        # 按R²排序
        valid_results = [r for r in results if r['status'] == 'success']
        sorted_results = sorted(
            valid_results,
            key=lambda x: x['result'].get('R2_macro_mean', x['result'].get('R2_macro', -999)),
            reverse=True
        )
        
        # Top 5
        lines.append(f"{'Rank':<6} {'Name':<40} {'R²':<15} {'MAE':<15}")
        lines.append("-" * 100)
        
        for rank, res in enumerate(sorted_results[:5], 1):
            config = res['config']
            result = res['result']
            
            r2 = result.get('R2_macro_mean', result.get('R2_macro', 0))
            r2_std = result.get('R2_macro_std', 0)
            mae = result.get('MAE_macro_mean', result.get('MAE_macro', 0))
            mae_std = result.get('MAE_macro_std', 0)
            
            if r2_std > 0:
                r2_str = f"{r2:.4f}±{r2_std:.4f}"
                mae_str = f"{mae:.4f}±{mae_std:.4f}"
            else:
                r2_str = f"{r2:.4f}"
                mae_str = f"{mae:.4f}"
            
            lines.append(f"{rank:<6} {config.name:<40} {r2_str:<15} {mae_str:<15}")
            
            # 显示详细配置
            lines.append(f"        Split: {config.split_strategy}, "
                        f"Train: {config.train_strategy}, "
                        f"LR: {config.lr_morph:.0e}, "
                        f"WD: {config.wd_morph:.0e}, "
                        f"Dropout: {config.dropout:.2f}")
        
        return lines


# ========================== 辅助类 ==========================
class ExperimentLogger:
    """实验日志"""
    
    def __init__(self, name, fold):
        self.name = name
        self.fold = fold
        self.logs = []
    
    def log_epoch(self, epoch, train_loss, val_metrics, best_val_r2):
        """记录每个epoch"""
        self.logs.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'best_val_r2': best_val_r2
        })
    
    def get_logs(self):
        return self.logs


# ========================== 主入口函数（可直接替换）==========================
def run_comprehensive_experiments_main(Cfg, data_dict, model_builder_func, variant):
    """
    主函数：运行所有实验
    
    参数:
        Cfg: 原始配置类
        data_dict: 包含s8, y_sparse, m_sparse等的字典
        model_builder_func: 构建模型的函数
        variant: 变体配置
    
    返回:
        所有实验的结果
    """
    # 创建实验管理器
    manager = ExperimentManager(save_root=Cfg.save_root + '_comprehensive')
    
    # 创建所有实验配置
    configs = manager.create_all_configs(base_seeds=Cfg.seeds)
    
    # 运行所有实验
    results = manager.run_all_experiments(
        data_dict=data_dict,
        model_builder=model_builder_func,
        variant=variant,
        Cfg=Cfg
    )
    
    return results, manager


if __name__ == "__main__":
    print("Comprehensive Experiment Framework Loaded Successfully!")
    print("\n使用方法:")
    print("1. 导入此模块: from comprehensive_experiments import *")
    print("2. 调用主函数: results, manager = run_comprehensive_experiments_main(...)")
    print("3. 查看报告: manager.generate_comprehensive_report(results)")
