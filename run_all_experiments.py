# -*- coding: utf-8 -*-
"""
完整实验运行脚本
===============

整合所有4种方法 + 消融实验 + 对比分析

使用方法：
  python run_all_experiments.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
from inverse_optimization_framework import (
    ExperimentConfig, ExperimentManager, set_seed,
    RecipeManager, ModelLoader, ExperimentLogger, Visualizer
)

# 导入各个方法
from method_1_pi_jgo import run_pi_jgo_method
from method_2_egbo import run_egbo_method

# 将方法注册到ExperimentManager
def integrate_methods(manager: ExperimentManager):
    """将具体方法实现集成到管理器"""
    
    def run_method_1():
        return run_pi_jgo_method(
            model_loader=manager.model_loader,
            recipe_manager=manager.recipe_mgr,
            config=manager.config,
            logger=manager.logger
        )
    
    def run_method_2():
        return run_egbo_method(
            model_loader=manager.model_loader,
            recipe_manager=manager.recipe_mgr,
            config=manager.config,
            logger=manager.logger
        )
    
    def run_method_3():
        print("\n  [CMOBO] Constrained Multi-objective Bayesian Optimization")
        print("  → Placeholder: will implement full CMOBO with BoTorch")
        return {'status': 'not_implemented'}
    
    def run_method_4():
        print("\n  [INN-Grad] Invertible Neural Network + Gradient")
        print("  → Placeholder: will train inverse network first")
        return {'status': 'not_implemented'}
    
    # 替换占位符方法
    manager.run_physics_informed_gradient = run_method_1
    manager.run_evolution_guided_bo = run_method_2
    manager.run_constrained_mobo = run_method_3
    manager.run_invertible_network = run_method_4


# ============ 消融实验 ============
def run_ablation_experiments(manager: ExperimentManager):
    """
    消融实验：
    1. 有/无StageA（物理一致性）
    2. 不同权重设置
    3. 不同多起点数量
    4. 不同时间加权策略
    """
    print("\n" + "="*80)
    print("消融实验")
    print("="*80)
    
    ablation_results = {}
    
    # 1. StageA消融
    print("\n[Ablation 1] 有/无StageA:")
    print("-" * 40)
    
    # 有StageA（默认）
    print("  → 运行with StageA...")
    config_with_a = manager.config
    result_with_a = run_pi_jgo_method(
        manager.model_loader,
        manager.recipe_mgr,
        config_with_a
    )
    ablation_results['with_stageA'] = result_with_a
    
    # 无StageA
    print("  → 运行without StageA...")
    # 临时禁用StageA
    original_stageA = manager.model_loader.stageA
    manager.model_loader.stageA = None
    
    result_without_a = run_pi_jgo_method(
        manager.model_loader,
        manager.recipe_mgr,
        config_with_a
    )
    ablation_results['without_stageA'] = result_without_a
    
    # 恢复StageA
    manager.model_loader.stageA = original_stageA
    
    print(f"\n  对比:")
    print(f"    With StageA:    Obj={result_with_a['best']['objective']:.6f}")
    print(f"    Without StageA: Obj={result_without_a['best']['objective']:.6f}")
    improvement = (result_without_a['best']['objective'] - result_with_a['best']['objective']) / result_without_a['best']['objective'] * 100
    print(f"    Improvement: {improvement:.2f}%")
    
    # 2. 权重消融
    print("\n[Ablation 2] 不同权重设置:")
    print("-" * 40)
    
    weight_configs = [
        (1.0, 1.0, 0.3, "平衡型"),
        (1.5, 1.0, 0.2, "d0优先"),
        (1.0, 1.5, 0.2, "d1优先"),
        (1.0, 1.0, 0.8, "w稳定优先"),
    ]
    
    for w1, w2, w3, name in weight_configs:
        print(f"  → 测试{name}: w1={w1}, w2={w2}, w3={w3}")
        
        config_temp = ExperimentConfig(
            stageA_model_path=manager.config.stageA_model_path,
            stageB_model_path=manager.config.stageB_model_path,
            old_data_path=manager.config.old_data_path,
            weight_d0=w1,
            weight_d1=w2,
            weight_w_std=w3,
            n_iterations=200,  # 减少迭代数加速
            n_multi_starts=10
        )
        
        result = run_pi_jgo_method(
            manager.model_loader,
            manager.recipe_mgr,
            config_temp
        )
        
        ablation_results[f'weights_{name}'] = result
        print(f"    → Obj={result['best']['objective']:.6f}, "
              f"d0={result['best']['metrics']['d0_weighted']:.4f}, "
              f"d1={result['best']['metrics']['d1_weighted']:.4f}, "
              f"w_std={result['best']['metrics']['w_std']:.4f}")
    
    # 3. 多起点数量消融
    print("\n[Ablation 3] 多起点数量影响:")
    print("-" * 40)
    
    n_starts_list = [5, 10, 20, 40]
    
    for n in n_starts_list:
        print(f"  → 测试n_starts={n}")
        
        config_temp = ExperimentConfig(
            stageA_model_path=manager.config.stageA_model_path,
            stageB_model_path=manager.config.stageB_model_path,
            old_data_path=manager.config.old_data_path,
            n_iterations=200,
            n_multi_starts=n
        )
        
        result = run_pi_jgo_method(
            manager.model_loader,
            manager.recipe_mgr,
            config_temp
        )
        
        ablation_results[f'n_starts_{n}'] = result
        print(f"    → Obj={result['best']['objective']:.6f}, Time={result['elapsed_time']:.2f}s")
    
    return ablation_results


# ============ 对比分析 ============
def run_comparison_analysis(results: dict, output_dir: str):
    """生成对比分析报告和图表"""
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)
    
    visualizer = Visualizer(output_dir)
    
    # 1. 收敛曲线对比（如果有历史数据）
    print("\n  → 生成收敛曲线...")
    # TODO: 实现
    
    # 2. Pareto前沿对比
    print("\n  → 生成Pareto前沿对比...")
    # TODO: 实现
    
    # 3. 生成对比表格
    print("\n  → 生成对比表格...")
    
    comparison_data = []
    
    for method_name, result in results.items():
        if 'best' in result:
            best = result['best']
            comparison_data.append({
                'Method': method_name,
                'Objective': best['objective'],
                'd0': best['metrics'].get('d0_weighted', 0),
                'd1': best['metrics'].get('d1_weighted', 0),
                'w_std': best['metrics'].get('w_std', 0),
                'Time(s)': result.get('elapsed_time', 0)
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Objective')
        
        print("\n" + "="*80)
        print("最终对比结果:")
        print("="*80)
        print(df.to_string(index=False))
        
        # 保存
        df.to_excel(os.path.join(output_dir, 'comparison_table.xlsx'), index=False)
        print(f"\n  ✓ 对比表格保存到: {output_dir}/comparison_table.xlsx")


# ============ 主函数 ============
def main():
    """主函数"""
    print("="*80)
    print("反向优化完整实验")
    print("="*80)
    print("\n📋 实验配置:")
    print("  - 方法数: 4 (PI-JGO, EGBO, CMOBO, INN-Grad)")
    print("  - 消融实验: 是")
    print("  - 对比分析: 是")
    print("  - 预计总时间: 2-4小时")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建配置
    config = ExperimentConfig(
        stageA_model_path="./runs_physics/phys_best_overall.pth",
        stageB_model_path="./runs_morph_old/morph_best_overall.pth",
        old_data_path="D:/data/pycharm/bosch/case.xlsx",
        
        target_d0=0.0,
        target_d1=0.0,
        target_w_std=0.0,
        
        weight_d0=1.0,
        weight_d1=1.0,
        weight_w_std=0.3,
        
        n_iterations=500,
        n_multi_starts=20,
        
        methods_to_run=["PI-JGO", "EGBO"],  # 先运行前2个
        ablation_experiments=True,
        comparison_experiments=True,
        
        output_dir="./inverse_optimization_results"
    )
    
    # 创建实验管理器
    print("\n🔧 初始化实验管理器...")
    manager = ExperimentManager(config)
    
    # 集成方法
    integrate_methods(manager)
    
    # 运行主要实验
    print("\n🚀 开始运行主要实验...")
    results = {}
    
    for method in config.methods_to_run:
        try:
            if method == "PI-JGO":
                result = manager.run_physics_informed_gradient()
            elif method == "EGBO":
                result = manager.run_evolution_guided_bo()
            elif method == "CMOBO":
                result = manager.run_constrained_mobo()
            elif method == "INN-Grad":
                result = manager.run_invertible_network()
            
            results[method] = result
        
        except Exception as e:
            print(f"  ✗ {method} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 运行消融实验
    if config.ablation_experiments:
        ablation_results = run_ablation_experiments(manager)
        results['ablation'] = ablation_results
    
    # 对比分析
    if config.comparison_experiments:
        run_comparison_analysis(results, config.output_dir)
    
    # 生成最终报告
    manager.generate_report(results)
    
    print("\n" + "="*80)
    print("✅ 所有实验完成!")
    print(f"📁 结果保存在: {config.output_dir}")
    print("="*80)
    
    # 输出推荐Recipe
    print("\n📊 推荐Recipe (Top 3):")
    print("-" * 80)
    
    all_candidates = []
    for method, result in results.items():
        if method == 'ablation':
            continue
        
        if 'best' in result:
            all_candidates.append({
                'method': method,
                'objective': result['best']['objective'],
                'recipe': result['best']['recipe'],
                'metrics': result['best']['metrics']
            })
    
    # 按目标值排序
    all_candidates = sorted(all_candidates, key=lambda x: x['objective'])
    
    for i, cand in enumerate(all_candidates[:3]):
        print(f"\nRank {i+1} ({cand['method']}):")
        print(f"  Objective: {cand['objective']:.6f}")
        print(f"  Metrics: d0={cand['metrics'].get('d0_weighted', 0):.4f}, "
              f"d1={cand['metrics'].get('d1_weighted', 0):.4f}, "
              f"w_std={cand['metrics'].get('w_std', 0):.4f}")
        print(f"  Recipe:")
        for j, param in enumerate(manager.recipe_mgr.PARAM_NAMES):
            print(f"    {param:12s}: {cand['recipe'][j]:.2f}")


if __name__ == "__main__":
    main()
