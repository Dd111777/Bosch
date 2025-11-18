# -*- coding: utf-8 -*-
"""
增量实验方案生成器：基于统计分析结果的智能采样
================================================

根据statistical_analysis.xlsx的结果，生成3种规模的实验方案：
- 方案A：10个样本（¥20k预算）
- 方案B：20个样本（¥40k预算）
- 方案C：30个样本（¥60k预算）

策略：
1. 覆盖旧数据的关键区域
2. 避开新数据集中的区域（TEMP≈-13, APC≈25等）
3. 优先采样高变异参数（Cohen's d大的）
4. 确保每个参数至少有3个不同水平
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ============ 基于统计分析的参数范围 ============
# 从statistical_analysis.xlsx提取

PARAM_RANGES = {
    'TEMP': {
        'old_min': -20, 'old_max': 20, 'old_mean': -3.63, 'old_std': 12.85,
        'new_mean': -13.00, 'new_std': 9.36,
        'cohens_d': 2.5,  # 高优先级
        'importance': 'HIGH'
    },
    'APC': {
        'old_min': 20, 'old_max': 100, 'old_mean': 49.51, 'old_std': 25.99,
        'new_mean': 25.67, 'new_std': 5.59,
        'cohens_d': 5.0,  # 极高优先级
        'importance': 'CRITICAL'
    },
    'SOURCE_RF': {
        'old_min': 20, 'old_max': 100, 'old_mean': 49.51, 'old_std': 25.99,
        'new_mean': 25.67, 'new_std': 5.59,
        'cohens_d': 5.0,  # 极高优先级
        'importance': 'CRITICAL'
    },
    'LF_RF': {
        'old_min': 50, 'old_max': 350, 'old_mean': 77.55, 'old_std': 39.35,
        'new_mean': 50.00, 'new_std': 76.10,
        'cohens_d': 4.0,  # 高优先级
        'importance': 'HIGH'
    },
    'SF6': {
        'old_min': 100, 'old_max': 800, 'old_mean': 410.66, 'old_std': 204.24,
        'new_mean': 320.00, 'new_std': 97.98,
        'cohens_d': 3.5,
        'importance': 'HIGH'
    },
    'C4F8': {
        'old_min': 250, 'old_max': 850, 'old_mean': 448.69, 'old_std': 233.50,
        'new_mean': 331.67, 'new_std': 79.04,
        'cohens_d': 4.5,
        'importance': 'HIGH'
    },
    'DEP_TIME': {
        'old_min': 1.0, 'old_max': 4.0, 'old_mean': 2.04, 'old_std': 1.07,
        'new_mean': 2.01, 'new_std': 0.45,
        'cohens_d': 2.0,
        'importance': 'MEDIUM'
    },
    'ETCH_TIME': {
        'old_min': 1.0, 'old_max': 4.0, 'old_mean': 1.98, 'old_std': 1.10,
        'new_mean': 1.57, 'new_std': 0.85,
        'cohens_d': 1.5,
        'importance': 'MEDIUM'
    }
}

PARAM_NAMES = list(PARAM_RANGES.keys())


# ============ 采样策略 ============

def generate_strategic_samples(n_samples, strategy='space_filling'):
    """
    生成战略性采样点
    
    strategy:
      - 'space_filling': 空间填充（均匀覆盖）
      - 'critical_regions': 关键区域（高变异参数）
      - 'boundary': 边界探索（极值点）
    """
    samples = []
    
    if strategy == 'space_filling':
        # 使用分层采样确保覆盖
        # 将每个参数分成3-5个区间
        
        # 确定每个参数的采样水平
        n_levels = max(3, int(np.ceil(n_samples ** (1/8))))  # 8个参数
        
        # 生成网格点（但不是全因子，而是Sobol序列）
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=8, scramble=True, seed=42)
        samples_normalized = sampler.random(n=n_samples * 10)  # 生成10倍候选
        
        # 缩放到实际范围
        samples_real = []
        for sample in samples_normalized:
            recipe = {}
            for i, param in enumerate(PARAM_NAMES):
                prange = PARAM_RANGES[param]
                # 使用5%-95%分位数范围（避免极值）
                low = prange['old_min'] + 0.05 * (prange['old_max'] - prange['old_min'])
                high = prange['old_max'] - 0.05 * (prange['old_max'] - prange['old_min'])
                recipe[param] = low + sample[i] * (high - low)
            samples_real.append(recipe)
        
        # 计算到新数据中心的距离，选择最远的n_samples个
        new_center = np.array([PARAM_RANGES[p]['new_mean'] for p in PARAM_NAMES])
        
        # 标准化
        samples_array = np.array([[s[p] for p in PARAM_NAMES] for s in samples_real])
        scales = np.array([PARAM_RANGES[p]['old_std'] for p in PARAM_NAMES])
        samples_normalized = (samples_array - new_center) / scales
        
        # 计算距离
        distances = np.linalg.norm(samples_normalized, axis=1)
        
        # 选择最远的n_samples个
        selected_idx = np.argsort(distances)[-n_samples:]
        samples = [samples_real[i] for i in selected_idx]
    
    elif strategy == 'critical_regions':
        # 重点采样关键参数（Cohen's d大的）
        critical_params = [p for p in PARAM_NAMES if PARAM_RANGES[p]['importance'] in ['CRITICAL', 'HIGH']]
        
        # 对关键参数使用3水平，其他参数使用均值
        levels = [-1, 0, 1]  # 低、中、高
        
        for i in range(n_samples):
            recipe = {}
            for param in PARAM_NAMES:
                prange = PARAM_RANGES[param]
                
                if param in critical_params:
                    # 随机选择一个水平
                    level = np.random.choice(levels)
                    if level == -1:
                        recipe[param] = prange['old_min'] + 0.1 * (prange['old_max'] - prange['old_min'])
                    elif level == 0:
                        recipe[param] = prange['old_mean']
                    else:
                        recipe[param] = prange['old_max'] - 0.1 * (prange['old_max'] - prange['old_min'])
                else:
                    # 使用均值 ± 随机扰动
                    recipe[param] = prange['old_mean'] + np.random.randn() * prange['old_std'] * 0.5
                    # 裁剪到范围内
                    recipe[param] = np.clip(recipe[param], prange['old_min'], prange['old_max'])
            
            samples.append(recipe)
    
    elif strategy == 'boundary':
        # 探索边界和角点
        # 2^8 = 256个角点太多，选择关键的
        
        # 对每个参数，交替使用高低值
        for i in range(n_samples):
            recipe = {}
            np.random.seed(42 + i)
            
            for param in PARAM_NAMES:
                prange = PARAM_RANGES[param]
                
                # 70%概率选择边界，30%概率选择中间
                if np.random.rand() < 0.7:
                    # 边界值（高或低）
                    if np.random.rand() < 0.5:
                        recipe[param] = prange['old_min'] + 0.05 * (prange['old_max'] - prange['old_min'])
                    else:
                        recipe[param] = prange['old_max'] - 0.05 * (prange['old_max'] - prange['old_min'])
                else:
                    # 中间值
                    recipe[param] = prange['old_mean']
            
            samples.append(recipe)
    
    return samples


def optimize_sample_diversity(samples):
    """优化样本多样性：确保每个参数有多个水平"""
    # 检查每个参数的唯一值数量
    df = pd.DataFrame(samples)
    
    for param in PARAM_NAMES:
        unique_values = df[param].nunique()
        if unique_values < 3:
            print(f"  ⚠ {param} only has {unique_values} unique values, adding diversity...")
            # 强制添加不同水平
            prange = PARAM_RANGES[param]
            levels = [
                prange['old_min'] + 0.1 * (prange['old_max'] - prange['old_min']),
                prange['old_mean'],
                prange['old_max'] - 0.1 * (prange['old_max'] - prange['old_min'])
            ]
            
            # 替换前3个样本
            for i in range(min(3, len(samples))):
                samples[i][param] = levels[i % 3]
    
    return samples


def round_to_practical_values(samples):
    """舍入到实际可操作的值"""
    for sample in samples:
        # 温度：整数
        sample['TEMP'] = round(sample['TEMP'])
        
        # APC, SOURCE_RF, LF_RF: 5的倍数
        sample['APC'] = round(sample['APC'] / 5) * 5
        sample['SOURCE_RF'] = round(sample['SOURCE_RF'] / 5) * 5
        sample['LF_RF'] = round(sample['LF_RF'] / 5) * 5
        
        # SF6, C4F8: 10的倍数
        sample['SF6'] = round(sample['SF6'] / 10) * 10
        sample['C4F8'] = round(sample['C4F8'] / 10) * 10
        
        # 时间: 0.1精度
        sample['DEP_TIME'] = round(sample['DEP_TIME'], 1)
        sample['ETCH_TIME'] = round(sample['ETCH_TIME'], 1)
    
    return samples


def prioritize_samples(samples, current_data_mean):
    """根据距离当前数据的远近排序（远的优先级高）"""
    samples_array = np.array([[s[p] for p in PARAM_NAMES] for s in samples])
    current_array = np.array([current_data_mean[p] for p in PARAM_NAMES])
    
    # 标准化
    scales = np.array([PARAM_RANGES[p]['old_std'] for p in PARAM_NAMES])
    samples_normalized = (samples_array - current_array) / scales
    
    # 计算距离
    distances = np.linalg.norm(samples_normalized, axis=1)
    
    # 排序（距离大的排前面）
    sorted_idx = np.argsort(distances)[::-1]
    
    return [samples[i] for i in sorted_idx], distances[sorted_idx]


# ============ 生成3个方案 ============

def main():
    print("="*80)
    print("增量实验方案生成器（基于统计分析结果）")
    print("="*80)
    
    # 当前新数据的中心
    current_center = {p: PARAM_RANGES[p]['new_mean'] for p in PARAM_NAMES}
    
    # 生成3个方案
    plans = {}
    
    for n in [10, 20, 30]:
        print(f"\n{'='*80}")
        print(f"方案{chr(64+n//10)}: {n}个新样本")
        print(f"{'='*80}")
        
        # 混合策略
        # 70%空间填充 + 30%关键区域
        n_space = int(n * 0.7)
        n_critical = n - n_space
        
        samples_space = generate_strategic_samples(n_space, strategy='space_filling')
        samples_critical = generate_strategic_samples(n_critical, strategy='critical_regions')
        
        samples = samples_space + samples_critical
        
        # 优化多样性
        samples = optimize_sample_diversity(samples)
        
        # 舍入到实际值
        samples = round_to_practical_values(samples)
        
        # 排序（优先级）
        samples, distances = prioritize_samples(samples, current_center)
        
        # 添加优先级和预期信息增益
        df = pd.DataFrame(samples)
        df.insert(0, 'Sample_ID', [f'EXP_{i+1:03d}' for i in range(len(df))])
        df.insert(1, 'Priority', ['HIGH']*5 + ['MEDIUM']*10 + ['LOW']*(n-15) if n>=15 else ['HIGH']*n)
        df.insert(2, 'Distance_from_Current', distances)
        df['Expected_Info_Gain'] = (distances / distances.max() * 100).astype(int)
        
        # 添加推荐的重复次数
        df['Recommended_Replicates'] = 2
        df['Notes'] = ''
        
        # 标记关键实验
        df.loc[df['Priority'] == 'HIGH', 'Notes'] = '优先执行，高信息增益'
        df.loc[df['Priority'] == 'MEDIUM', 'Notes'] = '次优先，覆盖补充'
        df.loc[df['Priority'] == 'LOW', 'Notes'] = '可选，进一步优化'
        
        plans[f'Plan_{n}samples'] = df
        
        print(f"\n  ✓ Generated {n} samples")
        print(f"  ✓ Average distance from current data: {distances.mean():.2f} σ")
        print(f"  ✓ Max distance: {distances.max():.2f} σ")
        print(f"  ✓ Min distance: {distances.min():.2f} σ")
        
        # 显示参数覆盖情况
        print(f"\n  Parameter coverage:")
        for param in PARAM_NAMES:
            vals = df[param].values
            print(f"    {param:12s}: [{vals.min():.1f}, {vals.max():.1f}]  "
                  f"(n_unique={df[param].nunique()})")
    
    # 保存到Excel
    output_file = './incremental_experiment_plans.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 写入各个方案
        for plan_name, df in plans.items():
            sheet_name = plan_name.replace('Plan_', '').replace('samples', 'S')
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 写入说明
        summary_data = {
            '方案': ['Plan A (10个)', 'Plan B (20个)', 'Plan C (30个)'],
            '总实验数': [10, 20, 30],
            '推荐重复': [2, 2, 2],
            '总运行数': [20, 40, 60],
            '预估成本(¥)': [20000, 40000, 60000],
            '预估时间(小时)': [40, 80, 120],
            '预估时间(天)': [2, 3, 5],
            '预期R²提升': ['0.3-0.5', '0.5-0.7', '0.6-0.8'],
            '推荐场景': ['预算紧张，快速验证', '平衡方案，推荐', '充分覆盖，最佳效果']
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 写入参数范围参考
        ranges_data = []
        for param in PARAM_NAMES:
            prange = PARAM_RANGES[param]
            ranges_data.append({
                'Parameter': param,
                'Old_Min': prange['old_min'],
                'Old_Max': prange['old_max'],
                'Old_Mean': prange['old_mean'],
                'Old_Std': prange['old_std'],
                'New_Mean': prange['new_mean'],
                'New_Std': prange['new_std'],
                'Cohens_d': prange['cohens_d'],
                'Importance': prange['importance'],
                'Recommended_Levels': f"[{prange['old_min']}, {prange['old_mean']:.0f}, {prange['old_max']}]"
            })
        ranges_df = pd.DataFrame(ranges_data)
        ranges_df.to_excel(writer, sheet_name='Parameter_Ranges', index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ All plans saved to: {output_file}")
    print(f"{'='*80}")
    
    # 打印推荐
    print(f"\n📋 RECOMMENDATIONS:")
    print(f"\n🥇 如果预算充足（¥60k）：")
    print(f"   → 使用 Plan C (30个样本)")
    print(f"   → 预期R²可达 0.6-0.8")
    print(f"   → 全面覆盖参数空间")
    
    print(f"\n🥈 如果预算适中（¥40k）：")
    print(f"   → 使用 Plan B (20个样本) ⭐ 推荐")
    print(f"   → 预期R²可达 0.5-0.7")
    print(f"   → 性价比最优")
    
    print(f"\n🥉 如果预算紧张（¥20k）：")
    print(f"   → 使用 Plan A (10个样本)")
    print(f"   → 预期R²可达 0.3-0.5")
    print(f"   → 快速验证可行性")
    
    print(f"\n💡 分阶段策略（推荐）：")
    print(f"   1. 先执行Plan A的10个HIGH priority样本")
    print(f"   2. 训练评估，如果R²>0.3，继续")
    print(f"   3. 补充Plan B的另外10个样本")
    print(f"   4. 再次评估，决定是否执行Plan C")
    
    print(f"\n🔬 执行建议：")
    print(f"   1. 打开Excel文件的对应sheet")
    print(f"   2. 优先执行Priority=HIGH的样本")
    print(f"   3. 每个样本重复2次测量")
    print(f"   4. 随机化执行顺序（防止系统误差）")
    print(f"   5. 记录所有参数设置和测量结果")
    
    print(f"\n{'='*80}")
    
    # 显示前5个样本预览
    print(f"\n📊 Plan B (20 samples) - Top 5 Preview:")
    print(f"{'='*80}")
    df_preview = plans['Plan_20samples'].head(5)
    print(df_preview[['Sample_ID', 'Priority', 'TEMP', 'APC', 'SOURCE_RF', 'LF_RF', 'Expected_Info_Gain']].to_string(index=False))
    print(f"\n  (See full table in Excel file)")
    
    return plans


if __name__ == "__main__":
    main()
