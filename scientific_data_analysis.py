# -*- coding: utf-8 -*-
"""
科学数据分析工具：新旧数据对比 + 样本增量设计
================================================

功能：
1. 深度统计分析（均值、方差、分布检验）
2. 参数空间覆盖度分析
3. 相关性分析
4. 主成分分析（PCA）
5. 聚类分析
6. 样本充足性评估
7. DOE（实验设计）建议

输出：
- 完整的PDF分析报告
- 详细的统计表格
- 可视化图表集
- DOE设计方案
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============ 配置 ============
class Config:
    old_excel = r"D:\data\pycharm\bosch\case.xlsx"
    new_excel = r"D:\data\pycharm\bosch\Bosch.xlsx"
    old_sheet = "case"
    new_sheet = "Sheet1"
    output_dir = "./scientific_data_analysis"
    
    # 列名匹配规则
    key_alias = {
        "temp": ["temp", "temperature"],
        "apc": ["apc", "apc(e2", "apc（e2"],
        "source_rf": ["source_rf", "sourcerf", "rfsource", "e2", "source_rf(e2", "source_rf（e2"],
        "lf_rf": ["lf_rf", "lfrf", "bias", "lf_rf(e2", "lf_rf（e2"],
        "sf6": ["sf6", "sf6(e2", "sf6（e2"],
        "c4f8": ["c4f8", "c4f8(dep", "c4f8（dep"],
        "dep_time": ["deptime", "dep_time", "dep time", "depositiontime"],
        "etch_time": ["etchtime", "etch_time", "etch time"],
    }
    
    param_names = ["APC", "SOURCE_RF", "LF_RF", "SF6", "C4F8", "DEP_TIME", "ETCH_TIME"]


# ============ 辅助函数 ============

def _canon(s: str) -> str:
    """标准化列名"""
    import re
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    return s


def _pick_one(df_cols, candidates):
    """从候选列名中匹配"""
    cols_c = {c: _canon(c) for c in df_cols}
    for c in df_cols:
        v = cols_c[c]
        for pat in candidates:
            if pat in v:
                return c
    return None


def get_static_columns(df, key_alias):
    """获取recipe参数列"""
    cols = list(df.columns)
    static_keys = []
    actual_names = []
    
    for key, aliases in key_alias.items():
        matched = _pick_one(cols, aliases)
        if matched:
            static_keys.append(matched)
            actual_names.append(key.upper())
        else:
            static_keys.append(None)
            actual_names.append(key.upper())
    
    return static_keys, actual_names


def load_data(excel_path, sheet_name, key_alias):
    """加载并清洗数据"""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    static_keys, param_names = get_static_columns(df, key_alias)
    
    # 提取数据
    data = []
    valid_names = []
    for col, name in zip(static_keys, param_names):
        if col is not None and col in df.columns:
            vals = df[col].values
            if not np.isnan(vals).all():
                data.append(vals)
                valid_names.append(name)
    
    if not data:
        return None, []
    
    data = np.array(data).T  # (N, K)
    
    # 移除NaN
    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]
    
    return data, valid_names


# ============ 统计分析 ============

def descriptive_statistics(old_data, new_data, param_names):
    """描述性统计"""
    stats_list = []
    
    for i, param in enumerate(param_names):
        old_vals = old_data[:, i]
        new_vals = new_data[:, i]
        
        # 基础统计
        stats_dict = {
            'Parameter': param,
            'Old_N': len(old_vals),
            'Old_Mean': np.mean(old_vals),
            'Old_Std': np.std(old_vals),
            'Old_Min': np.min(old_vals),
            'Old_Q1': np.percentile(old_vals, 25),
            'Old_Median': np.median(old_vals),
            'Old_Q3': np.percentile(old_vals, 75),
            'Old_Max': np.max(old_vals),
            'Old_CV': np.std(old_vals) / (np.mean(old_vals) + 1e-8),  # 变异系数
            'New_N': len(new_vals),
            'New_Mean': np.mean(new_vals),
            'New_Std': np.std(new_vals),
            'New_Min': np.min(new_vals),
            'New_Q1': np.percentile(new_vals, 25),
            'New_Median': np.median(new_vals),
            'New_Q3': np.percentile(new_vals, 75),
            'New_Max': np.max(new_vals),
            'New_CV': np.std(new_vals) / (np.mean(new_vals) + 1e-8),
        }
        
        # 分布差异
        stats_dict['Mean_Diff'] = stats_dict['New_Mean'] - stats_dict['Old_Mean']
        stats_dict['Std_Ratio'] = stats_dict['New_Std'] / (stats_dict['Old_Std'] + 1e-8)
        stats_dict['Range_Old'] = stats_dict['Old_Max'] - stats_dict['Old_Min']
        stats_dict['Range_New'] = stats_dict['New_Max'] - stats_dict['New_Min']
        stats_dict['Range_Ratio'] = stats_dict['Range_New'] / (stats_dict['Range_Old'] + 1e-8)
        
        # 标准化差异（Cohen's d）
        pooled_std = np.sqrt((stats_dict['Old_Std']**2 + stats_dict['New_Std']**2) / 2)
        stats_dict['Cohens_d'] = stats_dict['Mean_Diff'] / (pooled_std + 1e-8)
        
        # 统计检验
        # t检验（均值差异）
        t_stat, p_value_t = stats.ttest_ind(old_vals, new_vals)
        stats_dict['t_statistic'] = t_stat
        stats_dict['p_value_ttest'] = p_value_t
        
        # Levene检验（方差齐性）
        levene_stat, p_value_levene = stats.levene(old_vals, new_vals)
        stats_dict['levene_statistic'] = levene_stat
        stats_dict['p_value_levene'] = p_value_levene
        
        # Kolmogorov-Smirnov检验（分布差异）
        ks_stat, p_value_ks = stats.ks_2samp(old_vals, new_vals)
        stats_dict['ks_statistic'] = ks_stat
        stats_dict['p_value_ks'] = p_value_ks
        
        # 分布形状
        stats_dict['Old_Skewness'] = stats.skew(old_vals)
        stats_dict['Old_Kurtosis'] = stats.kurtosis(old_vals)
        stats_dict['New_Skewness'] = stats.skew(new_vals)
        stats_dict['New_Kurtosis'] = stats.kurtosis(new_vals)
        
        stats_list.append(stats_dict)
    
    return pd.DataFrame(stats_list)


def parameter_space_coverage(old_data, new_data, param_names):
    """参数空间覆盖度分析"""
    coverage_list = []
    
    for i, param in enumerate(param_names):
        old_vals = old_data[:, i]
        new_vals = new_data[:, i]
        
        old_min, old_max = np.min(old_vals), np.max(old_vals)
        new_min, new_max = np.min(new_vals), np.max(new_vals)
        
        # 计算新数据在旧数据范围内的比例
        in_range = np.sum((new_vals >= old_min) & (new_vals <= old_max))
        coverage_pct = 100 * in_range / len(new_vals)
        
        # 计算新数据覆盖旧数据范围的比例
        old_range = old_max - old_min
        new_range = new_max - new_min
        range_coverage = 100 * new_range / (old_range + 1e-8)
        
        # 空间采样密度（样本数/范围）
        old_density = len(old_vals) / (old_range + 1e-8)
        new_density = len(new_vals) / (new_range + 1e-8) if new_range > 1e-6 else 0
        
        coverage_list.append({
            'Parameter': param,
            'Old_Range': old_range,
            'New_Range': new_range,
            'Range_Coverage_%': range_coverage,
            'New_in_Old_Range_%': coverage_pct,
            'Old_Density': old_density,
            'New_Density': new_density,
            'Density_Ratio': new_density / (old_density + 1e-8),
            'Old_Min': old_min,
            'Old_Max': old_max,
            'New_Min': new_min,
            'New_Max': new_max,
        })
    
    return pd.DataFrame(coverage_list)


def correlation_analysis(old_data, new_data, param_names):
    """相关性分析"""
    old_corr = np.corrcoef(old_data.T)
    new_corr = np.corrcoef(new_data.T)
    
    # 相关性差异
    corr_diff = new_corr - old_corr
    
    # 平均绝对相关性
    old_mean_corr = np.mean(np.abs(old_corr[np.triu_indices_from(old_corr, k=1)]))
    new_mean_corr = np.mean(np.abs(new_corr[np.triu_indices_from(new_corr, k=1)]))
    
    return old_corr, new_corr, corr_diff, old_mean_corr, new_mean_corr


def pca_analysis(old_data, new_data, param_names, n_components=None):
    """主成分分析"""
    if n_components is None:
        n_components = min(len(param_names), 5)
    
    # 标准化
    scaler = StandardScaler()
    old_scaled = scaler.fit_transform(old_data)
    new_scaled = scaler.transform(new_data)
    
    # PCA
    pca = PCA(n_components=n_components)
    old_pca = pca.fit_transform(old_scaled)
    new_pca = pca.transform(new_scaled)
    
    # 解释方差
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # 载荷矩阵
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return old_pca, new_pca, explained_var, cumulative_var, loadings, pca


def cluster_analysis(old_data, new_data, n_clusters=5):
    """聚类分析"""
    scaler = StandardScaler()
    old_scaled = scaler.fit_transform(old_data)
    new_scaled = scaler.transform(new_data)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    old_labels = kmeans.fit_predict(old_scaled)
    new_labels = kmeans.predict(new_scaled)
    
    # 计算新数据在各簇中的分布
    old_cluster_counts = np.bincount(old_labels, minlength=n_clusters)
    new_cluster_counts = np.bincount(new_labels, minlength=n_clusters)
    
    old_cluster_pct = 100 * old_cluster_counts / len(old_labels)
    new_cluster_pct = 100 * new_cluster_counts / len(new_labels)
    
    return old_labels, new_labels, old_cluster_pct, new_cluster_pct, kmeans


def sample_sufficiency_analysis(old_data, new_data, param_names):
    """样本充足性分析"""
    n_params = len(param_names)
    old_n = len(old_data)
    new_n = len(new_data)
    
    # 经验法则
    min_samples_per_param = 10  # 每个参数至少10个样本
    recommended_min = n_params * min_samples_per_param
    
    # 基于方差稳定性的样本量估计
    old_cv = np.std(old_data, axis=0) / (np.mean(old_data, axis=0) + 1e-8)
    new_cv = np.std(new_data, axis=0) / (np.mean(new_data, axis=0) + 1e-8)
    
    # 估计达到CV<0.1所需的样本量（粗略估计）
    target_cv = 0.1
    estimated_n = []
    for i in range(n_params):
        if new_cv[i] > target_cv:
            # n ∝ 1/CV²
            n_needed = new_n * (new_cv[i] / target_cv)**2
            estimated_n.append(int(n_needed))
        else:
            estimated_n.append(new_n)
    
    max_n_needed = max(estimated_n)
    
    return {
        'old_n': old_n,
        'new_n': new_n,
        'n_params': n_params,
        'recommended_min': recommended_min,
        'estimated_n_for_stability': max_n_needed,
        'old_cv': old_cv,
        'new_cv': new_cv,
        'estimated_n_per_param': estimated_n
    }


# ============ 可视化 ============

def plot_comprehensive_analysis(old_data, new_data, param_names, output_dir):
    """生成综合分析图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    n_params = len(param_names)
    
    # 1. 分布对比（直方图 + 箱线图）
    fig = plt.figure(figsize=(24, 16))
    
    for i, param in enumerate(param_names):
        # 直方图
        ax1 = plt.subplot(4, n_params, i + 1)
        ax1.hist(old_data[:, i], bins=30, alpha=0.6, label='Old', color='blue', density=True)
        ax1.hist(new_data[:, i], bins=30, alpha=0.6, label='New', color='red', density=True)
        ax1.set_title(param, fontsize=10, fontweight='bold')
        ax1.set_xlabel('Value', fontsize=8)
        ax1.set_ylabel('Density', fontsize=8)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2 = plt.subplot(4, n_params, n_params + i + 1)
        bp = ax2.boxplot([old_data[:, i], new_data[:, i]], 
                          labels=['Old', 'New'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax2.set_title(param, fontsize=10)
        ax2.set_ylabel('Value', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Q-Q图（正态性检验）
        ax3 = plt.subplot(4, n_params, 2*n_params + i + 1)
        stats.probplot(new_data[:, i], dist="norm", plot=ax3)
        ax3.set_title(f'{param} Q-Q', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 核密度估计
        ax4 = plt.subplot(4, n_params, 3*n_params + i + 1)
        from scipy.stats import gaussian_kde
        if len(np.unique(old_data[:, i])) > 1:
            kde_old = gaussian_kde(old_data[:, i])
            x_old = np.linspace(old_data[:, i].min(), old_data[:, i].max(), 100)
            ax4.plot(x_old, kde_old(x_old), label='Old', color='blue', linewidth=2)
        if len(np.unique(new_data[:, i])) > 1:
            kde_new = gaussian_kde(new_data[:, i])
            x_new = np.linspace(new_data[:, i].min(), new_data[:, i].max(), 100)
            ax4.plot(x_new, kde_new(x_new), label='New', color='red', linewidth=2)
        ax4.set_title(f'{param} KDE', fontsize=10)
        ax4.set_xlabel('Value', fontsize=8)
        ax4.set_ylabel('Density', fontsize=8)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_distribution_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. 相关性矩阵对比
    old_corr, new_corr, corr_diff, _, _ = correlation_analysis(old_data, new_data, param_names)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    im1 = axes[0].imshow(old_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title('Old Data - Correlation Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(param_names)))
    axes[0].set_yticks(range(len(param_names)))
    axes[0].set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    axes[0].set_yticklabels(param_names, fontsize=10)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            axes[0].text(j, i, f'{old_corr[i,j]:.2f}', ha='center', va='center',
                        color='white' if abs(old_corr[i,j]) > 0.5 else 'black', fontsize=8)
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(new_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_title('New Data - Correlation Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(param_names)))
    axes[1].set_yticks(range(len(param_names)))
    axes[1].set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    axes[1].set_yticklabels(param_names, fontsize=10)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            axes[1].text(j, i, f'{new_corr[i,j]:.2f}', ha='center', va='center',
                        color='white' if abs(new_corr[i,j]) > 0.5 else 'black', fontsize=8)
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(corr_diff, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[2].set_title('Correlation Difference (New - Old)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(range(len(param_names)))
    axes[2].set_yticks(range(len(param_names)))
    axes[2].set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    axes[2].set_yticklabels(param_names, fontsize=10)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            axes[2].text(j, i, f'{corr_diff[i,j]:+.2f}', ha='center', va='center',
                        color='white' if abs(corr_diff[i,j]) > 0.3 else 'black', fontsize=8)
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_correlation_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. PCA分析
    old_pca, new_pca, explained_var, cumulative_var, loadings, pca_model = pca_analysis(
        old_data, new_data, param_names, n_components=min(len(param_names), 5)
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PC1 vs PC2 散点图
    axes[0, 0].scatter(old_pca[:, 0], old_pca[:, 1], alpha=0.5, s=20, label='Old', color='blue')
    axes[0, 0].scatter(new_pca[:, 0], new_pca[:, 1], alpha=0.8, s=50, label='New', color='red', marker='^')
    axes[0, 0].set_xlabel('PC1', fontsize=12)
    axes[0, 0].set_ylabel('PC2', fontsize=12)
    axes[0, 0].set_title('PCA: PC1 vs PC2', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 解释方差
    axes[0, 1].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, color='steelblue')
    axes[0, 1].plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Principal Component', fontsize=12)
    axes[0, 1].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0, 1].set_title('Scree Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(['Cumulative', 'Individual'], fontsize=10)
    
    # 载荷矩阵热力图
    im = axes[0, 2].imshow(loadings[:, :min(3, loadings.shape[1])], cmap='RdBu_r', aspect='auto')
    axes[0, 2].set_title('PCA Loadings', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Principal Component', fontsize=12)
    axes[0, 2].set_ylabel('Original Feature', fontsize=12)
    axes[0, 2].set_yticks(range(len(param_names)))
    axes[0, 2].set_yticklabels(param_names, fontsize=10)
    axes[0, 2].set_xticks(range(min(3, loadings.shape[1])))
    axes[0, 2].set_xticklabels([f'PC{i+1}' for i in range(min(3, loadings.shape[1]))], fontsize=10)
    plt.colorbar(im, ax=axes[0, 2])
    
    # PC1 vs PC3
    if old_pca.shape[1] > 2:
        axes[1, 0].scatter(old_pca[:, 0], old_pca[:, 2], alpha=0.5, s=20, label='Old', color='blue')
        axes[1, 0].scatter(new_pca[:, 0], new_pca[:, 2], alpha=0.8, s=50, label='New', color='red', marker='^')
        axes[1, 0].set_xlabel('PC1', fontsize=12)
        axes[1, 0].set_ylabel('PC3', fontsize=12)
        axes[1, 0].set_title('PCA: PC1 vs PC3', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    
    # PC2 vs PC3
    if old_pca.shape[1] > 2:
        axes[1, 1].scatter(old_pca[:, 1], old_pca[:, 2], alpha=0.5, s=20, label='Old', color='blue')
        axes[1, 1].scatter(new_pca[:, 1], new_pca[:, 2], alpha=0.8, s=50, label='New', color='red', marker='^')
        axes[1, 1].set_xlabel('PC2', fontsize=12)
        axes[1, 1].set_ylabel('PC3', fontsize=12)
        axes[1, 1].set_title('PCA: PC2 vs PC3', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
    
    # 载荷向量图（biplot）
    axes[1, 2].scatter(old_pca[:, 0], old_pca[:, 1], alpha=0.3, s=10, color='lightblue')
    axes[1, 2].scatter(new_pca[:, 0], new_pca[:, 1], alpha=0.6, s=30, color='lightcoral', marker='^')
    
    # 绘制载荷向量
    scale_factor = 3
    for i, param in enumerate(param_names):
        axes[1, 2].arrow(0, 0, loadings[i, 0]*scale_factor, loadings[i, 1]*scale_factor,
                        head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
        axes[1, 2].text(loadings[i, 0]*scale_factor*1.15, loadings[i, 1]*scale_factor*1.15,
                       param, fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    axes[1, 2].set_xlabel('PC1', fontsize=12)
    axes[1, 2].set_ylabel('PC2', fontsize=12)
    axes[1, 2].set_title('PCA Biplot', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 2].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_pca_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. 聚类分析
    old_labels, new_labels, old_cluster_pct, new_cluster_pct, kmeans_model = cluster_analysis(
        old_data, new_data, n_clusters=min(5, len(old_data)//10)
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 聚类分布对比
    x = np.arange(len(old_cluster_pct))
    width = 0.35
    axes[0].bar(x - width/2, old_cluster_pct, width, label='Old', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, new_cluster_pct, width, label='New', color='red', alpha=0.7)
    axes[0].set_xlabel('Cluster ID', fontsize=12)
    axes[0].set_ylabel('Percentage (%)', fontsize=12)
    axes[0].set_title('Cluster Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # PCA空间中的聚类
    scaler = StandardScaler()
    old_scaled = scaler.fit_transform(old_data)
    new_scaled = scaler.transform(new_data)
    pca_viz = PCA(n_components=2)
    old_pca_viz = pca_viz.fit_transform(old_scaled)
    new_pca_viz = pca_viz.transform(new_scaled)
    
    for cluster_id in range(len(old_cluster_pct)):
        mask_old = old_labels == cluster_id
        axes[1].scatter(old_pca_viz[mask_old, 0], old_pca_viz[mask_old, 1],
                       alpha=0.5, s=20, label=f'Old C{cluster_id}')
    
    for cluster_id in range(len(new_cluster_pct)):
        mask_new = new_labels == cluster_id
        if np.sum(mask_new) > 0:
            axes[1].scatter(new_pca_viz[mask_new, 0], new_pca_viz[mask_new, 1],
                           alpha=0.8, s=100, marker='^', edgecolors='black', linewidth=1.5,
                           label=f'New C{cluster_id}')
    
    axes[1].set_xlabel('PC1', fontsize=12)
    axes[1].set_ylabel('PC2', fontsize=12)
    axes[1].set_title('Clusters in PCA Space', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)
    
    # 簇间距离
    centers = kmeans_model.cluster_centers_
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(centers, metric='euclidean'))
    
    im = axes[2].imshow(distances, cmap='YlOrRd', aspect='auto')
    axes[2].set_title('Inter-Cluster Distances', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Cluster ID', fontsize=12)
    axes[2].set_ylabel('Cluster ID', fontsize=12)
    axes[2].set_xticks(range(len(old_cluster_pct)))
    axes[2].set_yticks(range(len(old_cluster_pct)))
    for i in range(len(old_cluster_pct)):
        for j in range(len(old_cluster_pct)):
            axes[2].text(j, i, f'{distances[i,j]:.2f}', ha='center', va='center',
                        color='white' if distances[i,j] > distances.max()/2 else 'black', fontsize=10)
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_cluster_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 5. 参数空间覆盖度
    coverage_df = parameter_space_coverage(old_data, new_data, param_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 范围对比
    x = np.arange(len(param_names))
    width = 0.35
    axes[0, 0].bar(x - width/2, coverage_df['Old_Range'], width, label='Old', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, coverage_df['New_Range'], width, label='New', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Parameter', fontsize=12)
    axes[0, 0].set_ylabel('Range', fontsize=12)
    axes[0, 0].set_title('Parameter Range Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 覆盖度百分比
    axes[0, 1].bar(x, coverage_df['Range_Coverage_%'], color='steelblue', alpha=0.7)
    axes[0, 1].axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% Coverage')
    axes[0, 1].set_xlabel('Parameter', fontsize=12)
    axes[0, 1].set_ylabel('Coverage (%)', fontsize=12)
    axes[0, 1].set_title('New Data Range Coverage of Old Data', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 采样密度对比
    axes[1, 0].bar(x - width/2, coverage_df['Old_Density'], width, label='Old', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, coverage_df['New_Density'], width, label='New', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Parameter', fontsize=12)
    axes[1, 0].set_ylabel('Density (samples/range)', fontsize=12)
    axes[1, 0].set_title('Sampling Density Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_yscale('log')
    
    # 范围可视化
    for i, param in enumerate(param_names):
        old_min = coverage_df.loc[i, 'Old_Min']
        old_max = coverage_df.loc[i, 'Old_Max']
        new_min = coverage_df.loc[i, 'New_Min']
        new_max = coverage_df.loc[i, 'New_Max']
        
        axes[1, 1].plot([old_min, old_max], [i-0.15, i-0.15], 'b-', linewidth=8, alpha=0.7, label='Old' if i==0 else '')
        axes[1, 1].plot([new_min, new_max], [i+0.15, i+0.15], 'r-', linewidth=8, alpha=0.7, label='New' if i==0 else '')
        
        # 标记重叠区域
        overlap_min = max(old_min, new_min)
        overlap_max = min(old_max, new_max)
        if overlap_min < overlap_max:
            axes[1, 1].plot([overlap_min, overlap_max], [i, i], 'g-', linewidth=4, alpha=0.9,
                           label='Overlap' if i==0 else '')
    
    axes[1, 1].set_yticks(range(len(param_names)))
    axes[1, 1].set_yticklabels(param_names, fontsize=10)
    axes[1, 1].set_xlabel('Parameter Value', fontsize=12)
    axes[1, 1].set_title('Parameter Range Overlap', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_coverage_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Generated 5 comprehensive analysis plots")


# ============ 主函数 ============

def main():
    print("="*70)
    print("科学数据分析工具：新旧数据深度对比")
    print("="*70)
    
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # 加载数据
    print("\n[1/5] Loading data...")
    old_data, old_names = load_data(Config.old_excel, Config.old_sheet, Config.key_alias)
    new_data, new_names = load_data(Config.new_excel, Config.new_sheet, Config.key_alias)
    
    if old_data is None or new_data is None:
        print("  ✗ Failed to load data")
        return
    
    print(f"  ✓ Old data: {old_data.shape[0]} samples × {old_data.shape[1]} parameters")
    print(f"  ✓ New data: {new_data.shape[0]} samples × {new_data.shape[1]} parameters")
    
    # 确保参数名一致
    if old_names != new_names:
        common_names = list(set(old_names) & set(new_names))
        print(f"  ⚠ Warning: Using {len(common_names)} common parameters")
        param_names = common_names
    else:
        param_names = old_names
    
    # 统计分析
    print("\n[2/5] Performing statistical analysis...")
    stats_df = descriptive_statistics(old_data, new_data, param_names)
    stats_df.to_excel(os.path.join(Config.output_dir, 'statistical_analysis.xlsx'), index=False)
    print("  ✓ Statistical analysis completed")
    
    # 参数空间覆盖度
    print("\n[3/5] Analyzing parameter space coverage...")
    coverage_df = parameter_space_coverage(old_data, new_data, param_names)
    coverage_df.to_excel(os.path.join(Config.output_dir, 'parameter_coverage.xlsx'), index=False)
    print("  ✓ Coverage analysis completed")
    
    # 样本充足性
    print("\n[4/5] Assessing sample sufficiency...")
    sufficiency = sample_sufficiency_analysis(old_data, new_data, param_names)
    
    sufficiency_df = pd.DataFrame({
        'Metric': ['Old N', 'New N', 'N Parameters', 'Recommended Min', 'Estimated N for Stability'],
        'Value': [sufficiency['old_n'], sufficiency['new_n'], sufficiency['n_params'],
                 sufficiency['recommended_min'], sufficiency['estimated_n_for_stability']]
    })
    sufficiency_df.to_excel(os.path.join(Config.output_dir, 'sample_sufficiency.xlsx'), index=False)
    print("  ✓ Sufficiency analysis completed")
    
    # 可视化
    print("\n[5/5] Generating comprehensive visualizations...")
    plot_comprehensive_analysis(old_data, new_data, param_names, Config.output_dir)
    
    # 生成总结报告
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Old dataset: {old_data.shape[0]} samples, {old_data.shape[1]} parameters")
    print(f"  New dataset: {new_data.shape[0]} samples, {new_data.shape[1]} parameters")
    
    print(f"\n🔍 Statistical Tests:")
    sig_params = stats_df[stats_df['p_value_ks'] < 0.05]['Parameter'].tolist()
    if sig_params:
        print(f"  ⚠ Significant distribution differences (p<0.05): {', '.join(sig_params)}")
    else:
        print(f"  ✓ No significant distribution differences detected")
    
    print(f"\n📏 Parameter Space Coverage:")
    low_coverage = coverage_df[coverage_df['Range_Coverage_%'] < 30]['Parameter'].tolist()
    if low_coverage:
        print(f"  🔴 Low coverage (<30%): {', '.join(low_coverage)}")
    else:
        print(f"  ✓ Adequate coverage for all parameters")
    
    print(f"\n📈 Sample Sufficiency:")
    print(f"  Current new samples: {sufficiency['new_n']}")
    print(f"  Recommended minimum: {sufficiency['recommended_min']}")
    print(f"  Estimated for stability: {sufficiency['estimated_n_for_stability']}")
    
    if sufficiency['new_n'] < sufficiency['recommended_min']:
        print(f"  🔴 INSUFFICIENT: Need {sufficiency['recommended_min'] - sufficiency['new_n']} more samples")
    elif sufficiency['new_n'] < sufficiency['estimated_n_for_stability']:
        print(f"  🟡 MARGINAL: {sufficiency['estimated_n_for_stability'] - sufficiency['new_n']} more samples recommended")
    else:
        print(f"  ✓ SUFFICIENT for current purposes")
    
    print(f"\n📁 Results saved to: {Config.output_dir}")
    print(f"  - statistical_analysis.xlsx (detailed statistics)")
    print(f"  - parameter_coverage.xlsx (coverage metrics)")
    print(f"  - sample_sufficiency.xlsx (sample size assessment)")
    print(f"  - 01_distribution_analysis.png")
    print(f"  - 02_correlation_analysis.png")
    print(f"  - 03_pca_analysis.png")
    print(f"  - 04_cluster_analysis.png")
    print(f"  - 05_coverage_analysis.png")
    
    print("\n" + "="*70)
    print("✅ Analysis completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
