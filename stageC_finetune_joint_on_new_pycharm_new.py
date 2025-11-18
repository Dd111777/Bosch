# -*- coding: utf-8 -*-

import os
import copy
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold

# ====== 依赖 util/模型（保持不变） ======
from physio_util import (
    set_seed, export_predictions_longtable, export_metrics_grid,
    write_summary_txt, heatmap, parity_scatter, residual_hist, save_manifest,
    metrics, transform_for_display, FAMILIES,
    excel_to_physics_dataset,
    load_new_excel_as_sparse_morph, build_sparse_batch
)
from phys_model import TemporalRegressor, PhysicsSeqPredictor


# ========================== 配置（30样本优化版） ==========================
class Cfg:
    # ==================== 数据路径（不变） ====================
    old_excel = r"D:\data\pycharm\bosch\case.xlsx"
    new_excel = r"D:\data\pycharm\bosch\Bosch.xlsx"
    save_root = "./runs_stageC_performance"  # 性能优化版

    # 预训练权重
    phys_ckpt_F = "./runs_phys_split/F_Flux/phys_best.pth"
    phys_ckpt_I = "./runs_phys_split/Ion_Flux/phys_best.pth"
    morph_ckpt = "./runs_morph_old/morph_best_overall.pth"

    test_size = 0.10  # ⬇️ 从0.15降到0.10 (3样本)
    val_size = 0.10  # ⬇️ 从0.15降到0.10 (3样本)
    split_random_state = 42

    # ==================== 训练参数（⬇️ 保守策略） ====================
    seed = 42
    seeds = [42, 43, 44]  # ✅ 恢复3 seeds
    max_epochs = 3000  # ⬆️ 增加到3000轮

    # 学习率（更激进）
    lr_morph = 3e-4  # ⬆️ 从1e-4提高到3e-4
    lr_phys = 1e-40  # ✅ 保持冻结
    lr_calib = 3e-4  # ⬆️ 从1e-4提高到3e-4

    # 权重衰减（适度正则化）
    wd_morph = 5e-3  # ⬆️ 从1e-3提高到5e-3
    wd_phys = 1e-2  # ✅ 保持
    wd_calib = 1e-5  # ⬇️ 从1e-4降到1e-5

    # 梯度裁剪
    batch_clip = 1.0  # ⬆️ 从0.5恢复到1.0

    # Batch size
    batch_size = 8
    use_full_batch = True

    dropout_morph = 0.3  # ⬇️ 从0.5降到0.3
    dropout_calib = 0.05  # ⬇️ 从0.1降到0.05

    loss_delta = 1.0
    loss_smooth_weight = 5e-3  # ⬆️ 从1e-3提高到5e-3
    mono_zmin_weight = 3e-3  # ⬆️ 从1e-3提高到3e-3

    # ==================== Ion相关 ====================
    ion_affine_default = {"a": 1.0, "b": 0.0, "c": 0.0}
    ion_learnable_lr = 5e-5  # ⬆️ 从1e-6恢复到5e-5
    ion_learnable_wd = 1e-6  # ⬇️ 从1e-5恢复到1e-6

    # ==================== 展示空间（不变） ====================
    unit_scale = 1
    flip_sign = False
    clip_nonneg = False
    min_display_value = 0.0
    family_sign = np.array([-1, +1, +1, +1, +1, +1], dtype=np.float32)
    sheet_name = "case"
    HEIGHT_FAMILY = "h1"

    use_smape = True
    mape_eps_nm = 0.001

    use_ema = True
    ema_decay = 0.995  # ⬆️ 从0.99提高到0.995

    early_stop = True
    early_stop_patience = 100  # ⬆️ 从50提高到100

    use_scheduler = True
    scheduler_type = "cosine_warmup"
    warmup_epochs = 50  # ⬆️ 从20增加到50
    min_lr_ratio = 0.001  # ⬇️ 从0.01降到0.001

    use_augmentation = True
    aug_noise_std = 0.005  # ⬇️ 从0.01降到0.005
    aug_time_jitter = False

    # ==================== 后校准 ====================
    calib_min_points = 5  # ⬇️ 从12降到5（样本少）
    calib_ridge = 1e-4  # Ridge正则化

    # ==================== 模型冻结策略（✅ 关键新增） ====================
    # 这些参数会在变体中覆盖
    default_freeze_phys = True  # 默认冻结物理网
    default_freeze_morph_encoder = True  # ✅ 默认冻结形貌encoder
    default_freeze_morph_heads = False  # 默认微调形貌heads

    variants = [
        # ===== 基础组 =====
        dict(
            name="baseline_calib_only",
            description="Only calibration head",
            use_adapter=False, use_derived=False, ion_gate="use",
            per_family_head=False, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=False,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=True,
            stagewise_unfreeze=False,
            expected_trainable_params=100,
        ),
        dict(
            name="baseline_with_heads",
            description="Calibration + morph heads",
            use_adapter=False, use_derived=False, ion_gate="use",
            per_family_head=False, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=False,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=1500,
        ),

        # ===== Adapter系列 =====
        dict(
            name="adapter_light",
            description="Lightweight adapter",
            use_adapter=True, use_derived=False, ion_gate="use",
            per_family_head=False, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=False,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=2000,
        ),
        dict(
            name="adapter_derived",
            description="Adapter + derived features",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=False, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=3000,
        ),

        # ===== Per-Family Head系列 =====
        dict(
            name="adapter_head",
            description="Adapter + per-family head",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=5000,
        ),
        dict(
            name="adapter_head_hetero",
            description="Adapter + per-family head + heteroscedastic",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=True, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=6000,
        ),
        dict(
            name="adapter_head_hetero_taskuncert",
            description="Full bells and whistles",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=True, task_uncertainty=True,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=7000,
        ),

        # ===== Ion Gate系列 =====
        dict(
            name="adapter_head_iongate",
            description="Ion gating mechanism",
            use_adapter=True, use_derived=True, ion_gate="gate",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=6000,
        ),
        dict(
            name="adapter_head_iongate_hetero",
            description="Ion gating + heteroscedastic + task uncertainty",
            use_adapter=True, use_derived=True, ion_gate="gate",
            per_family_head=True, hetero=True, task_uncertainty=True,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=7500,
        ),

        # ===== 后校准策略对比 =====
        dict(
            name="adapter_head_postcalib_kt",
            description="Per-kt posterior calibration",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_kt",
            learnable_ion=True,
            freeze_phys=True, freeze_morph_encoder=True, freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=5000,
        ),

        # ===== 解冻encoder（激进方案） =====
        dict(
            name="adapter_head_encoder",
            description="Unfreeze morph encoder (risky)",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, freeze_morph_encoder=False,  # 解冻！
            freeze_morph_heads=False,
            stagewise_unfreeze=False,
            expected_trainable_params=20000,
        ),

        # ===== 渐进式解冻（新增） =====
        dict(
            name="adapter_head_stagewise",
            description="Stagewise unfreezing strategy",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, freeze_morph_encoder=True,
            freeze_morph_heads=False,
            stagewise_unfreeze=True,  # 启用渐进解冻
            expected_trainable_params=5000,
        ),
    ]

    # ==================== 其他高级选项（可选） ====================
    # Ion门控策略ablation
    ion_gate_variants = ["use", "zero", "const", "smooth", "gate"]

    # 是否进行快速诊断检查
    quick_check = False

    # 日志和保存
    save_interval = 50  # 每50轮保存一次
    log_interval = 10  # 每10轮打印一次
    save_best_only = True  # 只保存最佳模型

    # 可视化
    plot_diagnostics = True  # 生成逐family诊断图
    plot_temporal_error = True  # 时序误差分析

# ========================== 辅助函数 ==========================
def print_config_summary():
    """打印配置摘要"""
    print("=" * 80)
    print("StageC Configuration Summary (Performance Optimized)")
    print("=" * 80)

    print("Data Split:")
    print(f"  - Train: {100 - (Cfg.test_size + Cfg.val_size) * 100:.0f}%")
    print(f"  - Val:   {Cfg.val_size * 100:.0f}%")
    print(f"  - Test:  {Cfg.test_size * 100:.0f}%")

    print("\nTraining:")
    print(f"  - Max Epochs: {Cfg.max_epochs}")
    print(f"  - Seeds: {Cfg.seeds}")
    print(f"  - LR Morph: {Cfg.lr_morph}")
    print(f"  - WD Morph: {Cfg.wd_morph}")
    print(f"  - Dropout: {Cfg.dropout_morph}")

    print("\nRegularization:")
    print(f"  - EMA Decay: {Cfg.ema_decay}")
    print(f"  - Early Stop Patience: {Cfg.early_stop_patience}")
    print(f"  - Data Augmentation: {Cfg.use_augmentation}")

    print(f"\nVariants: {len(Cfg.variants)}")
    for i, v in enumerate(Cfg.variants, 1):
        print(f"  {i}. {v['name']}: {v.get('description', 'N/A')}")
        # 安全获取可选字段
        r2_range = v.get('expected_r2_range', 'N/A')
        params = v.get('expected_trainable_params', 'N/A')
        if isinstance(params, int):
            print(f"     Expected R²: {r2_range}, Params: ~{params:,}")
        else:
            print(f"     Expected R²: {r2_range}, Params: {params}")

    print("=" * 80)
    print()

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def _norm_col(x, mean, std):
    """列归一化"""
    # 转换Tensor为numpy
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()

    return (x - mean) / (std + 1e-8)


def load_fallback_sparse(merged_excel_path, norm_mean, norm_std, time_values, families):
    if merged_excel_path.endswith('.csv'):
        df = pd.read_csv(merged_excel_path)
    else:
        df = pd.read_excel(merged_excel_path)

    # === 以下逻辑100%保留 ===
    s8_cols = [c for c in df.columns if str(c).lower().startswith("s") and str(c)[1:].isdigit()]
    s8_cols = sorted(s8_cols, key=lambda s: int(str(s)[1:]))[:7]
    s8_raw = df[s8_cols].astype(float).values
    s8 = _norm_col(s8_raw, norm_mean, norm_std).astype(np.float32)

    fams = list(families)
    if "zmin" not in fams:
        fams = ["zmin"] + fams
    T = len(time_values)
    fam2idx = {n: i for i, n in enumerate(fams)}
    B, K = s8.shape[0], len(fams)

    y_sparse = torch.zeros((B, K, T), dtype=torch.float32)
    m_sparse = torch.zeros((B, K, T), dtype=torch.bool)
    tv = np.array(time_values, dtype=float).tolist()

    def t_idx(tnum):
        return int(tv.index(float(tnum))) if float(tnum) in tv else max(0, min(T - 1, int(tnum) - 1))

    def _nm_to_um(vals):
        return np.nan_to_num(vals.astype(float), nan=np.nan) / 1000.0

    def _pick(df, cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    def _fill(k_idx, cands, times, negate=False):
        for n in times:
            col = _pick(df, cands(n))
            if col is None:
                continue
            vals = _nm_to_um(df[col].values)
            if negate:
                vals = -vals
            ti = t_idx(n)
            v_t = torch.tensor(vals, dtype=torch.float32)
            ok = torch.isfinite(v_t)
            if ok.any():
                y_sparse[:, k_idx, ti][ok] = v_t[ok]
                m_sparse[:, k_idx, ti][ok] = True

    # zmin
    if "zmin" in fams and "zmin_10" in df.columns:
        k = fam2idx["zmin"]
        vals = _nm_to_um(df["zmin_10"].values) * (-1.0)
        ti = t_idx(10)
        v_t = torch.tensor(vals, dtype=torch.float32)
        ok = torch.isfinite(v_t)
        if ok.any():
            y_sparse[:, k, ti][ok] = v_t[ok]
            m_sparse[:, k, ti][ok] = True

    # h1, d1, w
    def _cands_h(n):
        return [f"h1_{n}", f"h1{n}", f"h{n}", f"h_{n}", f"{n}thscallopheight"]

    def _cands_d(n):
        return [f"d1_{n}", f"d1{n}", f"d{n}", f"d_{n}", f"{n}thscallopdepth"]

    def _cands_w(n):
        return [f"w{n}", f"w_{n}", f"W{n}", f"W {n}", f"{n}thscallopwidth"]

    if "h1" in fams:
        _fill(fam2idx["h1"], _cands_h, [3, 5, 9], negate=False)
    if "d1" in fams:
        _fill(fam2idx["d1"], _cands_d, [3, 5, 9], negate=False)
    if "w" in fams:
        _fill(fam2idx["w"], _cands_w, [1, 3, 5, 9], negate=False)

    s8 = torch.tensor(s8, dtype=torch.float32)
    tvals = torch.tensor(np.array(time_values, dtype=np.float32), dtype=torch.float32)

    # 返回sample_ids用于划分
    sample_ids = np.arange(B)
    return s8, y_sparse, m_sparse, tvals, fams, sample_ids


def load_data_with_split(new_excel, norm_mean, norm_std, time_values, families,
                         test_size, val_size, seed):
    # 1. 加载数据（对齐逻辑保持不变）
    try:
        from physio_util import load_new_excel_as_sparse_morph, build_sparse_batch
        recs = load_new_excel_as_sparse_morph(new_excel, height_family=Cfg.HEIGHT_FAMILY)
        s8, y_sparse, m_sparse, tvals = build_sparse_batch(recs, norm_mean, norm_std, time_values)
        fams = families
        sample_ids = np.arange(s8.shape[0])
        print("[DataSplit] Loaded via util loader.")
    except Exception as e:
        print(f"[DataSplit] Fallback loader: {e}")
        s8, y_sparse, m_sparse, tvals, fams, sample_ids = load_fallback_sparse(
            new_excel, norm_mean, norm_std, time_values, families
        )

    # 2. 数据划分（新增）
    B = s8.shape[0]

    # 分层依据：每个样本的有效点数
    sample_counts = m_sparse.sum(dim=(1, 2)).numpy()
    strata = (sample_counts > np.median(sample_counts)).astype(int)

    # Train/Temp split
    train_idx, temp_idx = train_test_split(
        sample_ids, test_size=(test_size + val_size),
        random_state=seed, stratify=strata
    )

    # Val/Test split
    temp_strata = strata[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(test_size / (test_size + val_size)),
        random_state=seed, stratify=temp_strata
    )

    print(f"[DataSplit] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # 3. 重新计算统计量（仅用训练集）
    s8_train = s8[train_idx]
    # 反归一化到原始域
    s8_train_raw = s8_train * (norm_std + 1e-8) + norm_mean
    train_mean = s8_train_raw.mean(dim=0)
    train_std = s8_train_raw.std(dim=0).clamp_min(1e-8)

    # 全量数据用训练集统计量重新标准化
    s8_full_raw = s8 * (norm_std + 1e-8) + norm_mean
    s8_renorm = (s8_full_raw - train_mean) / train_std

    return {
        "s8": s8_renorm,
        "s8_raw": s8_full_raw,
        "orig_norm_mean": norm_mean,
        "orig_norm_std": norm_std,

        "y_sparse": y_sparse,
        "m_sparse": m_sparse,
        "tvals": tvals,
        "families": fams,
        "splits": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx
        },
        "train_stats": {
            "mean": train_mean,
            "std": train_std
        }
    }


# ========================== 数据统计和检查 ==========================
def print_data_statistics(data_dict):
    """打印数据集统计信息"""
    s8 = data_dict["s8"]
    y_sparse = data_dict["y_sparse"]
    m_sparse = data_dict["m_sparse"]
    fams = data_dict["families"]
    splits = data_dict["splits"]

    print("\n" + "=" * 80)
    print("Data Statistics")
    print("=" * 80)

    # 样本数
    B, K, T = y_sparse.shape
    print(f"Total samples: {B}")
    print(f"Families: {K} ({', '.join(fams)})")
    print(f"Time steps: {T}")

    # 划分
    print(f"\nData Split:")
    print(f"  Train: {len(splits['train'])} samples ({len(splits['train']) / B * 100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} samples ({len(splits['val']) / B * 100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} samples ({len(splits['test']) / B * 100:.1f}%)")

    # 逐Family有效点数
    print(f"\nValid points per family:")
    for k, fam in enumerate(fams):
        n_valid = m_sparse[:, k, :].sum().item()
        pct = n_valid / (B * T) * 100
        print(f"  {fam:8s}: {int(n_valid):4d} / {B * T:4d} ({pct:5.1f}%)")

    # 逐split有效点数
    print(f"\nValid points per split:")
    for split_name, idx in splits.items():
        n_valid = m_sparse[idx].sum().item()
        total = len(idx) * K * T
        pct = n_valid / total * 100 if total > 0 else 0
        print(f"  {split_name:5s}: {int(n_valid):5d} / {total:5d} ({pct:5.1f}%)")

    # 静态特征统计
    print(f"\nStatic features (s8) - train set:")
    s8_train = s8[splits["train"]]
    print(f"  Mean: {s8_train.mean(dim=0).numpy()}")
    print(f"  Std:  {s8_train.std(dim=0).numpy()}")

    print("=" * 80 + "\n")


def check_data_quality(data_dict):
    """检查数据质量并给出警告"""
    s8 = data_dict["s8"]
    y_sparse = data_dict["y_sparse"]
    m_sparse = data_dict["m_sparse"]
    splits = data_dict["splits"]

    warnings = []

    # 检查1: 训练集样本数
    n_train = len(splits["train"])
    if n_train < 20:
        warnings.append(f"⚠️ Train samples ({n_train}) < 20, high overfitting risk!")
    elif n_train < 30:
        warnings.append(f"⚠️ Train samples ({n_train}) < 30, consider increasing")

    # 检查2: 验证/测试集样本数
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    if n_val < 3 or n_test < 3:
        warnings.append(f"⚠️ Val ({n_val}) or Test ({n_test}) < 3, unreliable evaluation")

    # 检查3: 有效点分布
    for split_name, idx in splits.items():
        if len(idx) == 0:
            continue
        m_split = m_sparse[idx]
        valid_per_sample = m_split.sum(dim=(1, 2)).float().numpy()
        if valid_per_sample.min() < 3:
            warnings.append(f"⚠️ {split_name} set has samples with <3 valid points")

    # 检查4: NaN/Inf
    if torch.isnan(s8).any() or torch.isinf(s8).any():
        warnings.append(f"⚠️ s8 contains NaN or Inf")
    if torch.isnan(y_sparse[m_sparse]).any():
        warnings.append(f"⚠️ y_sparse contains NaN in valid positions")

    # 打印警告
    if warnings:
        print("\n" + "=" * 80)
        print("DATA QUALITY WARNINGS")
        print("=" * 80)
        for w in warnings:
            print(w)
        print("=" * 80 + "\n")
    else:
        print("[✓] Data quality check passed\n")

    return len(warnings) == 0


# ========================== 主加载流程 ==========================
def load_all_data(meta_old):
    print("\n" + "=" * 80)
    print("Loading Data for Stage C (30-Sample Optimized)")
    print("=" * 80)

    # 提取旧表统计量
    norm_mean = meta_old.get("norm_mean", torch.zeros(7))
    norm_std = meta_old.get("norm_std", torch.ones(7))
    time_values = meta_old.get("time_values", list(range(10)))
    families = meta_old.get("families", FAMILIES)

    if isinstance(norm_mean, dict):
        norm_mean = torch.tensor([norm_mean.get(f"s{i + 1}", 0.0) for i in range(7)])
    if isinstance(norm_std, dict):
        norm_std = torch.tensor([norm_std.get(f"s{i + 1}", 1.0) for i in range(7)])

    # 加载新表数据（带划分）
    data_dict = load_data_with_split(
        Cfg.new_excel,
        norm_mean, norm_std,
        time_values, families,
        Cfg.test_size, Cfg.val_size,
        Cfg.split_random_state
    )

    # 打印统计
    print_data_statistics(data_dict)

    # 质量检查
    check_data_quality(data_dict)

    return data_dict
# ========================== AUTO-SWEEP HELPERS（新增） ==========================
def _masked_data_only_family(data_dict: Dict, fam_index: int) -> Dict:
    """只保留第 fam_index 个 family 的有效掩码，其它 family 清零；不改动 y_sparse 本体。"""
    dd = copy.deepcopy(data_dict)
    m = dd["m_sparse"].clone()
    keep = torch.zeros_like(m)
    keep[:, fam_index:fam_index+1, :] = m[:, fam_index:fam_index+1, :]
    dd["m_sparse"] = keep
    return dd


def _decorate_variant_name(variant: Dict, suffix: str) -> Dict:
    """浅拷贝 variant，并在 name 后追加后缀，避免覆盖同名结果目录。"""
    v = dict(variant)
    v["name"] = f"{variant['name']}_{suffix}"
    return v


class CalibAffinePerChannel(nn.Module):
    """逐通道仿射校准"""

    def __init__(self, K, init_alpha=1.0, init_beta=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((K,), float(init_alpha)))
        self.beta = nn.Parameter(torch.full((K,), float(init_beta)))

    def forward(self, y):
        return self.alpha.view(1, -1, 1) * y + self.beta.view(1, -1, 1)


class CalibTimeConv(nn.Module):
    """时间卷积校准"""

    def __init__(self, K, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(K, K, kernel_size, padding=padding, groups=K, bias=True)
        with torch.no_grad():
            self.dw.weight.zero_()
            center = padding
            for k in range(K):
                self.dw.weight[k, 0, center] = 1.0
            self.dw.bias.zero_()

    def forward(self, y):
        return self.dw(y)


class CalibHybrid(nn.Module):
    """混合校准"""

    def __init__(self, K, kernel_size=3):
        super().__init__()
        self.aff = CalibAffinePerChannel(K)
        self.tcv = CalibTimeConv(K, kernel_size=kernel_size)

    def forward(self, y):
        return self.tcv(self.aff(y))


def build_calib_head(kind: str, K: int) -> nn.Module:
    """构建校准头"""
    if kind == "affine_per_channel":
        return CalibAffinePerChannel(K)
    elif kind == "time_conv":
        return CalibTimeConv(K)
    elif kind == "hybrid":
        return CalibHybrid(K)
    else:
        raise ValueError(f"Unknown calib_head: {kind}")


# ========================== 接口增强模块（保留原始实现） ==========================
class PhysAdapter(nn.Module):
    """深度可分离卷积适配器"""

    def __init__(self, in_ch=2, k=3):
        super().__init__()
        pad = (k - 1) // 2
        self.dw = nn.Conv1d(in_ch, in_ch, k, padding=pad, groups=in_ch, bias=True)
        self.pw = nn.Conv1d(in_ch, in_ch, 1, bias=True)
        with torch.no_grad():
            self.dw.weight.zero_()
            self.dw.bias.zero_()
            for c in range(in_ch):
                self.dw.weight[c, 0, pad] = 1.0
            self.pw.weight.zero_()
            for c in range(in_ch):
                self.pw.weight[c, c, 0] = 1.0
            self.pw.bias.zero_()

    def forward(self, x):
        return self.pw(self.dw(x))


class PhysFeaReducer(nn.Module):
    """派生特征提取器"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.pw = nn.Conv1d(5, 2, 1, bias=True)
        nn.init.zeros_(self.pw.weight)
        with torch.no_grad():
            self.pw.weight[0, 0, 0] = 1.0
            self.pw.weight[1, 1, 0] = 1.0
        nn.init.zeros_(self.pw.bias)

    def forward(self, F, I):
        logI = torch.log(I.clamp_min(self.eps))
        dF = F[:, :, 1:] - F[:, :, :-1]
        dI = I[:, :, 1:] - I[:, :, :-1]
        dF = torch.nn.functional.pad(dF, (1, 0), mode="replicate")
        dI = torch.nn.functional.pad(dI, (1, 0), mode="replicate")
        x = torch.cat([F, I, logI, dF, dI], dim=1)
        return self.pw(x)


class IonGate(nn.Module):
    """Ion门控机制"""

    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(2, 4, 1), nn.ReLU(inplace=True),
            nn.Conv1d(4, 1, 1), nn.Sigmoid()
        )

    def forward(self, F, I):
        pad = (self.k - 1) // 2
        w = torch.ones(1, 1, self.k, device=I.device) / self.k
        I_s = nn.functional.conv1d(nn.functional.pad(I, (pad, pad), mode="replicate"), w)
        gate = self.mlp(torch.cat([F, I], dim=1))
        return gate * I + (1.0 - gate) * I_s


class PerFamilyHead(nn.Module):
    """逐Family输出头"""

    def __init__(self, K, k=3):
        super().__init__()
        pad = (k - 1) // 2
        self.dw = nn.Conv1d(K, K, k, padding=pad, groups=K, bias=True)
        self.alpha = nn.Parameter(torch.ones(K))
        self.beta = nn.Parameter(torch.zeros(K))
        with torch.no_grad():
            self.dw.weight.zero_()
            for c in range(K):
                self.dw.weight[c, 0, pad] = 1.0
            self.dw.bias.zero_()

    def forward(self, y):
        y = self.dw(y)
        return self.alpha.view(1, -1, 1) * y + self.beta.view(1, -1, 1)


class HeteroHead(nn.Module):
    """异方差输出头（预测均值和方差）"""

    def __init__(self, K, per_family=True):
        super().__init__()
        self.per_family = per_family
        self.mu_head = PerFamilyHead(K) if per_family else nn.Identity()
        self.logv_head = nn.Sequential(
            nn.Conv1d(K, K, 1, groups=1, bias=True),
        )
        nn.init.zeros_(self.logv_head[0].weight)
        nn.init.constant_(self.logv_head[0].bias, math.log(1.0))

    def forward(self, y):
        mu = self.mu_head(y) if self.per_family else y
        logv = self.logv_head(mu.detach())
        return mu, logv


# ========================== Ion反变换 ==========================
class IonInverseTransform(nn.Module):
    """Ion反变换"""

    def __init__(self, init_abc: Dict[str, float], learnable: bool = False):
        super().__init__()
        a, b, c = init_abc.get("a", 1.0), init_abc.get("b", 0.0), init_abc.get("c", 0.0)
        if learnable:
            self.a = nn.Parameter(torch.tensor(float(a)))
            self.b = nn.Parameter(torch.tensor(float(b)))
            self.c = nn.Parameter(torch.tensor(float(c)))
        else:
            self.register_buffer("a", torch.tensor(float(a)), persistent=False)
            self.register_buffer("b", torch.tensor(float(b)), persistent=False)
            self.register_buffer("c", torch.tensor(float(c)), persistent=False)
        self.learnable = learnable

    def forward(self, z):
        arg = torch.clamp(self.a * z + self.b, min=-40.0, max=40.0)
        y = torch.exp(arg) - self.c
        y = torch.clamp(y, min=0.0, max=1e6)
        return y

    def reg(self):
        if not self.learnable:
            return 0.0
        return ((self.a - 1.0) ** 2 + (self.b ** 2) + (self.c ** 2))


# ========================== 模型冻结策略（✅ 关键新增） ==========================
def freeze_model_parts(model: nn.Module, freeze_encoder=True, freeze_heads=False, verbose=True):
    # 先解冻所有
    for param in model.parameters():
        param.requires_grad = True

    frozen_params = []

    # 冻结encoder
    if freeze_encoder:
        for name, param in model.named_parameters():
            if any(k in name for k in ['encoder', 'proj_in', 'pos']):
                param.requires_grad = False
                frozen_params.append(name)

    # 冻结heads
    if freeze_heads:
        for name, param in model.named_parameters():
            if 'heads' in name:
                param.requires_grad = False
                frozen_params.append(name)

    # 统计参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"  [Freeze] Frozen {len(frozen_params)} parameter groups")
        if len(frozen_params) > 0 and len(frozen_params) < 10:
            for name in frozen_params:
                print(f"    - {name}")
        print(f"  [Params] Trainable: {trainable:,} / Total: {total:,} ({100 * trainable / total:.1f}%)")

    return trainable


def freeze_physics_models(phys_F, phys_I, verbose=True):
    for param in phys_F.parameters():
        param.requires_grad = False
    for param in phys_I.parameters():
        param.requires_grad = False

    if verbose:
        total_F = sum(p.numel() for p in phys_F.parameters())
        total_I = sum(p.numel() for p in phys_I.parameters())
        print(f"  [Freeze] Physics models frozen: F={total_F:,}, I={total_I:,}")


def print_trainable_params(model_dict: Dict[str, nn.Module]):
    print("\n" + "=" * 80)
    print("Trainable Parameters Summary")
    print("=" * 80)

    total_trainable = 0
    total_all = 0

    for name, model in model_dict.items():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        total_trainable += trainable
        total_all += total

        status = "✓ Training" if trainable > 0 else "✗ Frozen"
        print(f"{name:20s}: {trainable:8,} / {total:8,} ({100 * trainable / max(total, 1):.1f}%) {status}")

    print("-" * 80)
    print(f"{'TOTAL':20s}: {total_trainable:8,} / {total_all:8,} ({100 * total_trainable / max(total_all, 1):.1f}%)")
    print("=" * 80 + "\n")


# ========================== EMA & Early Stopping ==========================
class EMA:
    def __init__(self, model: nn.Module, decay=0.99):
        self.decay = decay
        self.shadow = {}
        self.backup = None
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        # 备份一次
        if self.backup is None:
            self.backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self.backup is None:
            return
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = None

class EarlyStopper:
    """早停机制"""

    def __init__(self, patience=50):
        self.best = None
        self.patience = patience
        self.count = 0

    def step(self, val):
        if self.best is None or val < self.best:
            self.best = val
            self.count = 0
            return True
        else:
            self.count += 1
            return self.count < self.patience

    def is_improved(self, val):
        """检查是否改进"""
        return self.best is None or val < self.best


# ========================== 后校准（Post-Calibration） ==========================
class CalibrationParams:
    """校准参数容器（可序列化）"""

    def __init__(self, method: str):
        self.method = method
        self.params = {}

    def to_dict(self):
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d):
        obj = cls(d["method"])
        obj.params = d["params"]
        return obj


def fit_calibration_params(y_pred_disp, y_true_disp, mask, method="per_k",
                           min_points=5, ridge=1e-4):
    yp = y_pred_disp.detach().cpu().numpy()
    yt = y_true_disp.detach().cpu().numpy()
    mk = mask.detach().cpu().numpy().astype(bool)
    B, K, T = yp.shape

    calib = CalibrationParams(method)

    if method == "per_k":
        for k in range(K):
            xs = yp[:, k, :][mk[:, k, :]]
            ys = yt[:, k, :][mk[:, k, :]]
            good = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[good], ys[good]

            if xs.size < min_points:
                calib.params[f"k{k}"] = {"a": 1.0, "b": 0.0}
                continue

            # 线性回归: y_true = a * y_pred + b
            X = np.column_stack([xs, np.ones_like(xs)])
            try:
                coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
                a, b = float(coef[0]), float(coef[1])
            except Exception:
                XTX = X.T @ X + ridge * np.eye(2, dtype=X.dtype)
                XTy = X.T @ ys
                coef = np.linalg.solve(XTX, XTy)
                a, b = float(coef[0]), float(coef[1])

            calib.params[f"k{k}"] = {"a": a, "b": b}

    elif method == "per_kt":
        for k in range(K):
            for t in range(T):
                m = mk[:, k, t]
                xs = yp[m, k, t]
                ys = yt[m, k, t]
                good = np.isfinite(xs) & np.isfinite(ys)
                xs, ys = xs[good], ys[good]

                if xs.size < max(3, 2):  # 降低到3
                    calib.params[f"k{k}_t{t}"] = {"a": 1.0, "b": 0.0}
                    continue

                X = np.column_stack([xs, np.ones_like(xs)])
                try:
                    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
                    a, b = float(coef[0]), float(coef[1])
                except Exception:
                    XTX = X.T @ X + ridge * np.eye(2, dtype=X.dtype)
                    XTy = X.T @ ys
                    coef = np.linalg.solve(XTX, XTy)
                    a, b = float(coef[0]), float(coef[1])

                calib.params[f"k{k}_t{t}"] = {"a": a, "b": b}

    return calib


def apply_calibration_params(y_pred_disp, calib: CalibrationParams):
    yp = y_pred_disp.clone()
    B, K, T = yp.shape

    if calib.method == "per_k":
        for k in range(K):
            params = calib.params.get(f"k{k}", {"a": 1.0, "b": 0.0})
            yp[:, k, :] = params["a"] * yp[:, k, :] + params["b"]

    elif calib.method == "per_kt":
        for k in range(K):
            for t in range(T):
                params = calib.params.get(f"k{k}_t{t}", {"a": 1.0, "b": 0.0})
                yp[:, k, t] = params["a"] * yp[:, k, t] + params["b"]

    return yp


# ========================== 物理模型前向传播 ==========================
def phys_forward_raw(static_8, tvals, phys_F, phys_I, ion_transform, allow_grad=False):
    phys_F.eval()
    phys_I.eval()
    with torch.set_grad_enabled(allow_grad):
        f = phys_F(static_8, tvals)
        i_z = phys_I(static_8, tvals)
        f_ch = f[:, 0:1, :]
        z_ch = i_z[:, 1:2, :] if i_z.size(1) >= 2 else i_z[:, 0:1, :]
        i = ion_transform(z_ch)
        phys = torch.cat([f_ch, i], dim=1)
        return torch.nan_to_num(phys, nan=0.0, posinf=1e6, neginf=-1e6)


def phys_interface_pipeline(phys_raw, variant, adapters):
    F = phys_raw[:, 0:1, :]
    I = phys_raw[:, 1:2, :]

    # Ion策略
    mode = variant.get("ion_gate", "use")
    if mode == "zero":
        I_eff = torch.zeros_like(I)
    elif mode == "const":
        I_eff = torch.nan_to_num(I, nan=0.0).mean(dim=0, keepdim=True).expand_as(I)
    elif mode == "smooth":
        k = 5
        pad = (k - 1) // 2
        w = torch.ones(1, 1, k, device=I.device) / k
        I_eff = nn.functional.conv1d(nn.functional.pad(I, (pad, pad), mode="replicate"), w)
    elif mode == "gate":
        I_eff = adapters["gate"](F, I)
    else:
        I_eff = I

    x = torch.cat([F, I_eff], dim=1)

    # 派生特征
    if variant.get("use_derived", False):
        x = adapters["reducer"](F, I_eff)

    # PhysAdapter
    if variant.get("use_adapter", False):
        x = adapters["adapter"](x)

    return x


def train_single_variant_KFOLD(
    variant: Dict,
    data_dict: Dict,
    meta_old: Dict,
    device,
    seed: int,
    only_family: Optional[int] = None
):
    """
    5-FOLD 交叉验证训练。
    - 若 only_family 为 None：联合/Per-Head 训练（与 Block A 对应）
    - 若 only_family 为 int：仅对该 family 的有效标签训练（与 Block B 对应）
    返回值与主流程期望一致：包含 per_family / macro / micro / min_family 等键。
    """
    from physio_util import set_seed, metrics, transform_for_display
    set_seed(seed)

    # -------- 数据选择：整库 or 单 family 掩码 --------
    fams_full: List[str] = data_dict["families"]
    if only_family is not None:
        fam_name = fams_full[only_family]
        dd = _masked_data_only_family(data_dict, only_family)
        v = _decorate_variant_name(variant, f"ONLY{fam_name}")
        print("\n" + "=" * 80)
        print(f"🔄 5-FOLD CV Training (ONLY {fam_name}): {v['name']} | Seed: {seed}")
        print("=" * 80)
    else:
        dd = data_dict
        v = dict(variant)  # 浅拷贝
        print("\n" + "=" * 80)
        print(f"🔄 5-FOLD CV Training: {v['name']} | Seed: {seed}")
        print("=" * 80)

    # -------- 提取数据 --------
    s8_full = dd["s8"]
    y_full = dd["y_sparse"]
    m_full = dd["m_sparse"]
    tvals_full = dd["tvals"]
    fams: List[str] = dd["families"]
    B = s8_full.shape[0]

    # 分层变量：按有效点多少进行二分类
    sample_counts = m_full.sum(dim=(1, 2)).numpy()
    strata = (sample_counts > np.median(sample_counts)).astype(int)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    indices = np.arange(B)

    fold_results = []  # 保存每折的指标等

    # -------- 逐折训练 --------
    for fold, (train_idx, test_idx) in enumerate(kfold.split(indices, strata)):
        print(f"\n{'=' * 80}")
        print(f"📊 Fold {fold + 1}/5: Train={len(train_idx)}, Test={len(test_idx)}")
        print(f"{'=' * 80}")
        s8_raw_all = dd["s8_raw"]  # (B, 7) 原始域（未标准化）
        s8_train_raw = s8_raw_all[train_idx]
        train_mean = s8_train_raw.mean(dim=0)
        train_std = s8_train_raw.std(dim=0).clamp_min(1e-8)
        s8_renorm = (s8_raw_all - train_mean) / train_std

        # 当前 fold 的张量
        s8_train = s8_renorm[train_idx].to(device)
        y_train = y_full[train_idx].to(device)
        m_train = m_full[train_idx].to(device)
        s8_test = s8_renorm[test_idx].to(device)
        y_test = y_full[test_idx].to(device)
        m_test = m_full[test_idx].to(device)

        tvals = (tvals_full if tvals_full.dim() == 1 else tvals_full[0]).to(device)
        T = len(tvals)
        K = len(fams)

        # ------ 初始化模型与模块 ------
        phys_F, phys_I, ion_aff_init = build_phys_from_ckpt(
            Cfg.phys_ckpt_F, Cfg.phys_ckpt_I, device
        )
        freeze_physics_models(phys_F, phys_I, verbose=False)

        ion_tr = IonInverseTransform(
            ion_aff_init,
            learnable=v.get("learnable_ion", False)
        ).to(device)

        morph = TemporalRegressor(
            K=K, d_model=64, nhead=4, num_layers=2, dim_ff=128,
            dropout=Cfg.dropout_morph, T=T
        ).to(device)

        if os.path.exists(Cfg.morph_ckpt):
            ck = _safe_load(Cfg.morph_ckpt, map_location="cpu")
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            morph.load_state_dict(sd, strict=False)

        freeze_model_parts(
            morph,
            freeze_encoder=v.get("freeze_morph_encoder", True),
            freeze_heads=v.get("freeze_morph_heads", False),
            verbose=False
        )

        calib = build_calib_head(v["calib_head"], K=K).to(device)

        per_family_head = PerFamilyHead(K=K).to(device) \
            if v.get("per_family_head", False) else nn.Identity().to(device)

        hetero_head = HeteroHead(K=K, per_family=v.get("per_family_head", False)).to(device) \
            if v.get("hetero", False) else None

        task_logvars = nn.Parameter(torch.zeros(K, device=device)) \
            if v.get("task_uncertainty", False) else None

        adapters = {
            "adapter": PhysAdapter(2, k=3).to(device),
            "reducer": PhysFeaReducer().to(device),
            "gate": IonGate(k=5).to(device),
        }

        # ------ 优化器 / 调度 / EMA ------
        param_groups = []
        morph_params = [p for p in morph.parameters() if p.requires_grad]
        if morph_params:
            param_groups.append({'params': morph_params, 'lr': Cfg.lr_morph, 'weight_decay': Cfg.wd_morph})
        calib_params = [p for p in calib.parameters() if p.requires_grad]
        if calib_params:
            param_groups.append({'params': calib_params, 'lr': Cfg.lr_calib, 'weight_decay': Cfg.wd_calib})
        if not isinstance(per_family_head, nn.Identity):
            param_groups.append({'params': per_family_head.parameters(), 'lr': Cfg.lr_morph, 'weight_decay': Cfg.wd_morph})
        if v.get("learnable_ion", False):
            param_groups.append({'params': ion_tr.parameters(), 'lr': Cfg.ion_learnable_lr, 'weight_decay': Cfg.ion_learnable_wd})
        for ad in adapters.values():
            ad_params = [p for p in ad.parameters() if p.requires_grad]
            if ad_params:
                param_groups.append({'params': ad_params, 'lr': Cfg.lr_morph, 'weight_decay': Cfg.wd_morph})

        optimizer = torch.optim.AdamW(param_groups)
        if Cfg.use_scheduler:
            def lr_lambda(epoch):
                if epoch < Cfg.warmup_epochs:
                    return epoch / Cfg.warmup_epochs
                progress = (epoch - Cfg.warmup_epochs) / (Cfg.max_epochs - Cfg.warmup_epochs)
                return Cfg.min_lr_ratio + 0.5 * (1 - Cfg.min_lr_ratio) * (1 + np.cos(np.pi * progress))
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = None

        ema = EMA(morph, decay=Cfg.ema_decay) if Cfg.use_ema else None

        # ------ 训练循环 ------
        best_train_loss = float('inf')
        for epoch in range(1, Cfg.max_epochs + 1):
            morph.train()
            optimizer.zero_grad()

            phys_raw = phys_forward_raw(s8_train, tvals, phys_F, phys_I, ion_tr, allow_grad=False)
            phys_enh = phys_interface_pipeline(phys_raw, v, adapters)
            y_pred = morph(s8_train, phys_enh, tvals)
            if v.get("per_family_head", False):
                y_pred = per_family_head(y_pred)
            y_pred = calib(y_pred)

            if v.get("hetero", False) and hetero_head is not None:
                y_mu, y_logvar = hetero_head(y_pred)
                loss = hetero_nll(y_mu, y_logvar, y_train, m_train, task_logvars)
            else:
                mono_penalty = None
                if "zmin" in fams:
                    mono_penalty = {"k_idx": fams.index("zmin"), "weight": Cfg.mono_zmin_weight, "direction": "decrease"}
                loss, _ = masked_huber_with_channel_norm(
                    y_pred, y_train, m_train,
                    delta=Cfg.loss_delta, smooth_weight=Cfg.loss_smooth_weight,
                    mono_penalty=mono_penalty
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(morph.parameters(), Cfg.batch_clip)
            optimizer.step()
            if ema is not None: ema.update(morph)
            if scheduler is not None: scheduler.step()
            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
            if epoch % 100 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{Cfg.max_epochs}: Loss={loss.item():.4f}")

        # ------ 测试评估（使用 EMA 权重） ------
        morph.eval()
        with torch.no_grad():
            if ema is not None: ema.apply_to(morph)
            phys_test = phys_forward_raw(s8_test, tvals, phys_F, phys_I, ion_tr, allow_grad=False)
            phys_test_enh = phys_interface_pipeline(phys_test, v, adapters)
            y_test_pred = morph(s8_test, phys_test_enh, tvals)
            if v.get("per_family_head", False):
                y_test_pred = per_family_head(y_test_pred)
            y_test_pred = calib(y_test_pred)

            yhat_disp, ytrue_disp = transform_for_display(
                y_test_pred, y_test,
                family_sign=Cfg.family_sign, unit_scale=Cfg.unit_scale,
                flip_sign=Cfg.flip_sign, clip_nonneg=Cfg.clip_nonneg,
                min_display_value=Cfg.min_display_value
            )
            mts = metrics(yhat_disp, ytrue_disp, m_test)
            per_family, macro, micro, min_family = compute_per_family_metrics(mts, m_test, fams)
            if ema is not None: ema.restore(morph)

        fold_results.append({
            'fold': fold,
            'train_idx': train_idx.tolist(),
            'test_idx': test_idx.tolist(),
            'test_metrics': {
                'R2_macro': macro.get('R2', np.nan),
                'MAE_macro': macro.get('MAE', np.nan),
                'RMSE_macro': macro.get('RMSE', np.nan),
                # 仅保留每 family 的 R2，足够用于汇总与 min_family 计算
                'R2_per_family': {fam: {'R2': per_family[fam].get('R2', np.nan)} for fam in fams},
            },
            'train_loss_final': best_train_loss,
        })

        # 释放显存
        del morph, phys_F, phys_I, optimizer, scheduler, ema
        torch.cuda.empty_cache()

    # -------- 聚合 5 折结果，返回与主流程一致的结构 --------
    r2_values = [f['test_metrics']['R2_macro'] for f in fold_results]
    mae_values = [f['test_metrics']['MAE_macro'] for f in fold_results]
    rmse_values = [f['test_metrics']['RMSE_macro'] for f in fold_results]
    r2_mean, r2_std = np.nanmean(r2_values), np.nanstd(r2_values)
    mae_mean, mae_std = np.nanmean(mae_values), np.nanstd(mae_values)
    rmse_mean, rmse_std = np.nanmean(rmse_values), np.nanstd(rmse_values)

    # 每个 family 的 R2 取 5 折均值
    per_family_agg = {}
    for fam in fams:
        r2_list = []
        for fr in fold_results:
            fam_r2 = fr['test_metrics']['R2_per_family'].get(fam, {}).get('R2', np.nan)
            if not np.isnan(fam_r2): r2_list.append(fam_r2)
        per_family_agg[fam] = {'R2': float(np.nanmean(r2_list)) if r2_list else np.nan}

    # macro / micro（micro 此处不参与后续汇总，可留空壳）
    macro_agg = {'R2': float(r2_mean), 'MAE': float(mae_mean), 'RMSE': float(rmse_mean)}
    micro_agg = {}  # 占位，不影响 summary_report 的使用

    # 最差 family（R2 最低）
    r2_map = {fam: per_family_agg[fam].get('R2', -np.inf) for fam in fams}
    min_family = {'R2': min(r2_map, key=r2_map.get)}

    # 保存一个聚合 JSON（与原逻辑一致）
    save_root = os.path.join(Cfg.save_root, f"{v['name']}_seed{seed}")
    os.makedirs(save_root, exist_ok=True)
    final_results = {
        'variant_name': v['name'],
        'seed': seed,
        'n_folds': 5,
        'fold_results': fold_results,
        'aggregated_metrics': {
            'R2_macro_mean': r2_mean, 'R2_macro_std': r2_std,
            'MAE_macro_mean': mae_mean, 'MAE_macro_std': mae_std,
            'RMSE_macro_mean': rmse_mean, 'RMSE_macro_std': rmse_std,
        },
    }
    with open(os.path.join(save_root, 'kfold_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"[✓] Results saved to: {os.path.join(save_root, 'kfold_results.json')}\n")

    # —— 关键：按主流程需要返回四元组信息 —— #
    return {
        'per_family': per_family_agg,
        'macro': macro_agg,
        'micro': micro_agg,
        'min_family': min_family,
        'variant_name': v['name'],
    }

def masked_huber_with_channel_norm(y_pred, y_true, mask, delta=1.0,
                                   smooth_weight=1e-3, mono_penalty=None):
    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=-1e6)
    B, K, T = y_pred.shape
    device = y_pred.device

    # 有效掩码
    finite_mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    eff_mask = (mask.bool() & finite_mask).float()

    # 逐通道计算均值和标准差
    mean_k = torch.zeros(K, device=device)
    std_k = torch.ones(K, device=device)

    with torch.no_grad():
        for k in range(K):
            mk = eff_mask[:, k, :].bool()
            if mk.any():
                vt = y_true[:, k, :][mk]
                if vt.numel() > 1:
                    m = vt.mean()
                    s = vt.std()
                    if not torch.isfinite(m):
                        m = torch.tensor(0.0, device=device)
                    if not torch.isfinite(s):
                        s = torch.tensor(1.0, device=device)
                    mean_k[k] = m
                    std_k[k] = s.clamp_min(1.0)

    # 标准化
    y_true_n = torch.nan_to_num(
        (y_true - mean_k.view(1, K, 1)) / std_k.view(1, K, 1),
        nan=0.0, posinf=1e6, neginf=-1e6
    )
    y_pred_n = torch.nan_to_num(
        (y_pred - mean_k.view(1, K, 1)) / std_k.view(1, K, 1),
        nan=0.0, posinf=1e6, neginf=-1e6
    )

    # Huber损失
    diff = (y_pred_n - y_true_n)
    absd = diff.abs()
    huber = torch.where(
        absd <= delta,
        0.5 * diff * diff,
        delta * (absd - 0.5 * delta)
    )

    # 逐通道平均
    denom_k = eff_mask.sum(dim=(0, 2)).clamp_min(1.0)
    loss_main_per_k = (huber * eff_mask).sum(dim=(0, 2)) / denom_k

    # 按标准差加权（标准差大的family权重小）
    w_k = 1.0 / std_k.clamp_min(1.0)
    loss_main = (loss_main_per_k * w_k).mean()

    # 平滑正则化（二阶差分）
    loss_smooth = torch.tensor(0.0, device=device)
    if T >= 3 and smooth_weight > 0:
        d1 = y_pred_n[:, :, 1:] - y_pred_n[:, :, :-1]  # 一阶差分
        d2 = d1[:, :, 1:] - d1[:, :, :-1]  # 二阶差分
        loss_smooth = torch.nan_to_num((d2 ** 2).mean(), nan=0.0, posinf=0.0, neginf=0.0)

    # 单调性约束
    loss_mono = torch.tensor(0.0, device=device)
    if mono_penalty is not None and T >= 2:
        k_idx = mono_penalty.get("k_idx", None)
        w_mono = mono_penalty.get("weight", 0.0)
        direction = mono_penalty.get("direction", "decrease")

        if (k_idx is not None) and (w_mono > 0):
            d = y_pred[:, k_idx, 1:] - y_pred[:, k_idx, :-1]
            if direction == "decrease":
                # 要求单调递减，惩罚正的差分
                loss_mono = torch.nn.functional.relu(d).mean() * w_mono
            else:
                # 要求单调递增，惩罚负的差分
                loss_mono = torch.nn.functional.relu(-d).mean() * w_mono

    # 总损失
    loss = loss_main + smooth_weight * loss_smooth + loss_mono

    # 确保损失有效
    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=device)

    loss_dict = {
        "loss_main": loss_main.detach(),
        "loss_smooth": loss_smooth.detach(),
        "loss_mono": loss_mono.detach(),
        "loss_total": loss.detach()
    }

    return loss, loss_dict


def hetero_nll(y_mu, y_logvar, y_true, mask, task_logvars=None):

    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=-1e6)
    m = (mask.bool() & torch.isfinite(y_true)).float()

    # 添加任务不确定性
    if task_logvars is not None:
        y_logvar = y_logvar + task_logvars.view(1, -1, 1)

    # 负对数似然
    inv_var = torch.exp(-y_logvar).clamp_max(1e6)
    nll = 0.5 * ((y_mu - y_true) ** 2 * inv_var + y_logvar)

    # 按有效点平均
    nll = (nll * m).sum(dim=(0, 2)) / m.sum(dim=(0, 2)).clamp_min(1.0)

    return nll.mean()


# ========================== 学习率调度器 ==========================
def create_lr_scheduler_with_warmup(optimizer, max_epochs, warmup_epochs=20,
                                    min_lr_ratio=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup阶段：线性增长
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦退火阶段
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_current_lr(optimizer):
    """获取当前学习率"""
    return optimizer.param_groups[0]['lr']


# ========================== 训练辅助函数 ==========================
def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def clip_gradients(model, max_norm=1.0):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_device(model):
    """获取模型所在设备"""
    return next(model.parameters()).device


# ========================== 批次处理 ==========================
def prepare_batch(batch, device):
    return tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)


def collate_sparse_batch(samples):
    s8_list, y_list, m_list, t_list = zip(*samples)

    s8 = torch.stack(s8_list, dim=0)
    y_sparse = torch.stack(y_list, dim=0)
    m_sparse = torch.stack(m_list, dim=0)

    # 时间值可能是共享的
    if len(set([tuple(t.tolist()) for t in t_list])) == 1:
        tvals = t_list[0]
    else:
        tvals = torch.stack(t_list, dim=0)

    return s8, y_sparse, m_sparse, tvals


# ========================== 训练日志 ==========================
class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.epoch_losses = []
        self.epoch_metrics = []

    def log_step(self, epoch, step, total_steps, loss, lr):
        """记录训练步骤"""
        if step % self.log_interval == 0 or step == total_steps - 1:
            print(f"[Epoch {epoch}] Step {step}/{total_steps} | "
                  f"Loss: {loss:.4f} | LR: {lr:.6f}")

    def log_epoch(self, epoch, train_loss, val_metrics, best_metric=None):
        """记录epoch结果"""
        self.epoch_losses.append(train_loss)
        self.epoch_metrics.append(val_metrics)

        msg = f"[Epoch {epoch}] Train Loss: {train_loss:.4f}"

        if val_metrics:
            for k, v in val_metrics.items():
                msg += f" | Val {k}: {v:.4f}"

        if best_metric is not None:
            msg += f" | Best: {best_metric:.4f}"

        print(msg)

    def get_summary(self):
        """获取训练摘要"""
        return {
            "epoch_losses": self.epoch_losses,
            "epoch_metrics": self.epoch_metrics,
            "best_loss": min(self.epoch_losses) if self.epoch_losses else None,
        }


# ========================== 权重初始化 ==========================
def initialize_weights(model, method='xavier'):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            if method == 'xavier':
                nn.init.xavier_uniform_(param)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif method == 'normal':
                nn.init.normal_(param, mean=0, std=0.02)
        elif 'bias' in name:
            nn.init.zeros_(param)


# ========================== 检查点管理 ==========================
def save_checkpoint(save_path, model, optimizer, scheduler, epoch,
                    metrics, ema=None, variant=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'variant': variant,
    }

    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow

    torch.save(checkpoint, save_path)


def load_checkpoint(load_path, model, optimizer=None, scheduler=None,
                    ema=None, device='cpu'):
    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if ema and 'ema_shadow' in checkpoint:
        ema.shadow = checkpoint['ema_shadow']

    return checkpoint


# ========================== 模型推理 ==========================
@torch.no_grad()
def predict_batch(model, s8, phys, tvals, use_ema=False, ema=None):
    model.eval()

    # 临时应用EMA权重
    if use_ema and ema is not None:
        # 保存原始权重
        original_state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        ema.apply_to(model)

    # 预测
    pred = model(s8, phys, tvals)

    # 恢复原始权重
    if use_ema and ema is not None:
        for n, p in model.named_parameters():
            if n in original_state:
                p.data.copy_(original_state[n])

    return pred


def compute_per_family_metrics(mts: Dict, m_sparse: torch.Tensor, fams: List[str]):
    # 监督信号：每个(k,t)位置是否有有效数据
    sup = m_sparse.sum(dim=0).detach().cpu().numpy() > 0  # (K, T)

    # 逐Family统计
    per_family = {}
    for k, fam in enumerate(fams):
        per_family[fam] = {}
        for metric_name in ["R2", "MAE", "RMSE", "SMAPE", "MAPE", "MSE"]:
            if metric_name not in mts:
                continue

            grid = mts[metric_name]
            if torch.is_tensor(grid):
                grid = grid.detach().cpu().numpy()

            # 该Family的平均值（忽略无监督的时间点）
            fam_vals = grid[k, :]
            fam_sup = sup[k, :]

            if fam_sup.sum() > 0:
                per_family[fam][metric_name] = float(np.nanmean(fam_vals[fam_sup]))
            else:
                per_family[fam][metric_name] = np.nan

    # Macro平均（公平对待每个Family）
    macro = {}
    for metric_name in ["R2", "MAE", "RMSE", "SMAPE", "MAPE", "MSE"]:
        vals = [per_family[fam].get(metric_name, np.nan) for fam in fams]
        macro[metric_name] = float(np.nanmean(vals))

    # Micro平均（按样本数加权）
    micro = {}
    for metric_name in ["R2", "MAE", "RMSE", "SMAPE", "MAPE", "MSE"]:
        if metric_name not in mts:
            continue

        grid = mts[metric_name]
        if torch.is_tensor(grid):
            grid = grid.detach().cpu().numpy()

        # 全局平均（所有有监督的点）
        if sup.sum() > 0:
            micro[metric_name] = float(np.nanmean(grid[sup]))
        else:
            micro[metric_name] = np.nan

    # 最差Family（R²最低 / MAE最高）
    min_family = {}
    r2_vals = {fam: per_family[fam].get("R2", -np.inf) for fam in fams}
    min_family["R2"] = min(r2_vals, key=r2_vals.get)

    mae_vals = {fam: per_family[fam].get("MAE", np.inf) for fam in fams}
    min_family["MAE"] = max(mae_vals, key=mae_vals.get)

    return per_family, macro, micro, min_family


def print_per_family_report(per_family, macro, micro, min_family, fams,
                            title="Evaluation"):
    """★ 打印逐Family报告"""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}")

    # 逐Family
    print(f"\n{'Family':<10} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'SMAPE/MAPE':>12}")
    print("-" * 80)
    for fam in fams:
        r2 = per_family[fam].get("R2", np.nan)
        mae = per_family[fam].get("MAE", np.nan)
        rmse = per_family[fam].get("RMSE", np.nan)
        smape = per_family[fam].get("SMAPE", per_family[fam].get("MAPE", np.nan))
        print(f"{fam:<10} {r2:>8.4f} {mae:>8.2f} {rmse:>8.2f} {smape:>12.2f}")

    # 汇总
    print("-" * 80)
    print(f"{'Macro Avg':<10} {macro.get('R2', np.nan):>8.4f} "
          f"{macro.get('MAE', np.nan):>8.2f} "
          f"{macro.get('RMSE', np.nan):>8.2f} "
          f"{macro.get('SMAPE', macro.get('MAPE', np.nan)):>12.2f}")
    print(f"{'Micro Avg':<10} {micro.get('R2', np.nan):>8.4f} "
          f"{micro.get('MAE', np.nan):>8.2f} "
          f"{micro.get('RMSE', np.nan):>8.2f} "
          f"{micro.get('SMAPE', micro.get('MAPE', np.nan)):>12.2f}")
    print(f"{'Min Family':<10} {min_family['R2']:<8} "
          f"(R²={per_family[min_family['R2']].get('R2', np.nan):.4f})")
    print(f"{'=' * 80}\n")


# ========================== 可视化增强 ==========================
def plot_per_family_diagnostics(y_pred, y_true, mask, fams, T_values,
                                save_dir, title_prefix=""):
    ensure_dir(save_dir)

    yp = y_pred.detach().cpu().numpy()
    yt = y_true.detach().cpu().numpy()
    mk = mask.detach().cpu().numpy()

    K = len(fams)

    # 创建子图：2行 × K列
    fig, axes = plt.subplots(2, K, figsize=(4 * K, 8))
    if K == 1:
        axes = axes.reshape(2, 1)

    for k, fam in enumerate(fams):
        valid = mk[:, k, :].flatten()
        yp_k = yp[:, k, :].flatten()[valid]
        yt_k = yt[:, k, :].flatten()[valid]

        if len(yp_k) == 0:
            continue

        # =============== 上排：Parity Plot ===============
        ax1 = axes[0, k]
        ax1.scatter(yt_k, yp_k, alpha=0.5, s=10, c='blue')

        # 回归线
        if len(yp_k) > 1:
            try:
                from scipy.stats import linregress
                slope, intercept, r, *_ = linregress(yt_k, yp_k)
                x_line = np.linspace(yt_k.min(), yt_k.max(), 100)
                ax1.plot(x_line, slope * x_line + intercept, 'r-',
                         label=f'y={slope:.2f}x+{intercept:.1f}\nR²={r ** 2:.3f}')
            except:
                pass

        # 理想线 (y=x)
        ax1.plot([yt_k.min(), yt_k.max()], [yt_k.min(), yt_k.max()],
                 'k--', label='Ideal')
        ax1.set_xlabel('True')
        ax1.set_ylabel('Pred')
        ax1.set_title(f'{fam}')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # =============== 下排：Residual Histogram ===============
        ax2 = axes[1, k]
        residuals = yp_k - yt_k
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'μ={residuals.mean():.2f}, σ={residuals.std():.2f}')
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{title_prefix}_family_diagnostics.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Plot] Saved diagnostics to {save_path}")


def plot_temporal_error(y_pred, y_true, mask, fams, T_values,
                        save_dir, title_prefix=""):
    ensure_dir(save_dir)

    yp = y_pred.detach().cpu().numpy()
    yt = y_true.detach().cpu().numpy()
    mk = mask.detach().cpu().numpy().astype(bool)

    K = len(fams)
    T = len(T_values)

    # 创建子图：K行 × 1列
    fig, axes = plt.subplots(K, 1, figsize=(10, 3 * K), squeeze=False)

    for k, fam in enumerate(fams):
        ax = axes[k, 0]
        mae_t = []

        for t in range(T):
            valid = mk[:, k, t]
            if valid.sum() == 0:
                mae_t.append(np.nan)
                continue

            err = np.abs(yp[:, k, t][valid] - yt[:, k, t][valid])
            mae_t.append(float(err.mean()))

        # 绘制MAE曲线
        ax.plot(T_values, mae_t, marker='o', linewidth=2,
                markersize=8, label=fam, color='steelblue')
        ax.fill_between(T_values, mae_t, alpha=0.3, color='steelblue')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title(f'{fam} - Temporal MAE', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{title_prefix}_temporal_error.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Plot] Saved temporal error to {save_path}")


def plot_training_curves(train_losses, val_metrics, save_dir, title_prefix=""):
    ensure_dir(save_dir)

    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：训练损失
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, marker='o', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 右图：验证R²
    ax2 = axes[1]
    if val_metrics and len(val_metrics) > 0:
        val_r2 = [m.get('R2_macro', np.nan) for m in val_metrics]
        ax2.plot(epochs, val_r2, marker='s', linewidth=2,
                 color='green', label='Val R² (Macro)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²')
        ax2.set_title('Validation R²')
        ax2.legend()
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{title_prefix}_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Plot] Saved training curves to {save_path}")


# ========================== 结果保存 ==========================
def save_evaluation_results(save_dir, per_family, macro, micro, min_family,
                            variant_name=""):
    ensure_dir(save_dir)

    results = {
        "variant": variant_name,
        "per_family": per_family,
        "macro": macro,
        "micro": micro,
        "min_family": min_family,
    }

    save_path = os.path.join(save_dir, f"{variant_name}_evaluation.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Save] Evaluation results saved to {save_path}")

    return save_path


def save_predictions(save_dir, y_pred, y_true, mask, fams, T_values,
                     variant_name=""):
    ensure_dir(save_dir)

    save_dict = {
        'y_pred': y_pred.detach().cpu().numpy(),
        'y_true': y_true.detach().cpu().numpy(),
        'mask': mask.detach().cpu().numpy(),
        'families': fams,
        'time_values': T_values,
    }

    save_path = os.path.join(save_dir, f"{variant_name}_predictions.npz")
    np.savez_compressed(save_path, **save_dict)

    print(f"[Save] Predictions saved to {save_path}")

    return save_path


def generate_summary_report(save_dir, variant_results, fams):
    ensure_dir(save_dir)

    # 创建汇总表格
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("VARIANT COMPARISON REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")

    # 表头
    header = f"{'Variant':<30} {'R²_Macro':>10} {'MAE_Macro':>10} {'RMSE_Macro':>10} {'Min_R²_Family':>15}"
    report_lines.append(header)
    report_lines.append("-" * 100)

    # 逐变体
    best_r2 = -1e9
    best_variant = None

    for variant_name, (per_family, macro, micro, min_family) in variant_results.items():
        r2 = macro.get('R2', np.nan)
        mae = macro.get('MAE', np.nan)
        rmse = macro.get('RMSE', np.nan)
        min_fam = min_family.get('R2', 'N/A')

        line = f"{variant_name:<30} {r2:>10.4f} {mae:>10.2f} {rmse:>10.2f} {min_fam:>15}"
        report_lines.append(line)

        if not np.isnan(r2) and r2 > best_r2:
            best_r2 = r2
            best_variant = variant_name

    report_lines.append("-" * 100)
    report_lines.append(f"\nBest Variant: {best_variant} (R²={best_r2:.4f})")
    report_lines.append("=" * 100)

    # 保存报告
    report_path = os.path.join(save_dir, "summary_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # 打印报告
    print('\n'.join(report_lines))
    print(f"\n[Save] Summary report saved to {report_path}")

    return report_path


def _safe_load(path, map_location="cpu"):
    """安全加载checkpoint"""
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def _infer_arch_from_sd(sd: dict) -> Dict[str, int]:
    # 兜底
    T_default, d_model_default = 10, 128

    # pos 嵌入推断
    pos = sd.get("pos", None)
    if pos is not None and hasattr(pos, "shape") and pos.dim() == 3:
        _, T, d_model = pos.shape
    else:
        w = sd.get("input_proj.weight", None)
        if w is not None and hasattr(w, "shape"):
            d_model = w.shape[0]
        else:
            d_model = d_model_default
        T = T_default

    key_l1 = "encoder.layers.0.linear1.weight"
    if key_l1 in sd:
        dim_ff = sd[key_l1].shape[0]
    else:
        dim_ff = max(2 * d_model, 256)

    nhead = 8 if d_model % 8 == 0 else 4

    L = 0
    while any(k.startswith(f"encoder.layers.{L}.") for k in sd.keys()):
        L += 1
    num_layers = max(L, 1)

    return dict(T=T, d_model=d_model, nhead=nhead, dim_ff=dim_ff, num_layers=num_layers)

def build_phys_from_ckpt(ckpt_F_path, ckpt_I_path, device):
    ckf = _safe_load(ckpt_F_path, map_location="cpu")
    cki = _safe_load(ckpt_I_path, map_location="cpu")

    if not ckf or not cki:
        print(f"[WARN] Physics ckpt missing or empty. Using fallback tiny arch.")
        arch_F = arch_I = dict(T=10, d_model=128, nhead=4, dim_ff=256, num_layers=2)
        pf = PhysicsSeqPredictor(**arch_F).to(device)
        pi = PhysicsSeqPredictor(**arch_I).to(device)
        ion_aff = copy.deepcopy(Cfg.ion_affine_default)
        return pf, pi, ion_aff

    sd_F = ckf["model"] if isinstance(ckf, dict) and "model" in ckf else ckf
    sd_I = cki["model"] if isinstance(cki, dict) and "model" in cki else cki

    arch_F = _infer_arch_from_sd(sd_F) if sd_F else dict(T=10, d_model=128, nhead=4, dim_ff=256, num_layers=2)
    arch_I = _infer_arch_from_sd(sd_I) if sd_I else dict(T=10, d_model=128, nhead=4, dim_ff=256, num_layers=2)

    pf = PhysicsSeqPredictor(**arch_F).to(device)
    pi = PhysicsSeqPredictor(**arch_I).to(device)

    if sd_F: pf.load_state_dict(sd_F, strict=False)
    if sd_I: pi.load_state_dict(sd_I, strict=False)

    ion_aff = cki.get("ion_affine", copy.deepcopy(Cfg.ion_affine_default)) \
        if isinstance(cki, dict) else copy.deepcopy(Cfg.ion_affine_default)

    return pf, pi, ion_aff

# ========================== 单变体训练流程 ==========================
def train_single_variant(variant: Dict, data_dict: Dict, meta_old: Dict,
                         device, seed: int):
    set_seed(seed)
    variant_name = variant['name']
    save_dir = os.path.join(Cfg.save_root, f"{variant_name}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Training Variant: {variant_name} | Seed: {seed}")
    print("=" * 80)
    print(f"Description: {variant['description']}")
    print(f"Expected R²: {variant.get('expected_r2_range', 'N/A')}")
    print(f"Expected params: ~{variant['expected_trainable_params']:,}")

    # 解包数据
    s8_full = data_dict["s8"].to(device)
    y_full = data_dict["y_sparse"].to(device)
    m_full = data_dict["m_sparse"].to(device)
    tvals_full = data_dict["tvals"].to(device)
    fams = data_dict["families"]

    train_idx = data_dict["splits"]["train"]
    val_idx = data_dict["splits"]["val"]
    test_idx = data_dict["splits"]["test"]

    # 提取子集
    s8_train = s8_full[train_idx]
    y_train = y_full[train_idx]
    m_train = m_full[train_idx]

    s8_val = s8_full[val_idx]
    y_val = y_full[val_idx]
    m_val = m_full[val_idx]

    s8_test = s8_full[test_idx]
    y_test = y_full[test_idx]
    m_test = m_full[test_idx]

    tvals = tvals_full if tvals_full.dim() == 1 else tvals_full[0]
    T = len(tvals)
    K = len(fams)

    print(f"\nData: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # === 模型初始化 ===
    print("\n[1/6] Initializing models...")

    # 物理模型
    phys_F, phys_I, ion_aff_init = build_phys_from_ckpt(
        Cfg.phys_ckpt_F, Cfg.phys_ckpt_I, device
    )
    freeze_physics_models(phys_F, phys_I, verbose=True)

    # Ion变换
    ion_tr = IonInverseTransform(
        ion_aff_init,
        learnable=variant.get("learnable_ion", False)
    ).to(device)

    # 形貌模型
    morph = TemporalRegressor(
        K=K,
        d_model=64,  # 与stageB一致
        nhead=4,
        num_layers=2,
        dim_ff=128,
        dropout=Cfg.dropout_morph,
        T=T
    ).to(device)

    # 加载预训练权重
    if os.path.exists(Cfg.morph_ckpt):
        ck = _safe_load(Cfg.morph_ckpt, map_location="cpu")
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        morph.load_state_dict(sd, strict=False)
        print(f"  [✓] Loaded morph pretrained weights")

    # 冻结策略
    freeze_model_parts(
        morph,
        freeze_encoder=variant.get("freeze_morph_encoder", True),
        freeze_heads=variant.get("freeze_morph_heads", False),
        verbose=True
    )

    # 校准头
    calib = build_calib_head(variant["calib_head"], K=K).to(device)

    # 输出头（可选）
    per_family_head = PerFamilyHead(K=K).to(device) \
        if variant.get("per_family_head", False) \
        else nn.Identity().to(device)

    hetero_head = HeteroHead(K=K, per_family=variant.get("per_family_head", False)).to(device) \
        if variant.get("hetero", False) \
        else None

    task_logvars = nn.Parameter(torch.zeros(K, device=device)) \
        if variant.get("task_uncertainty", False) \
        else None

    # 适配器
    adapters = {
        "adapter": PhysAdapter(2, k=3).to(device),
        "reducer": PhysFeaReducer().to(device),
        "gate": IonGate(k=5).to(device),
    }

    # 打印参数统计
    model_dict = {
        "phys_F": phys_F,
        "phys_I": phys_I,
        "ion_tr": ion_tr,
        "morph": morph,
        "calib": calib,
    }
    print_trainable_params(model_dict)

    # === 优化器 ===
    print("\n[2/6] Setting up optimizer and scheduler...")

    params = [
        {"params": [p for p in morph.parameters() if p.requires_grad],
         "lr": Cfg.lr_morph, "weight_decay": Cfg.wd_morph},
        {"params": [p for p in calib.parameters() if p.requires_grad],
         "lr": Cfg.lr_calib, "weight_decay": Cfg.wd_calib},
    ]

    if ion_tr.learnable:
        params.append({
            "params": [p for p in ion_tr.parameters() if p.requires_grad],
            "lr": Cfg.ion_learnable_lr,
            "weight_decay": Cfg.ion_learnable_wd
        })

    if variant.get("per_family_head", False):
        params.append({
            "params": [p for p in per_family_head.parameters() if p.requires_grad],
            "lr": Cfg.lr_calib,
            "weight_decay": Cfg.wd_calib
        })

    if task_logvars is not None:
        params.append({
            "params": [task_logvars],
            "lr": Cfg.lr_calib,
            "weight_decay": 0.0
        })

    optimizer = torch.optim.AdamW(params)
    scheduler = create_lr_scheduler_with_warmup(
        optimizer,
        max_epochs=Cfg.max_epochs,
        warmup_epochs=Cfg.warmup_epochs,
        min_lr_ratio=Cfg.min_lr_ratio
    )

    scaler = torch.cuda.amp.GradScaler(enabled=False)  # 30样本不需要AMP

    # EMA和早停
    ema = EMA(morph, decay=Cfg.ema_decay) if Cfg.use_ema else None
    early_stopper = EarlyStopper(patience=Cfg.early_stop_patience) \
        if Cfg.early_stop else None

    logger = TrainingLogger(log_interval=Cfg.log_interval)

    # === 训练循环 ===
    print(f"\n[3/6] Training for {Cfg.max_epochs} epochs...")
    print(f"Full batch training: {Cfg.use_full_batch}")

    best_val_r2 = -1e9
    train_losses = []
    val_metrics_list = []

    for epoch in range(1, Cfg.max_epochs + 1):
        # ========== 训练 ==========
        morph.train()

        # 数据增强
        s8_aug = s8_train

        optimizer.zero_grad()

        # 物理前向
        phys_raw = phys_forward_raw(s8_aug, tvals, phys_F, phys_I, ion_tr,
                                    allow_grad=False)

        # 物理特征增强
        phys_aug = phys_raw

        # 接口增强
        phys_enh = phys_interface_pipeline(phys_aug, variant, adapters)

        # 形貌预测
        y_pred = morph(s8_aug, phys_enh, tvals)

        # 输出头
        if variant.get("per_family_head", False):
            y_pred = per_family_head(y_pred)

        # 校准
        y_pred = calib(y_pred)

        # 损失计算
        if variant.get("hetero", False) and hetero_head is not None:
            y_mu, y_logvar = hetero_head(y_pred)
            loss = hetero_nll(y_mu, y_logvar, y_train, m_train, task_logvars)
            loss_dict = {"loss_total": loss.detach()}
        else:
            # 单调性约束（如果需要）
            mono_penalty = None
            if "zmin" in fams:
                k_zmin = fams.index("zmin")
                mono_penalty = {
                    "k_idx": k_zmin,
                    "weight": Cfg.mono_zmin_weight,
                    "direction": "decrease"
                }

            loss, loss_dict = masked_huber_with_channel_norm(
                y_pred, y_train, m_train,
                delta=Cfg.loss_delta,
                smooth_weight=Cfg.loss_smooth_weight,
                mono_penalty=mono_penalty
            )

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(morph.parameters(), Cfg.batch_clip)
        optimizer.step()

        # EMA更新
        if ema is not None:
            ema.update(morph)

        scheduler.step()

        train_losses.append(loss.item())

        # ========== 验证 ==========
        if epoch % 10 == 0 or epoch == Cfg.max_epochs or epoch == 1:
            morph.eval()
            with torch.no_grad():
                # 使用EMA权重
                if ema is not None:
                    ema.apply_to(morph)

                # 验证集前向
                phys_val = phys_forward_raw(s8_val, tvals, phys_F, phys_I, ion_tr,
                                            allow_grad=False)
                phys_val_enh = phys_interface_pipeline(phys_val, variant, adapters)
                y_val_pred = morph(s8_val, phys_val_enh, tvals)

                if variant.get("per_family_head", False):
                    y_val_pred = per_family_head(y_val_pred)

                y_val_pred = calib(y_val_pred)

                # 转换到展示空间
                yhat_disp, ytrue_disp = transform_for_display(
                    y_val_pred, y_val,
                    family_sign=Cfg.family_sign,
                    unit_scale=Cfg.unit_scale,
                    flip_sign=Cfg.flip_sign,
                    clip_nonneg=Cfg.clip_nonneg,
                    min_display_value=Cfg.min_display_value
                )

                # 计算指标
                mts = metrics(yhat_disp, ytrue_disp, m_val)
                per_family, macro, micro, min_family = compute_per_family_metrics(
                    mts, m_val, fams
                )

                val_r2 = macro.get('R2', 0.0)
                val_metrics_list.append({"R2_macro": val_r2, "MAE_macro": macro.get('MAE', 0.0)})

                # 日志
                logger.log_epoch(
                    epoch, loss.item(),
                    {"R2": val_r2, "MAE": macro.get('MAE', 0.0)},
                    best_val_r2
                )

                # 保存最佳模型
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    save_checkpoint(
                        os.path.join(save_dir, "best_model.pth"),
                        morph, optimizer, scheduler, epoch,
                        {"val_r2": val_r2, "macro": macro},
                        ema, variant
                    )
                    print(f"  [✓] Saved best model (R²={val_r2:.4f})")

                # 早停检查
                if early_stopper is not None:
                    if not early_stopper.step(-val_r2):
                        print(f"[Early Stop] at epoch {epoch}")
                        break

    # === 后校准 ===
    print("\n[4/6] Post-calibration on training set...")

    morph.eval()
    with torch.no_grad():
        # 训练集预测
        phys_train = phys_forward_raw(s8_train, tvals, phys_F, phys_I, ion_tr,
                                      allow_grad=False)
        phys_train_enh = phys_interface_pipeline(phys_train, variant, adapters)
        y_train_pred = morph(s8_train, phys_train_enh, tvals)

        if variant.get("per_family_head", False):
            y_train_pred = per_family_head(y_train_pred)

        y_train_pred = calib(y_train_pred)

        # 转换到展示空间
        yhat_train_disp, ytrue_train_disp = transform_for_display(
            y_train_pred, y_train,
            family_sign=Cfg.family_sign,
            unit_scale=Cfg.unit_scale
        )

        # 拟合校准参数
        calib_params = fit_calibration_params(
            yhat_train_disp, ytrue_train_disp, m_train,
            method=variant.get("post_calib", "per_k"),
            min_points=Cfg.calib_min_points,
            ridge=Cfg.calib_ridge
        )

    # === 测试集评估 ===
    print("\n[5/6] Testing on test set...")

    morph.eval()
    with torch.no_grad():
        # 测试集预测
        phys_test = phys_forward_raw(s8_test, tvals, phys_F, phys_I, ion_tr,
                                     allow_grad=False)
        phys_test_enh = phys_interface_pipeline(phys_test, variant, adapters)
        y_test_pred = morph(s8_test, phys_test_enh, tvals)

        if variant.get("per_family_head", False):
            y_test_pred = per_family_head(y_test_pred)

        y_test_pred = calib(y_test_pred)

        # 转换到展示空间
        yhat_test_disp, ytrue_test_disp = transform_for_display(
            y_test_pred, y_test,
            family_sign=Cfg.family_sign,
            unit_scale=Cfg.unit_scale
        )

        # 应用后校准
        yhat_test_cal = apply_calibration_params(yhat_test_disp, calib_params)

        # 计算指标（校准前）
        mts_before = metrics(yhat_test_disp, ytrue_test_disp, m_test)
        pf_before, macro_before, micro_before, min_fam_before = \
            compute_per_family_metrics(mts_before, m_test, fams)

        # 计算指标（校准后）
        mts_after = metrics(yhat_test_cal, ytrue_test_disp, m_test)
        pf_after, macro_after, micro_after, min_fam_after = \
            compute_per_family_metrics(mts_after, m_test, fams)

        # 打印报告
        print_per_family_report(pf_before, macro_before, micro_before, min_fam_before,
                                fams, f"{variant_name} - Before Calibration")
        print_per_family_report(pf_after, macro_after, micro_after, min_fam_after,
                                fams, f"{variant_name} - After Calibration")

    # === 保存结果 ===
    print("\n[6/6] Saving results and plots...")

    save_evaluation_results(save_dir, pf_after, macro_after, micro_after,
                            min_fam_after, variant_name)
    save_predictions(save_dir, yhat_test_cal, ytrue_test_disp, m_test,
                     fams, tvals.cpu().numpy(), variant_name)

    plot_per_family_diagnostics(yhat_test_cal, ytrue_test_disp, m_test,
                                fams, tvals.cpu().numpy(), save_dir, variant_name)
    plot_temporal_error(yhat_test_cal, ytrue_test_disp, m_test,
                        fams, tvals.cpu().numpy(), save_dir, variant_name)

    print(f"\n[✓] Variant {variant_name} completed!")
    print(f"Final Test R² (Macro): {macro_after.get('R2', 0.0):.4f}")

    return {
        "variant_name": variant_name,
        "per_family": pf_after,
        "macro": macro_after,
        "micro": micro_after,
        "min_family": min_fam_after,
        "best_val_r2": best_val_r2,
        "save_dir": save_dir
    }


# ========================== 主函数 ==========================
# ========================== 主函数（替换为自动批跑版） ==========================
def main():
    """自动跑：5折(KFold) + 三分(TriSplit) × {联合/Per-Head/逐Family独立}，覆盖 Cfg.variants × Cfg.seeds"""
    print_config_summary()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")

    # 旧表 meta（用于 StageC 对齐）
    print("\n[Loading] Old table meta...")
    from physio_util import excel_to_physics_dataset
    try:
        _, meta_old = excel_to_physics_dataset(Cfg.old_excel, sheet_name=Cfg.sheet_name)
        print(f"  [✓] Loaded meta from {Cfg.old_excel}")
    except Exception as e:
        print(f"  [✗] Failed to load meta: {e}")
        print("  [INFO] Using default meta")
        meta_old = {
            "norm_mean": torch.zeros(7),
            "norm_std": torch.ones(7),
            "time_values": [1, 3, 5, 9, 10],
            "families": ["zmin", "h1", "d1", "w", "h2", "d2"]
        }

    # 新表数据（含固定三分划分）
    print("\n[Loading] New table data...")
    data_dict = load_all_data(meta_old)
    fams: List[str] = data_dict["families"]

    # 聚合所有结果，生成大汇总
    all_results = {}

    # === Sweep over variants × seeds ===
    for variant in Cfg.variants:
        for seed in Cfg.seeds:
            # ----------------------------
            # A) 5-FOLD CV（联合/Per-Head）
            # ----------------------------
            try:
                print("\n" + "="*80)
                print(f"[AUTO] KFold | JOINT/PER-HEAD | {variant['name']} | seed={seed}")
                kf_res = train_single_variant_KFOLD(variant, data_dict, meta_old, device, seed)
                key = f"{variant['name']}_seed{seed}_KFOLD"
                # 统一收集为 (per_family, macro, micro, min_family)
                all_results[key] = (
                    kf_res['per_family'],
                    kf_res['macro'],
                    kf_res['micro'],
                    kf_res['min_family']
                )
            except Exception as e:
                print(f"\n[ERROR] KFOLD joint/per-head failed: {variant['name']} s{seed}: {e}")
                import traceback; traceback.print_exc()

            # ----------------------------
            # B) 5-FOLD CV（逐 family 独立）
            # ----------------------------
            for k, fam in enumerate(fams):
                try:
                    print("\n" + "-"*80)
                    print(f"[AUTO] KFold | SEPARATE-FAMILY={fam} | {variant['name']} | seed={seed}")
                    dd_k = _masked_data_only_family(data_dict, k)
                    v_k = _decorate_variant_name(variant, f"ONLY{fam}")
                    kf_sep_res = train_single_variant_KFOLD(v_k, dd_k, meta_old, device, seed)
                    key = f"{v_k['name']}_seed{seed}_KFOLD"
                    all_results[key] = (
                        kf_sep_res['per_family'],
                        kf_sep_res['macro'],
                        kf_sep_res['micro'],
                        kf_sep_res['min_family']
                    )
                except Exception as e:
                    print(f"\n[ERROR] KFOLD separate family failed: {variant['name']} [{fam}] s{seed}: {e}")
                    import traceback; traceback.print_exc()

            # ----------------------------
            # C) TriSplit（联合/Per-Head）
            # 注：TriSplit 版本已由 train_single_variant 实现（使用 data_dict['splits']）
            # 为避免目录冲突，给 variant 临时加后缀 _TRI
            # ----------------------------
            try:
                print("\n" + "="*80)
                print(f"[AUTO] TRISPLIT | JOINT/PER-HEAD | {variant['name']} | seed={seed}")
                v_tri = _decorate_variant_name(variant, "TRI")
                tri_res = train_single_variant(v_tri, data_dict, meta_old, device, seed)
                key = f"{v_tri['name']}_seed{seed}"
                all_results[key] = (
                    tri_res['per_family'],
                    tri_res['macro'],
                    tri_res['micro'],
                    tri_res['min_family']
                )
            except Exception as e:
                print(f"\n[ERROR] TRISPLIT joint/per-head failed: {variant['name']} s{seed}: {e}")
                import traceback; traceback.print_exc()

            # ----------------------------
            # D) TriSplit（逐 family 独立）
            # ----------------------------
            for k, fam in enumerate(fams):
                try:
                    print("\n" + "-"*80)
                    print(f"[AUTO] TRISPLIT | SEPARATE-FAMILY={fam} | {variant['name']} | seed={seed}")
                    dd_k = _masked_data_only_family(data_dict, k)
                    v_tri_k = _decorate_variant_name(variant, f"TRI_ONLY{fam}")
                    tri_sep_res = train_single_variant(v_tri_k, dd_k, meta_old, device, seed)
                    key = f"{v_tri_k['name']}_seed{seed}"
                    all_results[key] = (
                        tri_sep_res['per_family'],
                        tri_sep_res['macro'],
                        tri_sep_res['micro'],
                        tri_sep_res['min_family']
                    )
                except Exception as e:
                    print(f"\n[ERROR] TRISPLIT separate family failed: {variant['name']} [{fam}] s{seed}: {e}")
                    import traceback; traceback.print_exc()

    # === 汇总报告 ===
    if all_results:
        print("\n" + "=" * 80)
        print("Generating Summary Report")
        print("=" * 80)
        generate_summary_report(Cfg.save_root, all_results, fams)

    print("\n[✓] All auto-sweeps done!")
    print(f"Results saved under: {Cfg.save_root}")



# ========================== 用于测试的主函数 ==========================
if __name__ == "__main__":
    main()
