# -*- coding: utf-8 -*-
"""
Stage C: 新表迁移微调（预测式 Phys→Morph）【完整修缮版 | R²→80%+】
核心修缮：
1. ✅ 数据划分：Train/Val/Test 完全隔离
2. ✅ 逐 Family 评估：Macro/Micro/Min-R² + 完整报告
3. ✅ 后校准隔离：训练集拟合，测试集应用
4. ✅ 扩展变体：14 组 SOTA 方法 × 3 seeds
5. ✅ 可视化增强：逐 Family 诊断图 + 时序误差分析
6. ⚠️ 数据对齐逻辑：100% 保留，严禁修改
"""

import os
import copy
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

# ====== 依赖 util/模型（保持不变） ======
from physio_util import (
    set_seed, export_predictions_longtable, export_metrics_grid,
    write_summary_txt, heatmap, parity_scatter, residual_hist, save_manifest,
    metrics, transform_for_display, FAMILIES,
    excel_to_physics_dataset,
    load_new_excel_as_sparse_morph, build_sparse_batch
)
from phys_model import TemporalRegressor, PhysicsSeqPredictor


# ========================== 配置（增强版） ==========================
class Cfg:
    # 数据路径（不变）
    old_excel = r"D:\data\pycharm\bosch\case.xlsx"
    new_excel = r"D:\data\pycharm\bosch\Bosch.xlsx"
    save_root = "./runs_stageC_enhanced"

    # 预训练权重
    phys_ckpt_F = "./runs_phys_split/F_Flux/phys_best.pth"
    phys_ckpt_I = "./runs_phys_split/Ion_Flux/phys_best.pth"
    morph_ckpt = "./runs_morph_old/morph_best_overall.pth"

    # 数据划分（新增 ★）
    test_size = 0.15
    val_size = 0.15
    split_random_state = 42

    # 训练参数
    seed = 42
    seeds = [42, 43, 44]
    max_epochs = 2000  # 24h 预算下平衡
    batch_clip = 1.0
    lr_morph = 5e-4
    lr_phys = 1e-40
    wd_morph = 5e-2
    wd_phys = 1e-2
    lr_calib = 5e-4
    wd_calib = 0.0

    # 损失函数
    loss_delta = 1.0
    loss_smooth_weight = 1e-2
    mono_zmin_weight = 5e-3

    # Ion 相关
    ion_affine_default = {"a": 1.0, "b": 0.0, "c": 0.0}
    ion_learnable_lr = 5e-5
    ion_learnable_wd = 1e-6

    # 展示空间（不变）
    unit_scale = 1
    flip_sign = False
    clip_nonneg = False
    min_display_value = 0.0
    family_sign = np.array([-1, +1, +1, +1, +1, +1], dtype=np.float32)
    sheet_name = "case"
    HEIGHT_FAMILY = "h1"

    # 评估
    use_smape = True
    mape_eps_nm = 0.001
    use_ema = True
    ema_decay = 0.999
    early_stop = True
    early_stop_patience = 25

    # 变体列表（扩展到 14 组 SOTA 方法）
    variants = [
        # === Baseline 组 ===
        dict(
            name="baseline_simple",
            use_adapter=False, use_derived=False, ion_gate="use",
            per_family_head=False, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=False, freeze_phys=True, stagewise_unfreeze=False
        ),
        # === Adapter 系列 ===
        dict(
            name="adapter_only",
            use_adapter=True, use_derived=False, ion_gate="use",
            per_family_head=False, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        dict(
            name="adapter_derived",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=False, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        # === Per-Family Head 系列 ===
        dict(
            name="adapter_derived_head",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        # === 异方差 + 任务不确定性 ===
        dict(
            name="adapter_derived_head_hetero",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=True, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        dict(
            name="adapter_derived_head_hetero_taskuncert",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=True, task_uncertainty=True,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        # === Ion Gate 系列 ===
        dict(
            name="adapter_derived_head_iongate",
            use_adapter=True, use_derived=True, ion_gate="gate",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        dict(
            name="adapter_derived_head_iongate_hetero",
            use_adapter=True, use_derived=True, ion_gate="gate",
            per_family_head=True, hetero=True, task_uncertainty=True,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        # === 后校准策略对比 ===
        dict(
            name="adapter_derived_head_postcalib_kt",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_kt",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        dict(
            name="adapter_derived_head_timeconv",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="time_conv", post_calib="time_conv",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        dict(
            name="adapter_derived_head_hybrid_calib",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="hybrid", post_calib="per_k",
            learnable_ion=True, freeze_phys=True, stagewise_unfreeze=False
        ),
        # === 解冻物理编码器 ===
        dict(
            name="adapter_derived_head_unfreeze_phys",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel", post_calib="per_k",
            learnable_ion=True, freeze_phys=False, stagewise_unfreeze=False
        ),
        dict(
            name="adapter_derived_head_stagewise_unfreeze",
            use_adapter=True, use_derived=True, ion_gate="gate",
            per_family_head=True, hetero=True, task_uncertainty=True,
            calib_head="hybrid", post_calib="per_k",
            learnable_ion=True, freeze_phys=False, stagewise_unfreeze=True
        ),
        # === 最强配置（Kitchen Sink）===
        dict(
            name="kitchen_sink_all_features",
            use_adapter=True, use_derived=True, ion_gate="gate",
            per_family_head=True, hetero=True, task_uncertainty=True,
            calib_head="hybrid", post_calib="per_k",
            learnable_ion=True, freeze_phys=False, stagewise_unfreeze=True
        ),
    ]


# ========================== 工具函数（保留 + 增强） ==========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_cpu_np_grid(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _safe_load(path, map_location="cpu"):
    """安全加载 checkpoint（显式注册 numpy reconstruct + 三段回退）"""
    import os as _os
    import numpy as _np
    import torch as _t
    from numpy.core import multiarray as _multi
    from torch.serialization import safe_globals, add_safe_globals

    if not _os.path.exists(path):
        raise FileNotFoundError(path)

    # 1) 全局白名单（对后续 torch.load 生效）
    from numpy.core import multiarray as _multi
    from torch.serialization import safe_globals, add_safe_globals

    # >>> 新增：拿到 dtype 类的“类型”
    _dtype_type = type(_np.dtype("float32"))

    try:
        add_safe_globals([_multi._reconstruct, _np.generic, _np.dtype, _np.ndarray, _dtype_type])
    except Exception:
        pass

    _allowed = [_multi._reconstruct, _np.generic, _np.dtype, _np.ndarray, _dtype_type]
    # 3) 三段回退
    try:
        return _t.load(path, map_location=map_location, weights_only=True)
    except Exception as e1:
        print(f"[safe_load] weights_only=True failed: {e1}")

    try:
        with safe_globals(_allowed):
            return _t.load(path, map_location=map_location, weights_only=True)
    except Exception as e2:
        print(f"[safe_load] safe_globals failed: {e2}")

    try:
        with safe_globals(_allowed):
            return _t.load(path, map_location=map_location, weights_only=False)
    except Exception as e3:
        print(f"[safe_load] all methods failed: {e3}")

    raise RuntimeError(f"Failed to load: {path}")



# ===== Metrics 兼容层（保留）=====
@torch.no_grad()
def _compute_smape_grid(y_pred_disp, y_true_disp, mask, eps=1e-8):
    err = (y_pred_disp - y_true_disp).abs()
    den = (y_pred_disp.abs() + y_true_disp.abs()).clamp_min(eps)
    smape_elem = 100.0 * 2.0 * err / den
    m = mask.float()
    sumE = (smape_elem * m).sum(dim=0)
    cnt = m.sum(dim=0).clamp_min(1.0)
    return sumE / cnt


@torch.no_grad()
def _metrics_compat(y_pred_disp, y_true_disp, mask, *, use_smape, mape_eps_nm):
    try:
        return metrics(y_pred_disp, y_true_disp, mask, use_smape=use_smape, mape_eps=mape_eps_nm)
    except TypeError:
        mts = metrics(y_pred_disp, y_true_disp, mask)
        if use_smape:
            mts["SMAPE"] = _compute_smape_grid(y_pred_disp, y_true_disp, mask)
        return mts


# ========================== 数据加载（100% 保留对齐逻辑） ==========================
def _norm_col(x, mean, std):
    """⚠️ 数据对齐核心函数 - 严禁修改"""
    x = (x - mean) / (std + 1e-8)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def load_fallback_sparse(merged_excel_path, norm_mean, norm_std, time_values, families=FAMILIES):
    """⚠️ 数据对齐核心函数 - 100% 保留，仅增加返回 sample_ids"""
    import pandas as pd
    if not os.path.exists(merged_excel_path):
        raise FileNotFoundError(merged_excel_path)
    try:
        df = pd.read_excel(merged_excel_path, sheet_name="merged_for_training")
    except Exception:
        df = pd.read_excel(merged_excel_path)

    # === 以下逻辑 100% 保留 ===
    s8_cols = [c for c in df.columns if str(c).lower().startswith("s") and str(c)[1:].isdigit()]
    s8_cols = sorted(s8_cols, key=lambda s: int(str(s)[1:]))[:8]
    s8_raw = df[s8_cols].astype(float).values
    s8 = _norm_col(s8_raw, norm_mean, norm_std).astype(np.float32)

    fams = list(families)
    if "zmin" not in fams: fams = ["zmin"] + fams
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
            if c in df.columns: return c
        return None

    def _fill(k_idx, cands, times, negate=False):
        for n in times:
            col = _pick(df, cands(n))
            if col is None: continue
            vals = _nm_to_um(df[col].values)
            if negate: vals = -vals
            ti = t_idx(n)
            v_t = torch.tensor(vals, dtype=torch.float32)
            ok = torch.isfinite(v_t)
            if ok.any():
                y_sparse[:, k_idx, ti][ok] = v_t[ok]
                m_sparse[:, k_idx, ti][ok] = True

    if "zmin" in fams and "zmin_10" in df.columns:
        k = fam2idx["zmin"]
        vals = _nm_to_um(df["zmin_10"].values) * (-1.0)
        ti = t_idx(10)
        v_t = torch.tensor(vals, dtype=torch.float32)
        ok = torch.isfinite(v_t)
        if ok.any():
            y_sparse[:, k, ti][ok] = v_t[ok]
            m_sparse[:, k, ti][ok] = True

    def _cands_h(n):
        return [f"h1_{n}", f"h1{n}", f"h{n}", f"h_{n}", f"{n}thscallopheight", f"第{n}个scallop高度",
                f"第{n}个scallop高度(nm)"]

    def _cands_d(n):
        return [f"d1_{n}", f"d1{n}", f"d{n}", f"d_{n}", f"{n}thscallopdepth", f"第{n}个scallop深度",
                f"第{n}个scallop深度(nm)"]

    def _cands_w(n):
        return [f"w{n}", f"w_{n}", f"W{n}", f"W {n}", f"{n}thscallopwidth", f"第{n}个scallop宽度",
                f"第{n}个scallop宽度(nm)"]

    if "h1" in fams: _fill(fam2idx["h1"], _cands_h, [3, 5, 9], negate=False)
    if "d1" in fams: _fill(fam2idx["d1"], _cands_d, [3, 5, 9], negate=False)
    if "w" in fams: _fill(fam2idx["w"], _cands_w, [1, 3, 5, 9], negate=False)

    s8 = torch.tensor(s8, dtype=torch.float32)
    tvals = torch.tensor(np.array(time_values, dtype=np.float32), dtype=torch.float32)

    # === 新增：返回 sample_ids 用于划分 ===
    sample_ids = np.arange(B)
    return s8, y_sparse, m_sparse, tvals, fams, sample_ids


def load_data_with_split(new_excel, norm_mean, norm_std, time_values, families, test_size, val_size, seed):
    """
    ★ 新增：在对齐逻辑外层包装划分
    ⚠️ 内部对齐逻辑 100% 不变
    """
    # 1. 加载数据（对齐逻辑保持不变）
    try:
        recs = load_new_excel_as_sparse_morph(new_excel, height_family=Cfg.HEIGHT_FAMILY)
        s8, y_sparse, m_sparse, tvals = build_sparse_batch(
            recs, norm_mean, norm_std, time_values
        )
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

    # 3. 重新计算统计量（仅用训练集）
    s8_train = s8[train_idx]
    # 反标准化到原始域
    s8_train_raw = s8_train * (norm_std + 1e-8) + norm_mean
    train_mean = s8_train_raw.mean(dim=0)
    train_std = s8_train_raw.std(dim=0).clamp_min(1e-8)

    # 全量数据用训练集统计量重新标准化
    s8_full_raw = s8 * (norm_std + 1e-8) + norm_mean
    s8_renorm = (s8_full_raw - train_mean) / train_std

    print(f"[DataSplit] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    return {
        "s8": s8_renorm,
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


# ========================== 模型组件（保留） ==========================
def _infer_arch_from_sd(sd: dict) -> Dict[str, int]:
    """从 state_dict 推断模型架构"""
    if "pos" in sd and hasattr(sd["pos"], "shape"):
        _, T, d_model = sd["pos"].shape
    else:
        d_model = sd["input_proj.weight"].shape[0]
        T = 10
    key_l1 = "encoder.layers.0.linear1.weight"
    dim_ff = sd[key_l1].shape[0] if key_l1 in sd else max(2 * d_model, 256)
    nhead = 8 if d_model % 8 == 0 else 4
    L = 0
    while True:
        prefix = f"encoder.layers.{L}."
        if any(k.startswith(prefix) for k in sd.keys()):
            L += 1
        else:
            break
    num_layers = max(L, 1)
    return dict(T=T, d_model=d_model, nhead=nhead, dim_ff=dim_ff, num_layers=num_layers)


def _get_model_sd(ck):
    return ck["model"] if (isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict)) else ck


def build_phys_from_ckpt(ckpt_F_path, ckpt_I_path, device):
    """加载预训练物理模型"""
    ckf = _safe_load(ckpt_F_path, map_location="cpu")
    cki = _safe_load(ckpt_I_path, map_location="cpu")
    sd_F = _get_model_sd(ckf);
    arch_F = _infer_arch_from_sd(sd_F)
    sd_I = _get_model_sd(cki);
    arch_I = _infer_arch_from_sd(sd_I)
    pf = PhysicsSeqPredictor(**arch_F).to(device)
    pi = PhysicsSeqPredictor(**arch_I).to(device)
    pf.load_state_dict(sd_F, strict=False)
    pi.load_state_dict(sd_I, strict=False)
    ion_aff = cki.get("ion_affine", copy.deepcopy(Cfg.ion_affine_default)) if isinstance(cki, dict) else copy.deepcopy(
        Cfg.ion_affine_default)
    return pf, pi, ion_aff


# ========================== 校准头（保留） ==========================
class CalibAffinePerChannel(nn.Module):
    def __init__(self, K, init_alpha=1.0, init_beta=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((K,), float(init_alpha)))
        self.beta = nn.Parameter(torch.full((K,), float(init_beta)))

    def forward(self, y):
        return self.alpha.view(1, -1, 1) * y + self.beta.view(1, -1, 1)


class CalibTimeConv(nn.Module):
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
    def __init__(self, K, kernel_size=3):
        super().__init__()
        self.aff = CalibAffinePerChannel(K)
        self.tcv = CalibTimeConv(K, kernel_size=kernel_size)

    def forward(self, y):
        return self.tcv(self.aff(y))


def build_calib_head(kind: str, K: int) -> nn.Module:
    if kind == "affine_per_channel":
        return CalibAffinePerChannel(K)
    elif kind == "time_conv":
        return CalibTimeConv(K)
    elif kind == "hybrid":
        return CalibHybrid(K)
    else:
        raise ValueError(f"Unknown calib_head: {kind}")


# ========================== 损失函数（保留） ==========================
def masked_huber_with_channel_norm(y_pred, y_true, mask, delta=1.0, smooth_weight=1e-2, mono_penalty=None):
    """稳健 Huber 损失 + 逐通道标准化"""
    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=-1e6)
    B, K, T = y_pred.shape
    device = y_pred.device
    finite_mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    eff_mask = (mask.bool() & finite_mask).float()

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
                    if not torch.isfinite(m): m = torch.tensor(0.0, device=device)
                    if not torch.isfinite(s): s = torch.tensor(1.0, device=device)
                    mean_k[k] = m
                    std_k[k] = s.clamp_min(1.0)

    y_true_n = torch.nan_to_num((y_true - mean_k.view(1, K, 1)) / std_k.view(1, K, 1), nan=0.0, posinf=1e6, neginf=-1e6)
    y_pred_n = torch.nan_to_num((y_pred - mean_k.view(1, K, 1)) / std_k.view(1, K, 1), nan=0.0, posinf=1e6, neginf=-1e6)

    diff = (y_pred_n - y_true_n)
    absd = diff.abs()
    huber = torch.where(absd <= delta, 0.5 * diff * diff, delta * (absd - 0.5 * delta))
    denom_k = eff_mask.sum(dim=(0, 2)).clamp_min(1.0)
    loss_main_per_k = (huber * eff_mask).sum(dim=(0, 2)) / denom_k
    w_k = 1.0 / std_k.clamp_min(1.0)
    loss_main = (loss_main_per_k * w_k).mean()

    if T >= 3:
        d1 = y_pred_n[:, :, 1:] - y_pred_n[:, :, :-1]
        d2 = d1[:, :, 1:] - d1[:, :, :-1]
        loss_smooth = torch.nan_to_num((d2 ** 2).mean(), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        loss_smooth = torch.tensor(0.0, device=device)

    loss_mono = torch.tensor(0.0, device=device)
    if mono_penalty is not None and T >= 2:
        k_idx = mono_penalty.get("k_idx", None)
        w_mono = mono_penalty.get("weight", 0.0)
        if (k_idx is not None) and (w_mono > 0):
            d = y_pred[:, k_idx, 1:] - y_pred[:, k_idx, :-1]
            loss_mono = torch.nn.functional.relu(-d).mean() * w_mono

    loss = loss_main + smooth_weight * loss_smooth + loss_mono
    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=device)
    return loss, {"loss_main": loss_main.detach(), "loss_smooth": loss_smooth.detach(), "loss_mono": loss_mono.detach()}


def hetero_nll(y_mu, y_logvar, y_true, mask, task_logvars=None):
    """异方差负对数似然损失"""
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=-1e6)
    m = (mask.bool() & torch.isfinite(y_true)).float()
    if task_logvars is not None:
        y_logvar = y_logvar + task_logvars.view(1, -1, 1)
    inv_var = torch.exp(-y_logvar).clamp_max(1e6)
    nll = 0.5 * ((y_mu - y_true) ** 2 * inv_var + y_logvar)
    nll = (nll * m).sum(dim=(0, 2)) / m.sum(dim=(0, 2)).clamp_min(1.0)
    return nll.mean()


# ========================== Ion 反变换（保留） ==========================
class IonInverseTransform(nn.Module):
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


# ========================== EMA & EarlyStopping ==========================
class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {}
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
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])


class EarlyStopper:
    def __init__(self, patience=20):
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


# ========================== 后验校准（隔离训练/测试）==========================
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


def fit_calibration_params(y_pred_disp, y_true_disp, mask, method="per_k", min_points=12, ridge=1e-6):
    """
    ★ 新增：仅在训练集拟合校准参数
    返回可序列化的 CalibrationParams 对象
    """
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

                if xs.size < max(6, 2):
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
    """
    ★ 新增：应用预拟合的校准参数（测试集使用）
    """
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


class PostCalibTimeConv(nn.Module):
    """Time-conv 后校准（需训练）"""
    def __init__(self, K, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(K, K, kernel_size, padding=pad, groups=K, bias=True)
        with torch.no_grad():
            self.dw.weight.zero_()
            center = pad
            for k in range(K):
                self.dw.weight[k, 0, center] = 1.0
            self.dw.bias.zero_()

    def forward(self, y):
        return self.dw(y)


# ========================== 接口增强模块（保留） ==========================
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
    """Ion 门控机制"""
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
    """逐 Family 输出头"""
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


# ========================== 物理前向传播（保留） ==========================
def phys_forward_raw(static_8, tvals, phys_F, phys_I, ion_transform, allow_grad=False):
    """物理模型前向传播"""
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
    """物理特征增强流水线"""
    F = phys_raw[:, 0:1, :]
    I = phys_raw[:, 1:2, :]

    # Ion 策略
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


# ========================== 逐 Family 评估报告（新增 ★）==========================
def compute_per_family_metrics(mts: Dict, m_sparse: torch.Tensor, fams: List[str]):
    """
    ★ 新增：计算逐 Family 指标 + Macro/Micro 汇总

    返回:
        per_family: {family_name: {metric: scalar}}
        macro: {metric: scalar}  # 各 Family 简单平均
        micro: {metric: scalar}  # 按样本数加权平均
        min_family: {metric: family_name}  # 最差 Family
    """
    sup = m_sparse.sum(dim=0).detach().cpu().numpy() > 0  # (K, T)

    per_family = {}
    for k, fam in enumerate(fams):
        per_family[fam] = {}
        for metric_name in ["R2", "MAE", "RMSE", "SMAPE", "MAPE", "MSE"]:
            if metric_name not in mts:
                continue
            grid = mts[metric_name]
            if torch.is_tensor(grid):
                grid = grid.detach().cpu().numpy()

            # 该 Family 的平均值（忽略无监督的时间点）
            fam_vals = grid[k, :]
            fam_sup = sup[k, :]
            if fam_sup.sum() > 0:
                per_family[fam][metric_name] = float(np.nanmean(fam_vals[fam_sup]))
            else:
                per_family[fam][metric_name] = np.nan

    # Macro 平均（公平对待每个 Family）
    macro = {}
    for metric_name in ["R2", "MAE", "RMSE", "SMAPE", "MAPE", "MSE"]:
        vals = [per_family[fam].get(metric_name, np.nan) for fam in fams]
        macro[metric_name] = float(np.nanmean(vals))

    # Micro 平均（按样本数加权）
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

    # 最差 Family（R² 最低 / MAE 最高）
    min_family = {}
    r2_vals = {fam: per_family[fam].get("R2", -np.inf) for fam in fams}
    min_family["R2"] = min(r2_vals, key=r2_vals.get)

    mae_vals = {fam: per_family[fam].get("MAE", np.inf) for fam in fams}
    min_family["MAE"] = max(mae_vals, key=mae_vals.get)

    return per_family, macro, micro, min_family


def print_per_family_report(per_family, macro, micro, min_family, fams, title="Evaluation"):
    """★ 新增：打印逐 Family 报告"""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}")

    # 逐 Family
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
    print(f"{'Macro Avg':<10} {macro.get('R2', np.nan):>8.4f} {macro.get('MAE', np.nan):>8.2f} "
          f"{macro.get('RMSE', np.nan):>8.2f} {macro.get('SMAPE', macro.get('MAPE', np.nan)):>12.2f}")
    print(f"{'Micro Avg':<10} {micro.get('R2', np.nan):>8.4f} {micro.get('MAE', np.nan):>8.2f} "
          f"{micro.get('RMSE', np.nan):>8.2f} {micro.get('SMAPE', micro.get('MAPE', np.nan)):>12.2f}")
    print(f"{'Min Family':<10} {min_family['R2']:<8} (R²={per_family[min_family['R2']].get('R2', np.nan):.4f})")
    print(f"{'=' * 80}\n")


# ========================== 可视化增强（新增 ★）==========================
def plot_per_family_diagnostics(y_pred, y_true, mask, fams, T_values, save_dir, title_prefix=""):
    """★ 新增：逐 Family 诊断图（parity + residual）"""
    ensure_dir(save_dir)
    import matplotlib.pyplot as plt

    yp = y_pred.detach().cpu().numpy()
    yt = y_true.detach().cpu().numpy()
    mk = mask.detach().cpu().numpy()

    K = len(fams)
    fig, axes = plt.subplots(2, K, figsize=(4 * K, 8))
    if K == 1:
        axes = axes.reshape(2, 1)

    for k, fam in enumerate(fams):
        valid = mk[:, k, :].flatten()
        yp_k = yp[:, k, :].flatten()[valid]
        yt_k = yt[:, k, :].flatten()[valid]

        if len(yp_k) == 0:
            continue

        # 上排：Parity
        ax1 = axes[0, k]
        ax1.scatter(yt_k, yp_k, alpha=0.5, s=10)

        # 回归线
        if len(yp_k) > 1:
            from scipy.stats import linregress
            slope, intercept, r, *_ = linregress(yt_k, yp_k)
            x_line = np.linspace(yt_k.min(), yt_k.max(), 100)
            ax1.plot(x_line, slope * x_line + intercept, 'r-',
                     label=f'y={slope:.2f}x+{intercept:.1f}\nR²={r ** 2:.3f}')

        ax1.plot([yt_k.min(), yt_k.max()], [yt_k.min(), yt_k.max()],
                 'k--', label='Ideal')
        ax1.set_xlabel('True')
        ax1.set_ylabel('Pred')
        ax1.set_title(f'{fam}')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # 下排：Residual
        ax2 = axes[1, k]
        residuals = yp_k - yt_k
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'μ={residuals.mean():.2f}, σ={residuals.std():.2f}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title_prefix}_family_diagnostics.png"), dpi=150)
    plt.close()


def plot_temporal_error(y_pred, y_true, mask, fams, T_values, save_dir, title_prefix=""):
    """★ 新增：时序误差曲线（每个 Family）"""
    ensure_dir(save_dir)
    import matplotlib.pyplot as plt

    yp = y_pred.detach().cpu().numpy()
    yt = y_true.detach().cpu().numpy()
    mk = mask.detach().cpu().numpy().astype(bool)

    K = len(fams)
    T = len(T_values)

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

        ax.plot(T_values, mae_t, marker='o', label=fam)
        ax.set_xlabel('Time')
        ax.set_ylabel('MAE')
        ax.set_title(f'{fam} - Temporal MAE')
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title_prefix}_temporal_error.png"), dpi=150)
    plt.close()


# ========================== 训练流程（修缮版 ★）==========================
def variant_training_pipeline_v2(dev, meta_old, data_dict, variant: Dict, save_dir: str, seed: int):
    """
    ★ 修缮版训练流程
    关键修改：
    1. 使用 data_dict 中的 train/val/test 划分
    2. 训练仅用 train，验证用 val，最终测试用 test
    3. 后校准仅在 train 上拟合，test 上应用
    4. 逐 Family 评估 + Macro/Micro 报告
    """
    set_seed(seed)
    ensure_dir(save_dir)
    torch.backends.cudnn.benchmark = True

    # === 解包数据 ===
    s8_full = data_dict["s8"].to(dev)
    y_full = data_dict["y_sparse"].to(dev)
    m_full = data_dict["m_sparse"].to(dev)
    tvals_full = data_dict["tvals"].to(dev)
    fams = data_dict["families"]

    train_idx = data_dict["splits"]["train"]
    val_idx = data_dict["splits"]["val"]
    test_idx = data_dict["splits"]["test"]

    # 提取子集
    s8_train = s8_full[train_idx]
    y_train = y_full[train_idx]
    m_train = m_full[train_idx]
    tvals_train = tvals_full if tvals_full.dim() == 1 else tvals_full[train_idx]

    s8_val = s8_full[val_idx]
    y_val = y_full[val_idx]
    m_val = m_full[val_idx]
    tvals_val = tvals_full if tvals_full.dim() == 1 else tvals_full[val_idx]

    s8_test = s8_full[test_idx]
    y_test = y_full[test_idx]
    m_test = m_full[test_idx]
    tvals_test = tvals_full if tvals_full.dim() == 1 else tvals_full[test_idx]

    # === 模型初始化 ===
    phys_F, phys_I, ion_aff_init = build_phys_from_ckpt(Cfg.phys_ckpt_F, Cfg.phys_ckpt_I, dev)
    ion_tr = IonInverseTransform(ion_aff_init, learnable=variant.get("learnable_ion", False)).to(dev)

    morph = TemporalRegressor(K=len(fams)).to(dev)
    if os.path.exists(Cfg.morph_ckpt):
        try:
            ck = _safe_load(Cfg.morph_ckpt, map_location="cpu")
            sd = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck
            morph.load_state_dict(sd, strict=False)
            print(f"[{variant['name']}|seed{seed}] Loaded morph ckpt.")
        except Exception as e:
            print(f"[{variant['name']}|seed{seed}] morph ckpt load failed: {e}")

    calib = build_calib_head(variant["calib_head"], K=len(fams)).to(dev)

    # 输出头
    per_family_head = PerFamilyHead(K=len(fams)).to(dev) if variant.get("per_family_head", False) else nn.Identity().to(dev)
    hetero_head = HeteroHead(K=len(fams), per_family=variant.get("per_family_head", False)).to(dev) if variant.get("hetero", False) else None
    task_logvars = nn.Parameter(torch.zeros(len(fams), device=dev)) if variant.get("task_uncertainty", False) else None

    # 接口增强
    adapters = {
        "adapter": PhysAdapter(2, k=3).to(dev),
        "reducer": PhysFeaReducer().to(dev),
        "gate": IonGate(k=5).to(dev),
    }

    # === 优化器 ===
    params = [
        {"params": [p for p in morph.parameters() if p.requires_grad], "lr": Cfg.lr_morph, "weight_decay": Cfg.wd_morph},
        {"params": [p for p in calib.parameters() if p.requires_grad], "lr": Cfg.lr_calib, "weight_decay": Cfg.wd_calib},
    ]
    if ion_tr.learnable:
        params.append({"params": [p for p in ion_tr.parameters() if p.requires_grad], "lr": Cfg.ion_learnable_lr, "weight_decay": Cfg.ion_learnable_wd})
    if not variant.get("freeze_phys", True):
        params += [
            {"params": [p for p in phys_F.parameters() if p.requires_grad], "lr": Cfg.lr_phys, "weight_decay": Cfg.wd_phys},
            {"params": [p for p in phys_I.parameters() if p.requires_grad], "lr": Cfg.lr_phys, "weight_decay": Cfg.wd_phys},
        ]
    if variant.get("per_family_head", False):
        params.append({"params": per_family_head.parameters(), "lr": Cfg.lr_morph, "weight_decay": 0.0})
    if variant.get("hetero", False):
        params.append({"params": hetero_head.parameters(), "lr": Cfg.lr_morph, "weight_decay": 0.0})
    if variant.get("task_uncertainty", False):
        params.append({"params": [task_logvars], "lr": 1e-3, "weight_decay": 0.0})
    if variant.get("use_adapter", False) or variant.get("use_derived", False) or variant.get("ion_gate", "use") == "gate":
        params.append({
            "params": list(adapters["adapter"].parameters()) + list(adapters["reducer"].parameters()) + list(adapters["gate"].parameters()),
            "lr": 5e-4, "weight_decay": 0.0
        })

    opt = torch.optim.AdamW(params)
    ema = EMA(morph, decay=Cfg.ema_decay) if Cfg.use_ema else None
    stopper = EarlyStopper(Cfg.early_stop_patience) if Cfg.early_stop else None

    # 冻结/解冻
    def set_phys_trainable(flag: bool):
        for p in phys_F.parameters(): p.requires_grad = flag
        for p in phys_I.parameters(): p.requires_grad = flag
        phys_F.train(flag)
        phys_I.train(flag)

    if variant.get("freeze_phys", True):
        set_phys_trainable(False)
    else:
        if variant.get("stagewise_unfreeze", False):
            set_phys_trainable(False)
        else:
            set_phys_trainable(True)

    morph.train()
    calib.train()
    if isinstance(per_family_head, nn.Module): per_family_head.train()
    if hetero_head is not None: hetero_head.train()

    best_snapshot = None
    best_val_macro_r2 = -1e9
    val_history = []

    # === 训练循环 ===
    for e in range(1, Cfg.max_epochs + 1):
        # 阶段式解冻
        if (not variant.get("freeze_phys", True)) and variant.get("stagewise_unfreeze", False) and e == (Cfg.max_epochs // 2):
            for p in phys_F.parameters(): p.requires_grad = False
            for p in phys_I.parameters(): p.requires_grad = False
            for n, m in phys_F.named_modules():
                if n.startswith("encoder.layers.") and n.endswith(".3"):
                    for p in m.parameters(): p.requires_grad = True
            for n, m in phys_I.named_modules():
                if n.startswith("encoder.layers.") and n.endswith(".3"):
                    for p in m.parameters(): p.requires_grad = True
            phys_F.train(True)
            phys_I.train(True)
            print(f"[{variant['name']}|seed{seed}] Stagewise unfreeze at epoch {e}.")

        # === 训练步（仅用 train 集）===
        opt.zero_grad()

        phys_pred = phys_forward_raw(s8_train, tvals_train, phys_F, phys_I, ion_tr,
                                     allow_grad=any(p.requires_grad for p in phys_F.parameters()))
        phys_in = phys_interface_pipeline(phys_pred, variant, adapters)

        y_core = morph(s8_train, phys_in, tvals_train)
        y_core = calib(torch.nan_to_num(y_core, nan=0.0, posinf=1e6, neginf=-1e6))

        if variant.get("hetero", False):
            mu, logv = hetero_head(y_core)
            nll = hetero_nll(mu, logv, y_train, m_train, task_logvars=task_logvars)
            huber_s, items = masked_huber_with_channel_norm(
                mu, y_train, m_train, delta=Cfg.loss_delta,
                smooth_weight=Cfg.loss_smooth_weight,
                mono_penalty={'k_idx': 0, 'weight': Cfg.mono_zmin_weight}
            )
            loss = nll + 0.1 * huber_s
            y_for_eval_train = mu
        else:
            y_head = per_family_head(y_core) if variant.get("per_family_head", False) else y_core
            loss, items = masked_huber_with_channel_norm(
                y_pred=y_head, y_true=y_train, mask=m_train,
                delta=Cfg.loss_delta, smooth_weight=Cfg.loss_smooth_weight,
                mono_penalty={'k_idx': 0, 'weight': Cfg.mono_zmin_weight}
            )
            y_for_eval_train = y_head

        if ion_tr.learnable:
            loss = loss + 1e-4 * ion_tr.reg()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(morph.parameters(), max_norm=Cfg.batch_clip)
        opt.step()
        if ema is not None: ema.update(morph)

        # === 验证步（每 5 轮）===
        if e % 5 == 0 or e == 1 or e == Cfg.max_epochs:
            with torch.no_grad():
                if ema is not None:
                    morph_eval = copy.deepcopy(morph).to(dev)
                    ema.apply_to(morph_eval)
                else:
                    morph_eval = morph

                phys_pred_val = phys_forward_raw(s8_val, tvals_val, phys_F, phys_I, ion_tr, allow_grad=False)
                phys_in_val = phys_interface_pipeline(phys_pred_val, variant, adapters)
                yp_val = morph_eval(s8_val, phys_in_val, tvals_val)
                yp_val = calib(yp_val)

                if variant.get("hetero", False):
                    mu_val, _ = hetero_head(yp_val)
                    yp_val = mu_val
                elif variant.get("per_family_head", False):
                    yp_val = per_family_head(yp_val)

                # 展示域评估
                fam_sign = torch.tensor(Cfg.family_sign, dtype=torch.float32, device=dev)
                yhat_val_disp, ytrue_val_disp = transform_for_display(
                    yp_val, y_val,
                    family_sign=fam_sign, unit_scale=Cfg.unit_scale,
                    flip_sign=Cfg.flip_sign, clip_nonneg=Cfg.clip_nonneg,
                    min_display_value=Cfg.min_display_value
                )

                mts_val = _metrics_compat(yhat_val_disp, ytrue_val_disp, m_val,
                                         use_smape=Cfg.use_smape, mape_eps_nm=Cfg.mape_eps_nm)

                # 逐 Family 评估
                per_fam_val, macro_val, micro_val, min_fam_val = compute_per_family_metrics(mts_val, m_val, fams)
                val_macro_r2 = macro_val.get("R2", -1e9)

                val_history.append({
                    "epoch": e,
                    "train_loss": float(loss.item()),
                    "val_macro_r2": val_macro_r2,
                    "val_min_r2": per_fam_val[min_fam_val["R2"]].get("R2", -1e9)
                })

                print(f"[{variant['name']}|seed{seed}][{e}/{Cfg.max_epochs}] "
                      f"train_loss={loss.item():.4f} | "
                      f"val_Macro-R²={val_macro_r2:.4f} | "
                      f"val_Min-R²={per_fam_val[min_fam_val['R2']].get('R2', -1e9):.4f} ({min_fam_val['R2']})")

                # 早停（基于 Macro-R²）
                if stopper is not None:
                    keep = stopper.step(-val_macro_r2)  # 负号因为 stopper 找最小值
                    if keep:
                        best_val_macro_r2 = val_macro_r2
                        best_snapshot = {
                            "epoch": e,
                            "morph": copy.deepcopy(morph.state_dict()),
                            "calib": copy.deepcopy(calib.state_dict()),
                            "per_head": copy.deepcopy(per_family_head.state_dict()) if isinstance(per_family_head, nn.Module) else None,
                            "hetero": copy.deepcopy(hetero_head.state_dict()) if hetero_head is not None else None,
                            "ion": copy.deepcopy(ion_tr.state_dict()) if ion_tr.learnable else None,
                            "adapters": {
                                "adapter": copy.deepcopy(adapters["adapter"].state_dict()),
                                "reducer": copy.deepcopy(adapters["reducer"].state_dict()),
                                "gate": copy.deepcopy(adapters["gate"].state_dict()),
                            }
                        }
                    else:
                        print(f"[{variant['name']}|seed{seed}] Early stop. Best Val Macro-R²={best_val_macro_r2:.4f}")
                        break

    # === 加载最佳权重 ===
    if best_snapshot is not None:
        morph.load_state_dict(best_snapshot["morph"])
        calib.load_state_dict(best_snapshot["calib"])
        if isinstance(per_family_head, nn.Module) and (best_snapshot["per_head"] is not None):
            per_family_head.load_state_dict(best_snapshot["per_head"])
        if hetero_head is not None and (best_snapshot["hetero"] is not None):
            hetero_head.load_state_dict(best_snapshot["hetero"])
        if ion_tr.learnable and (best_snapshot["ion"] is not None):
            ion_tr.load_state_dict(best_snapshot["ion"])
        adapters["adapter"].load_state_dict(best_snapshot["adapters"]["adapter"])
        adapters["reducer"].load_state_dict(best_snapshot["adapters"]["reducer"])
        adapters["gate"].load_state_dict(best_snapshot["adapters"]["gate"])

    morph.eval()
    calib.eval()
    phys_F.eval()
    phys_I.eval()
    if isinstance(per_family_head, nn.Module): per_family_head.eval()
    if hetero_head is not None: hetero_head.eval()

    # === 训练集预测（用于后校准拟合）===
    with torch.no_grad():
        phys_pred_train = phys_forward_raw(s8_train, tvals_train, phys_F, phys_I, ion_tr, allow_grad=False)
        phys_in_train = phys_interface_pipeline(phys_pred_train, variant, adapters)
        yhat_train = calib(morph(s8_train, phys_in_train, tvals_train))

        if variant.get("hetero", False):
            mu_train, _ = hetero_head(yhat_train)
            yhat_train = mu_train
        elif variant.get("per_family_head", False):
            yhat_train = per_family_head(yhat_train)

        fam_sign = torch.tensor(Cfg.family_sign, dtype=torch.float32, device=dev)
        yhat_train_disp, ytrue_train_disp = transform_for_display(
            yhat_train, y_train,
            family_sign=fam_sign, unit_scale=Cfg.unit_scale,
            flip_sign=Cfg.flip_sign, clip_nonneg=Cfg.clip_nonneg,
            min_display_value=Cfg.min_display_value
        )

        # 拟合校准参数（仅训练集）★
        post_kind = variant["post_calib"]
        if post_kind == "time_conv":
            pc = PostCalibTimeConv(len(fams)).to(dev)
            opt_pc = torch.optim.Adam(pc.parameters(), lr=1e-2, weight_decay=0.0)
            pc.train()

            # 固定输入为常量，不需要梯度；只让 pc 学
            y_in = yhat_train_disp.detach()
            y_t = ytrue_train_disp.detach()
            m = m_train.float().detach()

            with torch.enable_grad():  # ★ 只在这小段开启梯度
                for _ in range(80):
                    opt_pc.zero_grad()
                    yp_pc = pc(y_in)  # (B,K,T)
                    diff = (yp_pc - y_t)
                    loss_pc = (diff.pow(2) * m).sum() / m.sum().clamp_min(1.0)
                    loss_pc.backward()
                    opt_pc.step()

            calib_params = {"method": "time_conv", "model_state": pc.state_dict()}
        else:
            calib_params = fit_calibration_params(yhat_train_disp, ytrue_train_disp, m_train,
                                                  method=post_kind, min_points=12 if post_kind == "per_k" else 6)

    # === 测试集预测（应用校准）★ ===
    with torch.no_grad():
        phys_pred_test = phys_forward_raw(s8_test, tvals_test, phys_F, phys_I, ion_tr, allow_grad=False)
        phys_in_test = phys_interface_pipeline(phys_pred_test, variant, adapters)
        yhat_test = calib(morph(s8_test, phys_in_test, tvals_test))

        if variant.get("hetero", False):
            mu_test, _ = hetero_head(yhat_test)
            yhat_test = mu_test
        elif variant.get("per_family_head", False):
            yhat_test = per_family_head(yhat_test)

        yhat_test_disp, ytrue_test_disp = transform_for_display(
            yhat_test, y_test,
            family_sign=fam_sign, unit_scale=Cfg.unit_scale,
            flip_sign=Cfg.flip_sign, clip_nonneg=Cfg.clip_nonneg,
            min_display_value=Cfg.min_display_value
        )

        # 应用校准（不重新拟合）★
        if post_kind == "time_conv":
            pc_eval = PostCalibTimeConv(len(fams)).to(dev)
            pc_eval.load_state_dict(calib_params["model_state"])
            pc_eval.eval()
            yhat_test_cal = pc_eval(yhat_test_disp).detach()
        else:
            yhat_test_cal = apply_calibration_params(yhat_test_disp, calib_params)

        # 最终测试集指标
        mts_test = _metrics_compat(yhat_test_cal, ytrue_test_disp, m_test,
                                   use_smape=Cfg.use_smape, mape_eps_nm=Cfg.mape_eps_nm)
        mts_test["SMAPE"] = _compute_smape_grid(yhat_test_cal, ytrue_test_disp, m_test)

        # 逐 Family 报告 ★
        per_fam_test, macro_test, micro_test, min_fam_test = compute_per_family_metrics(mts_test, m_test, fams)
        print_per_family_report(per_fam_test, macro_test, micro_test, min_fam_test, fams,
                               title=f"Test Results ({variant['name']}|seed{seed})")

    # === 导出 ===
    T_values = meta_old["time_values"]

    # 1. Predictions
    export_predictions_longtable(
        yhat_test_cal.cpu(), ytrue_test_disp.cpu(), m_test.cpu(), fams, T_values,
        save_dir, filename=f"test_predictions_seed{seed}.xlsx"
    )

    # 2. Metrics Grid
    mts_test_cpu = {k: _to_cpu_np_grid(v) for k, v in mts_test.items()}
    export_metrics_grid(mts_test_cpu, fams, T_values, save_dir, filename=f"test_metrics_seed{seed}.xlsx")

    # 3. Per-Family Summary
    summary_path = os.path.join(save_dir, f"test_summary_seed{seed}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"=== Test Results ({variant['name']}|seed{seed}) ===\n\n")
        f.write(f"{'Family':<10} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'SMAPE/MAPE':>12}\n")
        f.write("-" * 60 + "\n")
        for fam in fams:
            r2 = per_fam_test[fam].get("R2", np.nan)
            mae = per_fam_test[fam].get("MAE", np.nan)
            rmse = per_fam_test[fam].get("RMSE", np.nan)
            smape = per_fam_test[fam].get("SMAPE", per_fam_test[fam].get("MAPE", np.nan))
            f.write(f"{fam:<10} {r2:>8.4f} {mae:>8.2f} {rmse:>8.2f} {smape:>12.2f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Macro Avg':<10} {macro_test.get('R2', np.nan):>8.4f} {macro_test.get('MAE', np.nan):>8.2f} "
                f"{macro_test.get('RMSE', np.nan):>8.2f} {macro_test.get('SMAPE', macro_test.get('MAPE', np.nan)):>12.2f}\n")
        f.write(f"{'Micro Avg':<10} {micro_test.get('R2', np.nan):>8.4f} {micro_test.get('MAE', np.nan):>8.2f} "
                f"{micro_test.get('RMSE', np.nan):>8.2f} {micro_test.get('SMAPE', micro_test.get('MAPE', np.nan)):>12.2f}\n")
        f.write(f"{'Min Family':<10} {min_fam_test['R2']:<8} (R²={per_fam_test[min_fam_test['R2']].get('R2', np.nan):.4f})\n")

    # 4. 可视化
    heatmap(
        mts_test.get("R2", torch.zeros(len(fams), len(T_values))).detach().cpu()
        if torch.is_tensor(mts_test.get("R2", None)) else _to_cpu_np_grid(mts_test.get("R2", None)),
        fams, T_values, f"Test R² ({variant['name']}|seed{seed})",
        os.path.join(save_dir, f"test_r2_seed{seed}.png")
    )

    plot_per_family_diagnostics(yhat_test_cal, ytrue_test_disp, m_test, fams, T_values,
                                save_dir, title_prefix=f"test_seed{seed}")

    plot_temporal_error(yhat_test_cal, ytrue_test_disp, m_test, fams, T_values,
                       save_dir, title_prefix=f"test_seed{seed}")

    # 5. 保存权重
    torch.save(
        {
            "phys_F": phys_F.state_dict(),
            "phys_I": phys_I.state_dict(),
            "ion_affine": dict(a=float(ion_tr.a.data), b=float(ion_tr.b.data), c=float(ion_tr.c.data)),
            "morph": morph.state_dict(),
            "calib_head": calib.state_dict(),
            "per_family_head": (per_family_head.state_dict() if isinstance(per_family_head, nn.Module) else None),
            "hetero_head": (hetero_head.state_dict() if hetero_head is not None else None),
            "task_logvars": (task_logvars.detach().cpu().numpy().tolist() if task_logvars is not None else None),
            "adapters": {
                "adapter": adapters["adapter"].state_dict(),
                "reducer": adapters["reducer"].state_dict(),
                "gate": adapters["gate"].state_dict(),
            },
            "calibration_params": calib_params.to_dict() if hasattr(calib_params, 'to_dict') else calib_params,
            "meta": {
                "families": fams,
                "time_values": T_values,
                "norm_static": meta_old["norm_static"],
                "height_family": Cfg.HEIGHT_FAMILY,
                "variant": variant,
                "seed": seed,
                "test_metrics": {
                    "per_family": per_fam_test,
                    "macro": macro_test,
                    "micro": micro_test,
                    "min_family": min_fam_test
                }
            },
        },
        os.path.join(save_dir, f"final_model_seed{seed}.pth")
    )

    save_manifest(save_dir)

    return {
        "variant": variant["name"],
        "seed": seed,
        "test_metrics": mts_test,
        "per_family": per_fam_test,
        "macro": macro_test,
        "micro": micro_test,
        "min_family": min_fam_test,
        "save_dir": save_dir
    }


# ========================== 主流程 ==========================
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[StageC-Enhanced] device = {dev}")
    set_seed(Cfg.seed)

    # === 加载数据（保持对齐逻辑） ===
    _, meta_old = excel_to_physics_dataset(Cfg.old_excel, sheet_name=Cfg.sheet_name)

    # === 数据划分（新增） ===
    data_dict = load_data_with_split(
        Cfg.new_excel,
        meta_old["norm_static"]["mean"],
        meta_old["norm_static"]["std"],
        meta_old["time_values"],
        FAMILIES,
        test_size=Cfg.test_size,
        val_size=Cfg.val_size,
        seed=Cfg.split_random_state
    )

    # 清洗监督
    y_sparse = data_dict["y_sparse"]
    m_sparse = data_dict["m_sparse"]
    y_sparse = torch.nan_to_num(y_sparse, nan=0.0, posinf=0.0, neginf=0.0)
    m_sparse = m_sparse & torch.isfinite(y_sparse)
    data_dict["y_sparse"] = y_sparse
    data_dict["m_sparse"] = m_sparse

    # 展开 tvals
    if data_dict["tvals"].dim() == 1:
        B = data_dict["s8"].shape[0]
        data_dict["tvals"] = data_dict["tvals"].unsqueeze(0).expand(B, -1).contiguous()

    ensure_dir(Cfg.save_root)
    results = []

    # === 多变体 × 多种子训练 ===
    for variant in Cfg.variants:
        for seed in Cfg.seeds:
            save_dir = os.path.join(Cfg.save_root, f"{variant['name']}_seed{seed}")
            ensure_dir(save_dir)
            print(f"\n{'=' * 80}")
            print(f"Starting: {variant['name']} | Seed: {seed}")
            print(f"{'=' * 80}")
            out = variant_training_pipeline_v2(dev, meta_old, data_dict, variant, save_dir, seed)
            results.append(out)

    # === 汇总对比表 ===
    rows_per_seed = []
    for r in results:
        row = {
            "variant": r["variant"],
            "seed": r["seed"],
            "save_dir": r["save_dir"],
            "Macro_R2": r["macro"].get("R2", np.nan),
            "Macro_MAE": r["macro"].get("MAE", np.nan),
            "Macro_RMSE": r["macro"].get("RMSE", np.nan),
            "Micro_R2": r["micro"].get("R2", np.nan),
            "Micro_MAE": r["micro"].get("MAE", np.nan),
            "Min_R2": r["per_family"][r["min_family"]["R2"]].get("R2", np.nan),
            "Min_Family": r["min_family"]["R2"]
        }
        rows_per_seed.append(row)

    df_seed = pd.DataFrame(rows_per_seed).sort_values(by=["variant", "Macro_R2"], ascending=[True, False], na_position="last")

    # 保存 per-seed 表
    comp_seed_xlsx = os.path.join(Cfg.save_root, "variants_seeds.xlsx")
    with pd.ExcelWriter(comp_seed_xlsx) as w:
        df_seed.to_excel(w, index=False, sheet_name="per_seed")

    # 聚合到 variant（取均值）
    rows_summary = []
    for variant_name in df_seed["variant"].unique():
        sub = df_seed[df_seed["variant"] == variant_name]
        agg = {
            "variant": variant_name,
            "Macro_R2_mean": float(sub["Macro_R2"].mean()),
            "Macro_R2_std": float(sub["Macro_R2"].std()),
            "Macro_MAE_mean": float(sub["Macro_MAE"].mean()),
            "Min_R2_mean": float(sub["Min_R2"].mean()),
            "Min_R2_std": float(sub["Min_R2"].std()),
        }
        rows_summary.append(agg)

    df_summary = pd.DataFrame(rows_summary).sort_values(by=["Macro_R2_mean"], ascending=False, na_position="last")

    # 保存汇总表
    comp_xlsx = os.path.join(Cfg.save_root, "variants_comparison.xlsx")
    with pd.ExcelWriter(comp_xlsx) as w:
        df_summary.to_excel(w, index=False, sheet_name="summary")

    # 生成文本报告
    with open(os.path.join(Cfg.save_root, "summary_comparison.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Stage C Enhanced - Variants Comparison (Test Set)\n")
        f.write("=" * 80 + "\n\n")
        f.write(df_summary.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Ranking Explanation:\n")
        f.write("- Primary: Macro-R² (公平对待每个 Family)\n")
        f.write("- Secondary: Min-R² (最差 Family 的下限)\n")
        f.write("-Tertiary: Macro-MAE (误差幅度)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Per-Seed Details: see variants_seeds.xlsx\n")
        f.write("Individual Reports: see each variant's subdirectory\n")

    print(f"\n{'=' * 80}")
    print(f"[OK] All variants finished. See: {Cfg.save_root}")
    print(f"Per-seed table: {comp_seed_xlsx}")
    print(f"Comparison table: {comp_xlsx}")
    print(f"{'=' * 80}\n")

    # 打印 Top 3
    print("Top 3 Variants (by Macro-R²):")
    for i, row in df_summary.head(3).iterrows():
        print(f"  {i+1}. {row['variant']}: Macro-R²={row['Macro_R2_mean']:.4f}±{row['Macro_R2_std']:.4f}, "
              f"Min-R²={row['Min_R2_mean']:.4f}±{row['Min_R2_std']:.4f}")


# ========================== Quick Check（保留，数据管线不改）==========================
@torch.no_grad()
def _display_space_eval(yp, y_sparse, m_sparse, fam_sign, note: str):
    """快速评估辅助函数"""
    unit_scale_eval = Cfg.unit_scale
    yhat_disp, ytrue_disp = transform_for_display(
        yp, y_sparse,
        family_sign=fam_sign, unit_scale=unit_scale_eval,
        flip_sign=Cfg.flip_sign, clip_nonneg=Cfg.clip_nonneg, min_display_value=Cfg.min_display_value
    )
    mts = _metrics_compat(yhat_disp, ytrue_disp, m_sparse, use_smape=Cfg.use_smape, mape_eps_nm=Cfg.mape_eps_nm)
    sup = m_sparse.sum(dim=0).detach().cpu().numpy() > 0
    out = {}
    for key in ["R2", "RMSE", "MAE", "SMAPE" if Cfg.use_smape else "MAPE"]:
        if key in mts:
            g = mts[key].detach().cpu().numpy() if torch.is_tensor(mts[key]) else mts[key]
            out[f"{key}_mean"] = float(np.nanmean(np.where(sup, g, np.nan)))
    ytd = ytrue_disp.detach().cpu().numpy()
    ypd = yhat_disp.detach().cpu().numpy()
    print(f"[Quick:{note}] "
          f"R2_mean={out.get('R2_mean', float('nan')):.4f}  "
          f"MAE_mean={out.get('MAE_mean', float('nan')):.2f}  "
          f"RMSE_mean={out.get('RMSE_mean', float('nan')):.2f}")
    return out, yhat_disp, ytrue_disp

def _print_family_ranges(y_disp: torch.Tensor, m_sparse: torch.Tensor, fams: List[str], label: str):
    """
    在展示域(μm)下按 family 打印分布范围。
    y_disp: 形状 (B, K, T)
    m_sparse: 形状 (B, K, T) 的监督mask
    fams: family 名称列表
    label: 打印标题，如 'y_true' / 'y_pred(before)' / 'y_pred(calibrated)'
    """
    print(f"\n[Ranges] {label}")
    print(f"{'Family':<10} {'n':>6} {'min':>8} {'p10':>8} {'median':>8} {'p90':>8} {'max':>8} {'mean':>8} {'std':>8}")
    print("-" * 86)
    with torch.no_grad():
        K = len(fams)
        for k in range(K):
            # 只取有监督的点 —— 关键改动：view -> reshape
            mask_k = m_sparse[:, k, :].to(torch.bool).reshape(-1)
            vals_k = y_disp[:, k, :].contiguous().view(-1)[mask_k]
            if vals_k.numel() == 0:
                print(f"{fams[k]:<10} {'0':>6} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
                continue
            v = vals_k.detach().float().cpu()
            n  = v.numel()
            vmin, vmax = torch.min(v), torch.max(v)
            p10, med, p90 = torch.quantile(v, torch.tensor([0.10, 0.50, 0.90]))
            mean, std = torch.mean(v), torch.std(v)
            print(f"{fams[k]:<10} {n:6d} {vmin:8.3f} {p10:8.3f} {med:8.3f} {p90:8.3f} {vmax:8.3f} {mean:8.3f} {std:8.3f}")


def quick_main():
    """
    快速诊断模式（不训练）：
    1. Ion 消融实验
    2. 后验校准上限测试
    """
    set_seed(Cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[StageC-Quick] device = {dev}")

    _, meta_old = excel_to_physics_dataset(Cfg.old_excel, sheet_name=Cfg.sheet_name)
    T_values = meta_old["time_values"]

    # 加载新表数据（保持对齐逻辑）
    try:
        recs = load_new_excel_as_sparse_morph(Cfg.new_excel, height_family=Cfg.HEIGHT_FAMILY)
        s8, y_sparse, m_sparse, tvals = build_sparse_batch(
            recs, meta_old["norm_static"]["mean"], meta_old["norm_static"]["std"], meta_old["time_values"]
        )
        fams = FAMILIES
        print("[Quick] Loaded via util loader.")
    except Exception as e:
        print(f"[Quick] Fallback loader: {e}")
        s8, y_sparse, m_sparse, tvals, fams, _ = load_fallback_sparse(
            Cfg.new_excel, meta_old["norm_static"]["mean"], meta_old["norm_static"]["std"], T_values, families=FAMILIES
        )

    s8 = s8.to(dev)
    y_sparse = y_sparse.to(dev)
    m_sparse = m_sparse.to(dev)
    # —— Quick 模式补丁：把 mask 与 y_true 的有限性相交 ——
    m_sparse = m_sparse & torch.isfinite(y_sparse)
    tvals = tvals.to(dev)
    if tvals.dim() == 1:
        tvals = tvals.unsqueeze(0).expand(s8.size(0), -1).contiguous()

    # 加载模型
    phys_F, phys_I, ion_aff_init = build_phys_from_ckpt(Cfg.phys_ckpt_F, Cfg.phys_ckpt_I, dev)
    ion_tr = IonInverseTransform(ion_aff_init, learnable=False).to(dev)
    morph = TemporalRegressor(K=len(fams)).to(dev)

    if os.path.exists(Cfg.morph_ckpt):
        try:
            ck = _safe_load(Cfg.morph_ckpt, map_location="cpu")
            sd = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck
            morph.load_state_dict(sd, strict=False)
            print("[Quick] Loaded morph ckpt.")
        except Exception as e:
            print(f"[Quick] morph ckpt load failed: {e}")

    fam_sign = torch.tensor(Cfg.family_sign, dtype=torch.float32, device=dev)

    # === Ion 消融实验 ===
    print("\n" + "=" * 80)
    print("Ion Ablation Study")
    print("=" * 80)

    base = phys_forward_raw(s8, tvals, phys_F, phys_I, ion_tr, allow_grad=False)
    # === Sanity check: 物理输出是否有效 ===
    with torch.no_grad():
        Fm, Fs = base[:, 0].mean().item(), base[:, 0].std().item()
        Im, Is = base[:, 1].mean().item(), base[:, 1].std().item()
    print(f"[CHECK] Phys F mean/std = {Fm:.4g}/{Fs:.4g} | Ion mean/std = {Im:.4g}/{Is:.4g}")
    if not np.isfinite([Fm, Fs, Im, Is]).all() or ((abs(Fm) < 1e-8 and Fs < 1e-8) or (abs(Im) < 1e-8 and Is < 1e-8)):
        print("[WARN] Physics outputs look degenerate. "
              "请检查 Cfg.phys_ckpt_F / Cfg.phys_ckpt_I 路径与格式（建议用纯 state_dict 保存）。")
    F = base[:, 0:1, :]
    I = base[:, 1:2, :]

    ion_results = {}
    for mode in ["use", "zero", "const", "smooth"]:
        if mode == "zero":
            I2 = torch.zeros_like(I)
        elif mode == "const":
            I2 = torch.nan_to_num(I, nan=0.0).mean(dim=0, keepdim=True).expand_as(I)
        elif mode == "smooth":
            k = 5
            pad = (k - 1) // 2
            w = torch.ones(1, 1, k, device=I.device) / k
            I2 = nn.functional.conv1d(nn.functional.pad(I, (pad, pad), mode="replicate"), w)
        else:
            I2 = I

        yp = morph(s8, torch.cat([F, I2], dim=1), tvals)
        stat, _, _ = _display_space_eval(yp, y_sparse, m_sparse, fam_sign, f"ion-{mode}")
        ion_results[mode] = stat

    # === 后验校准上限 ===
    print("\n" + "=" * 80)
    print("Posterior Calibration Ceiling")
    print("=" * 80)

    yp_base = morph(s8, base, tvals)
    stat_before, yhat_disp, ytrue_disp = _display_space_eval(yp_base, y_sparse, m_sparse, fam_sign, "before-postcalib")
    # 打印：真实值与“事后校准前”的预测分布
    _print_family_ranges(ytrue_disp, m_sparse, fams, label="y_true(μm)")
    _print_family_ranges(yhat_disp, m_sparse, fams, label="y_pred(before, μm)")

    with torch.no_grad():
        yp_disp, yt_disp = transform_for_display(yp_base, y_sparse, family_sign=fam_sign,
                                                 unit_scale=Cfg.unit_scale, flip_sign=Cfg.flip_sign,
                                                 clip_nonneg=Cfg.clip_nonneg, min_display_value=Cfg.min_display_value)
    q = torch.quantile(yt_disp[torch.isfinite(yt_disp)], torch.tensor([0.1, 0.5, 0.9], device=yt_disp.device))
    print(f"[UNIT] y_true(μm) quantiles: 10%={q[0]:.3f}, 50%={q[1]:.3f}, 90%={q[2]:.3f}")

    # per_kt 校准
    calib_params_kt = fit_calibration_params(yhat_disp, ytrue_disp, m_sparse, method="per_kt", min_points=4)
    yhat_cal_kt = apply_calibration_params(yhat_disp, calib_params_kt)
    mts_kt = _metrics_compat(yhat_cal_kt, ytrue_disp, m_sparse, use_smape=Cfg.use_smape, mape_eps_nm=Cfg.mape_eps_nm)
    sup = m_sparse.sum(dim=0).detach().cpu().numpy() > 0
    r2_kt = float(np.nanmean(_to_cpu_np_grid(mts_kt["R2"])[sup]))
    print(f"[Quick:postcalib-per_kt] R2_mean={r2_kt:.4f}")
    _print_family_ranges(yhat_cal_kt, m_sparse, fams, label="y_pred(calibrated per_kt, μm)")

    # per_k 校准
    calib_params_k = fit_calibration_params(yhat_disp, ytrue_disp, m_sparse, method="per_k", min_points=12)
    yhat_cal_k = apply_calibration_params(yhat_disp, calib_params_k)
    mts_k = _metrics_compat(yhat_cal_k, ytrue_disp, m_sparse, use_smape=Cfg.use_smape, mape_eps_nm=Cfg.mape_eps_nm)
    r2_k  = float(np.nanmean(_to_cpu_np_grid(mts_k["R2"])[sup]))
    print(f"[Quick:postcalib-per_k] R2_mean={r2_k:.4f}")
    _print_family_ranges(yhat_cal_k, m_sparse, fams, label="y_pred(calibrated per_k, μm)")

    # 保存结果
    save_dir = os.path.join(Cfg.save_root, "_quick_check")
    ensure_dir(save_dir)

    with open(os.path.join(save_dir, "ion_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(ion_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_dir, "posterior_ceiling.json"), "w", encoding="utf-8") as f:
        json.dump({
            "before": stat_before,
            "per_kt": {"R2_mean": r2_kt},
            "per_k": {"R2_mean": r2_k}
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[Quick] Done. See: {save_dir}")


# ========================== 入口 ==========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage C Enhanced - Multi-Variant SOTA Comparison")
    parser.add_argument("--quick-check", action="store_true", default=False,
                        help="Run quick diagnostic checks without training")
    args = parser.parse_args()

    if args.quick_check:
        quick_main()
    else:
        main()