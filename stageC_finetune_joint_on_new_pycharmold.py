# -*- coding: utf-8 -*-
"""
Stage C: 新表迁移微调（预测式 Phys→Morph）【SOTA 多变体对比版 | 24h 预算】
- 数据解析/对齐：完全沿用原项目（不得修改）
- Phys→Morph 接口增强：PhysAdapter + 派生特征 + Ion 门控（可开关）
- 主干：沿用 TemporalRegressor；输出侧加 Per-family Head（轻量分头）
- 损失：稳健 Huber / 异方差 NLL（可选）+ 多任务不确定性加权（可开）
- 后标定：per_(k,t) / per_k / 小卷积（可配）
- 训练：EMA、早停、三种子矩阵；自动导出 metrics/图像/pth 与对比表
- Quick 先验模式保留（--quick-check）
"""

import os
import copy
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# ====== 依赖 util/模型（保持你现有项目接口，严禁改动） ======
from physio_util import (
    set_seed, export_predictions_longtable, export_metrics_grid,
    write_summary_txt, heatmap, parity_scatter, residual_hist, save_manifest,
    metrics, transform_for_display, FAMILIES,
    excel_to_physics_dataset,
    load_new_excel_as_sparse_morph, build_sparse_batch
)
from phys_model import (
    TemporalRegressor,         # 形貌主干（保持不改）
    PhysicsSeqPredictor        # 物理编码器（保持不改）
)

# ========================== 基本配置 ==========================
class Cfg:
    # 数据路径（保持不改）
    old_excel = r"D:\data\pycharm\bosch\case.xlsx"
    new_excel = r"D:\data\pycharm\bosch\Bosch.xlsx"
    save_root = "./runs_stageC_compare"

    # 预训练权重
    phys_ckpt_F = "./runs_phys_split/F_Flux/phys_best.pth"
    phys_ckpt_I = "./runs_phys_split/Ion_Flux/phys_best.pth"
    morph_ckpt  = "./runs_morph_old/morph_best_overall.pth"  # 若无则尝试 morph_best.pth

    # 训练参数（默认，全局可被 variant 覆盖）
    seed = 42
    seeds = [42, 43, 44]      # 24h 预算：三种子复验
    max_epochs = 2000        # 视数据量与预算，建议 120~240
    batch_clip = 1.0
    freeze_phys = True
    lr_morph = 5e-4
    lr_phys  = 1e-4
    wd_morph = 5e-2
    wd_phys  = 1e-2

    # 校准头与损失
    lr_calib = 5e-4
    wd_calib = 0.0
    loss_delta = 1.0
    loss_smooth_weight = 1e-2
    mono_zmin_weight = 5e-3  # 若不需要单调可设 0.0

    # Ion 反变换默认常量
    ion_affine_default = {"a": 1.0, "b": 0.0, "c": 0.0}
    ion_learnable_lr = 5e-5
    ion_learnable_wd = 1e-6

    # 展示域设置（保持与 Stage B 一致）
    unit_scale = 1000.0
    flip_sign = False
    clip_nonneg = False
    min_display_value = 0.0
    family_sign = np.array([-1, +1, +1, +1, +1, +1], dtype=np.float32)  # zmin 翻正
    sheet_name = "case"
    HEIGHT_FAMILY = "h1"

    # 评估指标
    use_smape = True
    mape_eps_nm = 10.0

    # 早停 / EMA
    use_ema = True
    ema_decay = 0.999
    early_stop = True
    early_stop_metric = "MAE"
    early_stop_patience = 25

    # Quick 评估拨码（默认关闭，不改变现有行为）
    eval_unit_scale_override = None
    eval_disable_zmin_flip = False
    force_ion_mode = None  # "zero"/"const"/"smooth" 仅评估钩子

    # 变体列表（24h 预算：12~18 组 × 3 seeds）
    variants = [
        # 1) Baseline + per_(k,t) 后标定（参考）
        dict(
            name="baseline_affine_kt",
            use_adapter=False, use_derived=False, ion_gate="use",
            per_family_head=False, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel",
            post_calib="per_kt",
            learnable_ion=False,
            freeze_phys=True, stagewise_unfreeze=False
        ),
        # 2) Adapter + 派生特征 + 分头 + per_k
        dict(
            name="adapter_deriv_head_perk",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel",
            post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, stagewise_unfreeze=False
        ),
        # 3) 同上 + 异方差 + 任务不确定性
        dict(
            name="adapter_deriv_head_hetero_uncert_perk",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=True, task_uncertainty=True,
            calib_head="affine_per_channel",
            post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, stagewise_unfreeze=False
        ),
        # 4) Adapter + 派生 + 分头 + time_conv 后标定
        dict(
            name="adapter_deriv_head_timeconv_post",
            use_adapter=True, use_derived=True, ion_gate="use",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="time_conv",
            post_calib="time_conv",
            learnable_ion=True,
            freeze_phys=True, stagewise_unfreeze=False
        ),
        # 5) Adapter + 派生 + 分头 + gate Ion + per_k
        dict(
            name="adapter_deriv_head_gate_perk",
            use_adapter=True, use_derived=True, ion_gate="gate",
            per_family_head=True, hetero=False, task_uncertainty=False,
            calib_head="affine_per_channel",
            post_calib="per_k",
            learnable_ion=True,
            freeze_phys=True, stagewise_unfreeze=False
        ),
        # 6) 与 3) 同配置，解冻物理末层（阶段式）
        dict(
            name="adapter_deriv_head_hetero_uncert_perk_unfreeze",
            use_adapter=True, use_derived=True, ion_gate="gate",
            per_family_head=True, hetero=True, task_uncertainty=True,
            calib_head="hybrid",
            post_calib="per_k",
            learnable_ion=True,
            freeze_phys=False, stagewise_unfreeze=True
        ),
    ]


# ========================== 工具函数（原样 + 安全加载） ==========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _to_cpu_np_grid(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)

def _safe_load(path, map_location="cpu"):
    import numpy as _np
    import torch as _t
    from torch.serialization import safe_globals
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _allowed = []
    for _cand in ["core", "_core"]:
        try:
            _allowed.append(getattr(_np, f"{_cand}.multiarray")._reconstruct)
        except Exception:
            pass
    _allowed += [_np.generic, _np.dtype, _np.ndarray]
    try:
        return _t.load(path, map_location=map_location, weights_only=True)
    except Exception as e1:
        print(f"[safe_load] weights_only=True failed: {e1}")
    try:
        with safe_globals(_allowed):
            return _t.load(path, map_location=map_location, weights_only=True)
    except Exception as e2:
        print(f"[safe_load] with safe_globals + weights_only=True failed: {e2}")
    try:
        with safe_globals(_allowed):
            return _t.load(path, map_location=map_location, weights_only=False)
    except Exception as e3:
        print(f"[safe_load] with safe_globals + weights_only=False failed: {e3}")
    raise RuntimeError(f"Failed to load checkpoint safely: {path}")

# ===== metrics 兼容层 + 备选实现（SMAPE / 过滤版 MAPE） =====
@torch.no_grad()
def _compute_smape_grid(y_pred_disp: torch.Tensor, y_true_disp: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    err = (y_pred_disp - y_true_disp).abs()
    den = (y_pred_disp.abs() + y_true_disp.abs()).clamp_min(eps)
    smape_elem = 100.0 * 2.0 * err / den
    m = mask.float()
    sumE = (smape_elem * m).sum(dim=0)
    cnt  = m.sum(dim=0).clamp_min(1.0)
    return sumE / cnt

@torch.no_grad()
def _compute_mape_grid_with_eps(y_pred_disp: torch.Tensor, y_true_disp: torch.Tensor, mask: torch.Tensor, thr: float = 10.0, eps: float = 1e-8):
    ape = 100.0 * (y_pred_disp - y_true_disp).abs() / y_true_disp.abs().clamp_min(eps)
    valid = mask.bool() & (y_true_disp.abs() >= float(thr))
    m = valid.float()
    sumA = (ape * m).sum(dim=0)
    cnt  = m.sum(dim=0).clamp_min(1.0)
    out = sumA / cnt
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

@torch.no_grad()
def _metrics_compat(y_pred_disp, y_true_disp, mask, *, use_smape: bool, mape_eps_nm: float):
    try:
        return metrics(y_pred_disp, y_true_disp, mask, use_smape=use_smape, mape_eps=mape_eps_nm)
    except TypeError:
        mts = metrics(y_pred_disp, y_true_disp, mask)
        if use_smape:
            smape_grid = _compute_smape_grid(y_pred_disp, y_true_disp, mask)
            mts["SMAPE"] = smape_grid
        else:
            mape_grid = _compute_mape_grid_with_eps(y_pred_disp, y_true_disp, mask, thr=mape_eps_nm)
            mts["MAPE"] = mape_grid
        return mts

def _infer_arch_from_sd(sd: dict) -> Dict[str, int]:
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

# ========================== 绘图：每个 family 一张图，时间步着色 ==========================
def parity_scatter_per_family_timecolor(
    y_pred_disp: torch.Tensor,  # (B,K,T)
    y_true_disp: torch.Tensor,  # (B,K,T)
    mask: torch.Tensor,         # (B,K,T) bool
    fams: List[str],
    T_values: List[float],
    out_dir: str,
    title_prefix: str = "Parity"
):
    import os, math
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    y_pred_disp = y_pred_disp.detach().cpu()
    y_true_disp = y_true_disp.detach().cpu()
    mask = mask.detach().cpu()

    B, K, T = y_pred_disp.shape
    # 统一轴范围，便于不同 t 的点比较
    all_y = torch.cat([y_true_disp[mask], y_pred_disp[mask]])
    if all_y.numel() == 0:
        return
    y_min = float(torch.nanquantile(all_y, 0.01)) if all_y.numel() > 10 else float(all_y.min())
    y_max = float(torch.nanquantile(all_y, 0.99)) if all_y.numel() > 10 else float(all_y.max())
    if not math.isfinite(y_min) or not math.isfinite(y_max):
        y_min, y_max = -1.0, 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    pad = 0.05 * (y_max - y_min)
    lim_min, lim_max = y_min - pad, y_max + pad

    # 颜色映射：每个时间步一个颜色
    cmap = plt.get_cmap("tab20") if T <= 20 else plt.get_cmap("turbo")
    colors = [cmap(i % cmap.N) for i in range(T)]

    for k, fam in enumerate(fams):
        plt.figure(figsize=(6, 6), dpi=180)
        # 分时间步画散点
        for t in range(T):
            mkt = mask[:, k, t]
            if not mkt.any():
                continue
            x = y_true_disp[:, k, t][mkt].numpy()
            y = y_pred_disp[:, k, t][mkt].numpy()
            plt.scatter(x, y, s=10, alpha=0.5, color=colors[t], label=f"t={T_values[t]}")

        # 45° 参考线
        plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)

        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.xlabel("Ground Truth (display space)")
        plt.ylabel("Prediction (display space)")
        plt.title(f"{title_prefix} - {fam}")
        # t 比较多时，图例可能过密：超过 12 个时间步就放到外面
        if T <= 12:
            plt.legend(frameon=False, fontsize=8)
        else:
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=7)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_path = os.path.join(out_dir, f"parity_{fam}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def build_phys_from_ckpt(ckpt_F_path, ckpt_I_path, device):
    ckf = _safe_load(ckpt_F_path, map_location="cpu")
    cki = _safe_load(ckpt_I_path, map_location="cpu")
    sd_F = _get_model_sd(ckf); arch_F = _infer_arch_from_sd(sd_F)
    sd_I = _get_model_sd(cki); arch_I = _infer_arch_from_sd(sd_I)
    pf = PhysicsSeqPredictor(**arch_F).to(device)
    pi = PhysicsSeqPredictor(**arch_I).to(device)
    miss, unexp = pf.load_state_dict(sd_F, strict=False)
    if miss or unexp: print(f"[load F] missing={len(miss)} unexpected={len(unexp)}")
    miss, unexp = pi.load_state_dict(sd_I, strict=False)
    if miss or unexp: print(f"[load I] missing={len(miss)} unexpected={len(unexp)}")
    ion_aff = cki.get("ion_affine", copy.deepcopy(Cfg.ion_affine_default)) if isinstance(cki, dict) else copy.deepcopy(Cfg.ion_affine_default)
    return pf, pi, ion_aff

def _norm_col(x, mean, std):
    x = (x - mean) / (std + 1e-8)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def load_fallback_sparse(merged_excel_path, norm_mean, norm_std, time_values, families=FAMILIES):
    import pandas as pd
    if not os.path.exists(merged_excel_path):
        raise FileNotFoundError(merged_excel_path)
    try:
        df = pd.read_excel(merged_excel_path, sheet_name="merged_for_training")
    except Exception:
        df = pd.read_excel(merged_excel_path)
    s8_cols = [c for c in df.columns if str(c).lower().startswith("s") and str(c)[1:].isdigit()]
    s8_cols = sorted(s8_cols, key=lambda s: int(str(s)[1:]))[:8]
    s8_raw = df[s8_cols].astype(float).values
    s8 = _norm_col(s8_raw, norm_mean, norm_std).astype(np.float32)
    fams = list(families)
    if "zmin" not in fams: fams = ["zmin"] + fams
    T = len(time_values)
    fam2idx = {n:i for i,n in enumerate(fams)}
    B, K = s8.shape[0], len(fams)
    y_sparse = torch.zeros((B,K,T), dtype=torch.float32)
    m_sparse = torch.zeros((B,K,T), dtype=torch.bool)
    tv = np.array(time_values, dtype=float).tolist()
    def t_idx(tnum): return int(tv.index(float(tnum))) if float(tnum) in tv else max(0, min(T-1, int(tnum)-1))
    def _nm_to_um(vals): return np.nan_to_num(vals.astype(float), nan=np.nan) / 1000.0
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
                y_sparse[:,k_idx,ti][ok] = v_t[ok]
                m_sparse[:,k_idx,ti][ok] = True
    if "zmin" in fams and "zmin_10" in df.columns:
        k = fam2idx["zmin"]; vals = _nm_to_um(df["zmin_10"].values) * (-1.0)
        ti = t_idx(10); v_t = torch.tensor(vals, dtype=torch.float32)
        ok = torch.isfinite(v_t);
        if ok.any():
            y_sparse[:,k,ti][ok] = v_t[ok]; m_sparse[:,k,ti][ok] = True
    def _cands_h(n): return [f"h1_{n}", f"h1{n}", f"h{n}", f"h_{n}", f"{n}thscallopheight", f"第{n}个scallop高度", f"第{n}个scallop高度(nm)"]
    def _cands_d(n): return [f"d1_{n}", f"d1{n}", f"d{n}", f"d_{n}", f"{n}thscallopdepth",  f"第{n}个scallop深度", f"第{n}个scallop深度(nm)"]
    def _cands_w(n): return [f"w{n}", f"w_{n}", f"W{n}", f"W {n}", f"{n}thscallopwidth", f"第{n}个scallop宽度", f"第{n}个scallop宽度(nm)"]
    if "h1" in fams: _fill(fam2idx["h1"], _cands_h, [3,5,9], negate=False)
    if "d1" in fams: _fill(fam2idx["d1"], _cands_d, [3,5,9], negate=False)
    if "w"  in fams: _fill(fam2idx["w"],  _cands_w, [1,3,5,9], negate=False)
    s8 = torch.tensor(s8, dtype=torch.float32)
    tvals = torch.tensor(np.array(time_values, dtype=np.float32), dtype=torch.float32)
    return s8, y_sparse, m_sparse, tvals, fams

# ========================== 校准头（原样） ==========================
class CalibAffinePerChannel(nn.Module):
    def __init__(self, K, init_alpha=1.0, init_beta=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((K,), float(init_alpha)))
        self.beta  = nn.Parameter(torch.full((K,), float(init_beta)))
    def forward(self, y):
        return self.alpha.view(1,-1,1) * y + self.beta.view(1,-1,1)

class CalibTimeConv(nn.Module):
    def __init__(self, K, kernel_size=3):
        super().__init__()
        padding = (kernel_size-1)//2
        self.dw = nn.Conv1d(K, K, kernel_size, padding=padding, groups=K, bias=True)
        with torch.no_grad():
            self.dw.weight.zero_()
            center = padding
            for k in range(K):
                self.dw.weight[k,0,center] = 1.0
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

# ========================== 损失（稳健 / 异方差） ==========================
def masked_huber_with_channel_norm(
    y_pred, y_true, mask,
    delta=1.0, smooth_weight=1e-2, mono_penalty=None,
):
    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=-1e6)
    B, K, T = y_pred.shape
    device = y_pred.device
    finite_mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    eff_mask = (mask.bool() & finite_mask).float()
    mean_k = torch.zeros(K, device=device); std_k = torch.ones(K, device=device)
    with torch.no_grad():
        for k in range(K):
            mk = eff_mask[:,k,:].bool()
            if mk.any():
                vt = y_true[:,k,:][mk]
                if vt.numel() > 1:
                    m = vt.mean(); s = vt.std()
                    if not torch.isfinite(m): m = torch.tensor(0.0, device=device)
                    if not torch.isfinite(s): s = torch.tensor(1.0, device=device)
                    mean_k[k] = m; std_k[k] = s.clamp_min(1.0)
                else:
                    mean_k[k] = vt.mean(); std_k[k] = torch.tensor(1.0, device=device)
    y_true_n = torch.nan_to_num((y_true - mean_k.view(1,K,1))/std_k.view(1,K,1), nan=0.0, posinf=1e6, neginf=-1e6)
    y_pred_n = torch.nan_to_num((y_pred - mean_k.view(1,K,1))/std_k.view(1,K,1), nan=0.0, posinf=1e6, neginf=-1e6)
    diff = (y_pred_n - y_true_n)
    absd = diff.abs()
    huber = torch.where(absd <= delta, 0.5 * diff * diff, delta * (absd - 0.5 * delta))
    denom_k = eff_mask.sum(dim=(0,2)).clamp_min(1.0)
    loss_main_per_k = (huber * eff_mask).sum(dim=(0,2)) / denom_k
    w_k = 1.0 / std_k.clamp_min(1.0)
    loss_main = (loss_main_per_k * w_k).mean()
    if T >= 3:
        d1 = y_pred_n[:,:,1:] - y_pred_n[:,:,:-1]
        d2 = d1[:,:,1:] - d1[:,:,:-1]
        loss_smooth = torch.nan_to_num((d2**2).mean(), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        loss_smooth = torch.tensor(0.0, device=device)
    loss_mono = torch.tensor(0.0, device=device)
    if mono_penalty is not None and T >= 2:
        k_idx = mono_penalty.get("k_idx", None)
        w_mono = mono_penalty.get("weight", 0.0)
        if (k_idx is not None) and (w_mono > 0):
            d = y_pred[:,k_idx,1:] - y_pred[:,k_idx,:-1]
            loss_mono = torch.nn.functional.relu(-d).mean() * w_mono
    loss = loss_main + smooth_weight * loss_smooth + loss_mono
    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=device)
    return loss, {"loss_main": loss_main.detach(), "loss_smooth": loss_smooth.detach(), "loss_mono": loss_mono.detach()}

def hetero_nll(y_mu, y_logvar, y_true, mask, task_logvars=None):
    """
    y_mu, y_logvar: (B,K,T)；mask 同维；task_logvars: (K,) 可学的 task 不确定性（多任务加权）
    """
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=-1e6)
    m = (mask.bool() & torch.isfinite(y_true)).float()
    if task_logvars is not None:
        # 每 task 加一个全局 logσ²（K 维），用于不确定性加权
        y_logvar = y_logvar + task_logvars.view(1,-1,1)
    inv_var = torch.exp(-y_logvar).clamp_max(1e6)
    nll = 0.5*( (y_mu - y_true)**2 * inv_var + y_logvar )
    nll = (nll * m).sum(dim=(0,2)) / m.sum(dim=(0,2)).clamp_min(1.0)  # per K
    return nll.mean()

# ========================== Ion 反变换（原样） ==========================
class IonInverseTransform(nn.Module):
    def __init__(self, init_abc: Dict[str, float], learnable: bool = False):
        super().__init__()
        a, b, c = init_abc.get("a",1.0), init_abc.get("b",0.0), init_abc.get("c",0.0)
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
        return ((self.a-1.0)**2 + (self.b**2) + (self.c**2))

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

# ========================== 后验标定（原样） ==========================
def postcalib_per_kt(y_pred_disp, y_true_disp, mask, min_points=6, eps_var=1e-12, ridge=1e-6):
    yp = y_pred_disp.detach().cpu().numpy()
    yt = y_true_disp.detach().cpu().numpy()
    mk = mask.detach().cpu().numpy().astype(bool)
    B, K, T = yp.shape
    out = yp.copy()
    for k in range(K):
        for t in range(T):
            m = mk[:,k,t]
            xs = yp[m,k,t]; ys = yt[m,k,t]
            good = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[good]; ys = ys[good]
            if xs.size < max(min_points,2): continue
            if np.var(xs) < eps_var: continue
            x = xs.reshape(-1,1); y = ys.reshape(-1,1)
            A = np.concatenate([x, np.ones_like(x)], axis=1)
            try:
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                a = float(coef[0,0]); b = float(coef[1,0])
            except Exception:
                ATA = A.T @ A + ridge * np.eye(2, dtype=A.dtype)
                ATy = A.T @ y
                coef = np.linalg.solve(ATA, ATy)
                a = float(coef[0,0]); b = float(coef[1,0])
            out[:,k,t] = a * yp[:,k,t] + b
    return torch.tensor(out, dtype=y_pred_disp.dtype, device=y_pred_disp.device)

def postcalib_per_k(y_pred_disp, y_true_disp, mask, min_points=12, eps_var=1e-12, ridge=1e-6):
    yp = y_pred_disp.detach().cpu().numpy()
    yt = y_true_disp.detach().cpu().numpy()
    mk = mask.detach().cpu().numpy().astype(bool)
    B, K, T = yp.shape
    out = yp.copy()
    for k in range(K):
        xs = yp[:,k,:][mk[:,k,:]]
        ys = yt[:,k,:][mk[:,k,:]]
        good = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[good]; ys = ys[good]
        if xs.size < max(min_points,2) or np.var(xs) < eps_var:
            continue
        x = xs.reshape(-1,1); y = ys.reshape(-1,1)
        A = np.concatenate([x, np.ones_like(x)], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            a = float(coef[0,0]); b = float(coef[1,0])
        except Exception:
            ATA = A.T @ A + ridge * np.eye(2, dtype=A.dtype)
            ATy = A.T @ y
            coef = np.linalg.solve(ATA, ATy)
            a = float(coef[0,0]); b = float(coef[1,0])
        out[:,k,:] = a * yp[:,k,:] + b
    return torch.tensor(out, dtype=y_pred_disp.dtype, device=y_pred_disp.device)

class PostCalibTimeConv(nn.Module):
    def __init__(self, K, kernel_size=3):
        super().__init__()
        pad = (kernel_size-1)//2
        self.dw = nn.Conv1d(K, K, kernel_size, padding=pad, groups=K, bias=True)
        with torch.no_grad():
            self.dw.weight.zero_()
            center = pad
            for k in range(K):
                self.dw.weight[k,0,center] = 1.0
            self.dw.bias.zero_()
    def forward(self, y):
        return self.dw(y)

# ========================== Phys→Morph 接口增强 ==========================
class PhysAdapter(nn.Module):
    """ 深度可分离 1D 卷积 + per-channel 仿射，恒等初始化 """
    def __init__(self, in_ch=2, k=3):
        super().__init__()
        pad = (k-1)//2
        self.dw = nn.Conv1d(in_ch, in_ch, k, padding=pad, groups=in_ch, bias=True)
        self.pw = nn.Conv1d(in_ch, in_ch, 1, bias=True)
        # 恒等初始化
        with torch.no_grad():
            self.dw.weight.zero_()
            self.dw.bias.zero_()
            for c in range(in_ch):
                self.dw.weight[c,0,pad] = 1.0
            self.pw.weight.zero_()
            for c in range(in_ch):
                self.pw.weight[c,c,0] = 1.0
            self.pw.bias.zero_()
    def forward(self, x):  # (B,2,T)
        return self.pw(self.dw(x))

class PhysFeaReducer(nn.Module):
    """ 派生特征 [F,I,log(I+eps),dF,dI] -> 1x1 压回 2 通道 """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.pw = nn.Conv1d(5, 2, 1, bias=True)
        nn.init.zeros_(self.pw.weight)
        with torch.no_grad():
            self.pw.weight[0,0,0] = 1.0  # F -> F
            self.pw.weight[1,1,0] = 1.0  # I -> I
        nn.init.zeros_(self.pw.bias)
    def forward(self, F, I):
        logI = torch.log(I.clamp_min(self.eps))
        dF = F[:,:,1:] - F[:,:,:-1]; dI = I[:,:,1:] - I[:,:,:-1]
        dF = torch.nn.functional.pad(dF, (1,0), mode="replicate")
        dI = torch.nn.functional.pad(dI, (1,0), mode="replicate")
        x = torch.cat([F, I, logI, dF, dI], dim=1)
        return self.pw(x)

class IonGate(nn.Module):
    """ 极轻门控：根据 (F,I) 生成 gate ∈ [0,1] ，I_eff = gate*I + (1-gate)*I_smooth """
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(2, 4, 1), nn.ReLU(inplace=True),
            nn.Conv1d(4, 1, 1), nn.Sigmoid()
        )
    def forward(self, F, I):
        pad=(self.k-1)//2
        w = torch.ones(1,1,self.k, device=I.device)/self.k
        I_s = nn.functional.conv1d(nn.functional.pad(I,(pad,pad),mode="replicate"), w)
        gate = self.mlp(torch.cat([F,I], dim=1))
        return gate*I + (1.0-gate)*I_s

class PerFamilyHead(nn.Module):
    """ 逐 family 的轻量输出头（共享主干输出后再调） """
    def __init__(self, K, k=3):
        super().__init__()
        pad=(k-1)//2
        self.dw = nn.Conv1d(K, K, k, padding=pad, groups=K, bias=True)
        self.alpha = nn.Parameter(torch.ones(K))
        self.beta  = nn.Parameter(torch.zeros(K))
        with torch.no_grad():
            self.dw.weight.zero_()
            for c in range(K):
                self.dw.weight[c,0,pad] = 1.0
            self.dw.bias.zero_()
    def forward(self, y):  # (B,K,T)
        y = self.dw(y)
        return self.alpha.view(1,-1,1)*y + self.beta.view(1,-1,1)

class HeteroHead(nn.Module):
    """ 将 μ 通过 PerFamilyHead，再用一个并行分支输出 logσ² """
    def __init__(self, K, per_family=True):
        super().__init__()
        self.per_family = per_family
        self.mu_head = PerFamilyHead(K) if per_family else nn.Identity()
        self.logv_head = nn.Sequential(
            nn.Conv1d(K, K, 1, groups=1, bias=True),
        )
        nn.init.zeros_(self.logv_head[0].weight)
        nn.init.constant_(self.logv_head[0].bias, math.log(1.0))  # 初始 σ≈1
    def forward(self, y):  # y:(B,K,T) from morph+calib
        mu = self.mu_head(y) if self.per_family else y
        logv = self.logv_head(mu.detach())  # 避免梯度相互放大，可去掉 detach
        return mu, logv

def build_calib_head(kind: str, K: int) -> nn.Module:
    if kind == "affine_per_channel":
        return CalibAffinePerChannel(K)
    elif kind == "time_conv":
        return CalibTimeConv(K)
    elif kind == "hybrid":
        return CalibHybrid(K)
    else:
        raise ValueError(f"Unknown calib_head: {kind}")

# ========================== Phys 前向 + 接口增强流水线 ==========================
def phys_forward_raw(static_8, tvals, phys_F, phys_I, ion_transform: IonInverseTransform, allow_grad=False):
    phys_F.eval(); phys_I.eval()
    with torch.set_grad_enabled(allow_grad):
        f = phys_F(static_8, tvals)
        i_z = phys_I(static_8, tvals)
        f_ch = f[:,0:1,:]
        z_ch = i_z[:,1:2,:] if i_z.size(1)>=2 else i_z[:,0:1,:]
        i = ion_transform(z_ch)
        phys = torch.cat([f_ch, i], dim=1)
        return torch.nan_to_num(phys, nan=0.0, posinf=1e6, neginf=-1e6)  # (B,2,T)

def phys_interface_pipeline(phys_raw, variant, adapters):
    """
    phys_raw: (B,2,T); adapters: dict{'adapter','reducer','gate'}
    """
    F = phys_raw[:,0:1,:]; I = phys_raw[:,1:2,:]
    # Ion 策略
    mode = variant.get("ion_gate","use")
    if mode == "zero":
        I_eff = torch.zeros_like(I)
    elif mode == "const":
        I_eff = torch.nan_to_num(I, nan=0.0).mean(dim=0, keepdim=True).expand_as(I)
    elif mode == "smooth":
        k = 5; pad=(k-1)//2; w=torch.ones(1,1,k,device=I.device)/k
        I_eff = nn.functional.conv1d(nn.functional.pad(I,(pad,pad),mode="replicate"), w)
    elif mode == "gate":
        I_eff = adapters["gate"](F, I)
    else:
        I_eff = I
    x = torch.cat([F, I_eff], dim=1)
    # 派生特征 + 1x1 压回
    if variant.get("use_derived", False):
        x = adapters["reducer"](F, I_eff)
    # PhysAdapter
    if variant.get("use_adapter", False):
        x = adapters["adapter"](x)
    return x  # (B,2,T)

# ========================== 训练/评估核心 ==========================
def variant_training_pipeline(dev, meta_old, fams, s8, y_sparse, m_sparse, tvals, variant: Dict, save_dir: str, seed: int):
    set_seed(seed)
    ensure_dir(save_dir)
    torch.backends.cudnn.benchmark = True

    # 1) 模型 + 可学 Ion 反变换
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
            print(f"[{variant['name']}|seed{seed}] morph ckpt load failed, init new. {e}")

    calib = build_calib_head(variant["calib_head"], K=len(fams)).to(dev)

    # 输出头：per-family / 异方差
    per_family_head = PerFamilyHead(K=len(fams)).to(dev) if variant.get("per_family_head", False) else nn.Identity().to(dev)
    hetero_head = (HeteroHead(K=len(fams), per_family=variant.get("per_family_head", False)).to(dev)
                   if variant.get("hetero", False) else None)
    task_logvars = (nn.Parameter(torch.zeros(len(fams), device=dev))
                    if variant.get("task_uncertainty", False) else None)

    # 物理接口增强件
    adapters = {
        "adapter": PhysAdapter(2, k=3).to(dev),
        "reducer": PhysFeaReducer().to(dev),
        "gate": IonGate(k=5).to(dev),
    }

    # 优化器
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
    # 新增模块
    if variant.get("per_family_head", False):
        params.append({"params": per_family_head.parameters(), "lr": Cfg.lr_morph, "weight_decay": 0.0})
    if variant.get("hetero", False):
        params.append({"params": hetero_head.parameters(), "lr": Cfg.lr_morph, "weight_decay": 0.0})
    if variant.get("task_uncertainty", False):
        params.append({"params": [task_logvars], "lr": 1e-3, "weight_decay": 0.0})
    if variant.get("use_adapter", False) or variant.get("use_derived", False) or variant.get("ion_gate","use")=="gate":
        params.append({"params": list(adapters["adapter"].parameters())+list(adapters["reducer"].parameters())+list(adapters["gate"].parameters()),
                       "lr": 5e-4, "weight_decay": 0.0})

    opt = torch.optim.AdamW(params)

    ema = EMA(morph, decay=Cfg.ema_decay) if Cfg.use_ema else None
    stopper = EarlyStopper(Cfg.early_stop_patience) if Cfg.early_stop else None

    # 冻结/解冻
    def set_phys_trainable(flag: bool):
        for p in phys_F.parameters(): p.requires_grad = flag
        for p in phys_I.parameters(): p.requires_grad = flag
        phys_F.train(flag); phys_I.train(flag)

    if variant.get("freeze_phys", True):
        set_phys_trainable(False)
    else:
        if variant.get("stagewise_unfreeze", False):
            set_phys_trainable(False)
        else:
            set_phys_trainable(True)

    morph.train(); calib.train()
    if isinstance(per_family_head, nn.Module): per_family_head.train()
    if hetero_head is not None: hetero_head.train()

    best_snapshot = None
    best_mae = None
    fam_sign = torch.tensor(Cfg.family_sign, dtype=torch.float32, device=dev)

    for e in range(1, Cfg.max_epochs+1):
        opt.zero_grad()

        # 阶段式解冻
        if (not variant.get("freeze_phys", True)) and variant.get("stagewise_unfreeze", False) and e == (Cfg.max_epochs // 2):
            for p in phys_F.parameters(): p.requires_grad = False
            for p in phys_I.parameters(): p.requires_grad = False
            for n,m in phys_F.named_modules():
                if n.startswith("encoder.layers.") and n.endswith(".3"):
                    for p in m.parameters(): p.requires_grad = True
            for n,m in phys_I.named_modules():
                if n.startswith("encoder.layers.") and n.endswith(".3"):
                    for p in m.parameters(): p.requires_grad = True
            phys_F.train(True); phys_I.train(True)
            print(f"[{variant['name']}|seed{seed}] Stagewise unfreeze at epoch {e}.")

        # 前向：Phys -> 接口增强 -> Morph -> Calib -> Head/Hetero
        phys_pred = phys_forward_raw(s8, tvals, phys_F, phys_I, ion_tr, allow_grad=any(p.requires_grad for p in phys_F.parameters()))
        phys_in = phys_interface_pipeline(phys_pred, variant, adapters)

        y_core = morph(s8, phys_in, tvals)                  # (B,K,T)
        y_core = calib(torch.nan_to_num(y_core, nan=0.0, posinf=1e6, neginf=-1e6))  # (B,K,T)

        if variant.get("hetero", False):
            mu, logv = hetero_head(y_core)                 # (B,K,T),(B,K,T)
            # 多任务不确定性加权 + 异方差 NLL
            nll = hetero_nll(mu, logv, y_sparse, m_sparse, task_logvars=task_logvars)
            # 轻微平滑/单调（仅对 μ）
            huber_s, items = masked_huber_with_channel_norm(mu, y_sparse, m_sparse,
                                                            delta=Cfg.loss_delta, smooth_weight=Cfg.loss_smooth_weight,
                                                            mono_penalty={'k_idx': 0, 'weight': Cfg.mono_zmin_weight})
            loss = nll + 0.1*huber_s
            loss_items = {"loss_main": nll.detach(), "loss_smooth": items["loss_smooth"], "loss_mono": items["loss_mono"]}
            y_for_eval = mu
        else:
            y_head = per_family_head(y_core) if variant.get("per_family_head", False) else y_core
            loss, loss_items = masked_huber_with_channel_norm(
                y_pred=y_head, y_true=y_sparse, mask=m_sparse,
                delta=Cfg.loss_delta, smooth_weight=Cfg.loss_smooth_weight,
                mono_penalty={'k_idx': 0, 'weight': Cfg.mono_zmin_weight}
            )
            y_for_eval = y_head

        if ion_tr.learnable:
            loss = loss + 1e-4 * ion_tr.reg()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(morph.parameters(), max_norm=Cfg.batch_clip)
        opt.step()
        if ema is not None: ema.update(morph)

        # 验证（展示域）
        if e % 5 == 0 or e == 1 or e == Cfg.max_epochs:
            with torch.no_grad():
                if ema is not None:
                    morph_eval = copy.deepcopy(morph).to(dev)
                    ema.apply_to(morph_eval)
                else:
                    morph_eval = morph

                phys_pred_eval = phys_forward_raw(s8, tvals, phys_F, phys_I, ion_tr, allow_grad=False)
                phys_in_eval = phys_interface_pipeline(phys_pred_eval, variant, adapters)
                yp = morph_eval(s8, phys_in_eval, tvals)
                yp = calib(yp)
                if variant.get("hetero", False):
                    mu_eval, _ = hetero_head(yp)
                    yp = mu_eval
                elif variant.get("per_family_head", False):
                    yp = per_family_head(yp)

                fam_sign_eval = torch.tensor(Cfg.family_sign, dtype=torch.float32, device=dev)
                if Cfg.eval_disable_zmin_flip:
                    fam_sign_eval = fam_sign_eval.clone(); fam_sign_eval[0] = 1.0
                unit_scale_eval = Cfg.unit_scale if Cfg.eval_unit_scale_override is None else float(Cfg.eval_unit_scale_override)

                yhat_disp, ytrue_disp = transform_for_display(
                    yp, y_sparse,
                    family_sign=fam_sign_eval,
                    unit_scale=unit_scale_eval,
                    flip_sign=Cfg.flip_sign,
                    clip_nonneg=Cfg.clip_nonneg,
                    min_display_value=Cfg.min_display_value,
                )
                mts_base = _metrics_compat(
                    yhat_disp, ytrue_disp, m_sparse,
                    use_smape=Cfg.use_smape, mape_eps_nm=Cfg.mape_eps_nm
                )
                def _as_tensor_grid(x, device):
                    if torch.is_tensor(x): return x.to(device)
                    import numpy as _np
                    if isinstance(x, _np.ndarray): return torch.tensor(x, dtype=torch.float32, device=device)
                    return torch.tensor(x, dtype=torch.float32, device=device)
                mae_grid = mts_base.get("MAE", None)
                if mae_grid is not None:
                    mae_grid = _as_tensor_grid(mae_grid, dev)
                    mae_grid = torch.nan_to_num(mae_grid, nan=0.0, posinf=0.0, neginf=0.0)
                    sup = m_sparse.sum(dim=0)
                    mask_sup = (sup > 0).float()
                    denom = mask_sup.sum().clamp_min(1.0)
                    mae_scalar = float((mae_grid * mask_sup).sum().item() / denom.item())
                else:
                    mae_scalar = float(loss_items["loss_main"])
                print(f"[{variant['name']}|seed{seed}][{e}/{Cfg.max_epochs}] "
                      f"loss={loss.item():.6f} main={loss_items['loss_main']:.4f} "
                      f"smooth={loss_items['loss_smooth']:.4f} mono={loss_items['loss_mono']:.4f} "
                      f"valMAE={mae_scalar:.3f}")
                if stopper is not None:
                    keep = stopper.step(mae_scalar)
                    if keep:
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
                        best_mae = mae_scalar
                    else:
                        print(f"[{variant['name']}|seed{seed}] Early stop at epoch {e}. Best MAE={best_mae:.3f}")
                        break

    # 用最佳或最终权重导出
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

    morph.eval(); calib.eval(); phys_F.eval(); phys_I.eval()
    if isinstance(per_family_head, nn.Module): per_family_head.eval()
    if hetero_head is not None: hetero_head.eval()

    with torch.no_grad():
        phys_pred = phys_forward_raw(s8, tvals, phys_F, phys_I, ion_tr, allow_grad=False)
        phys_in = phys_interface_pipeline(phys_pred, variant, adapters)
        yhat = calib(morph(s8, phys_in, tvals))
        if variant.get("hetero", False):
            mu, _ = hetero_head(yhat)
            yhat = mu
        elif variant.get("per_family_head", False):
            yhat = per_family_head(yhat)

    fam_sign = torch.tensor(Cfg.family_sign, dtype=torch.float32, device=dev)
    if Cfg.eval_disable_zmin_flip:
        fam_sign = fam_sign.clone(); fam_sign[0] = 1.0
    unit_scale_eval = Cfg.unit_scale if Cfg.eval_unit_scale_override is None else float(Cfg.eval_unit_scale_override)

    yhat_disp, ytrue_disp = transform_for_display(
        yhat, y_sparse,
        family_sign=fam_sign,
        unit_scale=unit_scale_eval,
        flip_sign=Cfg.flip_sign,
        clip_nonneg=Cfg.clip_nonneg,
        min_display_value=Cfg.min_display_value,
    )

    # 后标定
    post_kind = variant["post_calib"]
    if post_kind == "per_kt":
        yhat_disp_cal = postcalib_per_kt(yhat_disp, ytrue_disp, m_sparse, min_points=6, ridge=1e-4)
    elif post_kind == "per_k":
        yhat_disp_cal = postcalib_per_k(yhat_disp, ytrue_disp, m_sparse, min_points=12, ridge=1e-4)
    elif post_kind == "time_conv":
        pc = PostCalibTimeConv(len(fams)).to(dev)
        opt_pc = torch.optim.Adam(pc.parameters(), lr=1e-2, weight_decay=0.0)
        for _ in range(80):
            opt_pc.zero_grad()
            yp_pc = pc(yhat_disp)
            diff = (yp_pc - ytrue_disp)
            loss_pc = (diff.pow(2) * m_sparse.float()).sum() / m_sparse.float().sum().clamp_min(1.0)
            loss_pc.backward(); opt_pc.step()
        yhat_disp_cal = pc(yhat_disp).detach()
    else:
        raise ValueError(f"Unknown post_calib: {post_kind}")

    mts = _metrics_compat(
        yhat_disp_cal, ytrue_disp, m_sparse,
        use_smape=Cfg.use_smape, mape_eps_nm=Cfg.mape_eps_nm
    )
    mts_all = dict(mts)
    mts_all["SMAPE"] = _compute_smape_grid(yhat_disp_cal, ytrue_disp, m_sparse)

    # 导出
    T_values = meta_old["time_values"]
    mts_all_cpu = {k: _to_cpu_np_grid(v) for k, v in mts_all.items()}
    export_metrics_grid(mts_all_cpu, fams, meta_old["time_values"], save_dir, filename=f"metrics_plus_seed{seed}.xlsx")

    export_predictions_longtable(
        yhat_disp_cal.cpu(), ytrue_disp.cpu(), m_sparse.cpu(), fams, T_values,
        save_dir, filename=f"predictions_seed{seed}.xlsx",
    )
    mts_cpu = {k: _to_cpu_np_grid(v) for k, v in mts.items()}
    export_metrics_grid(mts_cpu, fams, T_values, save_dir, filename=f"metrics_seed{seed}.xlsx")

    heatmap(
        (mts.get("R2", torch.zeros(len(fams), len(T_values))).detach().cpu()
         if torch.is_tensor(mts.get("R2", None)) else _to_cpu_np_grid(mts.get("R2", None))),
        fams, T_values, f"R2 ({variant['name']}|seed{seed})",
        os.path.join(save_dir, f"r2_seed{seed}.png")
    )
    parity_scatter_per_family_timecolor(
        yhat_disp_cal, ytrue_disp, m_sparse,
        fams=fams,
        T_values=meta_old["time_values"],
        out_dir=save_dir,
        title_prefix=f"Parity ({variant['name']}|seed{seed})"
    )
    residual_hist(yhat_disp_cal.cpu(), ytrue_disp.cpu(), m_sparse.cpu(),
                  os.path.join(save_dir, f"residual_seed{seed}.png"), f"Residuals ({variant['name']}|seed{seed})")

    # 保存权重包（含接口增强/头部）
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
            "meta": {
                "families": fams,
                "time_values": T_values,
                "norm_static": meta_old["norm_static"],
                "height_family": Cfg.HEIGHT_FAMILY,
                "variant": variant,
                "seed": seed
            },
        },
        os.path.join(save_dir, f"joint_new_last_seed{seed}.pth"),
    )
    save_manifest(save_dir)

    return {"variant": variant["name"], "metrics": mts, "save_dir": save_dir, "seed": seed}

# ========================== Quick Priors（保留，数据管线不改） ==========================
@torch.no_grad()
def _display_space_eval(yp, y_sparse, m_sparse, fam_sign, note: str):
    unit_scale_eval = Cfg.unit_scale if Cfg.eval_unit_scale_override is None else float(Cfg.eval_unit_scale_override)
    yhat_disp, ytrue_disp = transform_for_display(
        yp, y_sparse,
        family_sign=fam_sign, unit_scale=unit_scale_eval,
        flip_sign=Cfg.flip_sign, clip_nonneg=Cfg.clip_nonneg, min_display_value=Cfg.min_display_value
    )
    mts = _metrics_compat(yhat_disp, ytrue_disp, m_sparse, use_smape=Cfg.use_smape, mape_eps_nm=Cfg.mape_eps_nm)
    sup = m_sparse.sum(dim=0).detach().cpu().numpy() > 0
    out = {}
    for key in ["R2","RMSE","MAE","SMAPE" if Cfg.use_smape else "MAPE"]:
        if key in mts:
            g = mts[key].detach().cpu().numpy() if torch.is_tensor(mts[key]) else mts[key]
            out[f"{key}_mean"] = float(np.nanmean(np.where(sup, g, np.nan)))
    ytd = ytrue_disp.detach().cpu().numpy()
    ypd = yhat_disp.detach().cpu().numpy()
    print(f"[Quick:{note}] "
          f"R2_mean={out.get('R2_mean', float('nan')):.4f}  "
          f"MAE_mean={out.get('MAE_mean', float('nan')):.2f}  "
          f"RMSE_mean={out.get('RMSE_mean', float('nan')):.2f}")
    print(f"  y_true_disp mean/std = {np.nanmean(ytd):.3f}/{np.nanstd(ytd):.3f}  "
          f"y_pred_disp mean/std = {np.nanmean(ypd):.3f}/{np.nanstd(ypd):.3f}")
    return out, yhat_disp, ytrue_disp

def _assert_alignment(meta_old, tvals, fams, morph_ckpt_path: str):
    T_meta = len(meta_old["time_values"])
    if tvals.dim() == 1:
        assert tvals.numel() == T_meta, f"tvals length({tvals.numel()}) != meta T({T_meta})"
    else:
        assert tvals.shape[1] == T_meta, f"tvals.shape[1]({tvals.shape[1]}) != meta T({T_meta})"
    try:
        if os.path.exists(morph_ckpt_path):
            mk = _safe_load(morph_ckpt_path, map_location="cpu")
            meta_m = mk.get("meta", {})
            fam_train = meta_m.get("families", None)
            if fam_train is not None:
                assert list(fam_train) == list(fams), f"FAMILIES 不一致：train={fam_train} now={fams}"
    except Exception as e:
        print(f"[Quick] families 对齐未校验（ckpt 无 meta）: {e}")

def _ion_ablation_quick(dev, s8, tvals, phys_F, phys_I, ion_tr, morph, y_sparse, m_sparse, fam_sign):
    base = phys_forward_raw(s8, tvals, phys_F, phys_I, ion_tr, allow_grad=False)
    F = base[:,0:1,:]; I = base[:,1:2,:]
    modes = {}
    for mode in ["use","zero","const","smooth"]:
        if mode == "zero":
            I2 = torch.zeros_like(I)
        elif mode == "const":
            I2 = torch.nan_to_num(I, nan=0.0).mean(dim=0, keepdim=True).expand_as(I)
        elif mode == "smooth":
            k = 5; pad=(k-1)//2; w=torch.ones(1,1,k,device=I.device)/k
            I2 = nn.functional.conv1d(nn.functional.pad(I,(pad,pad),mode="replicate"), w)
        else:
            I2 = I
        yp = morph(s8, torch.cat([F,I2], dim=1), tvals)
        stat, _, _ = _display_space_eval(yp, y_sparse, m_sparse, fam_sign, f"ion-{mode}")
        modes[mode] = stat
    return modes

@torch.no_grad()
def _posterior_ceiling_quick(dev, s8, tvals, phys_F, phys_I, ion_tr, morph, y_sparse, m_sparse, fam_sign):
    base = phys_forward_raw(s8, tvals, phys_F, phys_I, ion_tr, allow_grad=False)
    yp = morph(s8, base, tvals)
    stat0, yhat_disp, ytrue_disp = _display_space_eval(yp, y_sparse, m_sparse, fam_sign, "baseline-before-postcalib")
    yhat_disp_cal = postcalib_per_kt(yhat_disp, ytrue_disp, m_sparse, min_points=4, ridge=1e-4)
    mts = _metrics_compat(yhat_disp_cal, ytrue_disp, m_sparse, use_smape=Cfg.use_smape, mape_eps_nm=Cfg.mape_eps_nm)
    sup = m_sparse.sum(dim=0).detach().cpu().numpy() > 0
    out = {}
    for key in ["R2","RMSE","MAE","SMAPE" if Cfg.use_smape else "MAPE"]:
        if key in mts:
            g = mts[key].detach().cpu().numpy() if torch.is_tensor(mts[key]) else mts[key]
            out[f"{key}_mean"] = float(np.nanmean(np.where(sup, g, np.nan)))
    print(f"[Quick:posterior-ceiling] R2_mean={out.get('R2_mean', float('nan')):.4f}")
    return {"baseline": stat0, "postcalib_per_kt": out}

def quick_main():
    set_seed(Cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[StageC-Quick] device = {dev}")
    _, meta_old = excel_to_physics_dataset(Cfg.old_excel, sheet_name=Cfg.sheet_name)
    T_values = meta_old["time_values"]
    try:
        recs = load_new_excel_as_sparse_morph(Cfg.new_excel, height_family=Cfg.HEIGHT_FAMILY)
        s8, y_sparse, m_sparse, tvals = build_sparse_batch(
            recs, meta_old["norm_static"]["mean"], meta_old["norm_static"]["std"], meta_old["time_values"]
        )
        fams = FAMILIES
        print("[StageC-Quick] Loaded sparse morph via util loader.")
    except Exception as e:
        print(f"[StageC-Quick] util loader failed -> {e} -> fallback loader.")
        s8, y_sparse, m_sparse, tvals, fams = load_fallback_sparse(
            Cfg.new_excel, meta_old["norm_static"]["mean"], meta_old["norm_static"]["std"], T_values, families=FAMILIES
        )
    s8 = s8.to(dev); y_sparse = y_sparse.to(dev); m_sparse = m_sparse.to(dev); tvals = tvals.to(dev)
    if tvals.dim() == 1: tvals = tvals.unsqueeze(0).expand(s8.size(0), -1).contiguous()
    _assert_alignment(meta_old, tvals, fams, Cfg.morph_ckpt)
    phys_F, phys_I, ion_aff_init = build_phys_from_ckpt(Cfg.phys_ckpt_F, Cfg.phys_ckpt_I, dev)
    ion_tr = IonInverseTransform(ion_aff_init, learnable=False).to(dev)
    morph = TemporalRegressor(K=len(fams)).to(dev)
    if os.path.exists(Cfg.morph_ckpt):
        try:
            ck = _safe_load(Cfg.morph_ckpt, map_location="cpu")
            sd = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck
            miss, unexp = morph.load_state_dict(sd, strict=False)
            if miss or unexp: print(f"[Quick] morph ckpt loaded with missing={len(miss)} unexpected={len(unexp)}")
        except Exception as e:
            print(f"[Quick] morph ckpt load failed, init new. {e}")
    fam_sign = torch.tensor(Cfg.family_sign, dtype=torch.float32, device=dev)
    if Cfg.eval_disable_zmin_flip:
        fam_sign = fam_sign.clone(); fam_sign[0] = 1.0

    with torch.no_grad():
        phys_pred = phys_forward_raw(s8, tvals, phys_F, phys_I, ion_tr, allow_grad=False)
        yp = morph(s8, phys_pred, tvals)
        _display_space_eval(yp, y_sparse, m_sparse, fam_sign, "baseline")

    ion_table = _ion_ablation_quick(dev, s8, tvals, phys_F, phys_I, ion_tr, morph, y_sparse, m_sparse, fam_sign)
    post_stats = _posterior_ceiling_quick(dev, s8, tvals, phys_F, phys_I, ion_tr, morph, y_sparse, m_sparse, fam_sign)

    save_dir = os.path.join(Cfg.save_root, "_quick_check")
    ensure_dir(save_dir)
    with open(os.path.join(save_dir, "ion_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(ion_table, f, indent=2, ensure_ascii=False)
    with open(os.path.join(save_dir, "posterior_ceiling.json"), "w", encoding="utf-8") as f:
        json.dump(post_stats, f, indent=2, ensure_ascii=False)
    print(f"[StageC-Quick] done. See: {save_dir}")

# ========================== 主流程（多变体 × 多种子） ==========================
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[StageC-Compare] device = {dev}")
    set_seed(Cfg.seed)

    # 旧表 meta（严格保留）
    _, meta_old = excel_to_physics_dataset(Cfg.old_excel, sheet_name=Cfg.sheet_name)
    T_values = meta_old["time_values"]

    # 新表（严格保留）
    try:
        recs = load_new_excel_as_sparse_morph(Cfg.new_excel, height_family=Cfg.HEIGHT_FAMILY)
        s8, y_sparse, m_sparse, tvals = build_sparse_batch(
            recs, meta_old["norm_static"]["mean"], meta_old["norm_static"]["std"], meta_old["time_values"]
        )
        fams = FAMILIES
        print("[StageC-Compare] Loaded sparse morph via util loader.")
    except Exception as e:
        print(f"[StageC-Compare] util loader failed -> {e} -> fallback loader.")
        s8, y_sparse, m_sparse, tvals, fams = load_fallback_sparse(
            Cfg.new_excel, meta_old["norm_static"]["mean"], meta_old["norm_static"]["std"], T_values, families=FAMILIES
        )

    # 清洗监督（严格保留）
    s8  = s8.to(dev); y_sparse = y_sparse.to(dev); m_sparse = m_sparse.to(dev); tvals = tvals.to(dev)
    y_sparse = torch.nan_to_num(y_sparse, nan=0.0, posinf=0.0, neginf=0.0)
    m_sparse = m_sparse & torch.isfinite(y_sparse)
    if tvals.dim() == 1:
        tvals = tvals.unsqueeze(0).expand(s8.size(0), -1).contiguous()

    ensure_dir(Cfg.save_root)
    results = []

    for variant in Cfg.variants:
        for seed in Cfg.seeds:
            save_dir = os.path.join(Cfg.save_root, f"{variant['name']}_seed{seed}")
            ensure_dir(save_dir)
            out = variant_training_pipeline(dev, meta_old, fams, s8, y_sparse, m_sparse, tvals, variant, save_dir, seed)
            results.append(out)

    # 汇总（对每组 variant 聚合三种子）
    import pandas as pd
    import numpy as _np
    rows = []
    key_metrics = ["R2", "RMSE", "MAE", "SMAPE" if Cfg.use_smape else "MAPE", "MSE"]
    # 收集每个 seed 的网格指标
    per_seed_rows = []
    for r in results:
        mts = r["metrics"]
        row = {"variant": r["variant"], "save_dir": r["save_dir"], "seed": r["seed"]}
        for key in key_metrics:
            if key not in mts: continue
            grid_np = _to_cpu_np_grid(mts[key])
            sup_np = m_sparse.sum(dim=0).detach().cpu().numpy()
            mask = (sup_np > 0)
            row[f"{key}_mean"] = float(_np.nanmean(_np.where(mask, grid_np, _np.nan))) if mask.sum() > 0 else _np.nan
        per_seed_rows.append(row)

    df_seed = pd.DataFrame(per_seed_rows).sort_values(by=["variant","R2_mean"], ascending=[True, False], na_position="last")
    comp_seed_xlsx = os.path.join(Cfg.save_root, "variants_seeds.xlsx")
    with pd.ExcelWriter(comp_seed_xlsx) as w:
        df_seed.to_excel(w, index=False, sheet_name="per_seed")

    # 聚合到 variant（取均值）
    for variant in df_seed["variant"].unique():
        sub = df_seed[df_seed["variant"]==variant]
        agg = {"variant": variant}
        for m in [c for c in df_seed.columns if c.endswith("_mean")]:
            agg[m] = float(sub[m].mean())
        rows.append(agg)
    df = pd.DataFrame(rows).sort_values(by=["R2_mean"], ascending=False, na_position="last")
    comp_xlsx = os.path.join(Cfg.save_root, "variants_comparison.xlsx")
    with pd.ExcelWriter(comp_xlsx) as w:
        df.to_excel(w, index=False, sheet_name="summary")

    with open(os.path.join(Cfg.save_root, "summary_comparison.txt"), "w", encoding="utf-8") as f:
        f.write("== Variants Comparison (supervised-only mean metrics; averaged over seeds) ==\n")
        f.write(df.to_string(index=False))
        f.write("\n\nNotes:\n")
        f.write("- R2_mean 越高越好；MAE/RMSE/SMAPE(MAPE) 越低越好。\n")
        f.write("- 每个变体×种子的完整导出见其子目录；per_seed 明细见 variants_seeds.xlsx\n")

    print(f"[OK] All variants finished. See: {Cfg.save_root}")
    print(f"Per-seed table: {comp_seed_xlsx}")
    print(f"Comparison table: {comp_xlsx}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-check", action="store_true", default=False,
                        help="运行无训练先验测试（不改 A/B，不进入多变体训练）。")
    args = parser.parse_args()
    if args.quick_check:
        quick_main()
    else:
        main()
