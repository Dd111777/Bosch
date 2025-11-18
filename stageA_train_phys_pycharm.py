# -*- coding: utf-8 -*-

import os, json, csv, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from physio_util import (
    set_seed, excel_to_physics_dataset, transform_for_display, metrics,
    export_predictions_longtable, export_metrics_grid, write_summary_txt,
    save_manifest, heatmap, parity_scatter, residual_hist
)
from phys_model import PhysicsSeqPredictor

# -------------------- 全局常量 --------------------
FAMILIES = ["F_Flux", "Ion_Flux"]

# -------------------- 配置 --------------------
class Cfg:
    # 数据
    old_excel = r"D:\data\pycharm\bosch\case.xlsx"
    sheet_name = "case"
    save_dir = "./runs_phys_split"
    seed = 42
    batch = 64
    max_epochs = 300
    val_ratio = 0.1

    d_model=64; nhead=4; num_layers=2; dim_ff=128; dropout=0.1

    lr = 1e-3
    weight_decay = 1e-3
    clip_grad_norm = 1.0
    warmup_epochs = 10
    use_cosine = True

    f_loss = "l1"
    eps_mask_F = 1e-3
    use_tv_reg_F = True
    tv_lambda_F = 1e-3

    use_learnable_affine_train_I = False
    use_affine_calibration_post = False
    cal_reg_lambda_I = 0.0
    eps_mask_I = 1e-3
    ion_use_log_mix = True
    ion_log_ramp_start = 20
    ion_log_ramp_end   = 80
    ion_log_p = 2.0
    ion_c_quantile = 0.10
    ion_c_min = 1e-6
    ion_weight_gamma_target = 3.5
    ion_weight_gamma_ramp_start = 10
    ion_weight_gamma_ramp_end   = 80
    ion_weight_cap = 15.0
    use_tv_reg_I = False
    tv_lambda_I = 0.0
    clamp_nonneg_on_export = False
    target_R2_F = 0.95
    target_R2_I = 0.90
def _make_loss(name):
    if name=="l1": return nn.L1Loss(reduction="none")
    if name=="l2": return nn.MSELoss(reduction="none")
    return nn.SmoothL1Loss(reduction="none")

def _to_serializable(x):
    import numpy as _np, torch as _t
    if isinstance(x, (int,float,str,bool)) or x is None: return x
    if isinstance(x, _np.integer):  return int(x)
    if isinstance(x, _np.floating): return float(x)
    if isinstance(x, _np.ndarray):  return x.tolist()
    if isinstance(x, _t.Tensor):    return x.detach().cpu().tolist()
    if isinstance(x, (list,tuple)): return [_to_serializable(i) for i in x]
    if isinstance(x, dict):         return {k:_to_serializable(v) for k,v in x.items()}
    return str(x)

def _clean_meta(meta: dict): return {k:_to_serializable(v) for k,v in meta.items()}

def _meta_torchify_for_display(meta):
    m = dict(meta)
    families = m.get("families", FAMILIES)
    def as_vec(v, default):
        import torch as _t, numpy as _np
        if _t.is_tensor(v): return v.float()
        if isinstance(v, dict):
            return _t.tensor([float(v.get(name, default)) for name in families], dtype=_t.float32)
        if isinstance(v, (list,tuple,_np.ndarray)):
            return _t.tensor(v, dtype=_t.float32)
        if isinstance(v, (int,float)):
            return _t.tensor([float(v)]*len(families), dtype=_t.float32)
        return _t.tensor([default]*len(families), dtype=_t.float32)
    m["family_sign"]  = as_vec(m.get("family_sign", 1.0), 1.0)
    m["family_scale"] = as_vec(m.get("family_scale", 1.0), 1.0)
    m["family_bias"]  = as_vec(m.get("family_bias", 0.0), 0.0)
    return m

# ---- 可视化/指标（保持不变） ----
def plot_timeseries_per_channel(save_dir, y_true, y_pred, mask, time_values=None, sample_ids=None, max_n=16):
    os.makedirs(os.path.join(save_dir, "timeseries"), exist_ok=True)
    B, C, T = y_true.shape; names = FAMILIES
    idxs = np.arange(B) if sample_ids is None else np.asarray(sample_ids); idxs = idxs[:max_n]
    t_axis = np.asarray(time_values) if time_values is not None else np.arange(T)
    for b in idxs:
        for c in range(C):
            valid = mask[b, c].astype(bool) if mask is not None else np.ones(T, bool)
            t = t_axis[valid]; gt = y_true[b, c, valid]; pd = y_pred[b, c, valid]
            plt.figure(figsize=(8,3)); plt.plot(t, gt, label="GT"); plt.plot(t, pd, "--", label="Pred")
            plt.xlabel("Time"); plt.ylabel(names[c]); plt.title(f"{names[c]}  sample#{b}"); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(save_dir,"timeseries",f"{names[c]}_pred_vs_gt_{b:04d}.png"), dpi=200); plt.close()

def parity_scatter_per_channel(save_dir, y_true, y_pred, mask, suffix=""):
    os.makedirs(os.path.join(save_dir, "parity"), exist_ok=True)
    names = FAMILIES; B, C, T = y_true.shape
    for c in range(C):
        valid = mask[:, c, :].reshape(-1).astype(bool) if mask is not None else np.ones(B*T, bool)
        gt = y_true[:, c, :].reshape(-1)[valid]; pd = y_pred[:, c, :].reshape(-1)[valid]
        if gt.size==0: continue
        lim_min = float(min(gt.min(), pd.min())); lim_max = float(max(gt.max(), pd.max()))
        plt.figure(figsize=(4,4)); plt.scatter(gt, pd, s=3, alpha=0.5); plt.plot([lim_min,lim_max],[lim_min,lim_max])
        ttl = f"Parity — {names[c]}" + (f" ({suffix})" if suffix else "")
        plt.title(ttl); plt.xlabel(f"{names[c]} GT"); plt.ylabel(f"{names[c]} Pred"); tag=f"_{suffix}" if suffix else ""
        plt.tight_layout(); plt.savefig(os.path.join(save_dir,"parity",f"parity_{names[c]}{tag}.png"), dpi=200); plt.close()

def channelwise_metrics(y_true, y_pred, mask, eps=1e-9):
    out={}; names=FAMILIES; B, C, T = y_true.shape
    for c in range(C):
        valid = mask[:, c, :].reshape(-1).astype(bool) if mask is not None else np.ones(B*T, bool)
        gt = y_true[:, c, :].reshape(-1)[valid]; pd = y_pred[:, c, :].reshape(-1)[valid]
        if gt.size==0: out[f"MAE_{names[c]}"]=out[f"RMSE_{names[c]}"]=out[f"R2_{names[c]}"]=np.nan; continue
        diff = pd-gt; ss_res=float(np.sum(diff**2)); ss_tot=float(np.sum((gt-np.mean(gt))**2))
        out[f"MAE_{names[c]}"]=float(np.mean(np.abs(diff)))
        out[f"RMSE_{names[c]}"]=float(np.sqrt(np.mean(diff**2)+eps))
        out[f"R2_{names[c]}"]=float(1-ss_res/(ss_tot+eps))
    return out

def ion_out(x, scale=1.0, beta=2.0):
    z0 = F.softplus(torch.zeros(1, device=x.device, dtype=x.dtype), beta=beta)
    return (F.softplus(x, beta=beta) - z0) * scale

# -------------------- 加权后仿射（新增函数） --------------------
def affine_calibrate_per_channel_weighted(y_true, y_pred, mask,
                                          gamma=4.0, cap=50.0, base_q=0.75):
    """
    对每个通道做加权线性校准： t ≈ a * p + b
    Ion_Flux 通道使用高值权重（同 ion_weight 逻辑），F_Flux 等权。
    """
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    m  = mask.detach().cpu().numpy().astype(bool)
    a = np.ones(2, dtype=np.float32); b = np.zeros(2, dtype=np.float32)

    for c in range(2):
        idx = m[:, c, :].reshape(-1)
        t = yt[:, c, :].reshape(-1)[idx]
        p = yp[:, c, :].reshape(-1)[idx]
        if t.size < 10:
            continue

        if c == 1:
            # Ion_Flux：构造权重（与训练同风格）
            pos = np.clip(t, 0, None)
            if (pos > 0).any():
                pq = np.quantile(pos[pos > 0], base_q)
            else:
                pq = 1.0
            w = np.minimum((pos / (pq + 1e-9)) ** gamma, cap).astype(np.float64)
            w = np.clip(w, 1e-6, None)
        else:
            # F_Flux：等权
            w = np.ones_like(t, dtype=np.float64)

        # 加权最小二乘：对 X、y 同时乘 sqrt(w)
        X = np.vstack([p, np.ones_like(p)]).T.astype(np.float64)
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(X * sw, (t[:, None] * sw), rcond=None)
        a[c], b[c] = float(coef[0, 0]), float(coef[1, 0])

    return a, b

def _enforce_monotone(y: np.ndarray) -> np.ndarray:
    """把 y 改成非降序（单调不减）。"""
    y_mono = y.copy()
    for i in range(1, len(y_mono)):
        if y_mono[i] < y_mono[i-1]:
            y_mono[i] = y_mono[i-1]
    return y_mono

def monotone_calibrate_ion(y_true: torch.Tensor,
                           y_pred: torch.Tensor,
                           mask: torch.Tensor,
                           n_bins: int = 20,
                           q_lo: float = 0.01,
                           q_hi: float = 0.99) -> tuple[torch.Tensor, dict]:
    """
    对 Ion 预测做单调分段线性标定：
      1) 以 y_pred 的分位点作为“横轴结点”，每个 bin 内取 y_true 的均值作为“纵轴结点”
      2) 对纵轴结点做“累积最大” -> 保证单调不降
      3) 对所有样本在结点间线性插值得到校准后的预测
    仅对 Ion 通道 (C=2 的第 2 个通道) 生效。
    返回：calibrated_pred, debug_info
    """
    # 拉到 CPU / numpy
    yp = y_pred.detach().cpu().numpy()   # (B,2,T)
    yt = y_true.detach().cpu().numpy()
    m  = mask.detach().cpu().numpy().astype(bool)

    B, C, T = yp.shape
    assert C == 2, "expect 2 channels [F_Flux, Ion_Flux]"
    ch = 1  # Ion 索引

    # 只取有效样本
    idx = m[:, ch, :].reshape(-1)
    p  = yp[:, ch, :].reshape(-1)[idx]
    t  = yt[:, ch, :].reshape(-1)[idx]
    if p.size < max(64, n_bins):
        # 数据太少，直接返回原预测
        return y_pred, {"used": False, "reason": "too_few_points"}

    # 去掉极端点，避免边界外插爆炸
    p_lo, p_hi = np.quantile(p, [q_lo, q_hi])
    sel = (p >= p_lo) & (p <= p_hi)
    p_use = p[sel]; t_use = t[sel]

    # 以预测的分位点做横轴结点
    qs = np.linspace(q_lo, q_hi, n_bins)
    x_knots = np.quantile(p_use, qs)

    # 每个 bin 的 y_true 平均作为纵轴结点
    y_knots = []
    edges = np.concatenate([[-np.inf], (x_knots[:-1] + x_knots[1:]) / 2.0, [np.inf]])
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        sel_bin = (p_use >= lo) & (p_use < hi)
        if sel_bin.sum() < 8:
            # 样本太少时用邻近预测值的回归近似：这里直接用中位数回填
            y_knots.append(np.median(t_use))
        else:
            y_knots.append(float(np.mean(t_use[sel_bin])))

    y_knots = np.array(y_knots, dtype=np.float64)

    # 纵轴结点强制单调不降（把 S 形拉直）
    y_knots_mono = _enforce_monotone(y_knots)

    # 对全体样本做线性插值
    def _interp_1d(xq: np.ndarray, xk: np.ndarray, yk: np.ndarray) -> np.ndarray:
        # np.interp 要求 xk 严格升序；x_knots 是分位数，天然升序
        return np.interp(xq, xk, yk, left=yk[0], right=yk[-1])

    # 构造新的 yhat（仅替换 Ion 通道）
    y_cal = yp.copy()
    y_cal[:, ch, :] = _interp_1d(yp[:, ch, :], x_knots, y_knots_mono)

    y_cal_t = torch.tensor(y_cal, dtype=y_pred.dtype, device=y_pred.device)

    dbg = {
        "used": True,
        "x_knots": x_knots.tolist(),
        "y_knots_raw": y_knots.tolist(),
        "y_knots_mono": y_knots_mono.tolist(),
        "q_lo": float(q_lo), "q_hi": float(q_hi), "n_bins": int(n_bins)
    }
    return y_cal_t, dbg

def apply_affine(y_pred, a, b):
    a = torch.tensor(a, dtype=y_pred.dtype, device=y_pred.device).view(1,2,1)
    b = torch.tensor(b, dtype=y_pred.dtype, device=y_pred.device).view(1,2,1)
    return a * y_pred + b

# -------------------- Ion 损失/权重（带日程） --------------------
def ion_log_loss(pred_ch, tgt_ch, ion_c=1e-4, p=2.0):
    lpred = torch.log(torch.clamp(pred_ch, min=1e-8) + ion_c)
    ltgt  = torch.log(torch.clamp(tgt_ch,  min=1e-8) + ion_c)
    diff = lpred - ltgt
    return diff.abs() if p == 1.0 else diff.pow(p)

def mix_l1_log_loss(pred_ch, tgt_ch, ion_c, alpha, p):
    """ alpha∈[0,1]: 0→纯L1，1→纯log-loss """
    l1 = torch.abs(pred_ch - tgt_ch)
    lg = ion_log_loss(pred_ch, tgt_ch, ion_c=ion_c, p=p)
    return (1.0 - alpha) * l1 + alpha * lg

def ramp_value(epoch, start, end, v0, v1):
    if epoch <= start: return v0
    if epoch >= end:   return v1
    t = (epoch - start) / max(1e-6, (end - start))
    return v0 + (v1 - v0) * 0.5 * (1 - np.cos(np.pi * t))

# -------------------- Ion 高值样本权重（替换原函数） --------------------
def ion_weight(tgt_ch, mask_ch, gamma=4.0, cap=50.0, base_q=0.75):
    if mask_ch.dim() == 3 and mask_ch.size(1) != 1:
        mask_ch = mask_ch[:, 1:2, :]
    with torch.no_grad():
        pos = torch.clamp(tgt_ch, min=0)
        if mask_ch.any():
            pq = torch.quantile(pos[mask_ch.bool()], base_q)
        else:
            pq = torch.tensor(1.0, device=tgt_ch.device)
        w_mag = torch.clamp((pos / (pq + 1e-9)) ** gamma, max=cap)
    return w_mag * mask_ch.float()

# ========== 新增：Ion 的 log 域变换 ==========
def to_log_domain(x, c):      # x>0,  c>0
    return torch.log(torch.clamp(x, min=1e-12) + c)

def from_log_domain(z, c):    # 反变换
    return torch.exp(z) - c

# -------------------- 学习率调度：Warmup + Cosine --------------------
def make_warmup_cosine(optimizer, total_epochs, warmup_epochs, base_lr, use_cosine=True):
    if not use_cosine:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -------------------- 训练单通道 --------------------
# ================== 替换函数 1：train_single_channel ==================
def train_single_channel(channel_idx, dataset, meta):
    """
    F_Flux: 保持你现有的 L1 + TV 等配置不变。
    Ion_Flux: 用 log 域回归，并在训练期学习全局仿射 (a,b)： y_log* ≈ a * z_pred + b
              其中 z_pred 是模型原始输出（不做 softplus），y_log*=log(y+c)。
              训练结束把 a,b,c 一起保存在 ckpt 里；验证/导出时再做反变换 exp(a*z+b)-c。
    """
    is_F = (channel_idx == 0)
    ch_name = FAMILIES[channel_idx]
    out_dir = os.path.join(Cfg.save_dir, ch_name); os.makedirs(out_dir, exist_ok=True)
    set_seed(Cfg.seed)

    # ===== 数据划分 =====
    N = len(dataset); nval = max(1, int(N * Cfg.val_ratio))
    tr_set, va_set = random_split(dataset, [N - nval, nval],
                                  generator=torch.Generator().manual_seed(Cfg.seed))
    tr = DataLoader(tr_set, batch_size=Cfg.batch, shuffle=True)
    va = DataLoader(va_set, batch_size=Cfg.batch, shuffle=False)
    T = int(meta["T"])

    # ===== 模型（Ion 可用独立的更大容量；未在 Cfg 里定义则回退到通用值）=====
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_F:
        model = PhysicsSeqPredictor(d_model=Cfg.d_model, nhead=Cfg.nhead, num_layers=Cfg.num_layers,
                                    dim_ff=Cfg.dim_ff, dropout=Cfg.dropout, T=T).to(dev)
    else:
        d_model  = getattr(Cfg, "ion_d_model", 192)
        nhead    = getattr(Cfg, "ion_nhead",   8)
        n_layers = getattr(Cfg, "ion_num_layers", 6)
        dim_ff   = getattr(Cfg, "ion_dim_ff",  384)
        dropout  = getattr(Cfg, "ion_dropout", 0.1)
        model = PhysicsSeqPredictor(d_model=d_model, nhead=nhead, num_layers=n_layers,
                                    dim_ff=dim_ff, dropout=dropout, T=T).to(dev)

    # ===== 损失/TV/F 配置 =====
    if is_F:
        base_loss = _make_loss(Cfg.f_loss)
        eps_mask  = Cfg.eps_mask_F
        use_tv_reg = Cfg.use_tv_reg_F
        tv_lambda  = Cfg.tv_lambda_F
    else:
        # Ion 的 log 域训练不再用 softplus，也不做样本加权/分箱正则；用 Huber 更稳
        eps_mask  = Cfg.eps_mask_I
        use_tv_reg = False
        tv_lambda  = 0.0

    # ======= Ion: 估计 log 平移常数 c =======
    ion_c = Cfg.ion_c_min
    if not is_F:
        with torch.no_grad():
            vals = []
            for _, phys_tgt, pmask, _ in tr:
                x = phys_tgt[:, 1:2, :]
                m = (pmask[:, 1:2, :] & (x > 0))
                if m.any(): vals.append(x[m].float())
                if len(vals) > 8: break
            if len(vals):
                allv = torch.cat(vals)
                ion_c = float(torch.quantile(allv, Cfg.ion_c_quantile).item())
            ion_c = max(ion_c, Cfg.ion_c_min)

    # ======= 优化器/调度 =======
    params = list(model.parameters())
    if not is_F:
        # Ion 同时学习全局仿射参数 a,b
        a_aff = nn.Parameter(torch.tensor(1.0, device=dev, dtype=torch.float32))
        b_aff = nn.Parameter(torch.tensor(0.0, device=dev, dtype=torch.float32))
        params += [a_aff, b_aff]
    opt_lr = Cfg.lr if is_F else getattr(Cfg, "ion_lr", 5e-4)
    wd     = Cfg.weight_decay if is_F else getattr(Cfg, "ion_weight_decay", 1e-4)
    opt = torch.optim.AdamW(params, lr=opt_lr, weight_decay=wd)
    sch = make_warmup_cosine(opt, Cfg.max_epochs, Cfg.warmup_epochs, opt_lr, use_cosine=Cfg.use_cosine)

    tr_hist, va_hist = [], []
    best = 1e9; best_path = os.path.join(out_dir, "phys_best.pth")
    saved_batch = False

    # =================== 训练循环 ===================
    for e in range(1, Cfg.max_epochs + 1):
        model.train(); s = 0.0; n = 0

        for s8, phys_tgt, pmask, tvals in tr:
            s8 = s8.to(dev); phys_tgt = phys_tgt.to(dev); pmask = pmask.to(dev); tvals = tvals.to(dev)

            if not saved_batch:
                np.savez_compressed(os.path.join(out_dir, "one_batch_debug.npz"),
                                    s8=s8.cpu().numpy(),
                                    tgt=phys_tgt.cpu().numpy(),
                                    mask=pmask.cpu().numpy(),
                                    t=tvals.cpu().numpy())
                saved_batch = True

            pred = model(s8, tvals)  # (B,2,T)
            pred_ch_raw = pred[:, channel_idx:channel_idx + 1, :]
            tgt_ch = phys_tgt[:, channel_idx:channel_idx + 1, :]
            mask_ch = (pmask[:, channel_idx:channel_idx + 1, :] &
                       (tgt_ch.abs() >= eps_mask))

            if mask_ch.sum() == 0:
                continue

            if is_F:
                # ========= F_Flux：与之前一致 =========
                pred_dom = pred_ch_raw
                loss_e = base_loss(pred_dom, tgt_ch)
                w = mask_ch.float()
                loss_main = (loss_e * w).sum() / (w.sum().clamp_min(1e-6))

                # TV（可选）
                if use_tv_reg and tv_lambda > 0:
                    tv = (pred_dom[:, :, 1:] - pred_dom[:, :, :-1]).abs()
                    denom = (mask_ch[:, :, 1:].float().sum()).clamp_min(1e-6)
                    tv = (tv * mask_ch[:, :, 1:].float()).sum() / denom
                    loss = loss_main + tv_lambda * tv
                else:
                    loss = loss_main
            else:
                # ========= Ion_Flux：log 域 + 学习 (a,b) =========
                pos_mask = (mask_ch & (tgt_ch > 0))
                if pos_mask.sum() == 0:
                    continue
                y_log = torch.log(torch.clamp(tgt_ch, min=1e-12) + ion_c)  # 目标
                z_pred = pred_ch_raw                                         # 模型输出
                z_adj  = a_aff * z_pred + b_aff                              # 学到的仿射

                loss = F.smooth_l1_loss(z_adj[pos_mask], y_log[pos_mask], reduction="mean")

            opt.zero_grad()
            loss.backward()
            if Cfg.clip_grad_norm and Cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), Cfg.clip_grad_norm)
            opt.step()

            s += loss.item() * s8.size(0); n += s8.size(0)

        trl = s / max(1, n); tr_hist.append(trl)
        sch.step()

        # =================== 验证 ===================
        model.eval(); s = 0.0; n = 0
        with torch.no_grad():
            for s8, phys_tgt, pmask, tvals in va:
                s8 = s8.to(dev); phys_tgt = phys_tgt.to(dev); pmask = pmask.to(dev); tvals = tvals.to(dev)
                pred = model(s8, tvals)
                pred_ch_raw = pred[:, channel_idx:channel_idx + 1, :]
                tgt_ch = phys_tgt[:, channel_idx:channel_idx + 1, :]
                mask_ch = (pmask[:, channel_idx:channel_idx + 1, :] &
                           (tgt_ch.abs() >= eps_mask))
                if mask_ch.sum() == 0:
                    continue

                if is_F:
                    pred_dom = pred_ch_raw
                    loss_e = base_loss(pred_dom, tgt_ch)
                    w = mask_ch.float()
                    loss_main = (loss_e * w).sum() / (w.sum().clamp_min(1e-6))
                    val_loss = loss_main
                else:
                    pos_mask = (mask_ch & (tgt_ch > 0))
                    if pos_mask.sum() == 0:
                        continue
                    y_log = torch.log(torch.clamp(tgt_ch, min=1e-12) + ion_c)
                    z_pred = pred_ch_raw
                    z_adj  = a_aff * z_pred + b_aff
                    val_loss = F.smooth_l1_loss(z_adj[pos_mask], y_log[pos_mask], reduction="mean")

                s += float(val_loss) * s8.size(0); n += s8.size(0)

        val = s / max(1, n); va_hist.append(val)
        if is_F:
            print(f"[A-{ch_name}][{e}/{Cfg.max_epochs}] train {trl:.4f} | val {val:.4f}")
        else:
            # 诊断：log 域 μ/σ；用 z_adj 对齐后再看
            with torch.no_grad():
                try:
                    s8_dbg, phys_tgt_dbg, pmask_dbg, tvals_dbg = next(iter(va))
                    s8_dbg = s8_dbg.to(dev); phys_tgt_dbg = phys_tgt_dbg.to(dev)
                    pmask_dbg = pmask_dbg.to(dev); tvals_dbg = tvals_dbg.to(dev)
                    z = model(s8_dbg, tvals_dbg)[:, 1:2, :]
                    tgt = phys_tgt_dbg[:, 1:2, :]
                    m = (pmask_dbg[:, 1:2, :] & (tgt > 0))
                    z_adj = a_aff * z + b_aff
                    y_log_dbg = torch.log(torch.clamp(tgt, min=1e-12) + ion_c)
                    zp = z_adj[m]; yt = y_log_dbg[m]
                    p_mean = float(zp.mean()); p_std = float(zp.std(unbiased=False))
                    t_mean = float(yt.mean()); t_std = float(yt.std(unbiased=False))
                    ratio = p_std / (t_std + 1e-12)
                    print(f"[A-{ch_name}][{e}/{Cfg.max_epochs}] train {trl:.4f} | val {val:.4f}")
                    print(f"[DBG-{ch_name}][{e}] log pred μ/σ={p_mean:.4f}/{p_std:.4f}  "
                          f"log tgt μ/σ={t_mean:.4f}/{t_std:.4f}  ratio={ratio:.3f}  ion_c={ion_c:.2e}  "
                          f"a={float(a_aff):.3f} b={float(b_aff):.3f}")
                except StopIteration:
                    pass

        # ====== 保存最优 ======
        if val < best:
            best = val
            ckpt = {
                "model": model.state_dict(),
                "meta": _clean_meta(meta),
                "hist": {"train": tr_hist, "val": va_hist}
            }
            if not is_F:
                ckpt["ion_affine"] = {"a": float(a_aff.detach().cpu()),
                                      "b": float(b_aff.detach().cpu()),
                                      "c": float(ion_c)}
            torch.save(ckpt, best_path)
            print("  -> saved", best_path)

    # 学习曲线
    with open(os.path.join(out_dir, "learning_curve.csv"), "w", newline="") as f:
        wcsv = csv.writer(f); wcsv.writerow(["epoch", "train", "val"])
        for i, (trv, vv) in enumerate(zip(tr_hist, va_hist), start=1):
            wcsv.writerow([i, trv, vv])

    return best_path


# ================== 替换函数 2：ion_inverse_for_export ==================
def ion_inverse_for_export(z_logits: torch.Tensor, ckpt: dict) -> torch.Tensor:
    """
    把 Ion 的模型输出 z (B,1,T) 还原到物理量域：
        y_hat = exp(a*z + b) - c
    a,b,c 来自训练 ckpt["ion_affine"]；若缺失则回退 a=1,b=0,c=Cfg.ion_c_min
    """
    aff = ckpt.get("ion_affine", None)
    if aff is None:
        a = 1.0; b = 0.0; c = getattr(Cfg, "ion_c_min", 1e-6)
    else:
        a = float(aff.get("a", 1.0))
        b = float(aff.get("b", 0.0))
        c = float(aff.get("c", getattr(Cfg, "ion_c_min", 1e-6)))

    a_t = torch.tensor(a, dtype=z_logits.dtype, device=z_logits.device).view(1, 1, 1)
    b_t = torch.tensor(b, dtype=z_logits.dtype, device=z_logits.device).view(1, 1, 1)
    y = torch.exp(a_t * z_logits + b_t) - c
    return torch.clamp(y, min=0.0)


def residual_bias_penalty(gt, pred, mask, n_bins=8, lam=0.02):
    """
    gt, pred, mask: (B,1,T) 张量，已在同一域（训练所用的域）
    作用：按 gt 分箱，惩罚各箱残差的均值（让每个数值区间的系统性偏差接近 0）
    """
    if lam <= 0:
        return gt.new_tensor(0.0)

    with torch.no_grad():
        g = gt[mask.bool()]
        if g.numel() < 64:
            return gt.new_tensor(0.0)
        qs = torch.quantile(g, torch.linspace(0, 1, n_bins + 1, device=g.device))
    res = (pred - gt)
    loss = gt.new_tensor(0.0)
    total_bins = 0
    for i in range(n_bins):
        lo, hi = qs[i], qs[i+1]
        sel = mask & (gt >= lo) & (gt <= hi)
        if sel.sum() < 16:
            continue
        mean_res = (res[sel].mean())
        loss = loss + mean_res.abs()
        total_bins += 1
    if total_bins == 0:
        return gt.new_tensor(0.0)
    return lam * (loss / total_bins)

# -------------------- 主流程 --------------------
def main():
    set_seed(Cfg.seed)
    os.makedirs(Cfg.save_dir, exist_ok=True)

    dataset, meta = excel_to_physics_dataset(Cfg.old_excel, sheet_name=Cfg.sheet_name)

    # 分通道独立训练
    best_f = train_single_channel(0, dataset, meta)
    best_i = train_single_channel(1, dataset, meta)

    # 载入最优权重，做验证集推理与导出
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_f = torch.load(best_f, map_location=dev)
    ckpt_i = torch.load(best_i, map_location=dev)
    T = int(meta["T"])

    # F：按通用超参
    model_f = PhysicsSeqPredictor(
        d_model=Cfg.d_model, nhead=Cfg.nhead, num_layers=Cfg.num_layers,
        dim_ff=Cfg.dim_ff, dropout=Cfg.dropout, T=T
    ).to(dev)

    # Ion：按“训练时使用”的专属超参来构建（优先从 ckpt 读取，其次从 Cfg 的 ion_*，最后回退默认）
    ion_arch = ckpt_i.get("ion_arch", {
        "d_model": getattr(Cfg, "ion_d_model", 192),
        "nhead": getattr(Cfg, "ion_nhead", 8),
        "num_layers": getattr(Cfg, "ion_num_layers", 6),
        "dim_ff": getattr(Cfg, "ion_dim_ff", 384),
        "dropout": getattr(Cfg, "ion_dropout", 0.1),
    })
    model_i = PhysicsSeqPredictor(
        d_model=ion_arch["d_model"], nhead=ion_arch["nhead"],
        num_layers=ion_arch["num_layers"], dim_ff=ion_arch["dim_ff"],
        dropout=ion_arch["dropout"], T=T
    ).to(dev)

    model_f.load_state_dict(ckpt_f["model"])
    model_i.load_state_dict(ckpt_i["model"])
    model_f.eval();
    model_i.eval()

    # ===== 验证集推理（Ion: log→linear 反变换 + 导出后仿射校准） =====
    N = len(dataset);
    nval = max(1, int(N * Cfg.val_ratio))
    _, va_set = random_split(dataset, [N - nval, nval],
                             generator=torch.Generator().manual_seed(Cfg.seed))
    va = DataLoader(va_set, batch_size=Cfg.batch, shuffle=False)

    preds_f, preds_i, trues, masks = [], [], [], []
    with torch.no_grad():
        for s8, phys_tgt, pmask, tvals in va:
            s8 = s8.to(dev);
            tvals = tvals.to(dev)
            pf = model_f(s8, tvals)[:, 0:1, :]

            # Ion: 先拿到 z，再做 exp(a*z+b)-c 的反变换
            zi = model_i(s8, tvals)[:, 1:2, :]
            pi = ion_inverse_for_export(zi, ckpt_i)

            preds_f.append(pf.detach().cpu())
            preds_i.append(pi.detach().cpu())
            trues.append(phys_tgt)
            masks.append(pmask)

    yhat = torch.cat([torch.cat(preds_f, 0), torch.cat(preds_i, 0)], dim=1)
    ytrue = torch.cat(trues, 0);
    mask = torch.cat(masks, 0)

    # ===== 显示域变换 =====
    meta_disp = _meta_torchify_for_display(meta)
    yhat_disp, ytrue_disp = transform_for_display(
        yhat, ytrue,
        family_sign=meta_disp["family_sign"],
        unit_scale=1000.0, flip_sign=False, clip_nonneg=False, min_display_value=0.0
    )
    mask_eps = mask.clone()
    mask_eps[:, 0:1, :] = mask[:, 0:1, :] & (ytrue_disp[:, 0:1, :].abs() >= Cfg.eps_mask_F)
    mask_eps[:, 1:2, :] = mask[:, 1:2, :] & (ytrue_disp[:, 1:2, :].abs() >= Cfg.eps_mask_I)

    yhat_disp, mono_dbg = monotone_calibrate_ion(
        y_true=ytrue_disp, y_pred=yhat_disp, mask=mask_eps,
        n_bins=24, q_lo=0.01, q_hi=0.99
    )
    if not mono_dbg.get("used", False):
        print("[CAL] Ion monotone calibration skipped:", mono_dbg.get("reason"))
    else:
        print("[CAL] Ion monotone calibration applied: "
              f"bins={mono_dbg['n_bins']} q=({mono_dbg['q_lo']:.2f},{mono_dbg['q_hi']:.2f})")

    # ===== 指标 & 导出 =====
    mts_all = metrics(yhat_disp, ytrue_disp, mask)
    mts_eps = metrics(yhat_disp, ytrue_disp, mask_eps)

    yhat_np = yhat_disp.detach().cpu().numpy()
    ytrue_np = ytrue_disp.detach().cpu().numpy()
    mask_np_all = mask.detach().cpu().numpy().astype(np.uint8)
    mask_np_eps = mask_eps.detach().cpu().numpy().astype(np.uint8)
    chm_eps = channelwise_metrics(ytrue_np, yhat_np, mask_np_eps)

    export_predictions_longtable(
        yhat_disp, ytrue_disp, mask,
        families=FAMILIES, time_values_1d=meta["time_values"],
        out_dir=Cfg.save_dir, filename="physics_predictions_split.xlsx"
    )
    export_metrics_grid(mts_all, FAMILIES, meta["time_values"],
                        out_dir=Cfg.save_dir, filename="physics_metrics_all.xlsx")
    export_metrics_grid(mts_eps, FAMILIES, meta["time_values"],
                        out_dir=Cfg.save_dir, filename="physics_metrics_eps.xlsx")

    os.makedirs(os.path.join(Cfg.save_dir, "all"), exist_ok=True)
    os.makedirs(os.path.join(Cfg.save_dir, "eps"), exist_ok=True)
    write_summary_txt(mts_all, FAMILIES, meta["time_values"], os.path.join(Cfg.save_dir, "all"))
    write_summary_txt(mts_eps, FAMILIES, meta["time_values"], os.path.join(Cfg.save_dir, "eps"))
    with open(os.path.join(Cfg.save_dir, "eps", "summary.txt"), "a", encoding="utf-8") as f:
        for k, v in chm_eps.items():
            f.write(f"{k}: {v}\n")

    plot_timeseries_per_channel(Cfg.save_dir, ytrue_np, yhat_np, mask_np_all,
                                time_values=meta["time_values"], max_n=16)
    parity_scatter_per_channel(Cfg.save_dir, ytrue_np, yhat_np, mask_np_all, suffix="all")
    parity_scatter_per_channel(Cfg.save_dir, ytrue_np, yhat_np, mask_np_eps, suffix="eps")
    heatmap(mts_all["RMSE"], FAMILIES, meta["time_values"], "Physics RMSE (ALL)",
            os.path.join(Cfg.save_dir, "physics_rmse_all.png"))
    heatmap(mts_eps["RMSE"], FAMILIES, meta["time_values"], "Physics RMSE (EPS)",
            os.path.join(Cfg.save_dir, "physics_rmse_eps.png"))
    parity_scatter(yhat_disp, ytrue_disp, mask,
                   os.path.join(Cfg.save_dir, "physics_scatter_all.png"), "Physics Parity (ALL)")
    residual_hist(yhat_disp, ytrue_disp, mask,
                  os.path.join(Cfg.save_dir, "physics_residual_all.png"), "Physics Residuals (ALL)")

    # ===== 达标检查 & 诊断（原样保留） =====
    r2_f = chm_eps.get("R2_F_Flux", None)
    r2_i = chm_eps.get("R2_Ion_Flux", None)
    need_diag = (r2_f is not None and r2_f < Cfg.target_R2_F) or (r2_i is not None and r2_i < Cfg.target_R2_I)
    diag_dir = os.path.join(Cfg.save_dir, "diagnostics");
    os.makedirs(diag_dir, exist_ok=True)
    if need_diag:
        for c, name in enumerate(FAMILIES):
            valid = mask_np_eps[:,c,:].reshape(-1).astype(bool)
            gt = ytrue_np[:,c,:].reshape(-1)[valid]; pd = yhat_np[:,c,:].reshape(-1)[valid]
            res = pd - gt
            plt.figure(figsize=(5,4)); plt.scatter(gt, res, s=3, alpha=0.5); plt.axhline(0, lw=1)
            plt.xlabel(f"{name} GT"); plt.ylabel("Residual (Pred-GT)"); plt.title(f"Residual vs GT — {name}")
            plt.tight_layout(); plt.savefig(os.path.join(diag_dir, f"residual_vs_gt_{name}.png"), dpi=200); plt.close()
            plt.figure(figsize=(5,4)); plt.hist(res, bins=60, alpha=0.8)
            plt.xlabel("Residual"); plt.ylabel("Count"); plt.title(f"Residual Histogram — {name}")
            plt.tight_layout(); plt.savefig(os.path.join(diag_dir, f"residual_hist_{name}.png"), dpi=200); plt.close()

        Tn = ytrue_np.shape[2]
        for c,name in enumerate(FAMILIES):
            se = ( (yhat_np[:,c,:]-ytrue_np[:,c,:])**2 ) * mask_np_eps[:,c,:]
            rmse_t = np.sqrt( np.sum(se, axis=0) / (np.sum(mask_np_eps[:,c,:], axis=0)+1e-9) )
            plt.figure(figsize=(7,3)); plt.plot(rmse_t)
            plt.xlabel("Time index"); plt.ylabel("RMSE"); plt.title(f"RMSE over Time — {name}")
            plt.tight_layout(); plt.savefig(os.path.join(diag_dir, f"rmse_over_time_{name}.png"), dpi=200); plt.close()

        for c,name in enumerate(FAMILIES):
            valid = mask_np_eps[:,c,:].astype(bool)
            se = ((yhat_np[:,c,:]-ytrue_np[:,c,:])**2) * valid
            mse_sample = np.sum(se, axis=1) / (np.sum(valid, axis=1)+1e-9)
            idx = int(np.argmax(mse_sample))
            t = np.arange(Tn)[valid[idx]]
            plt.figure(figsize=(8,3))
            plt.plot(t, ytrue_np[idx,c,valid[idx]], label="GT")
            plt.plot(t, yhat_np[idx,c,valid[idx]], "--", label="Pred")
            plt.legend(); plt.xlabel("Time"); plt.ylabel(name); plt.title(f"Worst Sample — {name}  #{idx}")
            plt.tight_layout(); plt.savefig(os.path.join(diag_dir, f"worst_sample_timeseries_{name}.png"), dpi=200); plt.close()

        np.savez_compressed(os.path.join(diag_dir, "tensors_display_space.npz"),
                            yhat=yhat_np, ytrue=ytrue_np, mask_all=mask_np_all, mask_eps=mask_np_eps,
                            time=np.asarray(meta["time_values"]))
        checklist = {
            "what_to_share": [
                "diagnostics/tensors_display_space.npz",
                "diagnostics/residual_vs_gt_F_Flux.png",
                "diagnostics/residual_vs_gt_Ion_Flux.png",
                "diagnostics/residual_hist_F_Flux.png",
                "diagnostics/residual_hist_Ion_Flux.png",
                "diagnostics/rmse_over_time_F_Flux.png",
                "diagnostics/rmse_over_time_Ion_Flux.png",
                "diagnostics/worst_sample_timeseries_F_Flux.png",
                "diagnostics/worst_sample_timeseries_Ion_Flux.png",
                "F_Flux/learning_curve.csv",
                "Ion_Flux/learning_curve.csv",
                "physics_predictions_split.xlsx",
                "physics_metrics_all.xlsx",
                "physics_metrics_eps.xlsx",
                "all/summary.txt",
                "eps/summary.txt"
            ],
            "note": "如未达标请把这些文件打包给我。"
        }
        with open(os.path.join(diag_dir, "please_share_these.json"), "w", encoding="utf-8") as f:
            json.dump(checklist, f, indent=2, ensure_ascii=False)
        print("\n[DIAG] 指标未达标，已在 diagnostics/ 生成额外诊断清单。")

    save_manifest(Cfg.save_dir)
    print("[OK] Stage A (split) done.")


if __name__ == "__main__":
    main()
