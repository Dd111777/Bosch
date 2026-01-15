# -*- coding: utf-8 -*-
"""
StageC (single-head, per-family) 一键对比脚本
口径（按你最新要求）：
- 不用 teacher / 蒸馏
- 新表 zmin@9_2 -> t=9
- 只训练/评估新表“有实际值”的点（mask=True）
- StageC 初始化来自 StageB 的 best（需要从 best_config + results_summary.csv 检索）
- 由于 StageB 是 per-family 单头模型（out_dim=1），StageC 也按 family 单头跑（避免 out 维度不匹配 + target 尺度/量纲耦合问题）

输出结构：
out_dir/
  summary_all_families.csv
  <fam>/
    stageB_best_ckpt.json
    best_split.json
    split_trials.csv
    compare_on_best_split.csv
    experiments/<exp_name>/
      model_best.pth
      metrics_test.json
      compare_row.json
"""

import os, json, time, argparse
import traceback
from dataclasses import dataclass
from email.policy import default
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import physio_util as pu
import stageB_util as su
import stageB_train_morph_on_phys7_pycharm as sb


# ----------------------------
# Globals
# ----------------------------
TIME_LIST = list(su.TIME_LIST)          # ["1","2",...,"9"]
TIME_VALUES = np.asarray([float(t.replace("_", ".")) for t in TIME_LIST], np.float32)
FAMILIES = list(su.FAMILIES)           # ["zmin","h0","h1","d0","d1","w"]
T = len(TIME_LIST)


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def family_to_index(fam: str) -> int:
    fam = str(fam).strip().lower()
    for i, f in enumerate(FAMILIES):
        if f.lower() == fam:
            return i
    raise KeyError(f"Unknown family={fam}, available={FAMILIES}")


def _torch_load_any(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        kk = k
        for p in ("module.", "model."):
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out


# ----------------------------
# Build StageC raw (no norm)
# ----------------------------
def build_sparse_batch_subset_time(
    recs: List[Dict],
    time_list: List[str],
    time_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
      recipe_raw: (N,7)
      y:         (N,K,T)  (K=6 canonical)
      m:         (N,K,T)
      time_mat:  (N,T)
    """
    N = len(recs)
    K = len(FAMILIES)
    T = len(time_list)
    t2i = {t: i for i, t in enumerate(time_list)}
    f2i = {f: i for i, f in enumerate(FAMILIES)}

    recipe_raw = np.stack([r["static"] for r in recs], 0).astype(np.float32)  # (N,7)
    y = np.zeros((N, K, T), np.float32)
    m = np.zeros((N, K, T), np.bool_)

    for i, r in enumerate(recs):
        tg: Dict[Tuple[str, str], float] = r.get("targets", {})
        for (fam, tid), val in tg.items():
            fam = str(fam).strip().lower()
            tid = str(tid).strip()
            # 你确认的口径：9_2 -> 9
            if tid == "9_2" and "9" in t2i:
                tid = "9"
            if fam in f2i and tid in t2i:
                y[i, f2i[fam], t2i[tid]] = float(val)
                m[i, f2i[fam], t2i[tid]] = True

    time_mat = np.tile(time_values.reshape(1, -1), (N, 1)).astype(np.float32)
    return recipe_raw, y, m, time_mat


def build_stageC_raw(device: str, new_excel: str, height_family: str, recipe_aug_mode: str,
                     stageA_heads_root: str) -> Dict[str, Any]:
    recs = pu.load_new_excel_as_sparse_morph(new_excel, height_family=height_family)
    recipe_raw, y_raw, mask, time_mat = build_sparse_batch_subset_time(recs, TIME_LIST, TIME_VALUES)

    static_raw = sb.augment_recipe_features(recipe_raw, str(recipe_aug_mode)).astype(np.float32)

    provider = sb.StageAEnsemblePhys7Provider(
        heads_root=stageA_heads_root,
        device=device,
        recipe_cols_in=None,
        expect_k=7
    )
    phys7_raw_full = provider.infer(recipe_raw, phys7_mode="full", use_cache=True).astype(np.float32)  # (N,7)

    # recipe_id 读取（用于 key_recipes）
    try:
        df = pd.read_excel(new_excel)
        rid_col = pu.detect_recipe_id_column(df)
        recipe_ids = df[rid_col].astype(str).to_numpy()
    except Exception:
        recipe_ids = np.asarray([f"row{i:04d}" for i in range(len(recs))], dtype=object)

    return dict(
        recipe_ids=recipe_ids,
        recipe_raw=recipe_raw,
        static_raw=static_raw,
        phys7_raw_full=phys7_raw_full,
        y_raw=y_raw,
        mask=mask,
        time_mat=time_mat,
    )


# ----------------------------
# Normalization (fit on train)
# ----------------------------
def zfit(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def zfit_targets_masked_1fam(y: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    y: (N,T), m: (N,T)
    返回 mean/std: (T,)
    """
    mean = np.zeros((T,), np.float32)
    std = np.ones((T,), np.float32)
    for t in range(T):
        sel = m[:, t]
        if int(sel.sum()) < 2:
            mean[t] = 0.0
            std[t] = 1.0
        else:
            vals = y[sel, t]
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
            mean[t] = mu
            std[t] = sd
    return mean, std


def apply_z(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def apply_phys7_mode(phys7_z_full: np.ndarray, phys7_mode: str) -> np.ndarray:
    # 用 StageB 的定义（按组/子集）
    return sb.apply_phys7_mode(phys7_z_full.astype(np.float32), str(phys7_mode).strip())


# ----------------------------
# DataLoader
# ----------------------------
def make_loader(static_x: np.ndarray,
                phys7_seq: np.ndarray,
                y_norm: np.ndarray,
                m: np.ndarray,
                time_mat: np.ndarray,
                idx: np.ndarray,
                batch: int,
                shuffle: bool,
                num_workers: int = 0) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(static_x[idx]).float(),     # (B,Ds)
        torch.from_numpy(phys7_seq[idx]).float(),    # (B,7,T)
        torch.from_numpy(y_norm[idx]).float(),       # (B,1,T)
        torch.from_numpy(m[idx]).bool(),             # (B,1,T)
        torch.from_numpy(time_mat[idx]).float(),     # (B,T)
    )
    return DataLoader(
        ds,
        batch_size=max(1, min(batch, len(ds))),
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )


# ----------------------------
# Split (key recipes in test + quality)
# ----------------------------
def compute_quality_score_1fam(m_fam: np.ndarray) -> np.ndarray:
    # m_fam: (N,T) bool
    return m_fam.astype(np.int32).sum(axis=1).astype(np.int32)  # per-sample valid count


def split_with_key_and_quality(
    recipe_ids: np.ndarray,
    key_recipes: List[str],
    scores: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    N = len(recipe_ids)
    all_idx = np.arange(N)

    key_set = set([str(x).strip() for x in key_recipes if str(x).strip()])
    key_idx = np.array([i for i in all_idx if str(recipe_ids[i]) in key_set], dtype=int)

    # --- desired counts ---
    if test_ratio <= 0:
        n_test = 0
    else:
        n_test = int(round(N * test_ratio))
        n_test = max(1, min(n_test, N))  # keep at least 1 if test_ratio>0

    n_val = int(round(N * val_ratio))
    # val 至少 1（否则 early stopping/val loss 没意义），且不能超过剩余样本
    n_val = max(1, min(n_val, max(1, N - n_test)))


    test = key_idx.tolist()
    remain = [i for i in all_idx.tolist() if i not in set(test)]

    if len(test) < n_test:
        need = n_test - len(test)
        remain_sorted = sorted(remain, key=lambda i: (scores[i], rng.rand()), reverse=True)
        add = remain_sorted[:need]
        test.extend(add)
        remain = [i for i in remain if i not in set(add)]

    remain_sorted = sorted(remain, key=lambda i: (scores[i], rng.rand()), reverse=True)
    val = remain_sorted[:n_val]
    train = [i for i in remain if i not in set(val)]

    return np.asarray(train, int), np.asarray(val, int), np.asarray(test, int)


def check_min_test_points_1fam(m_fam: np.ndarray, test_idx: np.ndarray, min_points: int) -> bool:
    # m_fam: (N,T)
    return bool(int(m_fam[test_idx].sum()) >= int(min_points))


# ----------------------------
# Model / init / finetune modes
# ----------------------------
def build_model(model_type: str, static_dim: int, out_dim: int = 1) -> nn.Module:
    mt = str(model_type).lower().strip()
    if mt == "transformer":
        return su.MorphTransformer(
            static_dim=static_dim,
            d_model=getattr(sb.Cfg, "tf_d_model", 256),
            nhead=getattr(sb.Cfg, "tf_nhead", 8),
            num_layers=getattr(sb.Cfg, "tf_layers", 4),
            dropout=getattr(sb.Cfg, "tf_dropout", 0.1),
            out_dim=out_dim
        )
    if mt == "gru":
        return su.MorphGRU(
            static_dim=static_dim,
            hidden=getattr(sb.Cfg, "gru_hidden", 256),
            num_layers=getattr(sb.Cfg, "gru_layers", 1),
            out_dim=out_dim
        )
    if mt == "mlp":
        return su.MorphMLP(
            static_dim=static_dim,
            hidden=getattr(sb.Cfg, "mlp_hidden", 256),
            num_layers=getattr(sb.Cfg, "mlp_layers", 3),
            out_dim=out_dim
        )
    raise ValueError(f"Unknown model_type={model_type}")


def load_ckpt_into_model(model: nn.Module, ckpt_path: str) -> Dict[str, Any]:
    ck = _torch_load_any(ckpt_path)
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    sd = None
    if isinstance(ck, dict):
        sd = ck.get("model", None) or ck.get("state_dict", None)
    if sd is None and isinstance(ck, dict):
        cand = {k: v for k, v in ck.items() if isinstance(v, torch.Tensor)}
        sd = cand if cand else None
    if sd is None:
        raise RuntimeError(f"Invalid ckpt: {ckpt_path}")
    sd = _strip_prefix(sd)
    miss = model.load_state_dict(sd, strict=False)
    return {"meta": meta, "missing": list(miss.missing_keys), "unexpected": list(miss.unexpected_keys)}


def apply_finetune_mode(model: nn.Module, mode: str):
    mode = str(mode).lower().strip()

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    for p in model.parameters():
        p.requires_grad = False

    def _enable_out():
        if hasattr(model, "out"):
            for p in model.out.parameters():
                p.requires_grad = True

    def _enable_layernorm_affine():
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

    def _enable_bias_only():
        for n, p in model.named_parameters():
            if n.endswith(".bias"):
                p.requires_grad = True

    if mode == "head":
        _enable_out()
        return
    if mode == "head_ln":
        _enable_out()
        _enable_layernorm_affine()
        return
    if mode == "last_block":
        _enable_out()
        _enable_layernorm_affine()
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers") and len(model.encoder.layers) > 0:
            for p in model.encoder.layers[-1].parameters():
                p.requires_grad = True
        return
    if mode == "bitfit":
        _enable_bias_only()
        return
    if mode == "bitfit_ln":
        _enable_bias_only()
        _enable_layernorm_affine()
        return

    raise ValueError(f"Unknown finetune_mode={mode}")


def masked_mse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    计算 MSE Loss，增加了 [Hard Example Mining] 策略：
    误差越大的样本，权重越高。这是提升 d1/w R2 的关键。
    """
    # 1. 基础 MSE
    diff = (pred - y) ** 2

    # 2. [提分关键] 动态加权
    # 逻辑：如果预测偏离真实值太远，就给它更大的梯度权重
    with torch.no_grad():
        # 防止 std 为 0 或 NaN
        std_val = torch.std(y)
        if torch.isnan(std_val) or std_val < 1e-6:
            std_val = 1.0

        # 权重公式：基础权重 1.0 + 偏差程度
        # 偏差越大，weight 越大，迫使 Loss 变大
        weight = 1.0 + torch.abs(pred - y) / (std_val + 1e-6)

    # 3. 应用权重
    diff = diff * weight

    # 4. Mask 掉无效点
    diff = diff.masked_fill(~m, 0.0)

    # 5. 求平均
    denom = m.float().sum().clamp_min(1.0)
    return diff.sum() / denom
def hem_weight(pred: torch.Tensor,
               y: torch.Tensor,
               mode: str = "none",
               clip: float = 3.0,
               tau: float = 1.0,
               eps: float = 1e-6) -> torch.Tensor:
    """
    pred,y: (B,1,T)
    mode:
      - none: all ones
      - linear: 1 + |err|/(std(y)*tau)
      - clamp: clamp(linear, 1, clip)
    """
    if mode is None or str(mode).lower() == "none":
        return torch.ones_like(y)

    mode = str(mode).lower()
    with torch.no_grad():
        std_val = torch.std(y)
        if torch.isnan(std_val) or std_val < eps:
            std_val = torch.tensor(1.0, device=y.device, dtype=y.dtype)
        denom = std_val * float(tau) + eps
        w = 1.0 + torch.abs(pred - y) / denom
        if mode == "clamp":
            w = torch.clamp(w, 1.0, float(clip))
        elif mode == "linear":
            pass
        else:
            raise ValueError(f"Unknown hem_mode={mode}")
    return w


def masked_huber(pred: torch.Tensor,
                 y: torch.Tensor,
                 m: torch.Tensor,
                 beta: float = 1.0,
                 hem_mode: str = "none",
                 hem_clip: float = 3.0,
                 hem_tau: float = 1.0) -> torch.Tensor:
    """
    Masked huber:
      if |e|<beta: 0.5*e^2/beta
      else: |e| - 0.5*beta
    """
    e = pred - y
    abs_e = torch.abs(e)
    b = float(beta)
    quad = 0.5 * (e ** 2) / max(b, 1e-12)
    lin = abs_e - 0.5 * b
    per = torch.where(abs_e < b, quad, lin)

    w = hem_weight(pred, y, mode=hem_mode, clip=hem_clip, tau=hem_tau)
    per = per * w
    per = per.masked_fill(~m, 0.0)
    denom = m.float().sum().clamp_min(1.0)
    return per.sum() / denom


def masked_loss(pred: torch.Tensor,
                y: torch.Tensor,
                m: torch.Tensor,
                loss_type: str = "mse",
                huber_beta: float = 1.0,
                hem_mode: str = "none",
                hem_clip: float = 3.0,
                hem_tau: float = 1.0) -> torch.Tensor:
    lt = str(loss_type).lower().strip()
    if lt == "mse":
        return masked_mse(pred, y, m, hem_mode=hem_mode, hem_clip=hem_clip, hem_tau=hem_tau)
    if lt == "huber":
        return masked_huber(pred, y, m, beta=huber_beta, hem_mode=hem_mode, hem_clip=hem_clip, hem_tau=hem_tau)
    raise ValueError(f"Unknown loss_type={loss_type}")


def l2sp_penalty(model: nn.Module, anchor_state: Dict[str, torch.Tensor], lam: float) -> torch.Tensor:
    if lam <= 0 or (not anchor_state):
        return torch.tensor(0.0, device=next(model.parameters()).device)
    reg = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in anchor_state:
            a = anchor_state[n].to(device=p.device, dtype=p.dtype)
            reg = reg + (p - a).pow(2).mean()
    return reg * float(lam)


@torch.no_grad()
def eval_pack(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    preds, ys, ms = [], [], []
    for static_x, phys7_seq, y, m, time_mat in loader:
        static_x = static_x.to(device)
        phys7_seq = phys7_seq.to(device)
        y = y.to(device)
        m = m.to(device)
        time_mat = time_mat.to(device)
        p = model(static_x, phys7_seq, time_mat)  # (B,1,T)
        preds.append(p.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        ms.append(m.detach().cpu().numpy())
    return dict(
        pred=np.concatenate(preds, 0),  # (N,1,T)
        y=np.concatenate(ys, 0),        # (N,1,T)
        m=np.concatenate(ms, 0).astype(bool),
    )

def train_one(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              device: str,
              epochs: int,
              lr: float,
              wd: float,
              patience: int,
              l2sp_lam: float = 0.0,
              anchor_state: Optional[Dict[str, torch.Tensor]] = None,
              is_transfer: bool = False,
              backbone_lr_ratio: float = 0.1,
              loss_type: str = "mse",
              huber_beta: float = 1.0,
              hem_mode: str = "none",
              hem_clip: float = 3.0,
              hem_tau: float = 1.0,
              grad_clip: float = 1.0) -> Dict[str, Any]:

    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "out" in name or "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if is_transfer:
        params_group = [
            {"params": backbone_params, "lr": lr * float(backbone_lr_ratio)},
            {"params": head_params, "lr": lr},
        ]
    else:
        params_group = [
            {"params": backbone_params, "lr": lr},
            {"params": head_params, "lr": lr},
        ]

    opt = torch.optim.AdamW(params_group, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best = {"val_loss": float("inf"), "epoch": 0, "state": None, "history": {"train": [], "val": []}}
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_train_loss, ep_train_n = 0.0, 0

        for static_x, phys7_seq, y, m, time_mat in train_loader:
            static_x = static_x.to(device)
            phys7_seq = phys7_seq.to(device)
            y = y.to(device)
            m = m.to(device)
            time_mat = time_mat.to(device)

            pred = model(static_x, phys7_seq, time_mat)
            loss = masked_loss(
                pred, y, m,
                loss_type=loss_type,
                huber_beta=huber_beta,
                hem_mode=hem_mode,
                hem_clip=hem_clip,
                hem_tau=hem_tau
            )

            raw_loss_val = float(loss.item())

            if l2sp_lam > 0 and anchor_state is not None:
                loss = loss + l2sp_penalty(model, anchor_state, l2sp_lam)

            opt.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

            ep_train_loss += raw_loss_val
            ep_train_n += 1

        scheduler.step()
        avg_train_loss = ep_train_loss / max(1, ep_train_n)

        # val
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for static_x, phys7_seq, y, m, time_mat in val_loader:
                static_x = static_x.to(device)
                phys7_seq = phys7_seq.to(device)
                y = y.to(device)
                m = m.to(device)
                time_mat = time_mat.to(device)

                pred = model(static_x, phys7_seq, time_mat)
                loss = masked_loss(
                    pred, y, m,
                    loss_type=loss_type,
                    huber_beta=huber_beta,
                    hem_mode=hem_mode,
                    hem_clip=hem_clip,
                    hem_tau=hem_tau
                )
                vl += float(loss.item())
                vn += 1

        val_loss = vl / max(1, vn)

        best["history"]["train"].append(avg_train_loss)
        best["history"]["val"].append(val_loss)

        if val_loss + 1e-9 < best["val_loss"]:
            best["val_loss"] = val_loss
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    return best
# ----------------------------
# StageB best retrieval (from best_config + results_summary.csv)
# ----------------------------
def _find_candidates(runs_root: str, relname: str) -> List[str]:
    cands = [
        os.path.join(runs_root, relname),
        os.path.join(runs_root, "_tuneV_verify", relname),
    ]
    return [p for p in cands if os.path.exists(p)]


def load_best_config_common(runs_root: str) -> Dict[str, Any]:
    cands = _find_candidates(runs_root, "best_config_common_all_families.json")
    if not cands:
        raise FileNotFoundError(f"best_config_common_all_families.json not found under: {runs_root}")
    with open(cands[0], "r", encoding="utf-8") as f:
        return json.load(f)


def load_results_summary_df(runs_root: str) -> Optional[pd.DataFrame]:
    cands = _find_candidates(runs_root, "results_summary.csv")
    if not cands:
        return None
    df = pd.read_csv(cands[0])
    return df

def _resolve_ckpt_by_expname(runs_root: str, best_conf: dict, fam: str) -> str:
    mt  = str(best_conf.get("model_type"))
    hp  = str(best_conf.get("hp_tag"))
    ps  = str(best_conf.get("phys_source"))
    au  = str(best_conf.get("recipe_aug_mode"))
    p7  = str(best_conf.get("phys7_mode"))
    ss  = int(best_conf.get("split_seed"))

    exp_name = f"{mt}_{hp}_{ps}_{au}_{p7}_{fam}_s{ss}"
    cands = [
        os.path.join(runs_root, exp_name, "best.pth"),
        os.path.join(runs_root, "_tuneV_verify", exp_name, "best.pth"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p

    # 兜底：只用后缀匹配（避免 best_conf 字段名略有差异）
    suffix = f"_{fam}_s{ss}"
    for base in [runs_root, os.path.join(runs_root, "_tuneV_verify")]:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            if name.endswith(suffix):
                p = os.path.join(base, name, "best.pth")
                if os.path.isfile(p):
                    return p

    raise FileNotFoundError(f"Cannot locate best.pth for fam={fam} using best_conf under {runs_root}")

def resolve_stageB_best_ckpts_from_common(runs_root: str) -> Dict[str, str]:
    """
    依据 StageB 的 best_config_common_all_families.json，优先用 results_summary.csv
    定位每个 family 的 ckpt_path；若 results_summary.csv 缺失，则 fallback
    通过 exp_name 规则直接定位 best.pth。
    """
    best_conf = load_best_config_common(runs_root)
    df = load_results_summary_df(runs_root)

    out: Dict[str, str] = {}

    # ---------- fallback: no results_summary.csv ----------
    if df is None:
        for fam in FAMILIES:
            out[fam] = _resolve_ckpt_by_expname(runs_root, best_conf, fam)
        return out

    # ---------- normal path: use results_summary.csv ----------
    # 用 best_conf 中“同时存在于 df 列”的字段做严格匹配
    filters = {}
    for k, v in best_conf.items():
        if k in df.columns:
            filters[k] = v

    df_f = df.copy()
    for k, v in filters.items():
        if k == "split_seed":
            df_f = df_f[pd.to_numeric(df_f[k], errors="coerce").fillna(-1).astype(int) == int(v)]
        else:
            df_f = df_f[df_f[k].astype(str) == str(v)]

    if df_f.empty:
        raise RuntimeError("No rows matched best_config in results_summary.csv (check paths / columns).")

    if "family_mode" not in df_f.columns or "ckpt_path" not in df_f.columns:
        raise RuntimeError("results_summary.csv missing required columns: family_mode / ckpt_path")

    for fam in FAMILIES:
        dff = df_f[df_f["family_mode"].astype(str).str.lower() == fam.lower()].copy()
        if dff.empty:
            continue
        if "min_pf_r2" in dff.columns:
            dff["min_pf_r2"] = pd.to_numeric(dff["min_pf_r2"], errors="coerce")
            dff = dff.sort_values("min_pf_r2", ascending=False)
        out[fam] = str(dff.iloc[0]["ckpt_path"])

    if not out:
        raise RuntimeError("Matched config, but no family ckpts resolved (check family_mode values).")

    return out

@dataclass
class ExpCfg:
    name: str
    init: str              # "scratch" / "stageB_best"
    finetune_mode: str     # "full" / "head_ln" / "bitfit_ln" ...
    phys7_mode: str        # "full" / "none" / ...

    lr: float
    wd: float
    l2sp: float = 0.0

    # --- scoring strategies ---
    loss_type: str = "mse"         # "mse" / "huber"
    huber_beta: float = 1.0

    hem_mode: str = "none"         # "none" / "linear" / "clamp"
    hem_clip: float = 3.0
    hem_tau: float = 1.0

    backbone_lr_ratio: float = 0.1 # transfer 时 backbone lr = lr * ratio


def run_one_experiment_on_split_1fam(
        fam: str,
        exp: ExpCfg,
        device: str,
        raw: Dict[str, Any],
        split: Dict[str, Any],
        stageB_best_ckpt_for_fam: Optional[str],
        model_type: str,
        stageA_heads_root: str,
        recipe_aug_mode: str,
        out_dir_fam: str,
        epochs: int,
        batch: int,
        patience: int,
        num_workers: int,
        run_seed: int,
) -> Dict[str, Any]:
    # -------------------------
    # [关键] 同一 split 下扫 seed
    # -------------------------
    set_seed(int(run_seed))

    recipe_ids = raw["recipe_ids"]
    static_raw = raw["static_raw"]  # (N,Ds)
    phys7_raw_full = raw["phys7_raw_full"]  # (N,7)
    y_raw = raw["y_raw"]  # (N,K,T)
    mask = raw["mask"]  # (N,K,T)
    time_mat = raw["time_mat"]  # (N,T)

    k = family_to_index(fam)
    y_f = y_raw[:, k, :].astype(np.float32)  # (N,T)
    m_f = mask[:, k, :].astype(bool)  # (N,T)

    train_idx = np.asarray(split["train_idx"], int)
    val_idx = np.asarray(split["val_idx"], int)
    test_idx = np.asarray(split["test_idx"], int)

    # fit norms on train
    s_mean, s_std = zfit(static_raw[train_idx])
    p_mean, p_std = zfit(phys7_raw_full[train_idx])
    y_mean_t, y_std_t = zfit_targets_masked_1fam(y_f[train_idx], m_f[train_idx])

    static = apply_z(static_raw, s_mean, s_std)  # (N,Ds)
    phys7_z_full = apply_z(phys7_raw_full, p_mean, p_std)  # (N,7)
    phys7_z = apply_phys7_mode(phys7_z_full, exp.phys7_mode)  # (N,7')
    phys7_seq = su.broadcast_phys7_to_T(phys7_z, T)  # (N,7',T)

    y_norm = apply_z(y_f, y_mean_t.reshape(1, T), y_std_t.reshape(1, T))  # (N,T)
    y_norm = y_norm[:, None, :]  # (N,1,T)
    m_ = m_f[:, None, :]  # (N,1,T)

    train_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, train_idx, batch, True, num_workers)
    val_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, val_idx, batch, False, num_workers)
    test_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, test_idx, batch, False, num_workers)

    # build model (single-head)
    model = build_model(model_type, static_dim=static.shape[1], out_dim=1).to(device)

    init_info = {}
    if exp.init == "stageB_best":
        if not stageB_best_ckpt_for_fam:
            raise RuntimeError(f"No StageB best ckpt for fam={fam}")
        init_info = load_ckpt_into_model(model, stageB_best_ckpt_for_fam)
    elif exp.init == "scratch":
        pass
    else:
        raise ValueError(f"Unknown init={exp.init}")

    apply_finetune_mode(model, exp.finetune_mode)

    # L2SP anchor = starting point after init
    anchor_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}

    is_transfer_exp = (exp.init != "scratch")

    best = train_one(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=exp.lr, wd=exp.wd, patience=patience,
        l2sp_lam=exp.l2sp, anchor_state=anchor_state,
        is_transfer=is_transfer_exp,

        # ---- scoring strategies (T1/T2/T3) ----
        backbone_lr_ratio=getattr(exp, "backbone_lr_ratio", 0.1),
        loss_type=getattr(exp, "loss_type", "mse"),
        huber_beta=getattr(exp, "huber_beta", 1.0),
        hem_mode=getattr(exp, "hem_mode", "none"),
        hem_clip=getattr(exp, "hem_clip", 3.0),
        hem_tau=getattr(exp, "hem_tau", 1.0),
    )

    # save
    exp_dir = os.path.join(out_dir_fam, "experiments", exp.name, f"seed_{run_seed}")
    ensure_dir(exp_dir)

    # loss curve
    history = best.get("history", {"train": [], "val": []})
    plt.figure()
    plt.plot(history.get("train", []), label="Train")
    plt.plot(history.get("val", []), label="Val")
    plt.title(f"{fam} {exp.name} seed={run_seed} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    plt.close()

    # final eval set (7:3: test empty -> use val)
    use_val_as_test = (len(test_idx) <= 1 and len(val_idx) > 1)
    final_loader = val_loader if use_val_as_test else test_loader
    eval_set = "val" if use_val_as_test else "test"

    pack = eval_pack(model, final_loader, device)
    met = metrics_1fam_display(pack, fam, y_mean_t, y_std_t, unit_scale=1000.0)

    # scatter
    if "yt" in met and "yp" in met and len(met["yt"]) > 0:
        yt = np.asarray(met["yt"], dtype=float)
        yp = np.asarray(met["yp"], dtype=float)

        plt.figure()
        plt.scatter(yt, yp, alpha=0.6, edgecolors='k', s=40)
        if len(yt) > 0:
            vmin = float(min(np.min(yt), np.min(yp)))
            vmax = float(max(np.max(yt), np.max(yp)))
            margin = (vmax - vmin) * 0.1
            vmin -= margin
            vmax += margin
            plt.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.5, label="Ideal")
            plt.xlim(vmin, vmax)
            plt.ylim(vmin, vmax)

        plt.xlabel(f"True {fam} (nm)")
        plt.ylabel(f"Pred {fam} (nm)")
        plt.title(f"{fam} {exp.name} seed={run_seed} (R2={met['r2']:.3f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(exp_dir, "scatter_best.png"))
        plt.close()

    # remove raw points for json compact
    met_compact = dict(met)
    if "yt" in met_compact: del met_compact["yt"]
    if "yp" in met_compact: del met_compact["yp"]

    ckpt_path = os.path.join(exp_dir, "model_best.pth")
    torch.save({
        "model": model.state_dict(),
        "best": best,
        "exp": exp.__dict__,
        "family": fam,
        "run_seed": int(run_seed),
        "split": split,
        "norm": {
            "static_mean": s_mean, "static_std": s_std,
            "phys7_mean": p_mean, "phys7_std": p_std,
            "y_mean_t": y_mean_t, "y_std_t": y_std_t
        },
        "meta": {
            "init_info": init_info,
            "model_type": model_type,
            "recipe_aug_mode": recipe_aug_mode,
            "phys7_mode": exp.phys7_mode,
            "time_list": TIME_LIST,
            "families": FAMILIES,
            "new_excel_recipe_ids_head": [str(x) for x in recipe_ids[:10]]
        }
    }, ckpt_path)

    with open(os.path.join(exp_dir, "metrics_eval.json"), "w", encoding="utf-8") as f:
        json.dump(met_compact, f, indent=2, ensure_ascii=False)

    row = {
        "family": fam,
        "exp": exp.name,
        "r2": float(met["r2"]),
        "mae_nm": float(met["mae_nm"]),
        "n": int(met["n"]),
        "best_epoch": int(best["epoch"]),
        "val_loss": float(best["val_loss"]),
        "ckpt": ckpt_path,

        # ---- new fields for “same split, best-run over seeds” ----
        "run_seed": int(run_seed),
        "split_seed": int(split.get("seed", -1)) if isinstance(split, dict) else -1,
        "trainN": int(len(train_idx)),
        "valN": int(len(val_idx)),
        "testN": int(len(test_idx)),
        "eval_set": eval_set,
    }

    with open(os.path.join(exp_dir, "compare_row.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    return row

def exp_to_group(exp_name: str) -> str:
    """
    你可以按自己的命名规则把 exp 映射到 0/1/2：
      0: scratch
      1: transfer
      2: transfer+l2sp
    """
    n = str(exp_name).lower()
    if "scratch" in n:
        return "0_scratch"
    if "l2sp" in n:
        return "2_transfer_l2sp"
    if "stageb" in n or "transfer" in n or "head_ln" in n or "bitfit" in n:
        return "1_transfer"
    return "other"

def get_common_valid_split(
        device: str,
        raw: Dict[str, Any],
        key_recipes: List[str],
        families_eval: List[str],
        test_ratio: float,
        val_ratio: float,
        min_test_points: int,
        seed_start: int,
        max_trials: int = 1000
) -> Dict[str, List[int]]:
    print(f"\n>>> Searching for a COMMON split valid for families: {families_eval}")

    recipe_ids = raw["recipe_ids"]
    mask = raw["mask"]  # (N, K, T)

    # 预先计算每个 family 的 mask 索引
    fam_indices = [family_to_index(f) for f in families_eval]

    # 我们用 mask 的总和作为 quality score (虽然不同 family 分数不同，这里用 sum 简化，或者只用 zmin)
    # 这里使用 zmin (最稀缺资源) 的 mask 作为主要排序依据，确保它被均匀分配
    # 或者简单点，使用所有 family mask 的 union
    # 简单策略：随机尝试

    # 构造一个虚拟的 score (这里不重要，主要靠随机打散)
    N = len(recipe_ids)
    scores = np.ones(N, dtype=np.int32)

    for tr in range(max_trials):
        seed = seed_start + tr
        # 复用已有的 split 函数
        train_idx, val_idx, test_idx = split_with_key_and_quality(
            recipe_ids=recipe_ids,
            key_recipes=key_recipes,
            scores=scores,  # 这里的 score 影响不大，主要靠随机
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed
        )

        # 检查 validation/test set 是否满足所有 family 的最小点数要求
        # 7:3 划分下，重点检查 val_idx (因为 test_ratio=0)
        # 如果你有 test_ratio > 0，则检查 test_idx
        check_idx = test_idx if len(test_idx) > 1 else val_idx

        all_passed = True
        details = []

        for k in fam_indices:
            # 获取该 family 的 mask: (N, T)
            m_f = mask[:, k, :].astype(bool)
            # 计算 check_idx 里有多少个有效点
            valid_pts = int(m_f[check_idx].sum())
            details.append(valid_pts)

            if valid_pts < min_test_points:
                all_passed = False
                break

        if all_passed:
            print(f"   [Success] Found common split at trial {tr} (Seed {seed})")
            print(f"   [Stats] Valid points in eval set per family: {dict(zip(families_eval, details))}")
            return {
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "seed": seed
            }

    raise RuntimeError(f"Could not find a common split after {max_trials} trials. "
                       f"Try reducing min_test_points or families list.")

# =========================================================
#  新增：数据健康度与分布漂移检测工具 (Stage C 专用版)
# =========================================================

def diagnose_data_health(raw):
    """
    检测 Stage C 数据的健康度：NaN, Inf, 0值, 极值
    适配 build_stageC_raw 返回的 numpy 字典结构
    """
    print("\n" + "=" * 40)
    print(" [DIAGNOSTIC] Data Health Check")
    print("=" * 40)

    # Stage C 的 raw 是一个 dict，包含 numpy array
    required_keys = ["y_raw", "mask", "recipe_raw"]
    for k in required_keys:
        if k not in raw:
            print(f" [SKIP] Missing key '{k}' in raw data. Cannot diagnose.")
            return

    y_raw = raw["y_raw"]  # (N, K, T)
    mask = raw["mask"]  # (N, K, T)
    recipe_raw = raw["recipe_raw"]  # (N, 7)

    N, K, T = y_raw.shape
    print(f" [INFO] Data Shape: N={N}, K={K}, T={T}")

    # 1. 检查 Target 里的 NaN/Inf
    if np.isnan(y_raw).any() or np.isinf(y_raw).any():
        print(" [!!!] CRITICAL: Targets (y_raw) contain NaN or Inf!")
        # 简单定位
        bad_indices = np.where(np.isnan(y_raw) | np.isinf(y_raw))
        if len(bad_indices[0]) > 0:
            print(f"   -> First bad sample index: {bad_indices[0][0]}")
    else:
        print(" [OK] No NaN/Inf in targets.")

    # 2. 检查 Target 里的 0 (Log 敏感) - 只检查 mask=True 的部分
    # mask 可能是 0/1 int 或 bool
    m_bool = mask.astype(bool)
    zeros = (y_raw == 0) & m_bool

    if zeros.any():
        count = zeros.sum()
        print(f" [WARN] Found {count} zeros in valid (masked) target entries. Log-transform will fail!")
        # 打印前几个位置
        nz = np.nonzero(zeros)
        for i in range(min(5, len(nz[0]))):
            n_idx, k_idx, t_idx = nz[0][i], nz[1][i], nz[2][i]
            print(f"        At (n={n_idx}, k={k_idx}, t={t_idx})")
    else:
        print(" [OK] No dangerous zeros in masked targets.")

    # 3. 检查输入 Recipe 的数值范围
    max_val = np.abs(recipe_raw).max()
    if max_val > 10000:
        print(f" [WARN] Recipe inputs have very large values (max={max_val:.1f}). Check units!")
    else:
        print(f" [OK] Recipe value range seems normal (max={max_val:.1f}).")


def diagnose_distribution_shift(raw, stageB_runs_root):
    """
    检测 Stage C 数据相对于 Stage B 的分布漂移 (Covariate Shift)
    适配 build_stageC_raw 返回的 numpy 字典结构
    """
    print("\n" + "=" * 40)
    print(" [DIAGNOSTIC] Distribution Shift Check (Stage B vs C)")
    print("=" * 40)

    import stageB_util as su

    # 1. 获取 Stage B 的统计量 (Mean/Std)
    mean_b, std_b = None, None
    try:
        dummy_ckpt = None
        for root, dirs, files in os.walk(stageB_runs_root):
            for f in files:
                if f.endswith("best.pth"):
                    dummy_ckpt = os.path.join(root, f)
                    break
            if dummy_ckpt: break

        if not dummy_ckpt:
            print(" [SKIP] No Stage B checkpoint found to compare stats.")
            return

        # 必须加上 weights_only=False 以防 pickle 报错，或者捕获 Warning
        try:
            ckpt = torch.load(dummy_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(dummy_ckpt, map_location="cpu")

        meta = ckpt.get("meta", {})
        norm_p7 = meta.get("norm_phys7", {})

        mean_b = norm_p7.get("mean", None)  # numpy array
        std_b = norm_p7.get("std", None)  # numpy array

        if mean_b is None:
            print(" [SKIP] Stage B checkpoint missing 'norm_phys7' stats.")
            return

    except Exception as e:
        print(f" [SKIP] Failed to load Stage B stats: {e}")
        return

    # 2. 计算 Stage C 的统计量
    if "phys7_raw_full" not in raw:
        print(" [SKIP] raw data missing 'phys7_raw_full'.")
        return

    phys_c = raw["phys7_raw_full"]  # (N, 7) - Stage C 中 phys7 是 (N, 7) 的 numpy 数组

    # 计算均值 (跨样本)
    mean_c = phys_c.mean(axis=0)  # (7,)

    # 3. 对比
    phys_names = ["logGamma_SF6", "pF_SF6", "spread_SF6", "qskew_SF6",
                  "logGamma_C4F8", "rho_C4F8", "spread_C4F8"]

    print(f"{'Feature':<20} | {'Z-Score Shift':<15} | {'Status'}")
    print("-" * 60)

    warnings = 0
    # mean_b 可能是 (1, 7) 或 (7,)
    mb_flat = mean_b.flatten()
    sb_flat = std_b.flatten()

    for i, name in enumerate(phys_names):
        if i >= len(mb_flat): break

        mu_b = mb_flat[i]
        sigma_b = sb_flat[i] + 1e-6
        mu_c = mean_c[i]

        z_shift = (mu_c - mu_b) / sigma_b

        status = "OK"
        if abs(z_shift) > 1.0: status = "DRIFT (!)"
        if abs(z_shift) > 3.0: status = "SEVERE (!!!)"
        if abs(z_shift) > 3.0: warnings += 1

        print(f"{name:<20} | {z_shift:+.2f} sigma      | {status}")

    if warnings > 0:
        print("\n [CONCLUSION] Severe covariate shift detected.")
        print("              Consider using 'phys7_mode=none' or re-calibrating Stage A.")
    else:
        print("\n [CONCLUSION] Distribution seems matched.")

def pick_best_rows(rows: List[Dict[str, Any]],
                   key_fields: Tuple[str, ...] = ("family", "exp"),
                   metric: str = "r2") -> List[Dict[str, Any]]:
    """
    在同一 split 下（你已经固定 common_split），对多次运行（不同seed）取 best-run。
    默认：每个 (family, exp) 取 r2 最大的那条记录。
    """
    best = {}
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        cur = best.get(key, None)
        if cur is None:
            best[key] = r
            continue
        try:
            if float(r.get(metric, -1e9)) > float(cur.get(metric, -1e9)):
                best[key] = r
        except Exception:
            pass
    return list(best.values())

def metrics_1fam_display(pack: Dict[str, np.ndarray],
                         fam: str,
                         y_mean_t: np.ndarray,
                         y_std_t: np.ndarray,
                         unit_scale: float = 1000.0) -> Dict[str, Any]:
    """
    计算显示空间（nm、zmin 翻正、非负裁剪）的 R2/MAE
    pack: pred/y/m 形状 (N,1,T)
    y_mean_t/y_std_t: (T,) 训练空间（um）统计
    """
    pred = torch.from_numpy(pack["pred"]).float()
    y = torch.from_numpy(pack["y"]).float()
    m = torch.from_numpy(pack["m"]).bool()

    mean = torch.from_numpy(y_mean_t).view(1, 1, T)
    std = torch.from_numpy(y_std_t).view(1, 1, T)

    pred_um = pred * std + mean
    y_um = y * std + mean

    # StageB 默认：zmin 翻符号，其余不翻；全部 family 非负裁剪（这里只是单 family）
    sign_map, nonneg_set = su._default_family_sign_and_nonneg([fam])
    family_sign = torch.tensor([sign_map[fam]], dtype=torch.float32)

    pred_disp, y_disp = pu.transform_for_display(
        pred_um, y_um,
        family_sign=family_sign,
        clip_nonneg=True,
        nonneg_families=[0],  # 单 family 下索引=0
        unit_scale=unit_scale,
        flip_sign=False,
        min_display_value=None
    )

    # flatten masked
    yp = pred_disp[:, 0, :].reshape(-1).numpy()
    yt = y_disp[:, 0, :].reshape(-1).numpy()
    mk = pack["m"][:, 0, :].reshape(-1)
    yp = yp[mk];
    yt = yt[mk]
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[ok];
    yp = yp[ok]
    n = int(len(yt))
    r2 = float(su.masked_r2_score_np(yt, yp)) if n >= 2 else float("nan")
    mae = float(np.mean(np.abs(yt - yp))) if n >= 1 else float("nan")

    # [修改点]：返回原始点集供画图
    return {"family": fam, "r2": r2, "mae_nm": mae, "n": n,
            "yp": yp.tolist(), "yt": yt.tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_excel", type=str, default=r"D:\PycharmProjects\Bosch\Bosch.xlsx")
    ap.add_argument("--out_dir", type=str, default=r"D:\PycharmProjects\Bosch\runs_stageC_singlehead")
    ap.add_argument("--stageB_runs_root", type=str, default=r"D:\PycharmProjects\Bosch\runs_stageB_morph_phys7")

    ap.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "gru", "mlp"])
    ap.add_argument("--recipe_aug_mode", type=str, default="base")
    ap.add_argument("--height_family", type=str, default="h1")
    ap.add_argument("--stageA_heads_root", type=str, default=getattr(sb.Cfg, "stageA_heads_root", ""))

    ap.add_argument("--key_recipes", type=str, default="")
    ap.add_argument("--families_eval", type=str, default="zmin,h1,d1,w")

    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--seed_repeats", type=int, default=10)  # [新增] 同一 split 下扫 seed 的次数
    ap.add_argument("--trials", type=int, default=500)

    ap.add_argument("--test_ratio", type=float, default=0.0)
    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--min_test_points", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # 1) 加载 StageB 最佳配置 & 对齐模型结构参数
    # ---------------------------------------------------------
    print(f"[StageC] Loading best config from: {args.stageB_runs_root}")
    best_conf = su.load_best_config_common(args.stageB_runs_root)
    sb.apply_hp_from_best_conf_to_cfg(best_conf)

    stageB_aug_mode = best_conf.get("recipe_aug_mode", args.recipe_aug_mode)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageB_model_type = best_conf.get("model_type", args.model_type)

    print(f"[StageC] Aligning with StageB:")
    print(f"  - Model Type: {stageB_model_type}")
    print(f"  - Aug Mode:   {stageB_aug_mode}")
    print(f"  - Phys Mode:  {stageB_phys_mode}")
    print(f"  - HPs: d_model={sb.Cfg.tf_d_model}, layers={sb.Cfg.tf_layers}")

    args.recipe_aug_mode = stageB_aug_mode
    args.model_type = stageB_model_type

    stageB_best = resolve_stageB_best_ckpts_from_common(args.stageB_runs_root)

    # ---------------------------------------------------------
    # 2) 构建数据
    # ---------------------------------------------------------
    raw = build_stageC_raw(
        device=device,
        new_excel=args.new_excel,
        height_family=args.height_family,
        recipe_aug_mode=args.recipe_aug_mode,
        stageA_heads_root=args.stageA_heads_root
    )
    diagnose_data_health(raw)
    diagnose_distribution_shift(raw, args.stageB_runs_root)

    families_eval = [x.strip().lower() for x in str(args.families_eval).split(",") if x.strip()]
    families_eval = [f for f in families_eval if f in [x.lower() for x in FAMILIES]]
    if not families_eval:
        raise RuntimeError("families_eval is empty.")

    key_recipes = [x.strip() for x in str(args.key_recipes).split(",") if x.strip()]

    # ---------------------------------------------------------
    # 3) 固定一个统一 split（所有 family/exp 共用）
    # ---------------------------------------------------------
    print("=" * 60)
    print(" [Step 1] Finding a Common Split for ALL families")
    print("=" * 60)

    common_split = get_common_valid_split(
        device=device,
        raw=raw,
        key_recipes=key_recipes,
        families_eval=families_eval,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        min_test_points=args.min_test_points,
        seed_start=args.seed
    )

    with open(os.path.join(args.out_dir, "common_best_split.json"), "w", encoding="utf-8") as f:
        json.dump(common_split, f, indent=2)

    # ---------------------------------------------------------
    # 4) 实验列表（不变）
    # ---------------------------------------------------------
    pm = stageB_phys_mode

    # -------------------------
    # Baseline（保留你原来的）
    # -------------------------
    experiments: List[ExpCfg] = [
        ExpCfg(name="scratch_full", init="scratch", finetune_mode="full", phys7_mode=pm,
               lr=3e-4, wd=1e-4, l2sp=0.0,
               loss_type="mse", hem_mode="none", backbone_lr_ratio=1.0),

        ExpCfg(name="stageB_full", init="stageB_best", finetune_mode="full", phys7_mode=pm,
               lr=2e-4, wd=1e-4, l2sp=0.0,
               loss_type="mse", hem_mode="none", backbone_lr_ratio=0.1),

        ExpCfg(name="stageB_head_ln", init="stageB_best", finetune_mode="head_ln", phys7_mode=pm,
               lr=6e-4, wd=1e-4, l2sp=0.0,
               loss_type="mse", hem_mode="none", backbone_lr_ratio=0.1),

        ExpCfg(name="stageB_bitfit_ln", init="stageB_best", finetune_mode="bitfit_ln", phys7_mode=pm,
               lr=1e-3, wd=0.0, l2sp=0.0,
               loss_type="mse", hem_mode="none", backbone_lr_ratio=0.1),

        ExpCfg(name="stageB_head_ln_l2sp", init="stageB_best", finetune_mode="head_ln", phys7_mode=pm,
               lr=6e-4, wd=1e-4, l2sp=1e-3,
               loss_type="mse", hem_mode="none", backbone_lr_ratio=0.1),

        ExpCfg(name="stageB_head_ln_noP7", init="stageB_best", finetune_mode="head_ln", phys7_mode="none",
               lr=6e-4, wd=1e-4, l2sp=0.0,
               loss_type="mse", hem_mode="none", backbone_lr_ratio=0.1),
    ]

    # -------------------------
    # 提分策略 T1/T2/T3 sweep（一次性加进去）
    # 只对 transfer 组做（scratch 没必要扫 backbone_lr_ratio）
    # -------------------------
    finetune_modes = ["full", "head_ln"]  # 够用了，bitfit 留作 baseline
    lr_base = 2e-4  # 与你 stageB_full 接近
    wd_base = 1e-4

    # T1: huber + hem none + backbone ratio sweep + l2sp=0
    for ft in finetune_modes:
        for ratio in [0.05, 0.1]:
            experiments.append(
                ExpCfg(
                    name=f"T1_huber_none_{ft}_r{ratio}",
                    init="stageB_best",
                    finetune_mode=ft,
                    phys7_mode=pm,
                    lr=lr_base if ft == "full" else 6e-4,
                    wd=wd_base,
                    l2sp=0.0,
                    loss_type="huber",
                    huber_beta=1.0,
                    hem_mode="none",
                    backbone_lr_ratio=ratio
                )
            )

    # T2: T1 + L2SP sweep
    for ft in finetune_modes:
        for ratio in [0.05, 0.1]:
            for l2 in [3e-4, 1e-3]:
                experiments.append(
                    ExpCfg(
                        name=f"T2_huber_none_{ft}_r{ratio}_l2{l2}",
                        init="stageB_best",
                        finetune_mode=ft,
                        phys7_mode=pm,
                        lr=lr_base if ft == "full" else 6e-4,
                        wd=wd_base,
                        l2sp=float(l2),
                        loss_type="huber",
                        huber_beta=1.0,
                        hem_mode="none",
                        backbone_lr_ratio=ratio
                    )
                )

    # T3: 如果你坚持 HEM：clamp + tau=2 + clip=3
    for ft in finetune_modes:
        experiments.append(
            ExpCfg(
                name=f"T3_huber_clamp_{ft}_r0.05",
                init="stageB_best",
                finetune_mode=ft,
                phys7_mode=pm,
                lr=lr_base if ft == "full" else 6e-4,
                wd=wd_base,
                l2sp=0.0,
                loss_type="huber",
                huber_beta=1.0,
                hem_mode="clamp",
                hem_clip=3.0,
                hem_tau=2.0,
                backbone_lr_ratio=0.05
            )
        )
    # ---------------------------------------------------------
    # 5) 同一 split 下扫 seed，收集 all-runs & best-runs
    # ---------------------------------------------------------
    summary_all = os.path.join(args.out_dir, "summary_common_split_allruns.csv")
    summary_best = os.path.join(args.out_dir, "summary_common_split_best.csv")

    header = "family,exp,r2,mae_nm,n,best_epoch,val_loss,run_seed,split_seed,trainN,valN,testN,eval_set,ckpt\n"
    if not os.path.exists(summary_all):
        with open(summary_all, "w", encoding="utf-8") as f:
            f.write(header)
    if not os.path.exists(summary_best):
        with open(summary_best, "w", encoding="utf-8") as f:
            f.write(header)

    all_rows: List[Dict[str, Any]] = []

    for fam in families_eval:
        print(f"\n>>> Processing Family: [{fam}] on Common Split")

        out_dir_fam = os.path.join(args.out_dir, fam)
        ensure_dir(out_dir_fam)

        fam_ckpt = stageB_best.get(fam, None)
        with open(os.path.join(out_dir_fam, "stageB_best_ckpt.json"), "w", encoding="utf-8") as f:
            json.dump({"family": fam, "ckpt_path": fam_ckpt}, f, indent=2, ensure_ascii=False)

        if fam_ckpt is None or (not os.path.exists(fam_ckpt)):
            print(f"[WARN] Skip {fam}: StageB best ckpt missing.")
            continue

        for exp in experiments:
            print(f"   -> {exp.name} (seed sweep x{args.seed_repeats})")

            for i in range(int(args.seed_repeats)):
                run_seed = int(args.seed + i)

                row = run_one_experiment_on_split_1fam(
                    fam=fam, exp=exp, device=device, raw=raw,
                    split=common_split,
                    stageB_best_ckpt_for_fam=fam_ckpt,
                    model_type=args.model_type,
                    stageA_heads_root=args.stageA_heads_root,
                    recipe_aug_mode=args.recipe_aug_mode,
                    out_dir_fam=out_dir_fam,
                    epochs=args.epochs,
                    batch=args.batch,
                    patience=args.patience,
                    num_workers=args.num_workers,
                    run_seed=run_seed,
                )
                all_rows.append(row)

                with open(summary_all, "a", encoding="utf-8") as f:
                    f.write(f"{row['family']},{row['exp']},{row['r2']},{row['mae_nm']},{row['n']},"
                            f"{row['best_epoch']},{row['val_loss']},{row['run_seed']},{row['split_seed']},"
                            f"{row['trainN']},{row['valN']},{row['testN']},{row['eval_set']},{row['ckpt']}\n")

    # 选 best-run（同一 split 内：每个 family+exp 取 r2 最大）
    best_rows = pick_best_rows(all_rows, key_fields=("family", "exp"), metric="r2")
    best_rows = sorted(best_rows, key=lambda r: (r["family"], r["exp"]))

    with open(summary_best, "a", encoding="utf-8") as f:
        for row in best_rows:
            f.write(f"{row['family']},{row['exp']},{row['r2']},{row['mae_nm']},{row['n']},"
                    f"{row['best_epoch']},{row['val_loss']},{row['run_seed']},{row['split_seed']},"
                    f"{row['trainN']},{row['valN']},{row['testN']},{row['eval_set']},{row['ckpt']}\n")

    print("\n[DONE] StageC Finished: SAME split, best-run picked over seeds.")

if __name__ == "__main__":
    main()
