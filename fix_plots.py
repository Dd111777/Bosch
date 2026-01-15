import numpy as np
import matplotlib.pyplot as plt

# -------------------------- IEEE 核心格式配置（适配旧版matplotlib）--------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 300,
    'figure.figsize': (12, 8)  # 适配屏幕预览的尺寸
    # 移除了旧版不支持的 savefig 相关rc参数
})

# -------------------------- 核心实验数据 --------------------------
# baseline: transformer+stageA_pred+time+full (R²=0.95695)
data = {
    '模型结构': {'transformer': 0.96078, 'gru': 0.945253, 'mlp': 0.93968},
    '物理特征来源': {'stageA_pred': 0.96078, 'none': 0.947022},
    '数据增强方式': {'time': 0.96078, 'rf': 0.948236, 'base': 0.950749, 'gas': 0.941388, 'squares': 0.946269},
    'Phys7模式': {'full': 0.96078, 'none': 0.947022, 'only_energy': 0.947155, 'only_flux': 0.94546}
}
baseline_r2 = 0.96078
baseline_color = '#D62728'  # 红色（突出baseline）
ablation_color = '#2C7FB8'  # 蓝色（消融组）

# -------------------------- 2x2子图布局 --------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Ablation Experiments on Morphology Regression Performance', fontsize=14, fontweight='bold')

# 子图1：模型结构消融
ax1 = axes[0, 0]
x1 = list(data['模型结构'].keys())
y1 = list(data['模型结构'].values())
colors1 = [baseline_color if x == 'transformer' else ablation_color for x in x1]
bars1 = ax1.bar(x1, y1, color=colors1)
ax1.set_xlabel('Model Type')
ax1.set_ylabel('Mean R²')
ax1.set_title('Ablation on Model Architecture')
ax1.axhline(y=baseline_r2, color='black', linestyle='--', label='Baseline')
ax1.legend()
ax1.set_ylim(0.925, 0.98)
# 添加数值标签
for bar in bars1:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005, f'{bar.get_height():.4f}', ha='center', fontsize=8)

# 子图2：物理特征来源消融
ax2 = axes[0, 1]
x2 = list(data['物理特征来源'].keys())
y2 = list(data['物理特征来源'].values())
colors2 = [baseline_color if x == 'stageA_pred' else ablation_color for x in x2]
bars2 = ax2.bar(x2, y2, color=colors2)
ax2.set_xlabel('Physical Feature Source')
ax2.set_ylabel('Mean R²')
ax2.set_title('Ablation on Physical Feature Source')
ax2.axhline(y=baseline_r2, color='black', linestyle='--', label='Baseline')
ax2.legend()
ax2.set_ylim(0.925, 0.98)
for bar in bars2:
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0002, f'{bar.get_height():.4f}', ha='center', fontsize=8)

# 子图3：数据增强方式消融
ax3 = axes[1, 0]
x3 = list(data['数据增强方式'].keys())
y3 = list(data['数据增强方式'].values())
colors3 = [baseline_color if x == 'time' else ablation_color for x in x3]
bars3 = ax3.bar(x3, y3, color=colors3)
ax3.set_xlabel('Recipe Augmentation Mode')
ax3.set_ylabel('Mean R²')
ax3.set_title('Ablation on Recipe Augmentation')
ax3.axhline(y=baseline_r2, color='black', linestyle='--', label='Baseline')
ax3.legend()
ax3.set_ylim(0.925, 0.98)
ax3.tick_params(axis='x', rotation=15)
for bar in bars3:
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003, f'{bar.get_height():.4f}', ha='center', fontsize=8)

# 子图4：Phys7模式消融
ax4 = axes[1, 1]
x4 = list(data['Phys7模式'].keys())
y4 = list(data['Phys7模式'].values())
colors4 = [baseline_color if x == 'full' else ablation_color for x in x4]
bars4 = ax4.bar(x4, y4, color=colors4)
ax4.set_xlabel('Phys7 Feature Mode')
ax4.set_ylabel('Mean R²')
ax4.set_title('Ablation on Phys7 Configuration')
ax4.axhline(y=baseline_r2, color='black', linestyle='--', label='Baseline')
ax4.legend()
ax4.set_ylim(0.925, 0.98)
ax4.tick_params(axis='x', rotation=15)
for bar in bars4:
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0002, f'{bar.get_height():.4f}', ha='center', fontsize=8)

# -------------------------- 保存+显示（把savefig参数移到这里，适配旧版matplotlib）--------------------------
plt.tight_layout()
# 关键修复：将bbox_inches、format等参数直接放在savefig中，而非rcParams
plt.savefig('ieee_ablation_plot.png', dpi=300, format='png', bbox_inches='tight')
# 若需PDF矢量图，替换为：
# plt.savefig('ieee_ablation_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')

# 显示图片
# plt.show()



import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import csv

def export_family_csv_all_t(
    pred_denorm_um: np.ndarray,
    y_denorm_um: np.ndarray,
    mask: np.ndarray,
    fam: str,
    out_csv: str,
    clip_nonneg: bool = True,
):
    """
    导出同一 family 的所有时刻合并数据到 CSV
    列：
      sample_idx, t_idx, t_val(optional), y_true_nm, y_pred_nm
    """
    fam_l = str(fam).lower()

    pred = pred_denorm_um
    ytru = y_denorm_um
    m = mask.astype(bool)

    # ---- 规范维度到 (N, K, T) ----
    if ytru.ndim == 1:
        ytru = ytru[:, None, None]
    elif ytru.ndim == 2:
        ytru = ytru[:, None, :]   # 默认 (N,T)
    elif ytru.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported y shape: {ytru.shape}")

    if pred.ndim == 1:
        pred = pred[:, None, None]
    elif pred.ndim == 2:
        pred = pred[:, None, :]
    elif pred.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported pred shape: {pred.shape}")

    N, K, T = ytru.shape

    if m.ndim == 1:
        m = m[:, None, None]
    elif m.ndim == 2:
        if m.shape[1] == T:
            m = m[:, None, :]
        else:
            m = m[:, :, None]
    elif m.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {m.shape}")

    # ---- 单 family 默认 K=1 ----
    k_idx = 0
    y_k = ytru[:, k_idx, :]     # (N,T)
    p_k = pred[:, k_idx, :]     # (N,T)
    m_k = m[:, 0, :]            # (N,T)

    # ---- 口径变换 + 单位换算 ----
    y_k = _display_transform_um(y_k, fam_l, clip_nonneg=clip_nonneg)
    p_k = _display_transform_um(p_k, fam_l, clip_nonneg=clip_nonneg)
    y_nm = y_k * 1000.0
    p_nm = p_k * 1000.0

    # ---- 生成 sample_idx / t_idx 网格 ----
    sample_idx = np.repeat(np.arange(N), T)         # (N*T,)
    t_idx = np.tile(np.arange(T), N)                # (N*T,)

    # ---- flatten 并按 mask 过滤 ----
    m_flat = m_k.reshape(-1)                        # (N*T,)
    y_flat = y_nm.reshape(-1)[m_flat]
    p_flat = p_nm.reshape(-1)[m_flat]
    s_flat = sample_idx[m_flat]
    t_flat = t_idx[m_flat]

    if y_flat.size == 0:
        print(f"[WARN] empty points after mask for fam={fam_l}, skip csv.")
        return

    out_csv = str(out_csv)
    Path(Path(out_csv).parent).mkdir(parents=True, exist_ok=True)

    # 如果 npz 里有 tvals 就写真实 t；没有就留空/用 idx
    tvals = None
    # 兼容可能的字段名
    for key in ["tvals", "t_values", "times"]:
        if key in getattr(mask, "files", []):
            tvals = mask[key]
    # 更稳：从 npz 加载时读；这里函数入参没有 npz 对象，所以默认 None
    # -> 我们先只输出 t_idx，用户需要真实 t 再加（见下方可选增强）

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "t_idx", "y_true_nm", "y_pred_nm"])
        for si, ti, yt, yp in zip(s_flat, t_flat, y_flat, p_flat):
            w.writerow([int(si), int(ti), float(yt), float(yp)])

    print(f"[OK] csv saved: {out_csv}  (rows={y_flat.size})")

# StageB plotting util (uses correct sign conventions internally)
from stageB_util import export_scatter_per_family
def calc_r2_mae(y_true_nm: np.ndarray, y_pred_nm: np.ndarray) -> Tuple[float, float]:
    """返回 (R2, MAE)；输入单位 nm，1D 数组"""
    y_true_nm = y_true_nm.astype(np.float64).ravel()
    y_pred_nm = y_pred_nm.astype(np.float64).ravel()
    if y_true_nm.size == 0:
        return float("nan"), float("nan")

    mae = float(np.mean(np.abs(y_pred_nm - y_true_nm)))

    # R2 = 1 - SSE/SST
    y_mean = float(np.mean(y_true_nm))
    sst = float(np.sum((y_true_nm - y_mean) ** 2))
    sse = float(np.sum((y_true_nm - y_pred_nm) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")
    return r2, mae


# --------------------------
# Path helpers
# --------------------------
def resolve_runs_root(runs_root: str) -> Path:
    p = Path(runs_root)
    if not p.is_absolute():
        p = (Path(__file__).resolve().parent / p).resolve()
    return p


def try_load_best_conf(runs_root: Path) -> Optional[dict]:
    cands = [
        runs_root / "_tuneV_verify" / "best_config_common_all_families.json",
        runs_root / "best_config_common_all_families.json",
    ]
    for p in cands:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] Failed to read best_conf: {p} ({e})")
                return None
    return None


# --------------------------
# Filename parsing
# --------------------------
def parse_eval_pack_filename(npz_path: Path) -> Optional[dict]:
    """
    Parse:
      eval_pack_{exp_name}_{split}.npz
    Where:
      exp_name = {mt}_{hp_tag}_{ps}_{aug}_{p7}_{fam}_s{seed}

    Note hp_tag itself contains underscores, so parse from right.
    """
    name = npz_path.name
    if not (name.startswith("eval_pack_") and name.endswith(".npz")):
        return None

    stem = name[:-4]  # remove .npz
    # split part after last underscore is split_name, but exp_name also has underscores
    # We know format: eval_pack_{exp_name}_{split}
    # So remove prefix, then split once from right.
    body = stem[len("eval_pack_"):]
    if "_" not in body:
        return None

    exp_name, split_name = body.rsplit("_", 1)
    toks = exp_name.split("_")
    if len(toks) < 7:
        return None

    seed_tok = toks[-1]          # s0
    fam = toks[-2]               # zmin / w / h1 ...
    p7 = toks[-3]                # full / Egy / Flux / no
    aug = toks[-4]               # b / t / g / r / sq / ph  (or original string)
    ps = toks[-5]                # stA / no / 0 ...
    mt = toks[0]                 # tf / gru / mlp

    if not seed_tok.startswith("s"):
        return None
    try:
        seed = int(seed_tok[1:])
    except Exception:
        return None

    hp_tag = "_".join(toks[1:-5])  # everything between mt and ps

    return dict(
        path=npz_path,
        exp_name=exp_name,
        split=split_name,
        mt=mt,
        hp_tag=hp_tag,
        ps=ps,
        aug=aug,
        p7=p7,
        fam=fam,
        seed=seed,
    )


def scan_eval_packs(runs_root: Path) -> List[dict]:
    packs = []
    for p in runs_root.rglob("eval_pack_*.npz"):
        info = parse_eval_pack_filename(p)
        if info is not None:
            packs.append(info)
    return packs


# --------------------------
# Grouping
# --------------------------
GroupKey = Tuple[str, str, str, str, str, int]  # (mt,hp_tag,ps,aug,p7,seed)

def group_packs(packs: List[dict], split: Optional[str] = None) -> Dict[GroupKey, List[dict]]:
    g: Dict[GroupKey, List[dict]] = {}
    for it in packs:
        if split is not None and it["split"] != split:
            continue
        key: GroupKey = (it["mt"], it["hp_tag"], it["ps"], it["aug"], it["p7"], it["seed"])
        g.setdefault(key, []).append(it)
    # sort each group by fam for stable output
    for k in g:
        g[k].sort(key=lambda x: x["fam"])
    return g


# --------------------------
# Best-conf interpretation (robust)
# --------------------------
def _extract_best_specs(best_conf: dict) -> List[dict]:
    """
    Return a list of "spec" dict(s) that describe which exp group(s) to plot.

    Supports:
      1) A single dict with keys like hp_tag/model_type/phys_source/recipe_aug_mode/phys7_mode/split_seed
      2) A dict containing per-family best (list/dict)
      3) A list of dict entries
    """
    if best_conf is None:
        return []

    # Case: list
    if isinstance(best_conf, list):
        # Expect each entry to contain hp_tag + split_seed, maybe family
        out = []
        for e in best_conf:
            if isinstance(e, dict) and ("hp_tag" in e or "best_hp_tag" in e):
                out.append(e)
        return out

    if not isinstance(best_conf, dict):
        return []

    # Common patterns
    # - per-family mapping
    for k in ["best_by_family", "by_family", "per_family", "families", "family_best", "best_family"]:
        v = best_conf.get(k, None)
        if isinstance(v, dict):
            # values are configs
            out = []
            for fam, conf in v.items():
                if isinstance(conf, dict):
                    c = dict(conf)
                    c.setdefault("family", fam)
                    out.append(c)
            if out:
                return out
        if isinstance(v, list):
            out = []
            for conf in v:
                if isinstance(conf, dict):
                    out.append(conf)
            if out:
                return out

    # - single best config
    if "hp_tag" in best_conf or "best_hp_tag" in best_conf:
        return [best_conf]

    return []


def _abbr_from_best_conf_entry(e: dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Try to reconstruct (mt, hp_tag, ps, aug, p7, seed, fam) from best_conf entry.
    We only *require* hp_tag and seed; others can be None (then we match broadly).
    """
    hp_tag = e.get("hp_tag", e.get("best_hp_tag", None))
    seed = e.get("split_seed", e.get("best_split_seed", e.get("seed", None)))
    fam = e.get("family", e.get("fam", None))

    # model_type -> mt abbreviation (best effort)
    mt_raw = e.get("model_type", e.get("mt", None))
    mt = None
    if isinstance(mt_raw, str):
        m = mt_raw.lower()
        mt = {"transformer": "tf", "gru": "gru", "mlp": "mlp", "tf": "tf"}.get(m, None)

    # phys_source -> ps
    ps_raw = e.get("phys_source", e.get("ps", None))
    ps = None
    if isinstance(ps_raw, str):
        s = ps_raw.lower()
        ps = {"stagea_pred": "stA", "stagea": "stA", "none": "no", "zero": "0", "sta": "stA", "no": "no", "0": "0"}.get(s, None)

    # recipe_aug_mode -> aug
    aug_raw = e.get("recipe_aug_mode", e.get("aug", None))
    aug = None
    if isinstance(aug_raw, str):
        a = aug_raw.lower()
        aug = {"base": "b", "time": "t", "gas": "g", "rf": "r", "squares": "sq", "phys": "ph",
               "b": "b", "t": "t", "g": "g", "r": "r", "sq": "sq", "ph": "ph"}.get(a, None)

    # phys7_mode -> p7
    p7_raw = e.get("phys7_mode", e.get("p7", None))
    p7 = None
    if isinstance(p7_raw, str):
        p = p7_raw.lower()
        p7 = {"full": "full", "only_energy": "Egy", "only_flux": "Flux", "none": "no",
              "egy": "Egy", "flux": "Flux", "no": "no"}.get(p, None)

    # normalize seed to int
    seed_i = None
    try:
        if seed is not None:
            seed_i = int(seed)
    except Exception:
        seed_i = None

    # normalize fam to lowercase (matches filenames)
    if isinstance(fam, str):
        fam = fam.lower()

    return mt, hp_tag, ps, aug, p7, seed_i, fam


def select_groups_from_best_conf(groups: Dict[GroupKey, List[dict]], best_conf: dict) -> List[Tuple[GroupKey, List[dict]]]:
    """
    Use best_conf to select which group(s) to plot.
    - If best_conf indicates per-family best, you may end up with multiple groups.
    - If best_conf indicates one best hp, select the matching group and plot all its families.
    """
    specs = _extract_best_specs(best_conf)
    if not specs:
        return []

    selected: Dict[GroupKey, List[dict]] = {}

    for e in specs:
        mt, hp_tag, ps, aug, p7, seed, fam = _abbr_from_best_conf_entry(e)
        if hp_tag is None:
            continue

        # Match groups by available constraints
        for k, items in groups.items():
            k_mt, k_hp, k_ps, k_aug, k_p7, k_seed = k
            if k_hp != hp_tag:
                continue
            if seed is not None and k_seed != seed:
                continue
            if mt is not None and k_mt != mt:
                continue
            if ps is not None and k_ps != ps:
                continue
            if aug is not None and k_aug != aug:
                continue
            if p7 is not None and k_p7 != p7:
                continue

            if fam is None:
                # select whole group
                selected[k] = items
            else:
                # select only that family pack within the group
                fam_items = [it for it in items if it["fam"].lower() == fam]
                if fam_items:
                    selected[k] = fam_items
            # do not break; allow multiple matches (rare) but fine

    return list(selected.items())


def select_group_latest(groups: Dict[GroupKey, List[dict]]) -> Optional[Tuple[GroupKey, List[dict]]]:
    """Pick the group whose any file is most recently modified."""
    best = None
    best_mtime = -1.0
    for k, items in groups.items():
        m = max(it["path"].stat().st_mtime for it in items)
        if m > best_mtime:
            best_mtime = m
            best = (k, items)
    return best


# --------------------------
# Plotting
# --------------------------
def plot_group(group_key: GroupKey,
               items: List[dict],
               out_subdir: str,
               clip_nonneg: bool = True) -> None:
    """
    items: list of packs (usually multiple families)
    """
    mt, hp_tag, ps, aug, p7, seed = group_key
    out_root = Path(out_subdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[GROUP] mt={mt} | hp={hp_tag} | ps={ps} | aug={aug} | p7={p7} | seed={seed}")
    print(f"[GROUP] families = {[it['fam'] for it in items]}")
    print(f"[OUT]   -> {out_root}")

    for it in items:
        npz_path: Path = it["path"]
        split_name = it["split"]
        exp_name = it["exp_name"]
        fam = it["fam"]

        try:
            z = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"[WARN] cannot load {npz_path}: {e}")
            continue

        need = ["pred_denorm_um", "y_denorm_um", "mask"]
        if any(k not in z.files for k in need):
            print(f"[WARN] skip {npz_path.name}: missing keys, have={list(z.files)}")
            continue

        pred_denorm_um = z["pred_denorm_um"]
        y_denorm_um = z["y_denorm_um"]
        m = z["mask"]

        # single-family plotting
        families = [str(fam).lower()]

        out_png = Path(out_root) / f"{fam.lower()}.png"
        out_csv = Path(out_root) / f"{fam.lower()}.csv"
        export_family_csv_all_t(
            pred_denorm_um=pred_denorm_um,
            y_denorm_um=y_denorm_um,
            mask=m,
            fam=fam,
            out_csv=str(out_csv),
            clip_nonneg=clip_nonneg,
        )

        r2, mae, npt = plot_scatter_family_all_t(
            pred_denorm_um=pred_denorm_um,
            y_denorm_um=y_denorm_um,
            mask=m,
            fam=fam,
            out_png=str(out_png),
            clip_nonneg=clip_nonneg,
            title_suffix="",  # IEEE 更推荐不在图里放标题
            annotate_metrics=False,  # 指标写在图内
        )

        print(f"[OK] {split_name} | {fam.lower():>6s} | R2={r2:.4f} | MAE={mae:.2f} nm | N={npt} | -> {out_png.name}")


def _display_transform_um(y_um: np.ndarray, fam: str, clip_nonneg: bool = True) -> np.ndarray:
    """
    和 StageB 的论文口径对齐（最关键：Zmin 翻符号；可选：非负裁剪）
    输入/输出单位：um
    """
    fam_l = str(fam).lower()

    # 1) Zmin 翻符号（你的 stageB_util 默认就是这样）
    if fam_l == "zmin":
        y_um = -y_um

    # 2) 非负裁剪（你之前默认全 family 都 clip；这里保持一致）
    if clip_nonneg:
        y_um = np.maximum(y_um, 0.0)

    return y_um

def plot_scatter_family_all_t(
    pred_denorm_um: np.ndarray,
    y_denorm_um: np.ndarray,
    mask: np.ndarray,
    fam: str,
    out_png: str,
    clip_nonneg: bool = True,
    title_suffix: str = "",
    annotate_metrics: bool = True,   # 是否把指标写在图里
):
    """
    同一 family 所有时刻合并一张散点图：
      x = y_true (nm) , y = y_pred (nm)
    返回： (r2, mae_nm, n_points)
    """
    fam_l = str(fam).lower()

    pred = pred_denorm_um
    ytru = y_denorm_um
    m = mask.astype(bool)

    # ---- 规范维度到 (N, K, T) ----
    if ytru.ndim == 1:
        ytru = ytru[:, None, None]
    elif ytru.ndim == 2:
        # 默认当作 (N,T)，K=1
        ytru = ytru[:, None, :]
    elif ytru.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported y shape: {ytru.shape}")

    if pred.ndim == 1:
        pred = pred[:, None, None]
    elif pred.ndim == 2:
        pred = pred[:, None, :]
    elif pred.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported pred shape: {pred.shape}")

    N, K, T = ytru.shape

    if m.ndim == 1:
        m = m[:, None, None]
    elif m.ndim == 2:
        if m.shape[1] == T:
            m = m[:, None, :]
        else:
            m = m[:, :, None]
    elif m.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {m.shape}")

    # ---- 单 family 默认 K=1，取 k_idx=0 ----
    k_idx = 0
    y_k = ytru[:, k_idx, :]     # (N,T)
    p_k = pred[:, k_idx, :]     # (N,T)
    m_k = m[:, 0, :]            # (N,T)

    # ---- 口径变换 + 单位换算 ----
    y_k = _display_transform_um(y_k, fam_l, clip_nonneg=clip_nonneg)
    p_k = _display_transform_um(p_k, fam_l, clip_nonneg=clip_nonneg)

    y_nm = y_k * 1000.0
    p_nm = p_k * 1000.0

    # ---- flatten across time ----
    y_flat = y_nm[m_k]
    p_flat = p_nm[m_k]

    if y_flat.size == 0:
        print(f"[WARN] empty points after mask for fam={fam_l}, skip.")
        return float("nan"), float("nan"), 0

    r2, mae = calc_r2_mae(y_flat, p_flat)
    npt = int(y_flat.size)

    # ---- IEEE-like plot styling ----
    out_png = str(out_png)
    Path(Path(out_png).parent).mkdir(parents=True, exist_ok=True)

    # IEEE 单栏常用宽度 ~3.5in；双栏 ~7.16in，这里按单栏图设置
    plt.figure(figsize=(3.5, 3.0), dpi=300)
    ax = plt.gca()

    # 散点：小点、轻透明
    ax.scatter(y_flat, p_flat, s=6, alpha=0.35)

    # y=x 参考线（细线）
    mn = float(np.min([y_flat.min(), p_flat.min()]))
    mx = float(np.max([y_flat.max(), p_flat.max()]))
    ax.plot([mn, mx], [mn, mx], linewidth=0.8)

    ax.set_xlabel("Measured (nm)", fontsize=8)
    ax.set_ylabel("Predicted (nm)", fontsize=8)

    # 等比例 + 统一范围
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect("equal", adjustable="box")

    # 刻度字体
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)

    # 边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # 不要大标题（IEEE 常用 caption），但你需要也可以保留 suffix
    if title_suffix:
        ax.set_title(f"{fam_l} {title_suffix}", fontsize=8)

    # 指标标注（放在图内左上角）
    if annotate_metrics:
        txt = f"$R^2$={r2:.3f}\nMAE={mae:.1f} nm\nN={npt}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=8)

    plt.tight_layout(pad=0.2)
    plt.savefig(out_png)
    plt.close()

    return r2, mae, npt

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="runs_stageB_morph_phys7",
                        help="Path to StageB runs root (relative to script or absolute).")
    parser.add_argument("--split", type=str, default="test", help="Which split to plot (test/val/train).")
    parser.add_argument("--pick", type=str, default="bestconf",
                        choices=["bestconf", "latest", "hp"],
                        help="How to select which run to plot.")
    parser.add_argument("--hp_tag", type=str, default=None,
                        help="When --pick hp: specify hp_tag exactly as in filenames.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional: restrict to a split_seed (e.g., 0).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for figures. Default: <RUNS_ROOT>/figs_scatter_fix")
    parser.add_argument("--clip_nonneg", action="store_true", default=True,
                        help="Clip non-negative for families (StageB paper convention). Default ON.")
    parser.add_argument("--no_clip_nonneg", action="store_false", dest="clip_nonneg",
                        help="Disable non-negative clipping.")
    args = parser.parse_args()

    runs_root = resolve_runs_root(args.runs_root)
    print("[DBG] cwd       =", os.getcwd())
    print("[DBG] scriptdir =", Path(__file__).resolve().parent)
    print("[DBG] RUNS_ROOT =", runs_root, "exists=", runs_root.exists())
    if not runs_root.exists():
        raise FileNotFoundError(f"RUNS_ROOT not exists: {runs_root}")

    packs = scan_eval_packs(runs_root)
    print(f"[DBG] scanned eval_pack npz = {len(packs)} under {runs_root}")
    if not packs:
        raise RuntimeError("No eval_pack_*.npz found. Check runs_root.")

    groups = group_packs(packs, split=args.split)
    print(f"[DBG] groups for split='{args.split}' = {len(groups)}")

    # Decide output root
    if args.out is None:
        out_root = runs_root / "figs_scatter_fix"
    else:
        out_root = resolve_runs_root(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_groups: List[Tuple[GroupKey, List[dict]]] = []

    if args.pick == "bestconf":
        best_conf = try_load_best_conf(runs_root)
        if best_conf is None:
            print("[WARN] best_config_common_all_families.json not found. Fallback to --pick latest.")
            g = select_group_latest(groups)
            if g is None:
                raise RuntimeError("No group available.")
            selected_groups = [g]
        else:
            selected_groups = select_groups_from_best_conf(groups, best_conf)
            if not selected_groups:
                print("[WARN] best_conf found but cannot match any eval_pack. Fallback to --pick latest.")
                g = select_group_latest(groups)
                if g is None:
                    raise RuntimeError("No group available.")
                selected_groups = [g]

    elif args.pick == "latest":
        g = select_group_latest(groups)
        if g is None:
            raise RuntimeError("No group available.")
        selected_groups = [g]

    elif args.pick == "hp":
        if not args.hp_tag:
            raise ValueError("--pick hp requires --hp_tag")
        # match by hp_tag and optional seed
        for k, items in groups.items():
            mt, hp_tag, ps, aug, p7, seed = k
            if hp_tag != args.hp_tag:
                continue
            if args.seed is not None and seed != args.seed:
                continue
            selected_groups.append((k, items))
        if not selected_groups:
            raise FileNotFoundError(f"No group matched hp_tag={args.hp_tag} seed={args.seed} split={args.split}")

    # Plot
    print(f"[DBG] selected groups = {len(selected_groups)}")
    for k, items in selected_groups:
        # If user requested seed restriction (bestconf/latest may return multiple seeds)
        if args.seed is not None and k[-1] != args.seed:
            continue

        # Put each group into its own folder name (sanitized)
        mt, hp_tag, ps, aug, p7, seed = k
        safe = f"{mt}__{hp_tag}__{ps}__{aug}__{p7}__s{seed}__{args.split}"
        safe = safe.replace(":", "_").replace("/", "_").replace("\\", "_")
        plot_group(k, items, out_subdir=str(out_root / safe), clip_nonneg=args.clip_nonneg)

    print("\n[DONE] All plots finished.")
    print("[OUT_ROOT]", out_root)


if __name__ == "__main__":
    main()
