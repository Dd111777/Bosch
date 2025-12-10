# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xlsxwriter
from matplotlib import pyplot as plt

from physio_util import (
    excel_to_morph_dataset_from_old,
    load_new_excel_as_sparse_morph,
    build_sparse_batch,
    FAMILIES,
    TIME_LIST,
)

from stageC_finetune_joint_on_new_pycharm_new import Cfg as CfgC

def basic_stats(x: np.ndarray) -> dict:
    """一维数组的基础统计，自动剔除 NaN。"""
    x = np.asarray(x).astype(float).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "p25": float(q25),
        "median": float(q50),
        "p75": float(q75),
        "max": float(x.max()),
    }


def analyze():
    old_excel = CfgC.old_excel
    new_excel = CfgC.new_excel
    os.makedirs(CfgC.save_root, exist_ok=True)
    out_excel = os.path.join(CfgC.save_root, "dataset_distribution.xlsx")

    print(f"[INFO] 使用旧表: {old_excel}")
    print(f"[INFO] 使用新表: {new_excel}")
    print(f"[INFO] 结果输出到: {out_excel}")

    # ---------- 1. 旧表：用现有工具函数加载 ----------
    ds_old, meta_old = excel_to_morph_dataset_from_old(
        old_excel,
        sheet_name=CfgC.sheet_name,
    )

    static_norm_old_t, phys_old_t, targets_old_norm_t, mask_old_t, time_mat_old_t = ds_old.tensors
    static_norm_old = static_norm_old_t.numpy()
    phys_old = phys_old_t.numpy()
    targets_old = targets_old_norm_t.numpy()
    mask_old = mask_old_t.numpy().astype(bool)
    time_mat_old = time_mat_old_t.numpy()

    # 反规范化静态参数
    norm_stat = meta_old["norm_static"]
    mean_static = norm_stat["mean"].numpy()    # (D_static,)
    std_static  = norm_stat["std"].numpy()     # (D_static,)
    static_old = static_norm_old * std_static[None, :] + mean_static[None, :]  # (N_old, D_static)

    # 反规范化形貌量
    norm_tgt = meta_old["norm_target"]
    mean_fam = norm_tgt["mean"].numpy()        # (K,)
    std_fam  = norm_tgt["std"].numpy()         # (K,)

    targets_old_raw = targets_old.copy()
    for k in range(len(FAMILIES)):
        targets_old_raw[:, k, :] = (
            targets_old_raw[:, k, :] * std_fam[k] + mean_fam[k]
        )

    # ---------- 2. 新表：稀疏形貌 ----------
    recs_new = load_new_excel_as_sparse_morph(new_excel)
    static_new = np.stack([r["static"] for r in recs_new], axis=0)  # (N_new, D_static)

    # 用旧表的 mean/std 构建稀疏 batch（只为对齐时间轴和形貌 family）
    time_values = meta_old["time_values"]
    _, targets_new_raw, mask_new, _ = build_sparse_batch(
        recs_new,
        norm_static_mean=mean_static,
        norm_static_std=std_static,
        time_values=time_values,
    )
    targets_new_raw = targets_new_raw.numpy()               # (N_new, K, T)
    mask_new        = mask_new.numpy().astype(bool)         # (N_new, K, T)

    # ---------- 3. 静态工艺参数：旧 vs 新 ----------
    # 注意：这里的名字只是示意，可以根据你 Excel 真实列名调整
    static_param_names = [
        "APC",
        "Source_RF",
        "LF_RF",
        "SF6",
        "C4F8",
        "DEP_time",
        "Etch_time",
    ]

    # 完全按数据读取结果来，不做额外假设
    D_static = static_new.shape[1]
    rows_static = []

    for j in range(D_static):
        name = static_param_names[j] if j < len(static_param_names) else f"param_{j}"
        stats_old = basic_stats(static_old[:, j])
        stats_new = basic_stats(static_new[:, j])

        row = {"param": name}
        for k, v in stats_old.items():
            row[f"old_{k}"] = v
        for k, v in stats_new.items():
            row[f"new_{k}"] = v

        rows_static.append(row)

    df_static = pd.DataFrame(rows_static)

    # ---------- 4. 形貌量：旧 vs 新（family × time） ----------
    # ---------- 4. 形貌量：旧 vs 新（family × time） ----------
    rows_morph = []
    K = len(FAMILIES)
    T = len(TIME_LIST)

    for k in range(K):
        fam = FAMILIES[k]
        for t_idx in range(T):
            tid = TIME_LIST[t_idx]

            # 旧表：mask_old = True 的位置才有真实标注
            vals_old = targets_old_raw[:, k, t_idx][mask_old[:, k, t_idx]]

            # 新表：只有部分 (fam,t) 有值
            vals_new = targets_new_raw[:, k, t_idx][mask_new[:, k, t_idx]]

            # ==== 单位与符号处理 ====
            if fam == "zmin":
                # zmin: 取绝对值，仍然是 µm
                vals_old = np.abs(vals_old)
                vals_new = np.abs(vals_new)
            else:
                # 其它形貌量: µm -> nm
                vals_old = vals_old * 1e3
                vals_new = vals_new * 1e3

            # 计算统计量（此时已经是“正值 + 正确单位”）
            stats_old = basic_stats(vals_old)
            stats_new = basic_stats(vals_new)

            row = {"family": fam, "time": tid}
            for kk, vv in stats_old.items():
                row[f"old_{kk}"] = vv
            for kk, vv in stats_new.items():
                row[f"new_{kk}"] = vv

            rows_morph.append(row)

    df_morph = pd.DataFrame(rows_morph)
    # ---------- 5. 物理量：F_Flux / Ion_Flux（仅旧表） ----------
    rows_phys = []
    N_old, _, T_phys = phys_old.shape
    chan_names = ["F_Flux", "Ion_Flux"]

    for c in range(2):
        cname = chan_names[c]
        # 整体分布
        stats_all = basic_stats(phys_old[:, c, :])
        row_all = {"channel": cname, "time": "all"}
        row_all.update({f"old_{k}": v for k, v in stats_all.items()})
        rows_phys.append(row_all)

        # 按时间步
        for t_idx in range(T_phys):
            tid = str(t_idx + 1) if t_idx < len(TIME_LIST) else f"{t_idx+1}"
            stats_t = basic_stats(phys_old[:, c, t_idx])
            row_t = {"channel": cname, "time": tid}
            row_t.update({f"old_{k}": v for k, v in stats_t.items()})
            rows_phys.append(row_t)

    df_phys = pd.DataFrame(rows_phys)

    # ---------- 6. 写入 Excel ----------
    with pd.ExcelWriter(out_excel, engine="xlsxwriter") as writer:
        df_static.to_excel(writer, sheet_name="static", index=False)
        df_morph.to_excel(writer, sheet_name="morph", index=False)
        df_phys.to_excel(writer, sheet_name="phys_old", index=False)

    print(f"[OK] 分布统计已写入: {out_excel}")

def set_ieee_style():
    """统一设置 IEEE 风格：小字号、黑白、细线。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
    })


# ---------- Fig 1: 静态工艺参数分布 ----------

def plot_static_from_excel(dist_path, save_path):
    """静态工艺参数平均值 ± 标准差，新旧表对比。"""
    df_static = pd.read_excel(dist_path, sheet_name="static")

    params = df_static["param"].tolist()
    old_mean = df_static["old_mean"].to_numpy(dtype=float)
    old_std  = df_static["old_std"].to_numpy(dtype=float)
    new_mean = df_static["new_mean"].to_numpy(dtype=float)
    new_std  = df_static["new_std"].to_numpy(dtype=float)

    # 防止 NaN 破坏 errorbar
    old_std = np.nan_to_num(old_std, nan=0.0)
    new_std = np.nan_to_num(new_std, nan=0.0)

    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.2))  # IEEE 单栏宽度约 3.5 in

    x = np.arange(len(params))
    width = 0.35

    ax.bar(
        x - width / 2, old_mean, width,
        yerr=old_std,
        error_kw=dict(lw=0.6, capsize=2),
        edgecolor="black", facecolor="white",
        label="Simuliaton table",
    )
    ax.bar(
        x + width / 2, new_mean, width,
        yerr=new_std,
        error_kw=dict(lw=0.6, capsize=2),
        edgecolor="black", facecolor="lightgray",
        hatch="//",
        label="tap-out table",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=30, ha="right")
    ax.set_ylabel("Process parameter (a.u.)")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlim(-0.5, len(params) - 0.5)

    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] 静态参数分布图已保存: {save_path}")


# ---------- Fig 2: zmin 单独一张图 ----------

def plot_morph_zmin_from_excel(dist_path, save_path):
    """zmin 的箱线图（使用 min/p25/median/p75/max），旧表 vs 新表。"""
    df_morph = pd.read_excel(dist_path, sheet_name="morph")

    df_z = df_morph[df_morph["family"] == "zmin"].copy()
    if df_z.empty:
        print("[WARN] morph 表中没有 family == 'zmin' 的行，跳过 zmin 分布图绘制")
        return

    # 优先选 new_count>0 的那一行（你现在就是 9_2），否则退回任何一行
    if "new_count" in df_z.columns and (df_z["new_count"].fillna(0) > 0).any():
        row = df_z.loc[df_z["new_count"].fillna(0).idxmax()]
    else:
        row = df_z.iloc[-1]

    # 从表里直接取 old/new 的五数概括
    old_stats = {
        "whislo": float(row["old_min"]),
        "q1":     float(row["old_p25"]),
        "med":    float(row["old_median"]),
        "q3":     float(row["old_p75"]),
        "whishi": float(row["old_max"]),
        "fliers": [],
        "label":  "Simuliaton table",
    }
    new_stats = {
        "whislo": float(row["new_min"]),
        "q1":     float(row["new_p25"]),
        "med":    float(row["new_median"]),
        "q3":     float(row["new_p75"]),
        "whishi": float(row["new_max"]),
        "fliers": [],
        "label":  "tap-out table",
    }

    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.0, 2.0))

    # Old / New 分别偏移一点
    stats_old = [old_stats]
    stats_new = [new_stats]
    pos_old = [0.4]
    pos_new = [1.5]

    ax.bxp(
        stats_old,
        positions=pos_old,
        widths=0.3,
        showfliers=False,
        boxprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
    )
    ax.bxp(
        stats_new,
        positions=pos_new,
        widths=0.3,
        showfliers=False,
        boxprops=dict(color="black"),
        whiskerprops=dict(color="black", linestyle="--"),
        capprops=dict(color="black", linestyle="--"),
        medianprops=dict(color="black", linestyle="--"),
    )

    ax.set_xticks([0.4, 1.5])
    ax.set_xticklabels(["Simuliaton table", "tap-out table"])
    ax.set_ylabel("zmin (µm)")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # 图例说明虚线的是新表
    ax.plot([], [], color="black", label="Simuliaton table")
    ax.plot([], [], color="black", linestyle="--", label="tap-out table")
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] zmin 箱线图已保存: {save_path}")

# ---------- Fig 3: 其他形貌参数（h0/h1/d0/d1/w） ----------

def plot_morph_others_from_excel(dist_path, save_path):
    """
    h0 / h1 / d0 / d1 / w 的箱线图（使用 min/p25/median/p75/max），
    在若干关键时间点 (3,5,9) 新旧表对比。
    """
    df_morph = pd.read_excel(dist_path, sheet_name="morph")

    target_families = ["h0", "h1", "d0", "d1", "w"]
    target_times = ["3", "5", "9"]

    labels = []
    stats_old_list = []
    stats_new_list = []

    for fam in target_families:
        for t in target_times:
            sel = (df_morph["family"] == fam) & (df_morph["time"] == t)
            if not sel.any():
                continue
            row = df_morph[sel].iloc[0]

            # 要么两边都有统计，要么至少 old 有；如果你想严格要求新旧都有，可以再加一层条件
            if pd.isna(row["old_median"]) and pd.isna(row["new_median"]):
                continue

            label = f"{fam}@{t}"
            labels.append(label)

            # Old
            stats_old = {
                "whislo": float(row["old_min"]),
                "q1":     float(row["old_p25"]),
                "med":    float(row["old_median"]),
                "q3":     float(row["old_p75"]),
                "whishi": float(row["old_max"]),
                "fliers": [],
                "label":  label,
            }
            stats_old_list.append(stats_old)

            # New（可能没有 new 数据，这里用 NaN 检查一下）
            if not pd.isna(row["new_median"]):
                stats_new = {
                    "whislo": float(row["new_min"]),
                    "q1":     float(row["new_p25"]),
                    "med":    float(row["new_median"]),
                    "q3":     float(row["new_p75"]),
                    "whishi": float(row["new_max"]),
                    "fliers": [],
                    "label":  label,
                }
            else:
                # 如果没有 new 数据，就让新表这边空着（也可以选择不画）
                stats_new = None
            stats_new_list.append(stats_new)

    M = len(labels)
    if M == 0:
        print("[WARN] morph 表中没有合适的 h0/h1/d0/d1/w 数据，跳过其他形貌箱线图绘制")
        return

    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    x = np.arange(M)
    pos_old = x - 0.15
    pos_new = x + 0.15

    # 只收集非 None 的 new stats
    stats_new_nonempty = []
    pos_new_nonempty = []
    for s, p in zip(stats_new_list, pos_new):
        if s is not None:
            stats_new_nonempty.append(s)
            pos_new_nonempty.append(p)

    # Old：实线箱
    ax.bxp(
        stats_old_list,
        positions=pos_old,
        widths=0.25,
        showfliers=False,
        boxprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
    )

    # New：虚线箱
    if stats_new_nonempty:
        ax.bxp(
            stats_new_nonempty,
            positions=pos_new_nonempty,
            widths=0.25,
            showfliers=False,
            boxprops=dict(color="black"),
            whiskerprops=dict(color="black", linestyle="--"),
            capprops=dict(color="black", linestyle="--"),
            medianprops=dict(color="black", linestyle="--"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Morphology (nm)")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlim(-0.5, M - 0.5)

    # 图例：实线 = Old, 虚线 = New
    ax.plot([], [], color="black", label="Simuliaton table")
    ax.plot([], [], color="black", linestyle="--", label="tap-out table")
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] 其他形貌箱线图已保存: {save_path}")

def main():
    save_root = CfgC.save_root
    dist_path = os.path.join(save_root, "dataset_distribution.xlsx")

    if not os.path.exists(dist_path):
        raise FileNotFoundError(
            f"找不到 {dist_path}，请先运行 analyze_distrbution.py 生成该文件。"
        )

    fig_static = os.path.join(save_root, "fig_static_dist.pdf")
    fig_zmin   = os.path.join(save_root, "fig_morph_zmin_dist.pdf")
    fig_other  = os.path.join(save_root, "fig_morph_others_dist.pdf")

    plot_static_from_excel(dist_path, fig_static)
    plot_morph_zmin_from_excel(dist_path, fig_zmin)
    plot_morph_others_from_excel(dist_path, fig_other)


if __name__ == "__main__":
    analyze()
    main()

