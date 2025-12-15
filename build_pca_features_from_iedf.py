# -*- coding: utf-8 -*-
import os
import re
import glob
import json
import numpy as np
import pandas as pd

# ========= 需要安装 =========
# pip install scikit-learn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================== 配置区 ==================
CASE_XLSX   = r"D:\PycharmProjects\Bosch\case.xlsx"
CASE_SHEET  = "case"
CASE_ID_COL = "input"     # 你的 case id 列

IEDF_ROOT   = r"D:\BaiduNetdiskDownload\TSV"

# 你真实结构：SF6 sheath2；C4F8 sheath1
TARGETS = {
    ("SF6",  "sheath2"): ["F_1p", "SF3_1p", "SF4_1p", "SF5_1p"],
    ("C4F8", "sheath1"): ["CF3_1p", "C2F3_1p"],
}

# 能量网格（建议 SF6 用更密；C4F8 能量低，用更少点）
NGRID = {
    ("SF6",  "sheath2"): 128,
    ("C4F8", "sheath1"): 64,
}

# PCA 维度（和你的 Phys7 对齐：k=7）
PCA_K = 7

# 输出
OUT_XLSX = r"D:\PycharmProjects\Bosch\case_with_pca7.xlsx"
OUT_META = r"D:\PycharmProjects\Bosch\pca7_manifest.json"

EPS = 1e-30

# ============================================================
def normalize_case_id(cid: str) -> str:
    cid = str(cid).strip()
    if re.fullmatch(r"\d+", cid):
        return f"cas{cid}"
    m = re.fullmatch(r"(?i)case(\d+)", cid)
    if m:
        return f"cas{m.group(1)}"
    return cid

def parse_gas_sheath_from_filename(fp: str):
    base = os.path.basename(fp)
    m = re.match(r"^([A-Za-z0-9]+)_(sheath\d+)_energy_distribution\.csv$", base)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def pick_energy_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "energy" in c.lower():
            return c
    return df.columns[0]

def pick_ion_col(df: pd.DataFrame, ion_prefix: str):
    pref = ion_prefix.strip().lower()
    for c in df.columns:
        if c.strip().lower().startswith(pref):
            return c
    return None

def trapz_compat(y, x):
    # numpy 新版用 trapezoid，老版用 trapz
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        return np.trapz(y, x)
    return fn(y, x)

def read_target_files_for_case(case_id: str):
    pattern = os.path.join(IEDF_ROOT, "scan*", str(case_id), "*_energy_distribution.csv")
    fps = sorted(glob.glob(pattern))
    got = {}
    for fp in fps:
        gas, sheath = parse_gas_sheath_from_filename(fp)
        if (gas, sheath) in TARGETS:
            got[(gas, sheath)] = fp
    return got

def make_energy_grid_from_file(fp: str, ngrid: int):
    df = pd.read_csv(fp)
    e_col = pick_energy_col(df)
    x = df[e_col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    x = np.sort(x)
    # 用文件自身范围构造 grid（不引入固定阈值，也不怕 C4F8 只有 12eV）
    return np.linspace(float(x[0]), float(x[-1]), int(ngrid), dtype=float)

def ion_shape_vector(fp: str, gas: str, sheath: str, ions: list[str], grid: np.ndarray):
    """
    返回按离子拼接的形状向量：
    对每个离子：y -> Gamma -> g=y/(Gamma+eps) -> 插值到 grid
    """
    df = pd.read_csv(fp)
    e_col = pick_energy_col(df)
    x = df[e_col].to_numpy(dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    if x.size < 2:
        return None

    # 排序
    idx = np.argsort(x)
    x = x[idx]

    vecs = []
    missing_ions = 0

    for ion in ions:
        c = pick_ion_col(df, ion)
        if c is None:
            # 离子列不存在：用全零（表示“无该离子形状信息”）
            vecs.append(np.zeros_like(grid, dtype=float))
            missing_ions += 1
            continue

        y = df[c].to_numpy(dtype=float)[ok][idx]
        y = np.where(np.isfinite(y), y, 0.0)
        y = np.maximum(y, 0.0)

        Gamma = float(trapz_compat(y, x))
        if not np.isfinite(Gamma) or Gamma <= 0:
            g = np.zeros_like(y, dtype=float)
        else:
            g = y / (Gamma + EPS)

        # 插值到 grid；超出范围填 0（形状外推为 0）
        gi = np.interp(grid, x, g, left=0.0, right=0.0)
        vecs.append(gi.astype(float))

    return np.concatenate(vecs, axis=0), missing_ions

def main():
    df_case = pd.read_excel(CASE_XLSX, sheet_name=CASE_SHEET)
    if CASE_ID_COL not in df_case.columns:
        raise KeyError(f"CASE_ID_COL='{CASE_ID_COL}' 不在列名中：{list(df_case.columns)}")

    case_ids = df_case[CASE_ID_COL].astype(str).map(normalize_case_id).tolist()

    # 先为每个 (gas,sheath) 选一个“参考文件”来生成能量 grid
    # 注意：这里用全数据找第一个存在的文件，grid 不是学习参数，风险很小；严格CV版见脚本B
    grids = {}
    for key in TARGETS:
        grids[key] = None

    # 找到每个 key 的首个可用文件
    for cid in case_ids:
        files = read_target_files_for_case(cid)
        for key in TARGETS:
            if grids[key] is None and key in files:
                grids[key] = make_energy_grid_from_file(files[key], NGRID[key])
        if all(grids[k] is not None for k in grids):
            break

    for k, g in grids.items():
        if g is None:
            raise RuntimeError(f"找不到任何 {k} 的 IEDF 文件，无法生成能量网格。请检查目录/文件名。")

    # 构建高维 IEDF 向量矩阵
    X_list = []
    ok_case = []
    miss_count = 0
    miss_ion_total = 0

    # 固定拼接顺序（保证可复现）
    keys_order = list(TARGETS.keys())  # [("SF6","sheath2"), ("C4F8","sheath1")]

    for cid in case_ids:
        files = read_target_files_for_case(cid)
        all_parts = []
        bad = False

        for key in keys_order:
            if key not in files:
                # 缺该文件：整段置零
                n = NGRID[key] * len(TARGETS[key])
                all_parts.append(np.zeros((n,), dtype=float))
                miss_count += 1
                continue

            fp = files[key]
            v, mi = ion_shape_vector(fp, key[0], key[1], TARGETS[key], grids[key])
            if v is None:
                bad = True
                break
            miss_ion_total += mi
            all_parts.append(v)

        if bad:
            # 极端坏样本，跳过（也可以改成全零）
            continue

        X_list.append(np.concatenate(all_parts, axis=0))
        ok_case.append(cid)

    X = np.vstack(X_list) if X_list else np.zeros((0, 1))
    if X.shape[0] == 0:
        raise RuntimeError("没有构建出任何样本向量，请检查 case_id 与目录对应关系。")

    # 标准化 + PCA
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_K, random_state=0)
    Z = pca.fit_transform(Xs)  # [n_samples, k]

    # 写回 Excel：按 case_id 对齐（ok_case 子集）
    df_pca = pd.DataFrame(Z, columns=[f"pca_{i+1}" for i in range(PCA_K)])
    df_pca["case_id"] = ok_case

    df_out = df_case.copy()
    df_out["_case_id_tmp_"] = df_out[CASE_ID_COL].astype(str).map(normalize_case_id)
    df_out = df_out.merge(df_pca, how="left", left_on="_case_id_tmp_", right_on="case_id")
    df_out = df_out.drop(columns=["_case_id_tmp_", "case_id"])

    os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)
    df_out.to_excel(OUT_XLSX, index=False)

    meta = {
        "case_xlsx": CASE_XLSX,
        "iedf_root": IEDF_ROOT,
        "targets": {f"{k[0]}_{k[1]}": v for k, v in TARGETS.items()},
        "ngrid": {f"{k[0]}_{k[1]}": int(NGRID[k]) for k in NGRID},
        "pca_k": PCA_K,
        "X_dim": int(X.shape[1]),
        "n_samples_used": int(X.shape[0]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "missing_file_segments_count": int(miss_count),
        "missing_ion_columns_total": int(miss_ion_total),
        "out_xlsx": OUT_XLSX,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {OUT_XLSX}")
    print(f"[OK] Meta : {OUT_META}")
    print(f"[INFO] PCA explained variance sum = {meta['explained_variance_ratio_sum']:.4f}")

if __name__ == "__main__":
    main()
