# -*- coding: utf-8 -*-
import os
import re
import glob
import json
import numpy as np
import pandas as pd

# ===================== 配置区（按你的路径改） =====================
CASE_XLSX   = r"D:\PycharmProjects\Bosch\case.xlsx"
CASE_SHEET  = "case"
CASE_ID_COL = "input"   # 你的 case id（cas1/cas2/...）在这一列


IEDF_ROOT   = r"D:\BaiduNetdiskDownload\TSV"

# 固定：根据你数据结构
TARGETS = {
    ("SF6",  "sheath2"): ["F_1p", "SF3_1p", "SF4_1p", "SF5_1p"],
    ("C4F8", "sheath1"): ["CF3_1p", "C2F3_1p"],
}

OUT_XLSX    = r"D:\PycharmProjects\Bosch\case_with_phys7.xlsx"
OUT_JSON    = r"D:\PycharmProjects\Bosch\phys7_manifest.json"

EPS = 1e-30  # 防止除零 / log(0)

# 双峰判别参数（不是能量阈值；是形状判别的“显著性”）
SMOOTH_WIN = 9                 # 移动平均窗口（奇数更好）
MIN_PEAK_SEP_BINS = 5          # 两峰最小间隔（按能量网格bin）
SECOND_PEAK_MIN_FRAC = 0.25    # 第二峰至少达到第一峰的 25%
VALLEY_RATIO_TH = 0.80         # 峰间谷值需“足够低”：valley/min(peak1,peak2) <= 0.80

# ===================== 工具函数 =====================
def canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    return s

def find_case_id_col(df: pd.DataFrame) -> str:
    cands = ["标识case", "case", "caseid", "id"]
    cols = list(df.columns)
    cc = {c: canon(c) for c in cols}
    for c in cols:
        v = cc[c]
        for k in cands:
            if canon(k) == v or canon(k) in v:
                return c
    raise KeyError(f"找不到 case id 列。现有列名：{cols}")

def parse_gas_sheath_from_filename(fp: str):
    base = os.path.basename(fp)
    m = re.match(r"^([A-Za-z0-9]+)_(sheath\d+)_energy_distribution\.csv$", base)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def pick_energy_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "energy" in canon(c):
            return c
    return df.columns[0]

def pick_ion_col(df: pd.DataFrame, ion_prefix: str):
    pref = canon(ion_prefix)
    for c in df.columns:
        if canon(c).startswith(pref):
            return c
    return None

def moving_average(y, win=9):
    win = int(win)
    if win <= 1:
        return y
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=float) / win
    return np.convolve(y, k, mode="same")

def local_maxima_indices(y):
    if len(y) < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

def cumulative_trapz(x, y):
    """
    返回 F[k] = ∫_{x0}^{xk} y dx  (梯形累计积分)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return np.zeros_like(x)
    dx = np.diff(x)
    area_seg = 0.5 * (y[:-1] + y[1:]) * dx
    F = np.concatenate([[0.0], np.cumsum(area_seg)])
    return F

def quantile_energy(x, y, q):
    """
    y>=0 的分布，返回满足累计比例为 q 的能量 E_q
    使用累计积分 + 线性插值
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    # 保证非负（数值噪声）
    y = np.maximum(y, 0.0)
    F = cumulative_trapz(x, y)
    tot = F[-1]
    if not np.isfinite(tot) or tot <= 0:
        return np.nan
    C = F / tot
    # 防止重复值导致 interp 失败：做单调化
    # 但 C 本来应单调非减；这里轻微修正
    C = np.maximum.accumulate(C)
    return float(np.interp(q, C, x))

def detect_bimodal_flag(x, g):
    """
    在归一化形状 g(E) 上判别“显著双峰”：
    - 两个局部极大值，间隔 >= MIN_PEAK_SEP_BINS
    - 第二峰高度 >= SECOND_PEAK_MIN_FRAC * 第一峰高度
    - 两峰间谷值 valley/min(peak1,peak2) <= VALLEY_RATIO_TH
    """
    x = np.asarray(x, float)
    g = np.asarray(g, float)
    if x.size < 3:
        return 0

    gs = moving_average(g, win=SMOOTH_WIN)
    pidx = local_maxima_indices(gs)
    if pidx.size < 2:
        return 0

    # 按峰高排序
    cand = pidx[np.argsort(gs[pidx])[::-1]]
    p1 = cand[0]
    p2 = None
    for j in cand[1:]:
        if abs(j - p1) >= MIN_PEAK_SEP_BINS:
            p2 = j
            break
    if p2 is None:
        return 0

    # 两峰排序（左->右）
    if p1 > p2:
        p1, p2 = p2, p1

    h1 = float(gs[p1])
    h2 = float(gs[p2])
    if h1 <= 0 or h2 <= 0:
        return 0
    if h2 < SECOND_PEAK_MIN_FRAC * h1:
        return 0

    # 谷值（两峰之间）
    mid = slice(p1, p2 + 1)
    v_rel = int(np.argmin(gs[mid]))
    v = p1 + v_rel
    valley = float(gs[v])

    ratio = valley / (min(h1, h2) + EPS)
    return 1 if ratio <= VALLEY_RATIO_TH else 0

def read_target_iedf_for_case(case_id: str):
    """
    返回 dict: {(gas,sheath): filepath}
    只选 TARGETS 里需要的文件；若缺失则该 key 不存在
    """
    pattern = os.path.join(IEDF_ROOT, "scan*", str(case_id), "*_energy_distribution.csv")
    fps = sorted(glob.glob(pattern))
    got = {}
    for fp in fps:
        gas, sheath = parse_gas_sheath_from_filename(fp)
        if (gas, sheath) in TARGETS:
            got[(gas, sheath)] = fp
    return got

def compute_phys7_from_file(fp: str, gas: str, sheath: str, ions: list[str]):
    """
    从一个 gas+sheath 的 IEDF 文件里提取：
    - per-ion Gamma
    - 聚合 f_agg(E) -> 归一化 g(E) -> E10/E90 spread
    - bimodal_flag（只对 SF6 sheath2 会被上层取用）
    """
    df = pd.read_csv(fp)
    e_col = pick_energy_col(df)
    x = df[e_col].to_numpy(dtype=float)

    # 排序 + 清理
    ok = np.isfinite(x)
    x = x[ok]
    if x.size < 3:
        return None

    sort_idx = np.argsort(x)
    x = x[sort_idx]

    # 聚合离子曲线
    gammas = {}
    f_agg = np.zeros_like(x, dtype=float)

    for ion in ions:
        c = pick_ion_col(df, ion)
        if c is None:
            gammas[ion] = np.nan
            continue
        y_full = df[c].to_numpy(dtype=float)[ok][sort_idx]
        y_full = np.where(np.isfinite(y_full), y_full, 0.0)
        y_full = np.maximum(y_full, 0.0)

        Gamma = float(trapz_compat(y_full, x)) if x.size >= 2 else np.nan

        gammas[ion] = Gamma
        f_agg += y_full

    Gamma_tot = float(trapz_compat(f_agg, x)) if x.size >= 2 else np.nan
    # 归一化形状
    g = f_agg / (Gamma_tot + EPS) if (np.isfinite(Gamma_tot) and Gamma_tot > 0) else np.full_like(f_agg, np.nan)

    # 分位点与展宽
    E10 = quantile_energy(x, g, 0.10) if np.all(np.isfinite(g)) else np.nan
    E50 = quantile_energy(x, g, 0.50) if np.all(np.isfinite(g)) else np.nan
    E90 = quantile_energy(x, g, 0.90) if np.all(np.isfinite(g)) else np.nan

    spread = float(E90 - E10) if (np.isfinite(E10) and np.isfinite(E90)) else np.nan

    qskew = (
        float((E90 + E10 - 2.0 * E50) / (spread + EPS))
        if (np.isfinite(E10) and np.isfinite(E50) and np.isfinite(E90) and np.isfinite(spread) and spread > 0)
        else np.nan
    )

    return {
        "Gamma_tot": Gamma_tot,
        "gammas": gammas,
        "E10": E10,
        "E50": E50,
        "E90": E90,
        "spread": spread,
        "qskew": qskew,
    }

def trapz_compat(y, x):
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        return np.trapz(y, x)
    return fn(y, x)

# ===================== 主流程 =====================
def main():
    df_case = pd.read_excel(CASE_XLSX, sheet_name=CASE_SHEET)
    case_col = CASE_ID_COL if CASE_ID_COL in df_case.columns else find_case_id_col(df_case)
    case_ids = df_case[case_col].astype(str).tolist()

    feat_rows = []
    missing = []

    for cid in case_ids:
        files = read_target_iedf_for_case(cid)

        row = {"case_id": cid}

        # ---------- SF6 sheath2 ----------
        key_sf6 = ("SF6", "sheath2")
        if key_sf6 in files:
            out = compute_phys7_from_file(files[key_sf6], "SF6", "sheath2", TARGETS[key_sf6])
            if out is not None:
                Gamma_tot = out["Gamma_tot"]
                row["logGamma_SF6_tot"] = float(np.log10(Gamma_tot + EPS)) if np.isfinite(Gamma_tot) else np.nan

                Gamma_F = out["gammas"].get("F_1p", np.nan)
                row["pF_SF6"] = float(Gamma_F / (Gamma_tot + EPS)) if (np.isfinite(Gamma_F) and np.isfinite(Gamma_tot)) else np.nan

                row["spread_SF6"] = out["spread"]
                row["qskew_SF6"] = out["qskew"]
            else:
                row.update({"logGamma_SF6_tot": np.nan, "pF_SF6": np.nan, "spread_SF6": np.nan, "qskew_SF6": np.nan})
        else:
            row.update({"logGamma_SF6_tot": np.nan, "pF_SF6": np.nan, "spread_SF6": np.nan, "qskew_SF6": np.nan})

        # ---------- C4F8 sheath1 ----------
        key_c4 = ("C4F8", "sheath1")
        if key_c4 in files:
            out = compute_phys7_from_file(files[key_c4], "C4F8", "sheath1", TARGETS[key_c4])
            if out is not None:
                Gamma_tot = out["Gamma_tot"]
                row["logGamma_C4F8_tot"] = float(np.log10(Gamma_tot + EPS)) if np.isfinite(Gamma_tot) else np.nan

                G1 = out["gammas"].get("CF3_1p", np.nan)
                G2 = out["gammas"].get("C2F3_1p", np.nan)
                # log 比值，避免数量级差
                row["rho_C4F8"] = float(np.log10((G1 + EPS) / (G2 + EPS))) if (np.isfinite(G1) and np.isfinite(G2)) else np.nan

                row["spread_C4F8"] = out["spread"]
            else:
                row.update({"logGamma_C4F8_tot": np.nan, "rho_C4F8": np.nan, "spread_C4F8": np.nan})
        else:
            row.update({"logGamma_C4F8_tot": np.nan, "rho_C4F8": np.nan, "spread_C4F8": np.nan})

        if (key_sf6 not in files) and (key_c4 not in files):
            missing.append(cid)

        feat_rows.append(row)

    df_feat = pd.DataFrame(feat_rows)

    # 合并回 case.xlsx（保持行顺序不变）
    df_out = df_case.copy()
    df_out["_case_id_tmp_"] = df_out[case_col].astype(str)
    df_out = df_out.merge(df_feat, how="left", left_on="_case_id_tmp_", right_on="case_id", validate="one_to_one")
    df_out = df_out.drop(columns=["_case_id_tmp_", "case_id"])

    os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)
    df_out.to_excel(OUT_XLSX, index=False)

    manifest = {
        "case_xlsx": CASE_XLSX,
        "iedf_root": IEDF_ROOT,
        "targets": {f"{k[0]}_{k[1]}": v for k, v in TARGETS.items()},
        "features": [
          "logGamma_SF6_tot","pF_SF6","spread_SF6","qskew_SF6",
          "logGamma_C4F8_tot","rho_C4F8","spread_C4F8"
        ]
        ,
        "params": {
            "EPS": EPS,
            "SMOOTH_WIN": SMOOTH_WIN,
            "MIN_PEAK_SEP_BINS": MIN_PEAK_SEP_BINS,
            "SECOND_PEAK_MIN_FRAC": SECOND_PEAK_MIN_FRAC,
            "VALLEY_RATIO_TH": VALLEY_RATIO_TH
        },
        "n_cases": int(len(case_ids)),
        "missing_both_files_cases": missing[:200],
        "missing_both_files_count": int(len(missing)),
        "out_xlsx": OUT_XLSX
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved merged dataset: {OUT_XLSX}")
    print(f"[OK] Saved manifest: {OUT_JSON}")
    if missing:
        print(f"[WARN] cases missing BOTH target files (show first 20): {missing[:20]} (total={len(missing)})")

if __name__ == "__main__":
    main()
