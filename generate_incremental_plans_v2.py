# -*- coding: utf-8 -*-

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ---------------------------
# Parameters
# ---------------------------
PARAM_NAMES: List[str] = [
    "APC", "SOURCE_RF", "LF_RF", "SF6", "C4F8", "DEP_TIME", "ETCH_TIME"
]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "APC": ["APC", "apc", "pressure", "APC（E2步骤）", "APC(E2步骤)"],
    "SOURCE_RF": ["SOURCE_RF", "source_RF", "Source", "source", "source_RF（E2步骤）", "source_RF(E2步骤)"],
    "LF_RF": ["LF_RF", "LF", "lf", "bias", "LF_RF（E2步骤）", "LF_RF(E2步骤)"],
    "SF6": ["SF6", "sf6", "SF6_sccm", "SF6（E2步骤）", "SF6(E2步骤)"],
    "C4F8": ["C4F8", "c4f8", "C4F8_sccm", "C4F8（DEP步骤）", "C4F8(DEP步骤)"],
    "DEP_TIME": ["DEP", "DEPtime", "DEP_time", "DEP time"],
    "ETCH_TIME": ["etch", "etchtime", "etch_time", "etch time"],
}


# ---------------------------
# Helpers
# ---------------------------
def _normalize_colname(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    return s


def _coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^0-9eE\.\+\-]+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def load_table(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        if p.name == path: p = Path.cwd() / path
        if not p.exists(): raise FileNotFoundError(f"Table not found: {path}")

    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, sheet_name=sheet if sheet else 0)
    elif p.suffix.lower() in [".csv", ".tsv", ".txt"]:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
    else:
        raise ValueError(f"Unsupported table type: {p.suffix} ({path})")
    df.columns = [_normalize_colname(c) for c in df.columns]
    return df


def map_param_columns(df: pd.DataFrame, params: List[str]) -> Dict[str, str]:
    colset = list(df.columns)
    mapping: Dict[str, str] = {}
    for p in params:
        aliases = COLUMN_ALIASES.get(p, [p])
        if p not in aliases: aliases.append(p)
        hit = None
        for alias in aliases:
            norm_alias = _normalize_colname(alias)
            if norm_alias in colset: hit = norm_alias; break
        if hit is None:
            for alias in aliases:
                norm_alias = _normalize_colname(alias)
                if len(norm_alias) < 2: continue
                for col in colset:
                    if norm_alias.lower() in col.lower(): hit = col; break
                if hit: break
        if hit is None: raise KeyError(f"Missing param column for '{p}'. Existing: {colset[:10]}...")
        mapping[p] = hit
    return mapping


# ---------------------------
# Stats
# ---------------------------
@dataclass
class DomainStats:
    name: str
    path: str
    sheet: Optional[str]
    mapping: Dict[str, str]
    stats_df: pd.DataFrame
    issues_df: pd.DataFrame
    raw_df: pd.DataFrame  # 包含原始数据


def extract_raw_data(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """提取干净的原始数据，包含参数列和 ID 列"""
    data = pd.DataFrame()

    # 1. 提取参数数值
    for param, col in mapping.items():
        data[param] = pd.to_numeric(df[col], errors='coerce')

    # 2. 智能寻找 ID 列
    possible_ids = ["input", "Input", "case", "Case", "Run", "run", "Experiment", "id", "ID", "配方名"]

    found_id = None
    for col in possible_ids:
        if col in df.columns:
            found_id = col
            break

    if found_id:
        data["input"] = df[found_id].astype(str).str.strip()
        print(f"[DEBUG] Found ID column: {found_id}")
    else:
        data["input"] = df.index.astype(str)
        print("[DEBUG] No ID column found, using index instead.")

    return data


def compute_param_stats(df: pd.DataFrame, param_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for p, col in param_map.items():
        x = _coerce_numeric(df[col])
        rows.append({
            "Param": p, "Column": col,
            "Count": int(x.notna().sum()), "Missing": int(x.isna().sum()),
            "Min": float(x.min()) if x.notna().any() else np.nan,
            "Max": float(x.max()) if x.notna().any() else np.nan,
            "Mean": float(x.mean()) if x.notna().any() else np.nan,
            "Std": float(x.std(ddof=1)) if x.notna().sum() > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def detect_quality_issues(stats_df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    for _, r in stats_df.iterrows():
        tags = []
        if r["Count"] == 0: tags.append("ALL_MISSING")
        issues.append(",".join(tags))
    out = stats_df.copy()
    out["Issues"] = issues
    return out


# ---------------------------------------------------------
# [EXPERT LOGIC] Override Ranges with User Knowledge
# ---------------------------------------------------------
def build_param_ranges(old_stats: pd.DataFrame, new_stats: Optional[pd.DataFrame] = None) -> Dict[str, dict]:
    old = old_stats.set_index("Param")
    new = new_stats.set_index("Param") if new_stats is not None else old
    ranges: Dict[str, dict] = {}

    EXTRAPOLATE = 0.1

    for p in PARAM_NAMES:
        rn = new.loc[p]
        n_min = float(rn["Min"]);
        n_max = float(rn["Max"])
        n_span = n_max - n_min

        # 1. Standard auto-calculated range
        safe_min = max(0, n_min - n_span * EXTRAPOLATE)
        safe_max = n_max + n_span * EXTRAPOLATE

        # 2. EXPERT OVERRIDES
        if p in ["SF6", "C4F8"]:
            safe_max = 500.0
            safe_min = max(safe_min, 150.0)

        elif p == "SOURCE_RF":
            safe_min = 2000.0
            safe_max = 3000.0

        ranges[p] = {
            "old_min": safe_min, "old_max": safe_max,
            "new_mean": float(rn["Mean"])
        }
    return ranges


# ---------------------------
# Plan Generation
# ---------------------------
def make_plan_local_exploration(
        domains: List[Optional[DomainStats]],
        n_samples: int,
        anchor_recipes: List[str],
        noise_ratios: Dict[str, float],
        param_ranges: Dict[str, dict]
) -> pd.DataFrame:
    # 1. 合并所有可用数据的 raw_df
    valid_dfs = [d.raw_df for d in domains if d is not None]
    if not valid_dfs:
        raise ValueError("No valid domains provided!")

    # 统一大表
    full_df = pd.concat(valid_dfs, axis=0, ignore_index=True)

    anchors = []

    # 2. 查找锚点
    for anchor_id in anchor_recipes:
        row = full_df[full_df["input"] == anchor_id]
        if row.empty:
            row = full_df[full_df["input"].str.contains(anchor_id, regex=False)]

        if not row.empty:
            anchors.append(row.iloc[0][PARAM_NAMES].values.astype(float))
            print(f"[INFO] Found anchor {anchor_id}")
        else:
            print(f"[WARN] Anchor {anchor_id} not found in ANY table!")

    if not anchors:
        raise ValueError("No valid anchors found!")

    anchors = np.vstack(anchors)

    # 3. 生成新点
    new_samples = []

    for _ in range(n_samples):
        base = anchors[np.random.choice(len(anchors))]

        noise = np.zeros_like(base)
        for i, param in enumerate(PARAM_NAMES):
            ratio = noise_ratios.get(param, 0.05)
            sigma = abs(base[i]) * ratio
            if sigma == 0: sigma = 0.1
            noise[i] = np.random.normal(0, sigma)

        candidate = base + noise

        for i, param in enumerate(PARAM_NAMES):
            # 1. Clipping
            pr = param_ranges.get(param)
            if pr:
                min_v = pr["old_min"]
                max_v = pr["old_max"]
            else:
                min_v, max_v = -np.inf, np.inf

            if param == "SOURCE_RF": min_v = max(min_v, 2000)

            val = np.clip(candidate[i], min_v, max_v)

            # 2. [关键修复] Rounding 整数/步长约束
            if param == "SOURCE_RF":
                # 整 100
                val = round(val / 100) * 100
            elif "TIME" in param:
                # 整 0.2
                val = round(val / 0.2) * 0.2
            else:
                # 其他 (APC, Gas, LF) 整 5
                val = round(val / 5) * 5

            candidate[i] = val

        new_samples.append(candidate)

    plan_df = pd.DataFrame(new_samples, columns=PARAM_NAMES)
    plan_df["Strategy"] = "Golden_Local"

    # 生成 Sample_ID
    plan_df["Sample_ID"] = [f"Local_{i + 1:02d}" for i in range(len(plan_df))]

    return plan_df


def write_report(
        out_xlsx: str,
        old_domain: DomainStats,
        new_domain: Optional[DomainStats],
        plans: Dict[str, pd.DataFrame],
) -> None:
    out_path = Path(out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_path.parent

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        old_domain.stats_df.to_excel(w, sheet_name=f"{old_domain.name}_Stats", index=False)
        if new_domain: new_domain.stats_df.to_excel(w, sheet_name=f"{new_domain.name}_Stats", index=False)

        for k, df in plans.items():
            df.to_excel(w, sheet_name=k[:31], index=False)

            # Formatted Plan
            fmt_rows = []
            cols = ["配方名", "步骤", "温度", "APC", "Source RF(边缘)", "LF RF 中心(边缘)", "Gas 中心(边缘)",
                    "Time(起始)"]

            for _, row in df.iterrows():
                sid = row.get("Sample_ID", "Unknown")
                p_apc = row["APC"]
                p_src = int(row["SOURCE_RF"])
                p_lf = int(row["LF_RF"])
                p_sf6 = int(row["SF6"])
                p_c4f8 = int(row["C4F8"])
                p_dep = row["DEP_TIME"]
                p_etch = row["ETCH_TIME"]

                fmt_rows.append({
                    "配方名": sid, "步骤": "Strike", "温度": -10, "APC": 40,
                    "Source RF(边缘)": "2200(0)", "LF RF 中心(边缘)": 0,
                    "Gas 中心(边缘)": "Ar 300 Sccm C4F8 250 Sccm", "Time(起始)": 2
                })
                fmt_rows.append({
                    "配方名": "", "步骤": "Dep", "温度": -10, "APC": 55,
                    "Source RF(边缘)": "2400(600)", "LF RF 中心(边缘)": 0,
                    "Gas 中心(边缘)": f"C4F8 {p_c4f8}", "Time(起始)": p_dep
                })
                fmt_rows.append({
                    "配方名": "", "步骤": "E1", "温度": -10, "APC": 20,
                    "Source RF(边缘)": "2400(0)", "LF RF 中心(边缘)": 140,
                    "Gas 中心(边缘)": "SF6 200", "Time(起始)": 1.0
                })
                fmt_rows.append({
                    "配方名": "", "步骤": "E2", "温度": -10, "APC": p_apc,
                    "Source RF(边缘)": f"{p_src}(0)", "LF RF 中心(边缘)": p_lf,
                    "Gas 中心(边缘)": f"SF6 {p_sf6}", "Time(起始)": p_etch
                })
                fmt_rows.append({c: "" for c in cols})

            fmt_df = pd.DataFrame(fmt_rows, columns=cols)
            fmt_sheet_name = f"Fmt_{k}"[:31]
            fmt_df.to_excel(w, sheet_name=fmt_sheet_name, index=False)

    print(f"\n[Export] Saving plans to CSV in: {out_dir}")
    for k, df in plans.items():
        csv_name = f"{k}.csv"
        df.to_csv(out_dir / csv_name, index=False, encoding='utf-8-sig')
        print(f"   -> {csv_name}")


# ---------------------------
# Main (Hardcoded)
# ---------------------------
def build_domain_stats(name: str, table_path: str, sheet: Optional[str]) -> DomainStats:
    df = load_table(table_path, sheet=sheet)
    mapping = map_param_columns(df, PARAM_NAMES)

    stats = compute_param_stats(df, mapping)
    issues = detect_quality_issues(stats)

    # 提取原始数据，用于查找锚点
    raw_df = extract_raw_data(df, mapping)

    return DomainStats(
        name=name, path=table_path, sheet=sheet, mapping=mapping,
        stats_df=stats, issues_df=issues, raw_df=raw_df
    )


def main():
    # --- HARDCODED PATHS ---
    old_table = r"D:\PycharmProjects\Bosch\case.xlsx"
    new_table = r"D:\PycharmProjects\Bosch\Bosch.xlsx"
    out_dir_path = r"D:\PycharmProjects\Bosch\new_plan"

    old_sheet = "case"
    new_sheet = "Sheet1"

    # 黄金锚点
    gold_anchors = ["B47", "B52", "B54"]

    # [修复] 包含 30 samples
    n_list = [10, 20, 30]
    seed = 20260114
    # -----------------------

    np.random.seed(seed)
    print("=" * 90)
    print("Expert-Driven Plan Generator (SF6/C4F8<=500, Source~2500)")
    print("=" * 90)

    old_domain = build_domain_stats("OLD", old_table, old_sheet)
    new_domain = build_domain_stats("NEW", new_table, new_sheet) if new_table else None

    # Use EXPERT ranges
    param_ranges = build_param_ranges(old_domain.stats_df, new_domain.stats_df if new_domain else None)

    noise_config = {
        "APC": 0.05,  # 压力波动 5%
        "SOURCE_RF": 0.05,
        "LF_RF": 0.10,  # 偏压可以波动大一点试试
        "SF6": 0.05,
        "C4F8": 0.05,  # 关键气体
        "DEP_TIME": 0.10,  # 时间参数通常线性度好，可以探索远一点
        "ETCH_TIME": 0.10
    }

    print(f"Generating plans focused around: {gold_anchors}")

    plans: Dict[str, pd.DataFrame] = {}
    for n in n_list:
        plans[f"Plan_{n}_Local"] = make_plan_local_exploration(
            domains=[old_domain, new_domain],
            n_samples=n,
            anchor_recipes=gold_anchors,
            noise_ratios=noise_config,
            param_ranges=param_ranges
        )
    out_dir = Path(out_dir_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_report = out_dir / "analysis_report.xlsx"
    write_report(str(out_report), old_domain, new_domain, plans)

    print("\n" + "=" * 90)
    print(f"✅ Report written: {out_report}")
    print(f"   Check 'Fmt_Plan_30_Local' for Machine-Ready Recipe Table.")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())