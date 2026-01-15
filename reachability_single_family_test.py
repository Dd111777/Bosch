# reachability_single_family_searchbest.py
# 单 family 可达性测试（含 searchbest）：
# - mode=fixed: 使用给定 split_json（best_split.json）训练并评估单 family
# - mode=search: 随机搜索 split，使该 family 的 test_R2 最大（StageB init + finetune 后评估）
#
# 依赖：与你的 stageC_finetune_joint_on_new_pycharm.py 同目录

import os
import json
import time
import argparse
import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    if y_true.size < 2:
        return float("nan")
    mu = float(np.mean(y_true))
    sst = float(np.sum((y_true - mu) ** 2))
    sse = float(np.sum((y_true - y_pred) ** 2))
    if sst <= 1e-12:
        return float("nan")
    return 1.0 - sse / sst


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def parse_key_recipes(s: str):
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for tok in s.replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def load_split_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_single_family_mask(mask: np.ndarray, k_sel: int) -> np.ndarray:
    m = mask.copy()
    for k in range(m.shape[1]):
        if k != k_sel:
            m[:, k, :] = 0
    return m


def family_points_display_nm(sc, pack: Dict[str, np.ndarray], y_mean, y_std,
                             k_sel: int, unit_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 flatten 后该 family 的 (y_true_nm, y_pred_nm)
    display 逻辑与 export_stageC_report 一致：family_sign=None -> 默认 zmin 翻正
    """
    pred_um = pack["pred_norm"] * (y_std + 1e-6) + y_mean
    y_um = pack["y_norm"] * (y_std + 1e-6) + y_mean
    m = pack["mask"].astype(bool)

    pred_t = torch.from_numpy(pred_um.astype(np.float32))
    y_t = torch.from_numpy(y_um.astype(np.float32))

    pred_disp, y_disp = sc.pu.transform_for_display(
        pred_t, y_t,
        family_sign=None,
        unit_scale=unit_scale,
        clip_nonneg=False
    )
    pred_disp = pred_disp.numpy()
    y_disp = y_disp.numpy()

    mk = m[:, k_sel, :].reshape(-1)
    yt = y_disp[:, k_sel, :].reshape(-1)[mk]
    yp = pred_disp[:, k_sel, :].reshape(-1)[mk]
    ok = np.isfinite(yt) & np.isfinite(yp)
    return yt[ok], yp[ok]


@torch.no_grad()
def eval_single_family_r2(sc, model, loader, y_mean, y_std, k_sel, unit_scale, device: str) -> Tuple[float, float, int]:
    pack = sc.eval_pack(model, loader, device=device)
    yt, yp = family_points_display_nm(sc, pack, y_mean, y_std, k_sel, unit_scale)
    r2 = masked_r2_np(yt, yp)
    mae = float(np.mean(np.abs(yt - yp))) if yt.size > 0 else float("nan")
    return float(r2), float(mae), int(yt.size)


def run_fixed(sc, args, out_dir: str):
    split = load_split_json(args.split_json)
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)
    test_idx = np.array(split["test_idx"], dtype=np.int64)

    data = sc.build_stageC_dataset(
        args.new_excel, args.stageA_ckpt, device=args.device,
        stageB_ckpt_for_norm=args.stageB_ckpt,
        recipe_aug_mode="time"
    )
    static_x, phys7_seq = data["static_x"], data["phys7_seq"]
    y_um, mask = data["y_um"], data["mask"]
    time_mat = data["time_mat"]

    families = list(sc.FAMILIES)
    k_sel = families.index(args.family)

    mask_single = make_single_family_mask(mask, k_sel)

    y_mean, y_std = sc.fit_y_norm(y_um, mask_single, train_idx)
    y_norm = (y_um - y_mean) / (y_std + 1e-6)

    train_loader = sc.make_loader(static_x, phys7_seq, y_norm, mask_single, time_mat, train_idx, args.batch)
    val_loader = sc.make_loader(static_x, phys7_seq, y_norm, mask_single, time_mat, val_idx, args.batch)
    test_loader = sc.make_loader(static_x, phys7_seq, y_norm, mask_single, time_mat, test_idx, args.batch)

    model = sc.build_morph_model(args.model_type, K=len(families), device=args.device, stageB_ckpt=args.stageB_ckpt)
    info = sc.load_morph_ckpt(model, args.stageB_ckpt)
    if not info.get("ok", False):
        raise RuntimeError("StageB ckpt 加载失败，无法做 C_transfer 单任务测试。")

    sc.train_one(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        early_patience=args.early_patience
    )

    train_r2, train_mae, ntr = eval_single_family_r2(sc, model, train_loader, y_mean, y_std, k_sel, args.unit_scale, args.device)
    test_r2, test_mae, nte = eval_single_family_r2(sc, model, test_loader, y_mean, y_std, k_sel, args.unit_scale, args.device)

    # 保存结果
    result = {
        "mode": "fixed",
        "family": args.family,
        "train_R2": train_r2,
        "test_R2": test_r2,
        "train_MAE": train_mae,
        "test_MAE": test_mae,
        "n_train_points": ntr,
        "n_test_points": nte,
        "split_json": args.split_json,
        "epochs": args.epochs, "lr": args.lr, "wd": args.wd, "batch": args.batch, "early_patience": args.early_patience
    }
    save_json(os.path.join(out_dir, "result_single_family.json"), result)

    # scatter（test）
    pack_te = sc.eval_pack(model, test_loader, device=args.device)
    yt, yp = family_points_display_nm(sc, pack_te, y_mean, y_std, k_sel, args.unit_scale)

    fig = plt.figure()
    plt.scatter(yt, yp, s=14, alpha=0.7)
    if yt.size > 0:
        lo = float(min(np.min(yt), np.min(yp)))
        hi = float(max(np.max(yt), np.max(yp)))
        plt.plot([lo, hi], [lo, hi], linewidth=1)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
    plt.xlabel("y_true (nm)")
    plt.ylabel("y_pred (nm)")
    plt.title(f"fixed | {args.family} | test_R2={test_r2:.3f} MAE={test_mae:.2f}nm")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"scatter_{args.family}.png"), dpi=200)
    plt.close(fig)

    print(f"[FIXED][{args.family}] train_R2={train_r2:.4f} test_R2={test_r2:.4f} "
          f"train_MAE={train_mae:.2f} test_MAE={test_mae:.2f}  points(test)={nte}")
    print(f"[Saved] {out_dir}")


def run_search(sc, args, out_dir: str):
    # 读数据一次
    data = sc.build_stageC_dataset(
        args.new_excel, args.stageA_ckpt, device=args.device,
        stageB_ckpt_for_norm=args.stageB_ckpt,
        recipe_aug_mode="time"
    )
    static_x, phys7_seq = data["static_x"], data["phys7_seq"]
    y_um, mask = data["y_um"], data["mask"]
    time_mat, recipe_ids = data["time_mat"], data["recipe_ids"]

    families = list(sc.FAMILIES)
    k_sel = families.index(args.family)

    mask_single = make_single_family_mask(mask, k_sel)

    # key recipes
    key_recipes = parse_key_recipes(args.key_recipes)
    existing_keys = []
    recipe_set = set([str(x).upper() for x in recipe_ids.tolist()])
    for k in key_recipes:
        if str(k).upper() in recipe_set:
            existing_keys.append(k)

    rng = np.random.default_rng(args.seed)
    N = int(len(recipe_ids))

    best = None
    best_rank = None

    trials_log = os.path.join(out_dir, "trials_log.csv")
    if not os.path.exists(trials_log):
        with open(trials_log, "w", encoding="utf-8-sig") as f:
            f.write("trial,train_n,val_n,test_n,test_R2,test_MAE,n_test_points,split_desc_json\n")

    for tr in range(1, args.trials + 1):
        # split（调用你原脚本里的函数，保证规则一致）
        train_idx, val_idx, test_idx = sc.random_subset_and_split(
            N, recipe_ids, existing_keys,
            args.test_ratio, args.val_ratio, args.drop_max_frac, rng
        )

        # 约束：test 必须有足够点数（否则 R2 没意义）
        n_test_points = int(mask_single[test_idx, k_sel, :].sum())
        if n_test_points < args.min_test_points:
            continue

        # norm：只用单 family mask
        y_mean, y_std = sc.fit_y_norm(y_um, mask_single, train_idx)
        y_norm = (y_um - y_mean) / (y_std + 1e-6)

        train_loader = sc.make_loader(static_x, phys7_seq, y_norm, mask_single, time_mat, train_idx, args.batch)
        val_loader = sc.make_loader(static_x, phys7_seq, y_norm, mask_single, time_mat, val_idx, args.batch)
        test_loader = sc.make_loader(static_x, phys7_seq, y_norm, mask_single, time_mat, test_idx, args.batch)

        # model: C_transfer init + finetune
        model = sc.build_morph_model(args.model_type, K=len(families), device=args.device, stageB_ckpt=args.stageB_ckpt)
        info = sc.load_morph_ckpt(model, args.stageB_ckpt)
        if not info.get("ok", False):
            continue

        sc.train_one(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            early_patience=args.early_patience
        )

        test_r2, test_mae, nte = eval_single_family_r2(sc, model, test_loader, y_mean, y_std, k_sel, args.unit_scale, args.device)

        split_desc = sc.describe_split(recipe_ids, train_idx, val_idx, test_idx)

        with open(trials_log, "a", encoding="utf-8-sig") as f:
            f.write(f"{tr},{len(train_idx)},{len(val_idx)},{len(test_idx)},{test_r2:.6f},{test_mae:.6f},{nte},"
                    f"\"{json.dumps(split_desc, ensure_ascii=False).replace('\"','\"\"')}\"\n")

        # rank：只最大化 test_R2（可达性测试的核心）
        # 先保证点数够（nte 大更稳），同 R2 时点数多更好
        rank = (float(test_r2), int(nte))

        if (best_rank is None) or (rank > best_rank):
            best_rank = rank
            best = {
                "mode": "search",
                "family": args.family,
                "trial": tr,
                "test_R2": float(test_r2),
                "test_MAE": float(test_mae),
                "n_test_points": int(nte),
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "split_desc": split_desc,
                "epochs": args.epochs, "lr": args.lr, "wd": args.wd, "batch": args.batch, "early_patience": args.early_patience,
                "search_cfg": {
                    "trials": args.trials,
                    "test_ratio": args.test_ratio,
                    "val_ratio": args.val_ratio,
                    "drop_max_frac": args.drop_max_frac,
                    "min_test_points": args.min_test_points,
                    "key_recipes": existing_keys
                }
            }
            save_json(os.path.join(out_dir, "best_split_single_family.json"), best)
            print(f"[SEARCH][best@{tr}] {args.family}: test_R2={test_r2:.4f}, MAE={test_mae:.2f}, n_test_points={nte}")

            # 早停条件：达到阈值就提前结束（你要的 >0.8）
            if test_r2 >= args.target_r2:
                break

    if best is None:
        raise RuntimeError("search 模式下没有找到任何满足 min_test_points 的 split。请降低 min_test_points 或调大 test_ratio。")

    # 用 best split 再跑一次“固定训练+保存散点”，方便复现（等价于 ablation 的单 family）
    tmp = {
        "train_idx": best["train_idx"],
        "val_idx": best["val_idx"],
        "test_idx": best["test_idx"],
        "split_desc": best["split_desc"],
    }
    tmp_path = os.path.join(out_dir, "best_split.json")
    save_json(tmp_path, tmp)

    args.split_json = tmp_path
    run_fixed(sc, args, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["fixed", "search"], default="search")
    ap.add_argument("--family", type=str, default="d1")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out_root", type=str, default="./runs_reachability")

    # fixed 模式
    ap.add_argument("--split_json", type=str, default="")

    # search 模式
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--test_ratio", type=float, default=0.25)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--drop_max_frac", type=float, default=0.35)
    ap.add_argument("--key_recipes", type=str, default="")
    ap.add_argument("--min_test_points", type=int, default=10, help="该 family 在 test 上至少要有多少有效点")

    # 训练超参（建议与你 ablation C_transfer 对齐）
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--early_patience", type=int, default=30)

    # 目标阈值（search 达到就提前停）
    ap.add_argument("--target_r2", type=float, default=0.80)

    # 路径
    ap.add_argument("--new_excel", type=str,default=r"D:\PycharmProjects\Bosch\Bosch.xlsx")
    ap.add_argument("--stageA_ckpt", type=str, default=r"D:\PycharmProjects\Bosch\runs_stageA_phys7\cv_transformer_seed2/phys7_best.pth")
    ap.add_argument("--stageB_ckpt", type=str, default="./runs_stageB_morph_phys7/model-transformer_phys-stageA_pred_aug-time_phys7-full_seed4/best_model-transformer_phys-stageA_pred_aug-time_phys7-full_seed4.pth")

    ap.add_argument("--unit_scale", type=float, default=1000.0)  # um->nm
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_type", type=str, default="transformer")

    args = ap.parse_args()

    # import 你的 stageC 脚本
    import stageC_finetune_joint_on_new_pycharm as sc

    families = list(sc.FAMILIES)
    if args.family not in families:
        raise RuntimeError(f"--family={args.family} 不在 FAMILIES={families}")

    if args.device == "cuda" and (not torch.cuda.is_available()):
        args.device = "cpu"

    set_all_seeds(args.seed)

    tag = args.tag.strip()
    if not tag:
        tag = f"{args.mode}_{args.family}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(os.path.join(args.out_root, tag))

    print(f"[Device] {args.device}")
    print(f"[MODE] {args.mode}")
    print(f"[INFO] family={args.family} out_dir={out_dir}")

    if args.mode == "fixed":
        if not args.split_json:
            raise RuntimeError("mode=fixed 必须提供 --split_json")
        run_fixed(sc, args, out_dir)
    else:
        run_search(sc, args, out_dir)


if __name__ == "__main__":
    main()
