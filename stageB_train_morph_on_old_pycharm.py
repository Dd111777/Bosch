# -*- coding: utf-8 -*-
"""
Stage B: 旧表训练形貌网络（逐 family 标准化 + 独立输出头）
- 训练损失：按 family 加权（仅统计有标注的位置）
- 评估&导出：既有整体，也有“逐 family 独立”产物与检查点
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from physio_util import (
    set_seed,
    excel_to_morph_dataset_from_old,
    transform_for_display,
    metrics,
    export_predictions_longtable,
    export_metrics_grid,
    write_summary_txt,
    save_manifest,
    heatmap,
    parity_scatter,
    residual_hist,
    FAMILIES,
)
from phys_model import TemporalRegressor


class Cfg:
    # 数据
    old_excel = r"D:\data\pycharm\bosch\case.xlsx"
    sheet_name = "case"
    save_dir = "./runs_morph_old"

    # 训练
    seed = 42
    batch = 64
    max_epochs = 200
    val_ratio = 0.1
    lr = 1e-3
    amp = False

    # 模型
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_ff = 128
    dropout = 0.3

    # 展示空间
    unit_scale = 1000.0               # μm → nm
    family_sign = np.array([-1, +1, +1, +1, +1, +1], dtype=np.float32)  # zmin 取正
    flip_sign = False
    clip_nonneg = False               # 如需开启，记得在 transform_for_display 传 nonneg_families
    min_display_value = None

    # 导出控制
    export_family_alone = True        # 导出每个 family 的独立文件/图
    nonneg_families = None            # 例：list(range(len(FAMILIES))) 让所有家族展示非负


def _masked_l1_per_family(pred, target, mask):
    """
    返回：
      loss_mean: 标量（按各 family 的有效点数加权平均）
      per_fam:   (K,) 张量，逐 family 的 L1（按自身有效点数平均）
      counts:    (K,) 张量，每个 family 的有效点数
    形状：
      pred/target/mask: (B,K,T)
    """
    with torch.no_grad():
        counts = mask.float().sum(dim=(0, 2))  # (K,)

    abs_e = torch.abs(pred - target) * mask.float()  # (B,K,T)
    per_fam = abs_e.sum(dim=(0, 2)) / counts.clamp_min(1.0)  # (K,)
    loss_mean = (per_fam * (counts > 0).float()).sum() / (counts > 0).float().sum().clamp_min(1.0)
    return loss_mean, per_fam, counts


def _maybe_denorm_targets(pred, trg, meta, device):
    if isinstance(meta, dict) and "norm_target" in meta:
        mean = meta["norm_target"]["mean"].to(device=device, dtype=pred.dtype)  # (K,)
        std  = meta["norm_target"]["std"].to(device=device, dtype=pred.dtype)   # (K,)
        pred = pred * std.view(1, -1, 1) + mean.view(1, -1, 1)
        trg  = trg  * std.view(1, -1, 1) + mean.view(1, -1, 1)
    return pred, trg


def _select_family(x, k):
    """从 (B,K,T) 张量中取出第 k 个 family → (B,1,T)"""
    return x[:, k:k+1, :]


def main():
    os.makedirs(Cfg.save_dir, exist_ok=True)
    set_seed(Cfg.seed)

    dataset, meta = excel_to_morph_dataset_from_old(Cfg.old_excel, sheet_name=Cfg.sheet_name)
    N = len(dataset)
    nval = max(1, int(N * Cfg.val_ratio))
    tr_set, va_set = random_split(dataset, [N - nval, nval])
    tr_loader = DataLoader(tr_set, batch_size=Cfg.batch, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=Cfg.batch, shuffle=False)
    T = int(meta["T"])

    model = TemporalRegressor(
        K=len(FAMILIES),
        d_model=Cfg.d_model,
        nhead=Cfg.nhead,
        num_layers=Cfg.num_layers,
        dim_ff=Cfg.dim_ff,
        dropout=Cfg.dropout,
        T=T,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=Cfg.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Cfg.max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=Cfg.amp)

    # 最佳记录：整体 + 逐 family
    best_overall = -1e9
    best_per_fam = {k: -1e9 for k in range(len(FAMILIES))}
    best_overall_path = os.path.join(Cfg.save_dir, "morph_best_overall.pth")
    best_fam_path = {k: os.path.join(Cfg.save_dir, f"morph_best_{FAMILIES[k]}.pth") for k in range(len(FAMILIES))}

    # ===== Train =====
    for epoch in range(1, Cfg.max_epochs + 1):
        model.train()
        tot_loss, num = 0.0, 0
        tr_loss_per_fam_acc = torch.zeros(len(FAMILIES), dtype=torch.float32, device=device)
        tr_count_per_fam_acc = torch.zeros(len(FAMILIES), dtype=torch.float32, device=device)

        for s8, phys, trg, msk, tvals in tr_loader:
            s8, phys, trg, msk, tvals = [x.to(device) for x in (s8, phys, trg, msk, tvals)]
            opt.zero_grad(set_to_none=True)

            if Cfg.amp:
                with torch.cuda.amp.autocast():
                    pred = model(s8, phys, tvals)
                    loss_mean, per_fam_l1, counts = _masked_l1_per_family(pred, trg, msk)
                scaler.scale(loss_mean).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(s8, phys, tvals)
                loss_mean, per_fam_l1, counts = _masked_l1_per_family(pred, trg, msk)
                loss_mean.backward()
                opt.step()

            tot_loss += float(loss_mean.item()) * s8.size(0); num += s8.size(0)
            tr_loss_per_fam_acc += per_fam_l1 * counts
            tr_count_per_fam_acc += counts

        sch.step()

        tr_l1 = tot_loss / max(1, num)
        tr_l1_per_fam = (tr_loss_per_fam_acc / tr_count_per_fam_acc.clamp_min(1)).detach().cpu().numpy()
        tr_l1_msg = " | ".join([f"{FAMILIES[k]}={tr_l1_per_fam[k]:.4f}" for k in range(len(FAMILIES))])

        # ===== Val (display space) =====
        model.eval()
        agg_mts = None
        with torch.no_grad():
            for s8, phys, trg, msk, tvals in va_loader:
                s8, phys, trg, msk, tvals = [x.to(device) for x in (s8, phys, trg, msk, tvals)]
                pred = model(s8, phys, tvals)
                pred, trg = _maybe_denorm_targets(pred, trg, meta, device)

                yhat_disp, ytrue_disp = transform_for_display(
                    pred, trg,
                    family_sign=Cfg.family_sign,
                    unit_scale=Cfg.unit_scale,
                    flip_sign=Cfg.flip_sign,
                    clip_nonneg=Cfg.clip_nonneg,
                    min_display_value=Cfg.min_display_value,
                    nonneg_families=Cfg.nonneg_families
                )
                mts = metrics(yhat_disp, ytrue_disp, msk)  # dict: each (K,T)
                if agg_mts is None:
                    agg_mts = {k: v.copy() for k, v in mts.items()}
                else:
                    # 验证集多 batch 时求平均
                    for k in agg_mts.keys():
                        agg_mts[k] += mts[k]
            # 按批数平均
            if agg_mts is not None:
                for k in agg_mts.keys():
                    agg_mts[k] = agg_mts[k] / max(1, len(va_loader))

        # 逐 family R2
        R2_grid = agg_mts["R2"]  # (K,T)
        R2_per_fam = np.nanmean(R2_grid, axis=1)  # (K,)
        R2_overall = np.nanmean(R2_per_fam)

        r2_msg = " | ".join([f"{FAMILIES[k]}={R2_per_fam[k]:.4f}" for k in range(len(FAMILIES))])
        print(f"[StageB][{epoch}/{Cfg.max_epochs}] train_L1={tr_l1:.4f} ({tr_l1_msg}) | "
              f"val_R2={R2_overall:.4f} ({r2_msg})")

        # 保存整体最佳
        if R2_overall > best_overall:
            best_overall = R2_overall
            torch.save({"model": model.state_dict(), "meta": meta}, best_overall_path)
            print(f"  -> saved overall best to {best_overall_path}")

        # 保存逐 family 最佳（基于该 family 的均值 R2）
        for k in range(len(FAMILIES)):
            if np.isnan(R2_per_fam[k]):
                continue
            if R2_per_fam[k] > best_per_fam[k]:
                best_per_fam[k] = R2_per_fam[k]
                torch.save({"model": model.state_dict(), "meta": meta, "best_family": FAMILIES[k]},
                           best_fam_path[k])
                print(f"  -> saved best for {FAMILIES[k]} to {best_fam_path[k]}")

    # ===== 综合导出（整体） =====
    model.eval()
    with torch.no_grad():
        # 拿验证集第一批导出（也可以合并多批）
        for s8, phys, trg, msk, tvals in va_loader:
            s8, phys, trg, msk, tvals = [x.to(device) for x in (s8, phys, trg, msk, tvals)]
            pred = model(s8, phys, tvals)
            pred, trg = _maybe_denorm_targets(pred, trg, meta, device)
            yhat_disp, ytrue_disp = transform_for_display(
                pred, trg,
                family_sign=Cfg.family_sign,
                unit_scale=Cfg.unit_scale,
                flip_sign=Cfg.flip_sign,
                clip_nonneg=Cfg.clip_nonneg,
                min_display_value=Cfg.min_display_value,
                nonneg_families=Cfg.nonneg_families
            )
            mts = metrics(yhat_disp, ytrue_disp, msk)

            # 整体导出（原样保留）
            export_predictions_longtable(
                yhat_disp, ytrue_disp, msk, FAMILIES, meta["time_values"],
                Cfg.save_dir, filename="predictions.xlsx"
            )
            export_metrics_grid(mts, FAMILIES, meta["time_values"], Cfg.save_dir, filename="metrics.xlsx")
            write_summary_txt(mts, FAMILIES, meta["time_values"], Cfg.save_dir)

            heatmap(mts["R2"], FAMILIES, meta["time_values"], "Morph R2",
                    os.path.join(Cfg.save_dir, "morph_r2.png"))
            parity_scatter(yhat_disp, ytrue_disp, msk,
                           os.path.join(Cfg.save_dir, "morph_scatter.png"), "Morph Parity")
            residual_hist(yhat_disp, ytrue_disp, msk,
                          os.path.join(Cfg.save_dir, "morph_residual.png"), "Morph Residuals")

            # ===== 逐 family 独立导出（关键）=====
            if Cfg.export_family_alone:
                for k, fam in enumerate(FAMILIES):
                    fam_dir = os.path.join(Cfg.save_dir, "family", fam)
                    os.makedirs(fam_dir, exist_ok=True)

                    yh_k = _select_family(yhat_disp, k)  # (B,1,T)
                    yt_k = _select_family(ytrue_disp, k) # (B,1,T)
                    m_k  = _select_family(msk, k)        # (B,1,T)
                    fam_list = [fam]

                    export_predictions_longtable(
                        yh_k, yt_k, m_k, fam_list, meta["time_values"],
                        fam_dir, filename=f"{fam}_predictions.xlsx"
                    )

                    # 针对单 family 计算一遍指标（方便该子目录下自洽）
                    mts_k = metrics(yh_k, yt_k, m_k)
                    export_metrics_grid(mts_k, fam_list, meta["time_values"],
                                        fam_dir, filename=f"{fam}_metrics.xlsx")
                    write_summary_txt(mts_k, fam_list, meta["time_values"], fam_dir)

                    # 作图（只画该 family）
                    # 热力图对单 family 是 1×T 的条状图，照样能看各时间点
                    heatmap(mts_k["R2"], fam_list, meta["time_values"], f"{fam} R2",
                            os.path.join(fam_dir, f"{fam}_r2.png"))
                    parity_scatter(
                        yh_k, yt_k, m_k,
                        os.path.join(fam_dir, f"{fam}_scatter.png"),
                        f"{fam} Parity"
                    )
                    residual_hist(
                        yh_k, yt_k, m_k,
                        os.path.join(fam_dir, f"{fam}_residual.png"),
                        f"{fam} Residuals"
                    )
            break  # 只导出一批

    save_manifest(Cfg.save_dir)
    print("[OK] Stage B done.")


if __name__ == "__main__":
    main()
