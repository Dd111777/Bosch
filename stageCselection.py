import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics.pairwise import pairwise_distances
from skopt import gp_minimize
from skopt.space import Real
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

# -------------------------- 导入StageB/C相关工具（关键对接） --------------------------
from physio_util import load_new_excel_as_sparse_morph  # 复用形态解析工具（移除未使用的import）
from stageC_finetune_joint_on_new_pycharm_new import Cfg as CfgC  # 复用StageC的新表路径配置


# -------------------------- 1. 配置类（与StageB/C参数对齐） --------------------------
class ExpConfig:
    """实验配置：严格对齐StageB/C的参数命名与映射关系"""
    # 1. 静态输入参数（顺序与physio_util.static_keys严格一致）
    static_params = {
        "APC": {"min": 10, "max": 100, "step": 5, "unit": "mT"},  # 对应apc
        "source_RF": {"min": 500, "max": 3500, "step": 100, "unit": "W"},  # 对应source_rf
        "LF_RF": {"min": 25, "max": 150, "step": 5, "unit": "W"},  # 对应lf_rf
        "SF6": {"min": 50, "max": 500, "step": 50, "unit": "sccm"},  # 对应sf6
        "C4F8": {"min": 50, "max": 500, "step": 50, "unit": "sccm"},  # 对应c4f8
        "DEP_time": {"min": 0.4, "max": 4.0, "step": 0.4, "unit": "s"},  # 对应dep_time
        "etch_time": {"min": 0.4, "max": 4.0, "step": 0.4, "unit": "s"}  # 对应etch_time
    }
    static_names = list(static_params.keys())  # 顺序必须与static_keys一致
    static_baseline = [25, 2500, 25, 300, 350, 2.0, 1.2]  # 初始基准recipe

    # 2. d目标约束（对应family="d1"，time="3/5/9"，与StageC映射一致）
    d_constraints = {
        "d1_3": {"min": 2000, "max": 2300, "unit": "μm", "center": 2150},  # (d1,3)
        "d1_5": {"min": 2100, "max": 2200, "unit": "μm", "center": 2150},  # (d1,5)
        "d1_9": {"min": 2100, "max": 2300, "unit": "μm", "center": 2200}  # (d1,9)
    }
    d_names = list(d_constraints.keys())
    d_weights = [0.3, 0.3, 0.4]  # 多目标权重（I3组会动态调整）

    # 3. 实验分组
    experiments = [
        # 维度1：mapping验证（仅toy模式）
        {"id": "C0", "dim": "mapping验证", "desc": "全d正确mapping", "seed": 42},
        {"id": "E1", "dim": "mapping验证", "desc": "漏d1_3映射", "seed": 42},
        {"id": "E2", "dim": "mapping验证", "desc": "d1_5单位错", "seed": 42},
        {"id": "E3", "dim": "mapping验证", "desc": "d1_5时间轴错", "seed": 42},
        # 维度2：短期改进（toy/real模式）
        {"id": "B0", "dim": "短期改进", "desc": "基线组（随机+单输出）", "seed": 42},
        {"id": "I1", "dim": "短期改进", "desc": "LHS采样+多输出", "seed": 42},
        {"id": "I2", "dim": "短期改进", "desc": "BO多输出", "seed": 42},
        {"id": "I3", "dim": "短期改进", "desc": "多d权重调优", "seed": 42},
        {"id": "I4", "dim": "短期改进", "desc": "组合组（LHS+BO+权重）", "seed": 42},
        # 维度3：中期融合（占位）
        {"id": "M1", "dim": "中期融合", "desc": "[占位]单独MFL", "seed": 42},
        {"id": "M2", "dim": "中期融合", "desc": "[占位]单独StageC", "seed": 42},
        {"id": "M3", "dim": "中期融合", "desc": "[占位]融合组", "seed": 42}
    ]

    # 4. 路径配置（与StageC共享数据路径）
    save_dir = "./stageC_results"
    excel_name = "stageC_selection_results.xlsx"
    real_meas_path = CfgC.new_excel  # 复用StageC的新表路径
    stageC_ckpt = "./ckpt/stageC_best.pth"  # StageC模型权重路径
    mfl_ckpt = "./ckpt/mfl_reverse.pth"  # MFL反向模型路径

    # 5. 实验参数
    n_candidates = 2000  # 候选池大小
    n_test = 10  # 单轮选点数量
    n_rounds = 3  # 多轮选点轮次（实战用）
    fig_dpi = 300


# -------------------------- 2. 基础工具函数 --------------------------
def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def sample_candidates(config: ExpConfig, exp: Dict, sample_type: str = "random") -> np.ndarray:
    """生成候选recipe（按设备步长量化）"""
    n = config.n_candidates
    dim = len(config.static_names)
    X = np.zeros((n, dim))

    # 采样
    for i, name in enumerate(config.static_names):
        min_val = config.static_params[name]["min"]
        max_val = config.static_params[name]["max"]
        if sample_type == "random":
            X[:, i] = np.random.uniform(min_val, max_val, n)
        elif sample_type == "LHS":
            partitions = np.linspace(min_val, max_val, n + 1)
            for j in range(n):
                X[j, i] = np.random.uniform(partitions[j], partitions[j + 1])
            np.random.shuffle(X[:, i])

    # 按设备步长量化（匹配物理调节能力）
    for i, name in enumerate(config.static_names):
        step = config.static_params[name]["step"]
        X[:, i] = np.round(X[:, i] / step) * step

    return X.astype(np.float32)


# -------------------------- 3. 代理模型接口（区分toy与真实） --------------------------
def surrogate_predict(config: ExpConfig, X: np.ndarray, exp: Dict,
                      mode: str = "analytic", aux: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """统一代理模型接口"""
    if mode == "analytic":
        return simulate_d_pred(config, X, exp)
    elif mode == "stageC":
        model = aux.get("stageC_model")
        return stageC_forward(model, X)
    elif mode == "gp":
        # 多输出GP：每个d目标对应一个模型
        gp_models: List[GaussianProcessRegressor] = aux["gp_models"]
        n = X.shape[0]
        n_targets = len(gp_models)

        mu = np.zeros((n, n_targets), dtype=np.float32)
        sigma = np.zeros((n, n_targets), dtype=np.float32)

        for j, gp in enumerate(gp_models):
            mu_j, sigma_j = gp.predict(X, return_std=True)  # mu_j, sigma_j: (n,)
            mu[:, j] = mu_j
            sigma[:, j] = sigma_j

        return mu, sigma
    else:
        raise ValueError(f"不支持的代理模型模式：{mode}")


def simulate_d_pred(config: ExpConfig, X: np.ndarray, exp: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """toy版d预测（解析公式）"""
    n = X.shape[0]
    pred_d = np.zeros((n, 3))
    sigma_d = np.ones((n, 3)) * 0.3

    # 物理关联公式
    for i in range(n):
        apc, source_rf, lf_rf, sf6, c4f8, dep, etch = X[i]
        pred_d[i, 0] = 2100 + 0.05 * source_rf + 0.1 * sf6 - 2 * apc + np.random.normal(0, 5)  # d1_3
        pred_d[i, 1] = 2150 + 0.04 * source_rf + 0.08 * sf6 - 1.8 * apc + np.random.normal(0, 4)  # d1_5
        pred_d[i, 2] = 2200 + 0.03 * source_rf + 0.06 * sf6 - 1.5 * apc + 0.5 * etch + np.random.normal(0, 3)  # d1_9

    # 模拟mapping错误
    if exp["id"] == "E1":
        pred_d[:, 0] = -1  # 漏d1_3
        sigma_d[:, 0] = -1
    elif exp["id"] == "E2":
        pred_d[:, 1] *= 1000  # 单位错误（nm未转μm）
        sigma_d[:, 1] *= 1000
    elif exp["id"] == "E3":
        pred_d[:, 1] += 50  # 时间轴错误
        sigma_d[:, 1] *= 2

    return pred_d, sigma_d


# -------------------------- 4. 真实场景接口（与StageB/C数据流对接） --------------------------
def load_real_measurements(config: ExpConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    从Bosch_new.xlsx读取真实实测数据（与StageC解析逻辑一致）
    返回：
    - X_meas: (N, 7) 静态参数 [APC, source_RF, LF_RF, SF6, C4F8, DEP_time, etch_time]
    - d_meas: (N, 3) 实测d值 [d1_3, d1_5, d1_9]
    """
    # 1. 用StageC同款工具解析新表（family="h1/d1"）
    recs = load_new_excel_as_sparse_morph(
        config.real_meas_path,
        height_family="h1"  # 与StageB/C保持一致
    )

    X_list, d_list = [], []
    for rec in recs:
        # 提取静态参数（与static_names顺序对应）
        static = rec["static"]  # shape: (7,)，顺序对应[apc, source_rf, ..., etch_time]
        X_list.append(static)

        # 提取d1_3/5/9（对应family="d1"，time="3"/"5"/"9"）
        tg = rec["targets"]  # dict: (family, tid) -> 实测值
        d_vals = []
        for tid in ["3", "5", "9"]:
            key = ("d1", tid)
            if key in tg and tg[key] is not None:
                d_vals.append(float(tg[key]))
            else:
                d_vals.append(np.nan)  # 标记缺失值
        d_list.append(d_vals)

    # 转换为数组并过滤缺失值
    X_meas = np.array(X_list, dtype=np.float32)
    d_meas = np.array(d_list, dtype=np.float32)
    mask = np.isfinite(d_meas).all(axis=1)  # 仅保留三个d都有效的样本
    X_meas = X_meas[mask]
    d_meas = d_meas[mask]

    print(f"加载真实实测数据：{X_meas.shape[0]}个有效样本")
    return X_meas, d_meas


def load_stageC_model(config: ExpConfig) -> torch.nn.Module:
    """
    加载StageC模型（当前为占位，需后续对接真实TemporalRegressor）
    真实实现需包含：
    - TemporalRegressor主体网络
    - per-family head（针对"d1"族）
    - 校准层（calib）
    """

    class DummyStageC(nn.Module):
        """占位模型，仅用于接口测试，不可用于真实场景"""

        def forward(self, x):
            # x: (N, 7) 静态参数
            w = torch.randn(x.shape[1], 3)  # 随机权重（仅toy）
            return x @ w, torch.ones(x.shape[0], 3) * 0.3  # 预测值+不确定性

    model = DummyStageC()
    # 注意：真实StageC模型加载需匹配网络结构，当前不加载ckpt避免报错
    # TODO: 对接真实模型时，实例化TemporalRegressor并加载config.stageC_ckpt
    return model


def stageC_forward(model: torch.nn.Module, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    StageC模型前向预测（当前未实现真实逻辑，禁止在real模式使用）
    真实实现需包含：
    1. 静态参数归一化（复用meta_old的norm_static）
    2. 调用StageA物理模型生成phys_enh
    3. 输入TemporalRegressor得到全时间序列预测
    4. 提取(d1,3)/(d1,5)/(d1,9)对应的预测值
    """
    raise NotImplementedError(
        "stageC_forward尚未对接真实StageC模型！\n"
        "请先实现：\n"
        "1) 静态参数归一化\n"
        "2) StageA物理特征生成\n"
        "3) TemporalRegressor前向计算\n"
        "4) 提取d1_3/5/9的预测值"
    )


def train_gp_surrogate(X_meas: np.ndarray, d_meas: np.ndarray) -> List[GaussianProcessRegressor]:
    """
    训练GP代理模型（基于实测数据）
    - 为每个d目标各训练一个独立的GP：d1_3 / d1_5 / d1_9
    """
    n_targets = d_meas.shape[1]
    models: List[GaussianProcessRegressor] = []

    for j in range(n_targets):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X_meas.shape[1])) \
                 + WhiteKernel(noise_level=1e-3)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,
            random_state=42 + j  # 每个目标一个随机种子，确保可复现
        )
        gp.fit(X_meas, d_meas[:, j])  # 单输出训练
        models.append(gp)

    return models


# -------------------------- 5. 选点核心逻辑 --------------------------
def calculate_score(config: ExpConfig, pred_d: np.ndarray, sigma_d: np.ndarray,
                    X_cand: np.ndarray, X_meas: np.ndarray, y_meas: np.ndarray = None) -> np.ndarray:
    """计算候选评分（EI + 多样性）"""
    n = pred_d.shape[0]
    score = np.zeros(n)

    # 1. 目标函数：距离区间中心的距离（越小越好）
    target_scores = np.zeros_like(pred_d)
    for j, d_name in enumerate(config.d_names):
        min_d = config.d_constraints[d_name]["min"]
        max_d = config.d_constraints[d_name]["max"]
        center_d = config.d_constraints[d_name]["center"]
        target_scores[:, j] = np.abs(pred_d[:, j] - center_d)
        # 超出区间的惩罚
        out_of_bounds = (pred_d[:, j] < min_d) | (pred_d[:, j] > max_d)
        target_scores[out_of_bounds, j] *= 5

    # 2. EI计算（基于目标函数）
    if y_meas is not None and y_meas.size > 0:
        # 实战模式：用实测最佳值
        target_best = np.array([
            np.abs(y_meas[:, j] - config.d_constraints[config.d_names[j]]["center"]).min()
            for j in range(3)
        ])
    else:
        # toy模式：用候选池最佳预测值
        target_best = np.array([
            target_scores[pred_d[:, j] != -1, j].min()
            if (pred_d[:, j] != -1).any() else np.inf
            for j in range(3)
        ])

    ei_per_d = np.zeros_like(pred_d)
    for j in range(3):
        valid = (pred_d[:, j] != -1) & (sigma_d[:, j] != -1)
        if not valid.any():
            continue
        mu = target_scores[valid, j]
        sigma = sigma_d[valid, j]
        improvement = target_best[j] - mu
        z = improvement / (sigma + 1e-8)
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei_per_d[valid, j] = np.maximum(ei, 0)

    # 3. 多样性权重（归一化后计算）
    if X_meas.size == 0:
        div_weight = np.ones(n)
    else:
        mins = np.array([config.static_params[n]["min"] for n in config.static_names])
        maxs = np.array([config.static_params[n]["max"] for n in config.static_names])
        X_c_norm = (X_cand - mins) / (maxs - mins + 1e-8)
        X_m_norm = (X_meas - mins) / (maxs - mins + 1e-8)
        dist = pairwise_distances(X_c_norm, X_m_norm, metric="euclidean")
        min_dist = dist.min(axis=1)
        div_weight = 1.0 / (1.0 + np.exp(-min_dist / 0.5))

    # 4. 综合评分（加权EI * 多样性）
    ei_weighted = np.dot(ei_per_d, config.d_weights)
    score = ei_weighted * div_weight

    # 5. 硬过滤：预测值严重超出物理目标区间的recipe直接丢弃（score=0）
    hard_out = np.zeros(n, dtype=bool)
    for j, d_name in enumerate(config.d_names):
        min_d = config.d_constraints[d_name]["min"]
        max_d = config.d_constraints[d_name]["max"]
        # 允许一点buffer，比如在区间两端各放宽 20%
        margin = 0.2 * (max_d - min_d)
        hard_out |= (pred_d[:, j] < (min_d - margin)) | (pred_d[:, j] > (max_d + margin))

    score[hard_out] = 0.0

    return score


def judge_d_qualified(config: ExpConfig, measured_d: np.ndarray) -> Tuple[List[bool], List[List[bool]]]:
    """判断d是否在目标区间内"""
    n = measured_d.shape[0]
    full_qualified = []
    single_qualified = []
    for i in range(n):
        single = []
        for j, name in enumerate(config.d_names):
            min_d = config.d_constraints[name]["min"]
            max_d = config.d_constraints[name]["max"]
            single.append(min_d <= measured_d[i, j] <= max_d)
        full_qualified.append(all(single))
        single_qualified.append(single)
    return full_qualified, single_qualified


# -------------------------- 6. 多轮主动选点 --------------------------
def active_selection_loop(config: ExpConfig, strategy_id: str, exp: Dict) -> List[Dict]:
    """多轮主动选点循环（实战模式）"""
    # 固定随机种子确保可复现
    np.random.seed(exp["seed"] + 1000)

    # 1. 加载初始实测数据（真实数据）
    X_meas, d_meas = load_real_measurements(config)
    if X_meas.size == 0:
        raise ValueError("未加载到有效实测数据，请检查real_meas_path是否正确")
    all_round_results = []

    for round_idx in range(config.n_rounds):
        print(f"\n===== 主动选点第{round_idx + 1}/{config.n_rounds}轮 =====")
        # 2. 选择代理模型（仅处理已实现的策略）
        if strategy_id in ["I2", "I4"]:
            # BO策略：用GP surrogate（每个d独立模型）
            gp_models = train_gp_surrogate(X_meas, d_meas)
            surrogate_mode = "gp"
            aux = {"gp_models": gp_models}
        elif strategy_id == "M1":
            # MFL策略（占位：未来替换为MFL反向模型）
            surrogate_mode = "analytic"
            aux = {}
        else:
            raise NotImplementedError(
                f"策略{strategy_id}尚未在real模式中实现，请在main中去掉该策略或补全实现。"
            )

        # 3. 生成候选池
        sample_type = "LHS" if strategy_id in ["I1", "I4", "M3"] else "random"
        X_cand = sample_candidates(config, exp, sample_type=sample_type)

        # 4. 预测与评分
        pred_d, sigma_d = surrogate_predict(config, X_cand, exp, mode=surrogate_mode, aux=aux)
        score = calculate_score(config, pred_d, sigma_d, X_cand, X_meas, y_meas=d_meas)

        # 5. 选择本轮推荐点
        top_idx = np.argsort(-score)[:config.n_test]
        X_next = X_cand[top_idx]
        pred_d_next = pred_d[top_idx]

        # 6. 记录本轮结果
        round_result = {
            "round": round_idx + 1,
            "X_recommended": X_next,
            "pred_d": pred_d_next,
            "X_measured": X_meas.copy(),
            "d_measured": d_meas.copy()
        }
        all_round_results.append(round_result)
        print(f"第{round_idx + 1}轮推荐完成，选点数量：{len(X_next)}")

        # 7. 模拟新增实测（真实场景中需替换为真实测量）
        simulated_d = simulate_d_measured(pred_d_next, exp)
        X_meas = np.vstack([X_meas, X_next])
        d_meas = np.vstack([d_meas, simulated_d])

    return all_round_results


# -------------------------- 7. 实验执行与结果保存 --------------------------
def run_single_experiment(config: ExpConfig, exp: Dict, run_mode: str = "toy") -> Dict:
    """执行单组实验（支持toy/real模式）"""
    print(f"\n{'=' * 50}")
    print(f"实验组：{exp['id']} | {exp['dim']} | {exp['desc']}")  # 修复文本拼写错误
    print(f"模式：{'消融实验' if run_mode == 'toy' else '实战选点'}")
    print(f"{'=' * 50}")

    # 统一随机种子
    np.random.seed(exp["seed"])
    # 保存原始权重，用于I3组动态调整后恢复
    orig_weights = config.d_weights.copy()

    try:
        # 针对I3组调整权重（体现多d权重调优）
        if exp["id"] == "I3":
            config.d_weights = [0.2, 0.4, 0.4]  # 更侧重d1_5和d1_9

        if run_mode == "toy":
            # toy模式：单轮消融实验
            sample_type = "LHS" if exp["id"] in ["I1", "I4"] else "random"
            X_cand = sample_candidates(config, exp, sample_type=sample_type)

            # 预测d
            if exp["id"] in ["I2", "I4"]:
                pred_d, sigma_d = bo_predict(config, X_cand, exp)
                model_type = "BO多输出"
            elif exp["id"] in ["B0"]:
                pred_d, sigma_d = surrogate_predict(config, X_cand, exp, mode="analytic")
                pred_d[:, [0, 1]] = -1  # 单输出（仅d1_9）
                sigma_d[:, [0, 1]] = -1
                model_type = "Analytic单输出"
            else:
                pred_d, sigma_d = surrogate_predict(config, X_cand, exp, mode="analytic")
                model_type = "Analytic多输出"

            # 评分与选点
            X_meas_init = np.array(config.static_baseline).reshape(1, -1)
            score = calculate_score(config, pred_d, sigma_d, X_cand, X_meas_init)
            top_idx = np.argsort(-score)[:config.n_test]
            X_top = X_cand[top_idx]
            pred_d_top = pred_d[top_idx]

            # 模拟实测
            measured_d_top = simulate_d_measured(pred_d_top, exp)
            full_qualified, single_qualified = judge_d_qualified(config, measured_d_top)

            # 计算指标
            full_rate = sum(full_qualified) / len(full_qualified) * 100
            single_rates = [sum(q[j] for q in single_qualified) / len(single_qualified) * 100 for j in range(3)]
            temp_noise = np.random.normal(0, 0.15, len(measured_d_top))
            avg_fluct = np.mean(np.abs(temp_noise))

            # 达标次数计算
            cum_qualified = 0
            required_trials = "无法达标"
            for i in range(len(full_qualified)):
                cum_qualified += full_qualified[i]
                if cum_qualified / (i + 1) * 100 >= 80:
                    required_trials = str(i + 1)
                    break

            return {
                "exp_info": exp,
                "top_candidates": {
                    "X": X_top,
                    "pred_d": pred_d_top,
                    "measured_d": measured_d_top,
                    "full_qualified": full_qualified,  # 全d达标
                    "single_qualified": single_qualified  # 每个d达标
                },
                "metrics": {
                    "full_qualified_rate": full_rate,
                    "single_rates": single_rates,
                    "avg_fluctuation": avg_fluct,
                    "required_trials": required_trials,
                    "model_type": model_type,
                    "sample_type": sample_type
                }
            }

        # 实战模式：多轮主动选点（不在if内部，避免出现 return 后的 else 语法错误）
        return {
            "exp_info": exp,
            "round_results": active_selection_loop(config, exp["id"], exp)
        }
    finally:
        # 恢复原始权重，避免影响其他实验组
        config.d_weights = orig_weights


def bo_predict(config: ExpConfig, X_cand: np.ndarray, exp: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """BO多输出预测（基于gp_minimize）"""
    space = [
        Real(config.static_params[name]["min"], config.static_params[name]["max"], name=name)
        for name in config.static_names
    ]

    def objective(params: List[float]) -> float:
        X = np.array(params).reshape(1, -1)
        pred_d, _ = surrogate_predict(config, X, exp, mode="analytic")
        target_scores = [
            np.abs(pred_d[0, j] - config.d_constraints[config.d_names[j]]["center"])
            for j in range(3)
        ]
        return np.dot(target_scores, config.d_weights)

    result = gp_minimize(
        func=objective, space=space, n_calls=200, n_initial_points=50,
        random_state=exp["seed"], acq_func="EI"
    )

    pred_d = np.zeros((X_cand.shape[0], 3))
    sigma_d = np.ones((X_cand.shape[0], 3)) * 0.25
    for i in range(X_cand.shape[0]):
        pred_i, _ = surrogate_predict(config, X_cand[i:i + 1], exp, mode="analytic")
        pred_d[i] = pred_i[0]
        # 归一化距离计算不确定性
        mins = np.array([config.static_params[n]["min"] for n in config.static_names])
        maxs = np.array([config.static_params[n]["max"] for n in config.static_names])
        x_norm = (X_cand[i] - mins) / (maxs - mins + 1e-8)
        best_norm = (np.array(result.x) - mins) / (maxs - mins + 1e-8)
        dist = np.linalg.norm(x_norm - best_norm)
        sigma_d[i] *= (1 + dist)
    return pred_d, sigma_d


def simulate_d_measured(pred_d: np.ndarray, exp: Dict) -> np.ndarray:
    """模拟实测d值（加入工艺噪声）"""
    noise_scale = [3, 2, 1.5]
    noise = np.random.normal(loc=0, scale=noise_scale, size=pred_d.shape)
    measured_d = pred_d + noise

    # 处理mapping错误
    if exp["id"] == "E1":
        measured_d[:, 0] = 2350 + np.random.normal(0, 10, size=len(measured_d))
    elif exp["id"] == "E2":
        measured_d[:, 1] /= 1000
    return measured_d


def save_results(config: ExpConfig, all_results: List[Dict]):
    """保存实验结果（修复达标标记bug）"""
    ensure_dir(config.save_dir)

    # 1. 详细结果Excel
    with pd.ExcelWriter(os.path.join(config.save_dir, config.excel_name), engine="openpyxl") as writer:
        # 实验参数表
        exp_params = []
        for res in all_results:
            exp = res["exp_info"]
            metrics = res.get("metrics", {})
            exp_params.append({
                "实验组ID": exp["id"],
                "维度": exp["dim"],
                "描述": exp["desc"],
                "模型类型": metrics.get("model_type", "N/A"),
                "采样方式": metrics.get("sample_type", "N/A"),
                "全d达标率(%)": metrics.get("full_qualified_rate", "N/A")
            })
        pd.DataFrame(exp_params).to_excel(writer, sheet_name="实验参数", index=False)

        # 候选详情表（仅toy模式有数据）
        details = []
        for res in all_results:
            exp = res["exp_info"]
            top = res.get("top_candidates", {})
            if not top:  # real模式无top_candidates，跳过
                continue
            # 遍历每个候选
            for i in range(len(top["X"])):
                row = {"实验组ID": exp["id"], "候选序号": i + 1}
                # 静态参数
                for j, name in enumerate(config.static_names):
                    row[name] = top["X"][i, j]
                # d值与达标情况（修复bug：使用single_qualified）
                for j, d_name in enumerate(config.d_names):
                    row[f"预测{d_name}"] = top["pred_d"][i, j]
                    row[f"实测{d_name}"] = top["measured_d"][i, j]
                    row[f"{d_name}达标"] = "是" if top["single_qualified"][i][j] else "否"
                row["全d达标"] = "是" if top["full_qualified"][i] else "否"
                details.append(row)
        pd.DataFrame(details).to_excel(writer, sheet_name="候选详情", index=False)

    # 2. 实战推荐清单（CSV，仅real模式有数据）
    recommendations = []
    for res in all_results:
        exp = res["exp_info"]
        rounds = res.get("round_results", [])
        for round_res in rounds:
            for i, x in enumerate(round_res["X_recommended"]):
                row = {
                    "实验组ID": exp["id"],
                    "轮次": round_res["round"],
                    "候选序号": i + 1
                }
                for j, name in enumerate(config.static_names):
                    row[name] = x[j]
                for j, d_name in enumerate(config.d_names):
                    row[f"预测{d_name}"] = round_res["pred_d"][i, j]
                recommendations.append(row)
    if recommendations:  # 仅当有数据时保存
        pd.DataFrame(recommendations).to_csv(
            os.path.join(config.save_dir, "推荐recipe清单.csv"), index=False
        )

    print(f"结果已保存至：{config.save_dir}")


# -------------------------- 8. 主函数 --------------------------
def main(run_mode: str = "toy"):
    """主函数：运行所有实验并保存结果"""
    config = ExpConfig()
    ensure_dir(config.save_dir)

    # 根据模式选择要跑的实验组
    if run_mode == "toy":
        exp_list = config.experiments  # toy模式跑所有组
    else:
        # 实战模式：
        #   I2 - 纯 BO + GP baseline
        #   I4 - LHS + BO + d权重调优（推荐用来补实测的策略）
        real_ids = ["I2", "I4"]
        exp_list = [e for e in config.experiments if e["id"] in real_ids]

    # 运行实验
    all_results = []
    for exp in exp_list:
        result = run_single_experiment(config, exp, run_mode=run_mode)
        all_results.append(result)

    # 保存结果
    save_results(config, all_results)
    print("\n所有实验完成！")


if __name__ == "__main__":
    # 模式选择：
    #   "toy"  - 消融实验（toy解析模型 + 映射错误注入）
    #   "real" - 实战选点（用真实new表 + GP surrogate，当前不依赖stageC_forward）
    # 真实补充实测时建议用 run_mode="real"
    main(run_mode="toy")
    # 想跑实战选点时改成：
    # main(run_mode="real")
