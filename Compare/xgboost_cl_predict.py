# -*- coding: utf-8 -*-
"""
XGBoost 回归模型预测动态俯仰翼型升力系数 CL
================================================
严格规则：
  - NACA4415 全部数据 → 测试集（完全独立，不参与训练/拟合）
  - 其余翼型全部数据 → 训练集
  - 保留所有原始攻角范围，不插值不外推不排序不平滑
  - CST 几何特征从 <翼型名>_CST.txt 自动读取作为模型输入
"""

import os
import re
import warnings
import glob
import traceback

import numpy as np
import pandas as pd
import h5py
from scipy import io as sio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ============================================================
# 全局配置
# ============================================================
ROOT_DIR = r"D:\AirfoilDesign\Data"
DYNAMIC_DIR = os.path.join(ROOT_DIR, "Dynamic")

TEST_AIRFOIL = "NACA4415"
RANDOM_STATE = 42
EPS = 1e-12

# 粗糙度编码：C=0，G=1（可根据需要调整）
ROUGHNESS_MAP = {"C": 0, "G": 1}

# XGBoost baseline 超参数（便于修改）
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# 输出文件
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_OUTPUT = os.path.join(OUTPUT_DIR, f"{TEST_AIRFOIL}_XGBoost_CL_predictions.csv")
MODEL_OUTPUT = os.path.join(OUTPUT_DIR, f"{TEST_AIRFOIL}_XGBoost_CL_model.json")
REPORT_OUTPUT = os.path.join(OUTPUT_DIR, f"{TEST_AIRFOIL}_XGBoost_CL_evaluation_report.txt")

warnings.filterwarnings("ignore")


# ============================================================
# 1. CST 文件解析
# ============================================================
def parse_cst_file(airfoil_name: str):
    """
    读取 <翼型名>_CST.txt，解析 22 个几何特征：
      N1, N2, A_u_1..A_u_9, z_u_TE, A_l_1..A_l_9, z_l_TE
    兼容中英文逗号/分号、科学计数法。
    返回 dict 或 None（失败时打印警告）。
    """
    cst_path = os.path.join(ROOT_DIR, f"{airfoil_name}_CST.txt")
    if not os.path.exists(cst_path):
        print(f"  [警告] 翼型 {airfoil_name} 缺少 CST 文件: {cst_path}，跳过该翼型")
        return None

    try:
        with open(cst_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            with open(cst_path, "r", encoding="gbk") as f:
                raw = f.read().strip()
    except Exception as e:
        print(f"  [警告] 读取 CST 文件失败 {cst_path}: {e}，跳过该翼型")
        return None

    # 统一符号：中文逗号/分号 → 英文
    text = (
        raw.replace("，", ",")
           .replace("；", ";")
           .replace("=", " = ")
    )

    def _first_float(pattern, text_str, field_name):
        m = re.search(pattern, text_str, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"未找到 {field_name}")
        return float(m.group(1))

    def _list_float(pattern, text_str, field_name, expected=9):
        m = re.search(pattern, text_str, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"未找到 {field_name} 列表")
        bracket = m.group(1)
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", bracket)
        nums = [float(x) for x in nums]
        if len(nums) != expected:
            raise ValueError(f"{field_name} 长度应为 {expected}，实际 {len(nums)}")
        return nums

    try:
        N1 = _first_float(r"N1\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text, "N1")
        N2 = _first_float(r"N2\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text, "N2")
        A_u = _list_float(r"A_u\s*=\s*\[([^\]]*)\]", text, "A_u", expected=9)
        z_u_TE = _first_float(r"z_u_TE\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text, "z_u_TE")
        A_l = _list_float(r"A_l\s*=\s*\[([^\]]*)\]", text, "A_l", expected=9)
        z_l_TE = _first_float(r"z_l_TE\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text, "z_l_TE")
    except ValueError as e:
        print(f"  [警告] 解析 CST 文件内容失败 {cst_path}: {e}，跳过该翼型")
        return None

    # 组装 22 维特征字典
    feat = {"N1": N1, "N2": N2}
    for i in range(9):
        feat[f"A_u_{i+1}"] = A_u[i]
    feat["z_u_TE"] = z_u_TE
    for i in range(9):
        feat[f"A_l_{i+1}"] = A_l[i]
    feat["z_l_TE"] = z_l_TE

    return feat


# ============================================================
# 2. MAT 文件读取（兼容 scipy 普通格式 和 h5py v7.3 格式）
# ============================================================
def load_mat_data(mat_path: str):
    """
    自动寻找目标矩阵：
      - 二维数值矩阵，至少 3 列，行数 > 列数
      - 排除 __header__ / __version__ / __globals__
      - 多个候选时取行数最多者
    返回 (alpha, cl) 两个 1D numpy 数组，或 (None, None) 表示失败。
    """
    data_matrix = None

    # 方式一：先尝试 scipy.io.loadmat（传统格式）
    try:
        mat = sio.loadmat(mat_path)
        candidates = []
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if not isinstance(v, np.ndarray):
                continue
            if v.ndim != 2:
                continue
            if v.shape[1] < 3:
                # 尝试转置
                if v.shape[0] >= 3 and v.shape[1] < v.shape[0]:
                    pass
                else:
                    continue
            arr = np.asarray(v, dtype=np.float64)
            # 确保行数 > 列数且列数 >= 3
            if arr.shape[0] < arr.shape[1] and arr.shape[1] >= arr.shape[0]:
                arr = arr.T
            if arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] > arr.shape[1]:
                candidates.append((arr.shape[0], arr))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            data_matrix = candidates[0][1]
    except NotImplementedError:
        # v7.3 HDF5 格式，换用 h5py
        data_matrix = None
    except Exception:
        data_matrix = None

    # 方式二：用 h5py 读取（v7.3 格式）
    if data_matrix is None:
        try:
            candidates = []
            with h5py.File(mat_path, "r") as f:
                def _collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            arr = np.asarray(obj[()], dtype=np.float64)
                        except Exception:
                            return
                        if arr.ndim != 2:
                            return
                        # h5py 列优先，MATLAB 数据通常 (cols, rows) 存储
                        # 统一保证 行数 > 列数 且 列数 >= 3
                        if arr.shape[0] < arr.shape[1]:
                            arr = arr.T
                        if arr.shape[1] >= 3 and arr.shape[0] > arr.shape[1]:
                            candidates.append((arr.shape[0], arr.copy()))
                f.visititems(_collect)
            if candidates:
                candidates.sort(key=lambda x: -x[0])
                data_matrix = candidates[0][1]
        except Exception:
            data_matrix = None

    if data_matrix is None:
        print(f"    [警告] 未找到合法数据矩阵: {mat_path}")
        return None, None

    alpha = data_matrix[:, 0].ravel()
    cl = data_matrix[:, 1].ravel()

    # 删除 NaN/Inf
    mask = np.isfinite(alpha) & np.isfinite(cl)
    if not mask.all():
        alpha = alpha[mask]
        cl = cl[mask]
    if len(alpha) < 3:
        print(f"    [警告] 有效样本过少: {mat_path}")
        return None, None

    return alpha, cl


# ============================================================
# 3. 文件名 / 工况名解析
# ============================================================
def parse_working_condition(dir_name: str):
    """
    解析工况目录名，如 C_RE_75 / G_RE_100 / C_RE_125
    返回 (roughness_str, Re_value) 或 (None, None)
    """
    m = re.match(r"^([CG])_RE_(\d+)$", dir_name.strip(), flags=re.IGNORECASE)
    if not m:
        return None, None
    roughness = m.group(1).upper()
    re_int = int(m.group(2))
    # RE_75 -> 0.75e6, RE_100 -> 1.0e6, RE_125 -> 1.25e6, RE_150 -> 1.5e6, RE_140 -> 1.4e6
    Re = re_int / 100.0 * 1e6
    return roughness, Re


def parse_mat_filename(fname: str):
    """
    解析 MAT 文件名，兼容两种格式：
      格式A（用户描述）：1_14_10_0.5_0.1.mat
        -> experiment_id, mean_aoa, amplitude, oscillation_frequency, decay_frequency
      格式B（实际数据）：1_8_5_v0.60_d0.038.mat
        -> 用 v / d 前缀提取频率
    返回 dict 或 None。
    """
    base = os.path.splitext(fname)[0].strip()

    # 优先尝试格式 B：带 v/d 前缀
    # pattern:  <exp>_<mean>_<amp>_v<osc>_d<decay>
    m = re.match(
        r"^(\d+)_([-+]?\d+(?:\.\d+)?)_([-+]?\d+(?:\.\d+)?)_v([-+]?\d+(?:\.\d+)?)_d([-+]?\d+(?:\.\d+)?)$",
        base,
    )
    if m:
        return {
            "experiment_id": int(m.group(1)),
            "mean_aoa": float(m.group(2)),
            "amplitude": float(m.group(3)),
            "oscillation_frequency": float(m.group(4)),
            "decay_frequency": float(m.group(5)),
        }

    # 回退格式 A：5 个下划线分割数字
    parts = base.split("_")
    if len(parts) == 5:
        try:
            return {
                "experiment_id": int(parts[0]),
                "mean_aoa": float(parts[1]),
                "amplitude": float(parts[2]),
                "oscillation_frequency": float(parts[3]),
                "decay_frequency": float(parts[4]),
            }
        except ValueError:
            pass

    print(f"    [警告] 无法解析 MAT 文件名: {fname}，跳过")
    return None


# ============================================================
# 4. 构建完整数据集（DataFrame）
# ============================================================
def build_dataset():
    """
    扫描 DYNAMIC_DIR 下所有翼型目录，读取所有合法 MAT，
    划分上升/下降段，附加 CST 特征，返回完整 DataFrame。
    """
    # 先枚举所有翼型目录
    if not os.path.isdir(DYNAMIC_DIR):
        raise FileNotFoundError(f"Dynamic 目录不存在: {DYNAMIC_DIR}")

    airfoil_dirs = []
    for name in sorted(os.listdir(DYNAMIC_DIR)):
        full = os.path.join(DYNAMIC_DIR, name)
        if os.path.isdir(full):
            # 跳过空目录或明显非翼型目录（不含任何 C_RE / G_RE 子目录的）
            has_wc = any(
                os.path.isdir(os.path.join(full, x)) and re.match(r"^[CG]_RE_\d+$", x, re.I)
                for x in os.listdir(full)
            )
            if has_wc:
                airfoil_dirs.append(name)
            else:
                print(f"[跳过] 非翼型数据目录或无有效工况: {name}")

    print(f"\n发现翼型目录 {len(airfoil_dirs)} 个: {airfoil_dirs}")

    rows = []
    file_counts = {"train": set(), "test": set()}
    airfoil_counts = {}

    for airfoil_name in airfoil_dirs:
        # 读取 CST（每个翼型读一次）
        cst_feat = parse_cst_file(airfoil_name)
        if cst_feat is None:
            continue

        airfoil_path = os.path.join(DYNAMIC_DIR, airfoil_name)
        airfoil_counts[airfoil_name] = 0

        # 遍历工况目录
        for wc_name in sorted(os.listdir(airfoil_path)):
            wc_path = os.path.join(airfoil_path, wc_name)
            if not os.path.isdir(wc_path):
                continue
            roughness, Re = parse_working_condition(wc_name)
            if roughness is None:
                # print(f"    [跳过] 非法工况目录名: {wc_name}")
                continue
            roughness_val = ROUGHNESS_MAP.get(roughness)
            if roughness_val is None:
                continue

            # 遍历所有 .mat 文件
            mat_files = sorted(glob.glob(os.path.join(wc_path, "*.mat")))
            for mat_path in mat_files:
                fname = os.path.basename(mat_path)
                meta = parse_mat_filename(fname)
                if meta is None:
                    continue

                alpha, cl = load_mat_data(mat_path)
                if alpha is None:
                    continue

                # 记录文件（相对路径作为标识）
                source_rel = os.path.join(airfoil_name, wc_name, fname)
                if airfoil_name == TEST_AIRFOIL:
                    file_counts["test"].add(source_rel)
                else:
                    file_counts["train"].add(source_rel)

                # 按最大攻角划分上下段（保留原始时间顺序）
                peak_index = int(np.argmax(alpha))

                # 上升段 direction=1
                alpha_up = alpha[: peak_index + 1]
                cl_up = cl[: peak_index + 1]
                for idx in range(len(alpha_up)):
                    row = {
                        # 动态/工况特征
                        "alpha": float(alpha_up[idx]),
                        "direction": 1,
                        "Re": float(Re),
                        "roughness": roughness_val,
                        "mean_aoa": float(meta["mean_aoa"]),
                        "amplitude": float(meta["amplitude"]),
                        "oscillation_frequency": float(meta["oscillation_frequency"]),
                        "decay_frequency": float(meta["decay_frequency"]),
                        # 目标
                        "CL": float(cl_up[idx]),
                        # 追踪信息
                        "experiment_id": int(meta["experiment_id"]),
                        "airfoil_name": airfoil_name,
                        "source_file": source_rel,
                        "original_index": idx,  # 先记段内索引，稍后调整为全局时间序
                    }
                    row.update(cst_feat)
                    rows.append(row)
                    airfoil_counts[airfoil_name] += 1

                # 下降段 direction=-1
                alpha_down = alpha[peak_index:]
                cl_down = cl[peak_index:]
                offset = peak_index  # 使得下降段 original_index 紧接上升段
                for idx in range(len(alpha_down)):
                    row = {
                        "alpha": float(alpha_down[idx]),
                        "direction": -1,
                        "Re": float(Re),
                        "roughness": roughness_val,
                        "mean_aoa": float(meta["mean_aoa"]),
                        "amplitude": float(meta["amplitude"]),
                        "oscillation_frequency": float(meta["oscillation_frequency"]),
                        "decay_frequency": float(meta["decay_frequency"]),
                        "CL": float(cl_down[idx]),
                        "experiment_id": int(meta["experiment_id"]),
                        "airfoil_name": airfoil_name,
                        "source_file": source_rel,
                        "original_index": offset + idx,
                    }
                    row.update(cst_feat)
                    rows.append(row)
                    airfoil_counts[airfoil_name] += 1

    if not rows:
        raise RuntimeError("未成功读取任何有效样本，请检查数据路径。")

    df = pd.DataFrame(rows)

    # original_index 已经按时间顺序（上段从0~peak，下段从peak~end）
    # 现在按 (source_file, original_index) 校验并确保顺序一致
    df = df.sort_values(by=["source_file", "original_index"], kind="mergesort").reset_index(drop=True)

    # 汇总文件计数
    meta = {
        "airfoil_counts": airfoil_counts,
        "train_files": file_counts["train"],
        "test_files": file_counts["test"],
    }
    return df, meta


# ============================================================
# 5. 训练/测试划分 与 评估
# ============================================================
FEATURE_COLS_BASE = [
    "alpha", "direction", "Re", "roughness",
    "mean_aoa", "amplitude", "oscillation_frequency", "decay_frequency",
    "N1", "N2",
] + [f"A_u_{i+1}" for i in range(9)] + ["z_u_TE"] + [f"A_l_{i+1}" for i in range(9)] + ["z_l_TE"]

TRACK_COLS = [
    "experiment_id", "airfoil_name", "source_file", "original_index",
]

TARGET_COL = "CL"


def split_train_test(df: pd.DataFrame):
    train_df = df[df["airfoil_name"] != TEST_AIRFOIL].copy()
    test_df = df[df["airfoil_name"] == TEST_AIRFOIL].copy()
    return train_df, test_df


def mean_relative_error(y_true, y_pred):
    """
    平均相对误差 MRE（Mean Relative Error）。
    注意：当真实值 y_true 极接近 0 甚至等于 0 时，MRE/MAPE 会数值爆炸。
    这里用 CL 物理上有意义的最小量级 CL_MRE_FLOOR = 1e-3 作为分母下限，
    从而避免少量近零样本（例如 CL≈0 的失速/过渡点）主导整体均值。
    """
    CL_MRE_FLOOR = 1e-3  # 对升力系数而言，0.001 是合理的相对误差参考底限
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    denom = np.where(np.abs(y_true) < CL_MRE_FLOOR, CL_MRE_FLOOR, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def evaluate(y_true, y_pred, tag=""):
    """计算 MAE / MRE / RMSE / R² 并打印一行。"""
    mae = mean_absolute_error(y_true, y_pred)
    mre = mean_relative_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    print(
        f"  {tag:<12s}  "
        f"MAE={mae:.6f}  "
        f"MRE={mre:.6f}  "
        f"RMSE={rmse:.6f}  "
        f"R²={r2:.6f}  "
        f"(n={n:,})"
    )
    return {"MAE": mae, "MRE": mre, "RMSE": rmse, "R2": r2, "n": n}


def _fmt_row(seg_label, metrics, n_total=None):
    """格式化一行报告。"""
    pct = f"({metrics['n']/n_total*100:.1f}%)" if n_total else ""
    return (
        f"  {seg_label:<12s} | "
        f"{metrics['MAE']:.6f} | "
        f"{metrics['MRE']:.6f} | "
        f"{metrics['RMSE']:.6f} | "
        f"{metrics['R2']:.6f} | "
        f"{metrics['n']:>7,} {pct}"
    )


def generate_report(train_df, test_df, model, feature_cols, target_col,
                    train_airfoils, airfoil_counts, meta_info,
                    csv_path, model_path, report_path):
    """
    生成包含 MAE / MRE / RMSE / R² 的完整评估报告，
    覆盖 训练集 / 测试集 的整体、上升段、下降段。
    同时打印到控制台并保存为 TXT 文件。
    """
    lines = []
    sep = "=" * 102
    sub_sep = "-" * 102

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(sep)
    log("  XGBoost 动态俯仰翼型升力系数 CL 预测 —— 模型评估报告")
    log(sep)
    import datetime
    log(f"  生成时间       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  测试翼型       : {TEST_AIRFOIL}")
    log(f"  训练翼型({len(train_airfoils):>2d}个) : {', '.join(train_airfoils)}")
    log(f"  随机种子       : {RANDOM_STATE}")
    log(f"  输入特征维数   : {len(feature_cols)}")

    # ====== 超参数 ======
    log("")
    log(sub_sep)
    log("  [1] XGBoost 超参数")
    log(sub_sep)
    for k, v in XGB_PARAMS.items():
        log(f"    {k:<22s} = {v}")

    # ====== 数据概况 ======
    log("")
    log(sub_sep)
    log("  [2] 数据集概况")
    log(sub_sep)
    log(f"    训练样本数         : {len(train_df):>,}")
    log(f"    测试样本数         : {len(test_df):>,}")
    log(f"    训练 MAT 文件数    : {len(meta_info['train_files']):>,}")
    log(f"    测试 MAT 文件数    : {len(meta_info['test_files']):>,}")
    log(f"    训练翼型样本明细  :")
    for a in train_airfoils:
        log(f"      - {a:<12s} {airfoil_counts[a]:>,} 样本")
    log(f"    测试翼型样本明细  :")
    log(f"      - {TEST_AIRFOIL:<12s} {airfoil_counts.get(TEST_AIRFOIL, 0):>,} 样本")

    # ====== 模型预测 ======
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float64)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["pred_CL"] = y_pred_train
    test_df["pred_CL"] = y_pred_test

    # ====== 训练集评估 ======
    log("")
    log(sub_sep)
    log("  [3] 训练集评估指标（用于诊断欠拟合/过拟合，不代表泛化能力）")
    log(sub_sep)
    hdr = ("  " + f"{'分段':<12s} | {'MAE':>8s} | {'MRE':>8s} | {'RMSE':>8s} | {'R²':>8s} | {'样本数':>10s}")
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))

    tr_all = evaluate(y_train, y_pred_train, "训练全部")
    log(_fmt_row("训练全部", tr_all))

    tr_up_mask = train_df["direction"] == 1
    tr_dn_mask = train_df["direction"] == -1
    if tr_up_mask.any():
        m_up = evaluate(
            train_df.loc[tr_up_mask, "CL"].values,
            train_df.loc[tr_up_mask, "pred_CL"].values,
            "训练上升",
        )
        log(_fmt_row("训练上升", m_up, tr_all["n"]))
    if tr_dn_mask.any():
        m_dn = evaluate(
            train_df.loc[tr_dn_mask, "CL"].values,
            train_df.loc[tr_dn_mask, "pred_CL"].values,
            "训练下降",
        )
        log(_fmt_row("训练下降", m_dn, tr_all["n"]))

    # ====== 测试集评估（核心） ======
    log("")
    log(sub_sep)
    log(f"  [4] 测试集评估指标（测试翼型: {TEST_AIRFOIL}，未见样本，代表真实泛化能力）")
    log(sub_sep)
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))

    te_all = evaluate(y_test, y_pred_test, "测试全部")
    log(_fmt_row("测试全部", te_all))

    te_up_mask = test_df["direction"] == 1
    te_dn_mask = test_df["direction"] == -1
    te_up_m = te_dn_m = None
    if te_up_mask.any():
        te_up_m = evaluate(
            test_df.loc[te_up_mask, "CL"].values,
            test_df.loc[te_up_mask, "pred_CL"].values,
            "测试上升",
        )
        log(_fmt_row("测试上升", te_up_m, te_all["n"]))
    if te_dn_mask.any():
        te_dn_m = evaluate(
            test_df.loc[te_dn_mask, "CL"].values,
            test_df.loc[te_dn_mask, "pred_CL"].values,
            "测试下降",
        )
        log(_fmt_row("测试下降", te_dn_m, te_all["n"]))

    # ====== 泛化诊断 ======
    log("")
    log(sub_sep)
    log("  [5] 训练 vs 测试 泛化诊断")
    log(sub_sep)
    gap_mae = te_all["MAE"] - tr_all["MAE"]
    gap_rmse = te_all["RMSE"] - tr_all["RMSE"]
    gap_r2  = tr_all["R2"]  - te_all["R2"]
    log(f"    MAE  差距 (测试-训练)  = {gap_mae:+.6f}  ({gap_mae/(tr_all['MAE']+EPS)*100:+.2f}%)")
    log(f"    RMSE 差距 (测试-训练)  = {gap_rmse:+.6f}  ({gap_rmse/(tr_all['RMSE']+EPS)*100:+.2f}%)")
    log(f"    R²   差距 (训练-测试)  = {gap_r2:+.6f}")
    if gap_r2 < 0.05:
        log("    → 判定：未见明显过拟合，泛化性能良好。")
    elif gap_r2 < 0.15:
        log("    → 判定：存在一定程度过拟合，可适当加强正则 (reg_alpha / reg_lambda) 或减少 n_estimators。")
    else:
        log("    → 判定：过拟合较明显，建议降低模型复杂度 (max_depth / n_estimators) 并加强正则。")

    # ====== 输出文件 ======
    log("")
    log(sub_sep)
    log("  [6] 输出文件")
    log(sub_sep)
    log(f"    XGBoost 模型  : {model_path}")
    log(f"    预测 CSV      : {csv_path}")
    log(f"    本评估报告    : {report_path}")
    log(sep)

    # 保存报告文件
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 把预测后的 test_df 返回给调用方以便后续 CSV 保存使用
    return test_df, te_all, te_up_m, te_dn_m


# ============================================================
# 6. 主函数
# ============================================================
def main():
    np.random.seed(RANDOM_STATE)

    # ---- 1. 构建数据集 ----
    print("=" * 70)
    print("步骤 1：扫描目录并构建数据集")
    print("=" * 70)
    df, meta = build_dataset()
    train_df, test_df = split_train_test(df)

    # ---- 2. 打印统计信息 ----
    print("\n" + "=" * 70)
    print("步骤 2：数据集统计")
    print("=" * 70)

    airfoil_counts = meta["airfoil_counts"]
    train_airfoils = sorted([a for a in airfoil_counts.keys() if a != TEST_AIRFOIL])
    test_airfoils = sorted([a for a in airfoil_counts.keys() if a == TEST_AIRFOIL])

    print(f"\n训练翼型 ({len(train_airfoils)} 个):")
    for a in train_airfoils:
        print(f"  - {a}: {airfoil_counts[a]} 样本")
    print(f"\n测试翼型: {TEST_AIRFOIL}")
    if test_airfoils:
        print(f"  - {TEST_AIRFOIL}: {airfoil_counts[TEST_AIRFOIL]} 样本")
    else:
        print(f"  [警告] 未在 Dynamic 目录下找到 {TEST_AIRFOIL} 的任何数据！")

    print(f"\n训练 MAT 文件数: {len(meta['train_files'])}")
    print(f"测试 MAT 文件数: {len(meta['test_files'])}")
    print(f"训练样本数: {len(train_df):,}")
    print(f"测试样本数: {len(test_df):,}")
    print(f"输入特征数量: {len(FEATURE_COLS_BASE)}")

    # 工况统计
    def _stat(df_sub, title):
        if len(df_sub) == 0:
            return
        print(f"\n[{title}] 工况统计 (前20项):")
        for col, label in [
            ("Re", "Re"),
            ("roughness", "roughness"),
            ("mean_aoa", "mean_aoa"),
            ("amplitude", "amplitude"),
        ]:
            vc = df_sub[col].value_counts().sort_index()
            print(f"  {label}:")
            for k, v in list(vc.items())[:20]:
                if col == "Re":
                    k_show = f"{k/1e6:.2f}e6"
                elif col == "roughness":
                    inv = {v: k for k, v in ROUGHNESS_MAP.items()}
                    k_show = inv.get(int(k), str(k))
                else:
                    k_show = str(k)
                print(f"    {k_show:<10s} -> {v:>8,} 样本")

    _stat(train_df, "训练集")
    _stat(test_df, "测试集")

    if len(train_df) == 0:
        raise RuntimeError("训练集为空，请检查 TEST_AIRFOIL 设置和数据目录。")
    if len(test_df) == 0:
        raise RuntimeError(f"测试集为空：Dynamic 下无 {TEST_AIRFOIL} 数据或全部被跳过。")

    # ---- 3. 训练 XGBoost ----
    print("\n" + "=" * 70)
    print("步骤 3：训练 XGBoost 回归模型")
    print("=" * 70)
    X_train = train_df[FEATURE_COLS_BASE].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)

    print(f"X_train shape: {X_train.shape}  (训练样本 × {len(FEATURE_COLS_BASE)} 特征)")
    print(f"X_test  shape: ({len(test_df):,}, {len(FEATURE_COLS_BASE)})  (测试样本 × {len(FEATURE_COLS_BASE)} 特征)")
    print(f"XGBoost 参数: {XGB_PARAMS}")

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    # 保存模型
    model.save_model(MODEL_OUTPUT)
    print(f"模型已保存: {MODEL_OUTPUT}")

    # ---- 4. 生成完整评估报告（MAE / MRE / RMSE / R²，分整体/上升/下降，训练/测试对照） ----
    print("\n" + "=" * 102)
    print(f"步骤 4：生成模型评估报告（测试翼型 = {TEST_AIRFOIL}）")
    print("=" * 102)

    test_df_pred, te_all, te_up_m, te_dn_m = generate_report(
        train_df=train_df,
        test_df=test_df,
        model=model,
        feature_cols=FEATURE_COLS_BASE,
        target_col=TARGET_COL,
        train_airfoils=train_airfoils,
        airfoil_counts=airfoil_counts,
        meta_info=meta,
        csv_path=CSV_OUTPUT,
        model_path=MODEL_OUTPUT,
        report_path=REPORT_OUTPUT,
    )

    # ---- 5. 保存 CSV ----
    print("\n" + "=" * 70)
    print("步骤 5：保存预测 CSV")
    print("=" * 70)

    # 输出列顺序：前三列严格 alpha, true_CL, pred_CL
    output_cols = [
        "alpha", "CL", "pred_CL",
        "direction", "Re", "roughness", "mean_aoa", "amplitude",
        "oscillation_frequency", "decay_frequency",
        "experiment_id", "airfoil_name", "source_file", "original_index",
    ]
    out_df = test_df_pred[output_cols].rename(columns={"CL": "true_CL"})
    # 按 source_file 分组，每个文件内按 original_index 恢复原始时间顺序
    out_df = out_df.sort_values(by=["source_file", "original_index"], kind="mergesort").reset_index(drop=True)

    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    out_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"CSV 已保存: {CSV_OUTPUT}")
    print(f"CSV 行数: {len(out_df):,}")
    print(f"CSV 前3列: {list(out_df.columns[:3])}")
    print(f"涉及 MAT 文件数: {out_df['source_file'].nunique()}")

    # ---- 6. 最终汇总 ----
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"测试翼型       : {TEST_AIRFOIL}")
    print(f"训练翼型数量   : {len(train_airfoils)}")
    print(f"训练样本数     : {len(train_df):,}")
    print(f"测试样本数     : {len(test_df):,}")
    print(f"模型文件       : {MODEL_OUTPUT}")
    print(f"预测 CSV 文件  : {CSV_OUTPUT}")
    print(f"评估报告文件   : {REPORT_OUTPUT}")
    if te_all is not None:
        print("\n测试集核心指标（再强调）:")
        print(f"  整体    —  MAE={te_all['MAE']:.6f}  MRE={te_all['MRE']:.6f}  RMSE={te_all['RMSE']:.6f}  R²={te_all['R2']:.6f}")
        if te_up_m is not None:
            print(f"  上升段  —  MAE={te_up_m['MAE']:.6f}  MRE={te_up_m['MRE']:.6f}  RMSE={te_up_m['RMSE']:.6f}  R²={te_up_m['R2']:.6f}")
        if te_dn_m is not None:
            print(f"  下降段  —  MAE={te_dn_m['MAE']:.6f}  MRE={te_dn_m['MRE']:.6f}  RMSE={te_dn_m['RMSE']:.6f}  R²={te_dn_m['R2']:.6f}")
    print("\n全部完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        raise
