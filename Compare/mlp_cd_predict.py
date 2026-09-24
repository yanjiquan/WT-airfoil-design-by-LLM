# -*- coding: utf-8 -*-
"""
PyTorch MLP 回归模型预测动态俯仰翼型阻力系数 CD
===================================================
严格实验约束：
  - NACA4415 100% 最终测试集，不参与任何训练/Scaler拟合/超参
  - 其余全部翼型 100% 作为训练集（不划分验证集）
  - 不插值不外推不平滑不排序；所有工况和攻角范围全部保留
  - 22 维 CST 几何特征从 <翼型名>_CST.txt 自动读取
  - 目标 CD 取自 MAT 数据矩阵第 3 列（索引 2）：列0=alpha, 列1=CL, 列2=CD
  - X 与 y 分别用 StandardScaler（仅训练集 fit），评估时对预测值反标准化
"""

import os
import re
import sys
import glob
import pickle
import random
import warnings
import traceback
import datetime

# --- 若 PyTorch 安装在脚本同级 _pytorch_deps 目录，自动加入 sys.path ---
_LOCAL_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pytorch_deps")
if os.path.isdir(_LOCAL_DEPS) and _LOCAL_DEPS not in sys.path:
    sys.path.insert(0, _LOCAL_DEPS)

import numpy as np
import pandas as pd
import h5py
from scipy import io as sio

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 全局配置
# ============================================================
ROOT_DIR = r"D:\AirfoilDesign\Data"
DYNAMIC_DIR = os.path.join(ROOT_DIR, "Dynamic")

TEST_AIRFOIL = "NACA4415"
RANDOM_STATE = 42
EPS = 1e-12

# 粗糙度编码
ROUGHNESS_MAP = {"C": 0, "G": 1}

# MLP 超参数 baseline（便于修改）
MLP_HIDDEN = [128, 128, 64]
DROPOUT = 0.10
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PRINT_EVERY_EPOCH = 5

# 输出路径
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_OUTPUT = os.path.join(OUTPUT_DIR, f"{TEST_AIRFOIL}_MLP_CD_predictions.csv")
MODEL_OUTPUT = os.path.join(OUTPUT_DIR, f"{TEST_AIRFOIL}_MLP_CD_model.pt")
ARTIFACTS_OUTPUT = os.path.join(OUTPUT_DIR, f"{TEST_AIRFOIL}_MLP_CD_artifacts.pkl")  # X_scaler/y_scaler/feature_names
REPORT_OUTPUT = os.path.join(OUTPUT_DIR, f"{TEST_AIRFOIL}_MLP_CD_evaluation_report.txt")

warnings.filterwarnings("ignore")


# ============================================================
# 0. 固定所有随机种子（Python / NumPy / PyTorch / CUDA）
# ============================================================
def seed_everything(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. CST 文件解析（复用上一版已验证逻辑）
# ============================================================
def parse_cst_file(airfoil_name: str):
    cst_path = os.path.join(ROOT_DIR, f"{airfoil_name}_CST.txt")
    if not os.path.exists(cst_path):
        print(f"  [警告] 翼型 {airfoil_name} 缺少 CST 文件: {cst_path}，跳过该翼型")
        return None
    raw = ""
    try:
        with open(cst_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            with open(cst_path, "r", encoding="gbk") as f:
                raw = f.read().strip()
    except Exception as e:
        print(f"  [警告] 读取 CST 文件失败 {cst_path}: {e}，跳过该翼型")
        return None

    text = (
        raw.replace("，", ",")
           .replace("；", ";")
           .replace("=", " = ")
    )

    def _ff(pat, fname):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"未找到 {fname}")
        return float(m.group(1))

    def _lf(pat, fname, expected=9):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"未找到 {fname} 列表")
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", m.group(1))]
        if len(nums) != expected:
            raise ValueError(f"{fname} 长度应为 {expected}，实际 {len(nums)}")
        return nums

    try:
        N1 = _ff(r"N1\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", "N1")
        N2 = _ff(r"N2\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", "N2")
        A_u = _lf(r"A_u\s*=\s*\[([^\]]*)\]", "A_u", 9)
        z_u_TE = _ff(r"z_u_TE\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", "z_u_TE")
        A_l = _lf(r"A_l\s*=\s*\[([^\]]*)\]", "A_l", 9)
        z_l_TE = _ff(r"z_l_TE\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", "z_l_TE")
    except ValueError as e:
        print(f"  [警告] 解析 CST 内容失败 {cst_path}: {e}，跳过该翼型")
        return None

    feat = {"N1": N1, "N2": N2}
    for i in range(9):
        feat[f"A_u_{i+1}"] = A_u[i]
    feat["z_u_TE"] = z_u_TE
    for i in range(9):
        feat[f"A_l_{i+1}"] = A_l[i]
    feat["z_l_TE"] = z_l_TE
    return feat


# ============================================================
# 2. MAT 数据读取（兼容 scipy 与 h5py v7.3）
# ============================================================
def load_mat_data(mat_path: str):
    """
    数据矩阵列约定：列0 = alpha(攻角)，列1 = CL(升力系数)，列2 = CD(阻力系数)
    返回 (alpha, cd) 或 (None, None)。
    """
    data_matrix = None
    try:
        mat = sio.loadmat(mat_path)
        cands = []
        for k, v in mat.items():
            if k.startswith("__") or not isinstance(v, np.ndarray) or v.ndim != 2:
                continue
            arr = np.asarray(v, dtype=np.float64)
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T
            if arr.shape[1] >= 3 and arr.shape[0] > arr.shape[1]:
                cands.append((arr.shape[0], arr))
        if cands:
            cands.sort(key=lambda x: -x[0])
            data_matrix = cands[0][1]
    except Exception:
        data_matrix = None

    if data_matrix is None:
        try:
            cands = []
            with h5py.File(mat_path, "r") as f:
                def _c(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            arr = np.asarray(obj[()], dtype=np.float64)
                        except Exception:
                            return
                        if arr.ndim != 2:
                            return
                        if arr.shape[0] < arr.shape[1]:
                            arr = arr.T
                        if arr.shape[1] >= 3 and arr.shape[0] > arr.shape[1]:
                            cands.append((arr.shape[0], arr.copy()))
                f.visititems(_c)
            if cands:
                cands.sort(key=lambda x: -x[0])
                data_matrix = cands[0][1]
        except Exception:
            data_matrix = None

    if data_matrix is None:
        print(f"    [警告] 未找到合法数据矩阵: {mat_path}")
        return None, None

    alpha = data_matrix[:, 0].ravel()
    cd = data_matrix[:, 2].ravel()  # 第 3 列：阻力系数 CD
    mask = np.isfinite(alpha) & np.isfinite(cd)
    if not mask.all():
        alpha = alpha[mask]
        cd = cd[mask]
    if len(alpha) < 3:
        print(f"    [警告] 有效样本过少: {mat_path}")
        return None, None
    return alpha, cd


# ============================================================
# 3. 工况 / 文件名解析
# ============================================================
def parse_working_condition(dir_name: str):
    m = re.match(r"^([CG])_RE_(\d+)$", dir_name.strip(), flags=re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1).upper(), int(m.group(2)) / 100.0 * 1e6


def parse_mat_filename(fname: str):
    base = os.path.splitext(fname)[0].strip()
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
# 4. 构建完整 DataFrame（含所有翼型所有样本）
# ============================================================
FEATURE_COLS_BASE = (
    ["alpha", "direction", "Re", "roughness", "mean_aoa", "amplitude",
     "oscillation_frequency", "decay_frequency", "N1", "N2"]
    + [f"A_u_{i+1}" for i in range(9)]
    + ["z_u_TE"]
    + [f"A_l_{i+1}" for i in range(9)]
    + ["z_l_TE"]
)
TARGET_COL = "CD"
TRACK_COLS = ["experiment_id", "airfoil_name", "source_file", "original_index"]


def build_dataset():
    if not os.path.isdir(DYNAMIC_DIR):
        raise FileNotFoundError(f"Dynamic 目录不存在: {DYNAMIC_DIR}")
    airfoil_dirs = []
    for name in sorted(os.listdir(DYNAMIC_DIR)):
        full = os.path.join(DYNAMIC_DIR, name)
        if os.path.isdir(full) and any(
            os.path.isdir(os.path.join(full, x)) and re.match(r"^[CG]_RE_\d+$", x, re.I)
            for x in os.listdir(full)
        ):
            airfoil_dirs.append(name)
        else:
            if os.path.isdir(full):
                print(f"[跳过] 非翼型数据目录或无有效工况: {name}")
    print(f"发现翼型目录 {len(airfoil_dirs)} 个: {airfoil_dirs}")

    rows = []
    airfoil_counts = {}
    file_counts = {"train_dev": set(), "test": set()}

    for airfoil_name in airfoil_dirs:
        cst_feat = parse_cst_file(airfoil_name)
        if cst_feat is None:
            continue
        airfoil_path = os.path.join(DYNAMIC_DIR, airfoil_name)
        airfoil_counts[airfoil_name] = 0

        for wc_name in sorted(os.listdir(airfoil_path)):
            wc_path = os.path.join(airfoil_path, wc_name)
            if not os.path.isdir(wc_path):
                continue
            roughness, Re = parse_working_condition(wc_name)
            if roughness is None:
                continue
            rval = ROUGHNESS_MAP.get(roughness)
            if rval is None:
                continue

            for mat_path in sorted(glob.glob(os.path.join(wc_path, "*.mat"))):
                fname = os.path.basename(mat_path)
                meta = parse_mat_filename(fname)
                if meta is None:
                    continue
                alpha, cd = load_mat_data(mat_path)
                if alpha is None:
                    continue
                source_rel = os.path.join(airfoil_name, wc_name, fname)
                if airfoil_name == TEST_AIRFOIL:
                    file_counts["test"].add(source_rel)
                else:
                    file_counts["train_dev"].add(source_rel)

                peak = int(np.argmax(alpha))
                # 上升段
                for idx in range(peak + 1):
                    r = {
                        "alpha": float(alpha[idx]), "direction": 1,
                        "Re": float(Re), "roughness": rval,
                        "mean_aoa": float(meta["mean_aoa"]),
                        "amplitude": float(meta["amplitude"]),
                        "oscillation_frequency": float(meta["oscillation_frequency"]),
                        "decay_frequency": float(meta["decay_frequency"]),
                        "CD": float(cd[idx]),
                        "experiment_id": int(meta["experiment_id"]),
                        "airfoil_name": airfoil_name,
                        "source_file": source_rel,
                        "original_index": idx,
                    }
                    r.update(cst_feat)
                    rows.append(r)
                    airfoil_counts[airfoil_name] += 1
                # 下降段
                for idx in range(len(alpha) - peak):
                    orig_i = peak + idx
                    r = {
                        "alpha": float(alpha[orig_i]), "direction": -1,
                        "Re": float(Re), "roughness": rval,
                        "mean_aoa": float(meta["mean_aoa"]),
                        "amplitude": float(meta["amplitude"]),
                        "oscillation_frequency": float(meta["oscillation_frequency"]),
                        "decay_frequency": float(meta["decay_frequency"]),
                        "CD": float(cd[orig_i]),
                        "experiment_id": int(meta["experiment_id"]),
                        "airfoil_name": airfoil_name,
                        "source_file": source_rel,
                        "original_index": orig_i,
                    }
                    r.update(cst_feat)
                    rows.append(r)
                    airfoil_counts[airfoil_name] += 1

    if not rows:
        raise RuntimeError("未成功读取任何有效样本。")
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["source_file", "original_index"], kind="mergesort").reset_index(drop=True)
    return df, {"airfoil_counts": airfoil_counts, "file_counts": file_counts}


# ============================================================
# 5. 训练 / 测试 划分（无验证集：全部非测试翼型作训练集）
# ============================================================
def split_train_test(df: pd.DataFrame):
    """
    严格按翼型划分：
      test = TEST_AIRFOIL 的全部数据（完全隔离）
      train = 其余全部翼型的全部数据（不划分验证集）
    """
    test_df = df[df["airfoil_name"] == TEST_AIRFOIL].copy()
    train_df = df[df["airfoil_name"] != TEST_AIRFOIL].copy()
    train_airfoils = sorted(train_df["airfoil_name"].unique().tolist())
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        {"train_airfoils": train_airfoils},
    )


# ============================================================
# 6. PyTorch Dataset / DataLoader
# ============================================================
class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
    ds = NumpyDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=(device.type == "cuda"))


# ============================================================
# 7. MLP 模型
# ============================================================
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=None, dropout: float = DROPOUT, out_dim: int = 1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = MLP_HIDDEN
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# 8. 训练（无验证集，固定轮数训练，风格与 XGBoost baseline 一致）
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        bs = Xb.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


def train_mlp(model, train_loader):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_history = []

    print(f"\n开始训练 MLP：epochs={MAX_EPOCHS}, batch={BATCH_SIZE}, "
          f"lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}")
    print(f"设备：{device}")
    print(f"{'Epoch':>6s}  {'Train Loss':>14s}")

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        loss_history.append(tr_loss)
        if epoch == 1 or epoch % PRINT_EVERY_EPOCH == 0 or epoch == MAX_EPOCHS:
            print(f"{epoch:>6d}  {tr_loss:>14.6f}")

    print(f"训练完成：共 {MAX_EPOCHS} epochs，最终 train loss = {loss_history[-1]:.8f}")
    return MAX_EPOCHS, loss_history[-1], loss_history


# ============================================================
# 9. 逆标准化后的预测 & 评估指标
# ============================================================
def mean_relative_error(y_true, y_pred):
    """MRE：对 CD 采用 1e-3 作为分母下限，避免近零/负值点爆炸。"""
    CD_MRE_FLOOR = 1e-3
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    denom = np.where(np.abs(y_true) < CD_MRE_FLOOR, CD_MRE_FLOOR, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def compute_metrics(y_true, y_pred, tag=""):
    mae = mean_absolute_error(y_true, y_pred)
    mre = mean_relative_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    n = len(y_true)
    if tag:
        print(
            f"  {tag:<12s}  "
            f"MAE={mae:.6f}  MRE={mre:.6f}  RMSE={rmse:.6f}  R²={r2:.6f}  (n={n:,})"
        )
    return {"MAE": mae, "MRE": mre, "RMSE": rmse, "R2": r2, "n": n}


@torch.no_grad()
def predict_raw(model, X, batch_size=4096):
    """在任意 numpy X 上执行模型前向，返回 (n_samples,1) 原始标准化尺度下的预测。"""
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32))
    out = np.empty(X_t.shape[0], dtype=np.float32)
    for i in range(0, X_t.shape[0], batch_size):
        xb = X_t[i:i + batch_size].to(device)
        out[i:i + batch_size] = model(xb).detach().cpu().numpy()
    return out.reshape(-1, 1)


# ============================================================
# 10. 工况统计 & 报告工具
# ============================================================
def print_condition_stats(df: pd.DataFrame, title: str):
    if len(df) == 0:
        return
    print(f"\n[{title}] 工况统计 (前20项):")
    for col, label in [("Re", "Re"), ("roughness", "roughness"),
                       ("mean_aoa", "mean_aoa"), ("amplitude", "amplitude")]:
        vc = df[col].value_counts().sort_index()
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


def _fmt_row(seg_label, metrics, n_total=None):
    pct = f"({metrics['n']/n_total*100:.1f}%)" if n_total else ""
    return (
        f"  {seg_label:<12s} | "
        f"{metrics['MAE']:.6f} | "
        f"{metrics['MRE']:.6f} | "
        f"{metrics['RMSE']:.6f} | "
        f"{metrics['R2']:.6f} | "
        f"{metrics['n']:>7,} {pct}"
    )


def generate_report(artifacts, split_info, meta,
                    train_mets,
                    te_all, te_up, te_dn, total_epochs, final_train_loss):
    lines = []
    sep = "=" * 104
    sub = "-" * 104

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(sep)
    log("  PyTorch MLP 动态俯仰翼型阻力系数 CD 预测 —— 模型评估报告")
    log(sep)
    log(f"  生成时间         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  测试翼型         : {TEST_AIRFOIL}")
    log(f"  训练翼型 ({len(split_info['train_airfoils']):>2d}个) : {', '.join(split_info['train_airfoils'])}")
    log(f"  数据划分方式     : 除测试翼型外全部翼型 100% 作为训练集（无验证集）")
    log(f"  随机种子         : {RANDOM_STATE}")
    log(f"  输入特征维数     : {len(FEATURE_COLS_BASE)}")
    log(f"  运行设备         : {device}")
    log(f"  训练轮数         : {total_epochs} epochs   (最终 train_loss = {final_train_loss:.8f})")

    log("")
    log(sub)
    log("  [1] MLP 架构 & 超参数")
    log(sub)
    log(f"    隐层结构               : {MLP_HIDDEN}")
    log(f"    Dropout                : {DROPOUT}")
    log(f"    Batch Size             : {BATCH_SIZE}")
    log(f"    Learning Rate          : {LEARNING_RATE}")
    log(f"    Weight Decay (AdamW)   : {WEIGHT_DECAY}")
    log(f"    Max Epochs             : {MAX_EPOCHS}")
    log(f"    损失函数               : MSELoss")
    log(f"    优化器                 : AdamW")

    ac = meta["airfoil_counts"]
    log("")
    log(sub)
    log("  [2] 数据集概况")
    log(sub)
    log(f"    训练样本数             : {artifacts['n_train']:>,}")
    log(f"    测试样本数             : {artifacts['n_test']:>,}")
    log(f"    训练 MAT 文件数        : {artifacts['n_train_files']:>,}")
    log(f"    测试 MAT 文件数        : {artifacts['n_test_files']:>,}")
    log(f"    各翼型样本明细:")
    for a, n in sorted(ac.items()):
        tag = " [TEST]" if a == TEST_AIRFOIL else " [TRAIN]"
        log(f"      - {a:<12s} {n:>7,} 样本{tag}")

    log("")
    log(sub)
    log("  [3] 训练集评估（inverse后真实尺度，拟合程度参考，不代表泛化能力）")
    log(sub)
    hdr = ("  " + f"{'分段':<12s} | {'MAE':>8s} | {'MRE':>8s} | {'RMSE':>8s} | {'R²':>8s} | {'样本数':>10s}")
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))
    log(_fmt_row("训练全部", train_mets))

    log("")
    log(sub)
    log(f"  [4] 测试集评估指标 ★（翼型 {TEST_AIRFOIL}，inverse后真实尺度，代表泛化能力）")
    log(sub)
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))
    n_te = te_all["n"]
    log(_fmt_row("测试全部", te_all))
    if te_up is not None:
        log(_fmt_row("测试上升", te_up, n_te))
    if te_dn is not None:
        log(_fmt_row("测试下降", te_dn, n_te))

    log("")
    log(sub)
    log("  [5] 输出文件")
    log(sub)
    log(f"    MLP 模型权重    : {MODEL_OUTPUT}")
    log(f"    Scaler/元数据   : {ARTIFACTS_OUTPUT}  (X_scaler, y_scaler, feature_names)")
    log(f"    预测 CSV        : {CSV_OUTPUT}")
    log(f"    本评估报告      : {REPORT_OUTPUT}")
    log(sep)

    os.makedirs(os.path.dirname(REPORT_OUTPUT), exist_ok=True)
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# 11. 主函数
# ============================================================
def main():
    seed_everything(RANDOM_STATE)

    # ---- 1. 构建数据集 ----
    print("=" * 70)
    print("步骤 1：扫描目录并构建 DataFrame")
    print("=" * 70)
    df, meta = build_dataset()

    # ---- 2. 划分 train / test（无验证集） ----
    print("\n" + "=" * 70)
    print("步骤 2：按翼型划分 train / test（全部非测试翼型作训练集）")
    print("=" * 70)
    train_df, test_df, split_info = split_train_test(df)

    print(f"划分方式: 除 {TEST_AIRFOIL} 外全部翼型 100% 作为训练集（无验证集）")
    print(f"训练翼型 ({len(split_info['train_airfoils'])}): {split_info['train_airfoils']}")
    print(f"测试翼型         : {TEST_AIRFOIL}")
    print(f"训练样本 / 测试样本 : {len(train_df):>,} / {len(test_df):>,}")
    print(f"输入特征数       : {len(FEATURE_COLS_BASE)}")
    print_condition_stats(train_df, "训练集")
    print_condition_stats(test_df, "测试集")

    if len(train_df) == 0:
        raise RuntimeError("训练集为空，检查数据划分设置。")
    if len(test_df) == 0:
        raise RuntimeError(f"测试集为空：Dynamic 目录下无 {TEST_AIRFOIL} 数据或全部被跳过。")

    # ---- 3. 标准化：Scaler 只在训练集 fit ----
    print("\n" + "=" * 70)
    print("步骤 3：StandardScaler（仅训练集 fit；X 与 y 分别标准化）")
    print("=" * 70)
    X_train_raw = train_df[FEATURE_COLS_BASE].values.astype(np.float64)
    y_train_raw = train_df[TARGET_COL].values.astype(np.float64).reshape(-1, 1)
    X_test_raw = test_df[FEATURE_COLS_BASE].values.astype(np.float64)
    y_test_raw = test_df[TARGET_COL].values.astype(np.float64).reshape(-1, 1)

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = X_scaler.fit_transform(X_train_raw)
    y_train_s = y_scaler.fit_transform(y_train_raw)
    X_test_s = X_scaler.transform(X_test_raw)

    print(f"X_train_s shape : {X_train_s.shape}")
    print(f"X_test_s  shape : {X_test_s.shape}")
    print("Scaler 仅在训练集上拟合完成。")

    # ---- 4. DataLoader + MLP 模型 ----
    print("\n" + "=" * 70)
    print("步骤 4：构建 DataLoader & MLP 模型（目标：CD 阻力系数）")
    print("=" * 70)
    train_loader = make_loader(X_train_s, y_train_s.ravel(), batch_size=BATCH_SIZE, shuffle=True)

    model = MLPRegressor(in_dim=len(FEATURE_COLS_BASE), hidden_dims=MLP_HIDDEN, dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"总参数量: {n_params:,}")

    # ---- 5. 训练（固定轮数，无 Early Stopping） ----
    print("\n" + "=" * 70)
    print("步骤 5：训练 MLP（固定轮数，无验证集）")
    print("=" * 70)
    total_epochs, final_train_loss, loss_history = train_mlp(model, train_loader)

    # 保存模型权重
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": len(FEATURE_COLS_BASE),
        "hidden_dims": MLP_HIDDEN,
        "dropout": DROPOUT,
        "total_epochs": total_epochs,
        "final_train_loss": final_train_loss,
    }, MODEL_OUTPUT)
    print(f"MLP 权重已保存: {MODEL_OUTPUT}")

    # ---- 6. 训练集评估（inverse 真实尺度，拟合程度参考） ----
    print("\n" + "=" * 70)
    print("步骤 6：训练集评估（inverse 后真实尺度）")
    print("=" * 70)
    pred_tr_s = predict_raw(model, X_train_s)
    y_train_pred = y_scaler.inverse_transform(pred_tr_s).ravel()

    print("\n[训练集]")
    train_mets = compute_metrics(y_train_raw.ravel(), y_train_pred, "训练全部")

    # ---- 7. 最终 NACA4415 测试 ★ ----
    print("\n" + "=" * 70)
    print(f"步骤 7：最终测试（测试翼型 {TEST_AIRFOIL}）—— 完全未见样本")
    print("=" * 70)
    pred_te_s = predict_raw(model, X_test_s)
    y_test_pred = y_scaler.inverse_transform(pred_te_s).ravel()
    y_test_true = y_test_raw.ravel()

    test_df_out = test_df.copy()
    test_df_out["pred_CD"] = y_test_pred

    print("\n[测试集整体指标]")
    te_all = compute_metrics(y_test_true, y_test_pred, "测试全部")

    up_mask = test_df_out["direction"] == 1
    dn_mask = test_df_out["direction"] == -1
    te_up = te_dn = None
    if up_mask.any():
        print("\n[测试集上升段]")
        te_up = compute_metrics(
            test_df_out.loc[up_mask, "CD"].values,
            test_df_out.loc[up_mask, "pred_CD"].values,
            "测试上升",
        )
    if dn_mask.any():
        print("\n[测试集下降段]")
        te_dn = compute_metrics(
            test_df_out.loc[dn_mask, "CD"].values,
            test_df_out.loc[dn_mask, "pred_CD"].values,
            "测试下降",
        )

    # ---- 8. 保存 CSV ----
    print("\n" + "=" * 70)
    print("步骤 8：保存预测 CSV")
    print("=" * 70)
    output_cols = [
        "alpha", "CD", "pred_CD",
        "direction", "Re", "roughness", "mean_aoa", "amplitude",
        "oscillation_frequency", "decay_frequency",
        "experiment_id", "airfoil_name", "source_file", "original_index",
    ]
    out_df = test_df_out[output_cols].rename(columns={"CD": "true_CD"})
    out_df = out_df.sort_values(by=["source_file", "original_index"], kind="mergesort").reset_index(drop=True)
    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    out_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"CSV 已保存: {CSV_OUTPUT}")
    print(f"CSV 行数: {len(out_df):,}")
    print(f"CSV 前3列: {list(out_df.columns[:3])}")
    print(f"涉及 MAT 文件数: {out_df['source_file'].nunique()}")

    # ---- 9. 保存 Scaler + Feature Names 元数据 ----
    artifacts = {
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "feature_names": list(FEATURE_COLS_BASE),
        "target_name": TARGET_COL,
        "test_airfoil": TEST_AIRFOIL,
        "random_state": RANDOM_STATE,
        "mlp_hidden": MLP_HIDDEN,
        "dropout": DROPOUT,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_files": train_df["source_file"].nunique(),
        "n_test_files": test_df["source_file"].nunique(),
    }
    with open(ARTIFACTS_OUTPUT, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Scaler & 元数据 已保存: {ARTIFACTS_OUTPUT}")

    # ---- 10. 生成完整评估报告 ----
    print("\n" + "=" * 104)
    print("步骤 9：生成模型评估报告")
    print("=" * 104)
    generate_report(
        artifacts=artifacts,
        split_info=split_info,
        meta=meta,
        train_mets=train_mets,
        te_all=te_all, te_up=te_up, te_dn=te_dn,
        total_epochs=total_epochs, final_train_loss=final_train_loss,
    )

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("最终汇总")
    print("=" * 70)
    print(f"测试翼型         : {TEST_AIRFOIL}")
    print(f"训练翼型数量     : {len(split_info['train_airfoils'])}")
    print(f"训练轮数         : {total_epochs} epochs")
    print(f"MLP 模型文件     : {MODEL_OUTPUT}")
    print(f"Scaler/元数据    : {ARTIFACTS_OUTPUT}")
    print(f"预测 CSV 文件    : {CSV_OUTPUT}")
    print(f"评估报告文件     : {REPORT_OUTPUT}")
    if te_all is not None:
        print("\n★ 测试集 NACA4415 最终指标（真实尺度）:")
        print(f"  整体   —  MAE={te_all['MAE']:.6f}  MRE={te_all['MRE']:.6f}  RMSE={te_all['RMSE']:.6f}  R²={te_all['R2']:.6f}  (n={te_all['n']:,})")
        if te_up is not None:
            print(f"  上升段 —  MAE={te_up['MAE']:.6f}  MRE={te_up['MRE']:.6f}  RMSE={te_up['RMSE']:.6f}  R²={te_up['R2']:.6f}  (n={te_up['n']:,})")
        if te_dn is not None:
            print(f"  下降段 —  MAE={te_dn['MAE']:.6f}  MRE={te_dn['MRE']:.6f}  RMSE={te_dn['RMSE']:.6f}  R²={te_dn['R2']:.6f}  (n={te_dn['n']:,})")
    print("\n全部完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
