# -*- coding: utf-8 -*-
"""
匿名化数据集（onlyVar）的误差计算程序。
仅读取 generated_predictions_*.jsonl 主数据，
从匿名化 input（x1~x14）中提取 x7（AOA）和 x8（trend），
输出 CL 的误差与其它指标（MRE、MAE、RMSE、R²、样本量）。

参考思路：eval_metrics_NACA.py（适配匿名化变量格式）
"""
import json
import re
import numpy as np

# sklearn 可选；若未安装则用 numpy 原生实现同等指标
try:
    from sklearn.metrics import r2_score as _r2_score, mean_squared_error as _mse
except Exception:  # pragma: no cover
    _r2_score = None
    _mse = None


def r2_score(y_true, y_pred):
    if _r2_score is not None:
        return _r2_score(y_true, y_pred)
    ss_res = float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    ss_tot = float(np.sum((np.asarray(y_true) - np.mean(np.asarray(y_true))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def mean_squared_error(y_true, y_pred):
    if _mse is not None:
        return _mse(y_true, y_pred)
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

# =============================
# 配置
# =============================
DATA_PATH = r"D:\AirfoilDesign\PythonPram\Result\Qwen3\eval\0831_onkey\generated_predictions_NACA.jsonl"

# AOA 过滤区间（与参考脚本一致：4°~24°）
AOA_MIN = 4.0
AOA_MAX = 24.0


# =============================
# 解析工具
# =============================
def extract_cl_value(text):
    """
    从 predict/label 文本中提取 CL 数值。
    兼容 Qwen3 思考标签格式：<think>...</think>\n\n0.79
    """
    # 去除思考内容（</think> 之后为实际输出）
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    # 提取第一个浮点数
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text.strip())
    if m:
        return float(m.group())
    return None


def parse_record(d):
    """
    解析单条匿名化 JSON 记录，返回 (aoa, trend, pred_val, label_val)。
    trend: 1=上升, -1=下降
    失败返回 None。
    """
    prompt = d["prompt"]
    predict = d.get("predict", "")
    label = d.get("label", "")

    # ----- 从 input JSON 提取 x7（AOA）-----
    m_aoa = re.search(r'"x7":\s*([-+]?\d*\.?\d+)', prompt)
    if m_aoa is None:
        return None
    try:
        aoa = float(m_aoa.group(1))
    except Exception:
        return None

    # ----- 从 input JSON 提取 x8（trend）-----
    m_trend = re.search(r'"x8":\s*([-+]?\d+)', prompt)
    if m_trend is None:
        trend = "未知"
    else:
        val = int(m_trend.group(1))
        if val == 1:
            trend = "上升"
        elif val == -1:
            trend = "下降"
        else:
            trend = "未知"

    # ----- 预测值 / 标签值 -----
    pred_val = extract_cl_value(predict)
    label_val = extract_cl_value(label)
    if pred_val is None or label_val is None:
        return None

    return aoa, trend, pred_val, label_val


# =============================
# 指标计算
# =============================
def compute_metrics(predict, label):
    """
    返回 {MRE(%), MAE, RMSE, R2, N}。
    - MRE：在 |label| > eps 的样本上计算相对误差，然后平均。
    - 相对误差分母过小（<1e-8）的样本不计入 MRE。
    """
    predict = np.asarray(predict, dtype=float)
    label = np.asarray(label, dtype=float)
    n_total = len(label)
    if n_total == 0:
        return {"N": 0, "MRE(%)": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    eps = 1e-8
    mask = np.abs(label) > eps
    if mask.any():
        rel_err = np.abs((predict[mask] - label[mask]) / label[mask]) * 100.0
        mre = float(np.mean(rel_err))
        mre_used = int(mask.sum())
    else:
        mre = np.nan
        mre_used = 0

    mae = float(np.mean(np.abs(predict - label)))
    rmse = float(np.sqrt(mean_squared_error(label, predict)))
    if n_total >= 2 and np.std(label) > 0:
        r2 = float(r2_score(label, predict))
    else:
        r2 = np.nan

    return {
        "N": n_total,
        "MRE_used": mre_used,
        "MRE(%)": mre,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def fmt(m):
    """单行格式化输出指标字典"""
    if m["N"] == 0:
        return "N=0 (无有效样本)"
    line = f"N={m['N']}"
    if not np.isnan(m["MRE(%)"]):
        line += f", MRE={m['MRE(%)']:.2f}%"
        if m["MRE_used"] != m["N"]:
            line += f"(MRE有效样本N={m['MRE_used']})"
    else:
        line += ", MRE=N/A"
    line += f", MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}"
    if not np.isnan(m["R2"]):
        line += f", R²={m['R2']:.4f}"
    else:
        line += ", R²=N/A"
    return line


# =============================
# 主流程
# =============================
def main():
    # 1) 读取 JSONL
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # 2) 解析，按趋势分组（匿名化数据集仅 CL，无 CD）
    groups = {"all": [], "上升": [], "下降": []}
    parse_fail = 0
    aoa_filtered = 0
    for d in records:
        parsed = parse_record(d)
        if parsed is None:
            parse_fail += 1
            continue
        aoa, trend, pv, lv = parsed
        if not (AOA_MIN <= aoa <= AOA_MAX):
            aoa_filtered += 1
            continue
        groups["all"].append((pv, lv))
        if trend in ("上升", "下降"):
            groups[trend].append((pv, lv))

    # 3) 输出报告
    print("=" * 70)
    print(f"数据文件：{DATA_PATH}")
    print(f"总记录数：{len(records)}   AOA区间：[{AOA_MIN}°, {AOA_MAX}°]")
    print(f"解析失败：{parse_fail}   因AOA区间过滤：{aoa_filtered}")
    print("=" * 70)

    g = groups
    if not g["all"]:
        print("\n[CL] 无有效样本")
    else:
        print(f"\n升力系数 CL")
        print("-" * 50)
        for scope_cn, key in (("总体", "all"), ("上升段", "上升"), ("下降段", "下降")):
            data = g[key]
            if not data:
                print(f"  {scope_cn:<6s}: 无有效样本")
                continue
            pred = [x[0] for x in data]
            lbl = [x[1] for x in data]
            m = compute_metrics(pred, lbl)
            print(f"  {scope_cn:<6s}: {fmt(m)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
