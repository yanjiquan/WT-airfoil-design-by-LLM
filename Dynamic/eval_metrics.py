# -*- coding: utf-8 -*-
"""
仅读取 generated_predictions_*.jsonl 主数据（不读取 interp 插值数据），
输出升/阻力系数的误差与其它指标（MRE、MAE、RMSE、R²、样本量）。

参考思路：PltGene_95_Scatter_New.py
"""
import json
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
DATA_PATH = r"D:\AirfoilDesign\PythonPram\Result\Qwen3\eval\0826-nohis\generated_predictions_NACA.jsonl"

# AOA 过滤区间（与参考脚本一致：4°~24°）
AOA_MIN = 4.0
AOA_MAX = 24.0


# =============================
# 解析工具
# =============================
def _extract_float(text, start_key, end_key_list):
    """
    从 text 中按 start_key -> 第一个命中的 end_key 截取数值字符串，并转 float。
    """
    part = text.split(start_key, 1)[1]
    val_str = None
    min_idx = len(part) + 1
    for ek in end_key_list:
        idx = part.find(ek)
        if 0 <= idx < min_idx:
            min_idx = idx
            val_str = part[:idx]
    if val_str is None:
        # 兜底：取到行尾
        val_str = part.strip()
    return float(val_str.strip())


def parse_record(d):
    """
    解析单条 JSON 记录，返回 (task, aoa, trend, pred_val, label_val)。
    task ∈ {"CL", "CD", "UNKNOWN"}，失败返回 None。
    """
    prompt = d["prompt"]
    predict = d.get("predict", "")
    label = d.get("label", "")

    # ----- 任务类型（CL / CD） -----
    if "升力系数。" in prompt:
        task = "CL"
        val_key = "升力系数为："
    elif "阻力系数。" in prompt:
        task = "CD"
        val_key = "阻力系数为："
    else:
        # 兜底：从 prompt 末尾的任务问句判断
        if "升力系数" in prompt:
            task = "CL"
            val_key = "升力系数为："
        elif "阻力系数" in prompt:
            task = "CD"
            val_key = "阻力系数为："
        else:
            return None

    # ----- 攻角 -----
    try:
        # 先匹配任务问句里的“攻角为X°”（避免误命中 CST 参数中其它攻角字段）
        aoa = _extract_float(prompt, "攻角为：", ["°"])
    except Exception:
        try:
            aoa = _extract_float(prompt, "攻角为", ["°"])
        except Exception:
            return None

    # ----- 趋势 -----
    if "趋势为：上升" in prompt:
        trend = "上升"
    elif "趋势为：下降" in prompt:
        trend = "下降"
    else:
        trend = "未知"

    # ----- 预测值 / 标签值 -----
    try:
        pred_val = _extract_float(predict, val_key, ["，", "。", "\n"])
        label_val = _extract_float(label, val_key, ["，", "。", "\n"])
    except Exception:
        return None

    # 标签值为 0 的记录避免被 MRE 除 0 影响（这里保留，MRE 时 mask）
    return task, aoa, trend, pred_val, label_val


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
    line = (
        f"N={m['N']}"
    )
    if not np.isnan(m["MRE(%)"]):
        line += f", MRE={m['MRE(%)']:.2f}%"
        if m["MRE_used"] != m["N"]:
            line += f"(MRE有效样本N={m['MRE_used']})"
    else:
        line += ", MRE=N/A"
    line += (
        f", MAE={m['MAE']:.4f}"
        f", RMSE={m['RMSE']:.4f}"
    )
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

    # 2) 解析，按任务分组
    groups = {"CL": {"all": [], "上升": [], "下降": []},
              "CD": {"all": [], "上升": [], "下降": []}}
    parse_fail = 0
    aoa_filtered = 0
    for d in records:
        parsed = parse_record(d)
        if parsed is None:
            parse_fail += 1
            continue
        task, aoa, trend, pv, lv = parsed
        if not (AOA_MIN <= aoa <= AOA_MAX):
            aoa_filtered += 1
            continue
        groups[task]["all"].append((pv, lv))
        if trend in ("上升", "下降"):
            groups[task][trend].append((pv, lv))

    # 3) 输出报告
    print("=" * 70)
    print(f"数据文件：{DATA_PATH}")
    print(f"总记录数：{len(records)}   AOA区间：[{AOA_MIN}°, {AOA_MAX}°]")
    print(f"解析失败：{parse_fail}   因AOA区间过滤：{aoa_filtered}")
    print("=" * 70)

    for task_name in ("CL", "CD"):
        g = groups[task_name]
        if not g["all"]:
            print(f"\n[{task_name}] 无有效样本")
            continue
        task_cn = "升力系数 CL" if task_name == "CL" else "阻力系数 CD"
        print(f"\n{task_cn}")
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
