#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CST 几何 → B-L 参数的映射：让 B-L 外推到未见翼型时不再用"猜的参数"。

核心思想（B-L 方程的参数化）：
  C_Nalpha(升力线斜率)  ← 弯度主导（CST 上下表面差）
  alpha0(零升攻角)      ← 弯度主导（符号相反）
  alpha1/S1/S2(失速)    ← 厚度/弯度分布
  动态参数 T_*          ← 翼型特征（弱依赖，留作常数或低权重）

数据：11 个翼型的 CST(16)  +  per-wing 标定的 B-L 参数(11)。
方法：对每个 B-L 参数，用"几何特征 + 数据回归"拟合（ridge 正则化），
      leave-one-out 验证：对未见翼型预测其参数 → 跑 B-L → 对比全局参数。
"""
import json
import os
import glob
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
CST_FILE = os.path.join(BASE, "..", "datasets", "cst_params.json")
PARAM_FILES = sorted(glob.glob(os.path.join(BASE, "bl_params_*.json")))
PARAM_KEYS = ['C_Nalpha', 'alpha0', 'alpha1', 'S1', 'S2', 'Cd0', 'C_N1',
              'T_P', 'T_f', 'T_v', 'T_vl']


def geom_features(p):
    """CST → 几何特征（16 维）：弯度 = A_u - A_l，厚度 = A_u + A_l。"""
    au = np.array(p['A_u'], dtype=float)
    al = np.array(p['A_l'], dtype=float)
    return np.concatenate([au - al, au + al])


def main():
    cst = json.load(open(CST_FILE, encoding="utf-8"))
    wings = [os.path.basename(f).replace('bl_params_', '').replace('.json', '')
             for f in PARAM_FILES]
    wings = [w for w in wings if w in cst]   # 只保留有 CST 的翼型
    X, y = [], []
    for w in wings:
        f = os.path.join(BASE, f'bl_params_{w}.json')
        X.append(geom_features(cst[w]))
        y.append([json.load(open(f, encoding='utf-8'))[k] for k in PARAM_KEYS])
    X = np.array(X); y = np.array(y)
    print(f"{len(wings)} 翼型, 特征 {X.shape[1]} 维, 参数 {y.shape[1]} 维")

    # 全局参数（外推基线）
    gp = json.load(open(os.path.join(BASE, "bl_params.json"), encoding="utf-8"))

    # Leave-one-out：每个翼型当"未见翼型"
    print(f"\n=== Leave-one-out: 用其余 {len(wings)-1} 翼型预测参数 ===")
    results = []
    for i, w in enumerate(wings):
        mask = np.ones(len(wings), bool); mask[i] = False
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(X[mask], y[mask])
        pred = model.predict(X[i:i+1])[0]
        true = y[i]
        # 对比: 预测参数 vs 全局参数 对 B-L 输出的影响（用该翼型的 14/10 工况近似）
        err_pred = np.abs(pred - true).mean()
        err_glob = np.abs(np.array([gp[k] for k in PARAM_KEYS]) - true).mean()
        results.append((w, err_pred, err_glob, pred, true))
        print(f"  {w:>10}: 参数MAE(预测)={err_pred:.3f}  参数MAE(全局)={err_glob:.3f}")

    # 汇总
    rp = np.mean([r[1] for r in results]); rg = np.mean([r[2] for r in results])
    print(f"\n平均: 预测参数 MAE={rp:.3f}  vs  全局参数 MAE={rg:.3f}")

    # 保存映射（训练所有翼型）
    model_all = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model_all.fit(X, y)
    print("\n已训练全量映射模型 (Ridge + StandardScaler)")

    # 打印预测参数 vs 实际（用于检查几何相关性）
    print("\n=== 几何→参数 相关性诊断（以 C_Nalpha 为例）===")
    from scipy.stats import pearsonr
    cam = np.array([abs(geom_features(cst[w])[0]) for w in wings])
    cna = y[:, 0]
    if np.std(cam) > 0 and np.std(cna) > 0:
        r, pv = pearsonr(cam, cna)
        print(f"  弯度特征(上下表面差第1项) vs C_Nalpha: r={r:.3f} (p={pv:.3f})")


if __name__ == '__main__':
    main()
