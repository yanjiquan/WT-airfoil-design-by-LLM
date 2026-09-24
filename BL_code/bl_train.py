#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B-L 模型参数标定：用 train.jsonl 拟合全局 B-L 参数（14 维）。

标定参数（全局共享，14 维）：
  C_Nalpha, alpha0, alpha1, S1, S2, Cd0, C_N1, T_P, T_f, T_v, T_vl,
  dalpha1, eta, D_f   ← 新增 3 个此前固定的经验参数
固定参数：M 由 Re 推算，其余（K0/K1/K2 死参数、indicial 系数、sigmaf/sigmav 表）不标。
优化目标：所有 (翼型×工况) 序列的 B-L 预测 CL vs CFD cl 的残差平方和。
方法：scipy least_squares（多起点）→ 可选 differential_evolution。
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import least_squares

from bl_model import BLModel
from bl_data import load_records, build_sequences, seq_alpha_rad, seq_time

NU = 1.5e-5   # 空气运动粘度 [m^2/s]（Re=V·c/ν，c=1）
SOUND = 340.0  # 音速 [m/s]

PARAM_KEYS = ['C_Nalpha', 'alpha0', 'alpha1', 'S1', 'S2', 'Cd0', 'C_N1',
              'T_P', 'T_f', 'T_v', 'T_vl', 'dalpha1', 'eta', 'D_f']
# 物理合理性 bounds：新增 dalpha1∈[0,0.15]、eta∈[0.9,1.0]、D_f∈[3,15]
X0 = [6.5, 0.003, 0.26, 0.052, 0.040, 0.007, 1.45, 1.7, 3.0, 6.0, 7.0,
      0.04, 0.965, 8.0]
BOUNDS = ([3.0, -0.05, 0.10, 0.02, 0.01, 0.003, 0.8, 0.5, 1.0, 2.0, 2.0,
           0.0, 0.90, 3.0],
          [8.0, 0.05, 0.40, 0.09, 0.09, 0.020, 2.0, 6.0, 8.0, 12.0, 12.0,
           0.15, 1.00, 15.0])


def _model(params, Re):
    M = (Re * NU) / SOUND  # c=1
    p = dict(zip(PARAM_KEYS, params))
    p['M'] = M
    return BLModel(**p)


def residual(params, seqs):
    """所有序列所有点的 (pred - obs)，1D 数组。"""
    out = []
    for s in seqs:
        m = _model(params, s['Re'])
        t = seq_time(s)
        alpha = seq_alpha_rad(s)
        r = m.simulate(t, alpha)
        pred = r['CL']
        obs = np.array(s['cl_mean'])
        mask = np.isfinite(pred) & np.isfinite(obs)
        out.append(pred[mask] - obs[mask])
    return np.concatenate(out) if out else np.zeros(1)


def fit(seqs, x0=None, n_restarts=3, max_nfev=2000):
    best = None
    rng = np.random.default_rng(0)
    starts = [X0 if x0 is None else x0]
    for _ in range(n_restarts - 1):
        starts.append(np.array(BOUNDS[0]) + rng.random(len(PARAM_KEYS)) *
                      (np.array(BOUNDS[1]) - np.array(BOUNDS[0])))
    for s0 in starts:
        res = least_squares(residual, s0, args=(seqs,), bounds=BOUNDS,
                            max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        if best is None or res.cost < best.cost:
            best = res
    return best


def mae_of(params, seqs):
    r = residual(params, seqs)
    return float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r**2)))


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "datasets", "train.jsonl")
    if len(sys.argv) > 1:
        path = sys.argv[1]
    print(f"加载: {path}")
    recs = load_records(path)
    seqs = build_sequences(recs)
    print(f"{len(recs)} 条记录 → {len(seqs)} 个序列")

    # 默认参数基线
    m0 = X0
    mae0, rmse0 = mae_of(m0, seqs)
    print(f"\n默认参数(NACA0012)基线: MAE={mae0:.4f} RMSE={rmse0:.4f}")

    print("标定中 (least_squares 多起点, 14 维)...")
    best = fit(seqs, n_restarts=3, max_nfev=2000)
    p = best.x
    mae, rmse = mae_of(p, seqs)
    print(f"\n标定完成: MAE={mae:.4f} RMSE={rmse:.4f}")
    print("参数:")
    for k, v in zip(PARAM_KEYS, p):
        print(f"  {k:<10} = {v:.5f}")

    out = dict(zip(PARAM_KEYS, [float(x) for x in p]))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "bl_params.json"), "w"),
              indent=2)
    print("\n参数已保存: bl_params.json")
