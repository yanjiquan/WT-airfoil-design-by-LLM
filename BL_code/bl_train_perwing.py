#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-wing B-L 标定：每个 test 翼型用自己的数据单独标定参数。

数据源：low.jsonl + high.jsonl 中该翼型（CST 指纹匹配 cst_params.json）的所有记录
        —— 这些是 train 集外的数据，覆盖该翼型全工况（含 14°/10° 大振幅）。
标定：对每个翼型独立 least_squares（11 维，同 bl_train 的 PARAM_KEYS），
      保存为 bl_params_{wing}.json。
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

from bl_model import BLModel
from bl_data import load_records, build_sequences, seq_alpha_rad, seq_time
from bl_data import parse_record

NU = 1.5e-5
SOUND = 340.0

BASE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(BASE, "..", "datasets")
CST = os.path.join(DATASETS, "cst_params.json")

PARAM_KEYS = ['C_Nalpha', 'alpha0', 'alpha1', 'S1', 'S2', 'Cd0', 'C_N1',
              'T_P', 'T_f', 'T_v', 'T_vl']
X0 = [6.5, 0.003, 0.26, 0.052, 0.040, 0.007, 1.45, 1.7, 3.0, 6.0, 7.0]
BOUNDS = ([3.0, -0.05, 0.10, 0.02, 0.01, 0.003, 0.8, 0.5, 1.0, 2.0, 2.0],
          [8.0,  0.05, 0.40, 0.09, 0.09, 0.020, 2.0, 6.0, 8.0, 12.0, 12.0])

_RE = {
    'au': re.compile(r"A_u = \[(.*?)\]"),
    'al': re.compile(r"A_l = \[(.*?)\]"),
}


def load_fp2name():
    cst = json.load(open(CST, encoding="utf-8"))
    out = {}
    for n, p in cst.items():
        out[",".join(f"{a:.6f}" for a in p["A_u"]) + "|"
            + ",".join(f"{a:.6f}" for a in p["A_l"])] = n
    return out


def extract_fp(inp):
    au = _RE['au'].search(inp)
    al = _RE['al'].search(inp)
    if not (au and al):
        return None
    return (",".join(f"{float(x):.6f}" for x in au.group(1).split(",")) + "|"
            + ",".join(f"{float(x):.6f}" for x in al.group(1).split(",")))


def _model(params, Re):
    p = dict(zip(PARAM_KEYS, params))
    p['M'] = (Re * NU) / SOUND
    return BLModel(**p)


def residual(params, seqs):
    out = []
    for s in seqs:
        m = _model(params, s['Re'])
        t = seq_time(s)
        alpha = seq_alpha_rad(s)
        r = m.simulate(t, alpha)
        obs = np.array(s['cl_mean'])
        mask = np.isfinite(r['CL']) & np.isfinite(obs)
        out.append(r['CL'][mask] - obs[mask])
    return np.concatenate(out) if out else np.zeros(1)


def fit_wing(seqs, n_restarts=3, max_nfev=2000):
    best = None
    rng = np.random.default_rng(0)
    starts = [X0]
    for _ in range(n_restarts - 1):
        starts.append(np.array(BOUNDS[0]) + rng.random(len(PARAM_KEYS)) *
                      (np.array(BOUNDS[1]) - np.array(BOUNDS[0])))
    for s0 in starts:
        res = least_squares(residual, s0, args=(seqs,), bounds=BOUNDS,
                            max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        if best is None or res.cost < best.cost:
            best = res
    return best


def main():
    fp2name = load_fp2name()
    wings = defaultdict(list)
    for fname in ['low.jsonl', 'high.jsonl']:
        path = os.path.join(DATASETS, fname)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            w = fp2name.get(extract_fp(r['input']), 'unknown')
            if w != 'unknown':
                wings[w].append(parse_record(r))
    print(f"low/high 中翼型: {sorted(wings.keys())}")

    for wing, recs in wings.items():
        seqs = build_sequences(recs)
        print(f"\n=== {wing}: {len(recs)} 条 → {len(seqs)} 序列 ===")
        m0 = X0
        r0 = residual(m0, seqs)
        mae0 = float(np.mean(np.abs(r0)))
        best = fit_wing(seqs)
        p = best.x
        r1 = residual(p, seqs)
        mae1 = float(np.mean(np.abs(r1)))
        print(f"  默认基线 MAE={mae0:.4f} → 标定 MAE={mae1:.4f}")
        for k, v in zip(PARAM_KEYS, p):
            print(f"  {k:<10} = {v:.5f}")
        out = dict(zip(PARAM_KEYS, [float(x) for x in p]))
        json.dump(out, open(os.path.join(BASE, f"bl_params_{wing}.json"), "w"),
                  indent=2)
        print(f"  已保存 bl_params_{wing}.json")


if __name__ == '__main__':
    main()
