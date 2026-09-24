#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B-L test 验证 + 画图 + 覆盖率（对齐 draw_uncertainty.py）。

- 用标定好的全局 B-L 参数在 test.jsonl（3 未见翼型）上验证外推精度
- 每翼型 Cl / Cd 各一张图：横轴归一化周期相位，Times New Roman，
  黑散点 = CFD 真实，蓝线/带 = B-L 预测均值 + 95% 区间
- 95% 区间 = B-L 参数扰动蒙特卡洛 (P2.5/P97.5)
- 覆盖率 = 真实值落在区间内的比例 → 95%CI.csv
"""
import json
import os
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

from bl_model import BLModel
from bl_data import load_records, build_sequences

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(BASE, "..", "datasets", "train.jsonl")
TEST = os.path.join(BASE, "..", "datasets", "test.jsonl")
PARAMS = os.path.join(BASE, "bl_params.json")
CST = os.path.join(BASE, "..", "datasets", "cst_params.json")
OUT = BASE

NU = 1.5e-5
SOUND = 340.0
PARAM_KEYS = ['C_Nalpha', 'alpha0', 'alpha1', 'S1', 'S2', 'Cd0', 'C_N1',
              'T_P', 'T_f', 'T_v', 'T_vl']
N_MC = 40
PERT = 0.03   # 参数相对扰动幅度（蒙特卡洛采样）
SEED = 0


def norm_phase(aoa, trend, a_min, a_max):
    span = a_max - a_min
    if span <= 0:
        return 0.5
    if trend == 'up':
        return 0.5 * (aoa - a_min) / span
    return 0.5 + 0.5 * (a_max - aoa) / span


def _fp6(cst):
    au, al = cst.split("|")
    return ",".join(f"{float(x):.6f}" for x in au.split(",")) + "|" + \
           ",".join(f"{float(x):.6f}" for x in al.split(","))


def load_fp2name():
    cst = json.load(open(CST, encoding="utf-8"))
    out = {}
    for n, p in cst.items():
        out[",".join(f"{a:.6f}" for a in p["A_u"]) + "|"
            + ",".join(f"{a:.6f}" for a in p["A_l"])] = n
    return out


def make_model(params, Re):
    M = (Re * NU) / SOUND
    p = dict(params)
    p['M'] = M
    return BLModel(**p)


def predict_mc(seq, params, rng):
    """蒙特卡洛参数扰动仿真，返回每点 CL/CD 分布 (N_MC × N)。"""
    n = len(seq['alpha'])
    CL = np.zeros((N_MC, n)); CD = np.zeros((N_MC, n))
    for i in range(N_MC):
        pm = dict(params)
        for k in PARAM_KEYS:
            pm[k] = params[k] + rng.normal(0, PERT * abs(params[k]) + 1e-4)
        m = make_model(pm, seq['Re'])
        w = 2 * np.pi * seq['f']
        t = np.array(seq['wt']) / w
        alpha = np.deg2rad(np.array(seq['alpha']))
        r = m.simulate(t, alpha)
        CL[i] = r['CL']; CD[i] = r['CD']
    return CL, CD


def plot_qty(wing, wseqs, qty, obs_key, params, out_path, rng):
    """单翼型单量：CFD 散点 + B-L 预测均值/95%CI（横轴归一化周期相位）。"""
    bins = defaultdict(list)        # bin -> B-L 预测值
    t_pts = []                      # (phase, true)
    for s in wseqs:
        a_min, a_max = s['mean'] - s['amp'], s['mean'] + s['amp']
        CL, CD = predict_mc(s, params, rng)
        pred = CL if qty == 'Cl' else CD
        obs = np.array(s[obs_key])
        n = len(s['alpha'])
        for j in range(n):
            tr = 'up' if s['trend'][j] == '上升' else 'down'
            ph = norm_phase(s['alpha'][j], tr, a_min, a_max)
            b = round(ph * 20) / 20
            if np.isfinite(pred[:, j]).all():
                bins[b].extend(pred[:, j].tolist())
            if obs[j] is not None:
                t_pts.append((ph, obs[j]))

    fig, ax = plt.subplots(figsize=(10, 6))
    if t_pts:
        ax.scatter([p[0] for p in t_pts], [p[1] for p in t_pts],
                   s=8, c='black', alpha=0.6, label='CFD data')
    xs = sorted(bins)
    if xs:
        pm = [np.mean(bins[p]) for p in xs]
        pl = [np.percentile(bins[p], 2.5) for p in xs]
        ph_ = [np.percentile(bins[p], 97.5) for p in xs]
        ax.plot(xs, pm, 'b-o', lw=2, ms=4, label='B-L mean')
        ax.fill_between(xs, pl, ph_, color='blue', alpha=0.22,
                        label='B-L 95% CI (MC)')
    ax.axvline(0.5, color='gray', ls=':', lw=1)
    ax.set_xlabel('Normalized Cycle Phase', fontsize=13)
    ax.set_ylabel(qty, fontsize=13)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_title(f'{wing} — Beddoes-Leishman {qty} (test)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return bins, t_pts


def coverage(bins, t_pts):
    """每个真实值点落在其 bin 的预测区间内比例。"""
    covered = total = 0
    for ph, t in t_pts:
        b = round(ph * 20) / 20
        v = bins.get(b)
        if v is None or len(v) < 3:
            continue
        lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
        total += 1
        if lo <= t <= hi:
            covered += 1
    return covered, total


def main():
    rng = np.random.default_rng(SEED)
    params = json.load(open(PARAMS, encoding="utf-8"))
    fp2name = load_fp2name()

    # 训练残差参考（全局 MAE 报告用）
    train_seqs = build_sequences(load_records(TRAIN))
    test_seqs = build_sequences(load_records(TEST))
    print(f"train {len(train_seqs)} 序列, test {len(test_seqs)} 序列")

    by_wing = defaultdict(list)
    for s in test_seqs:
        by_wing[fp2name.get(_fp6(s['cst']), 'unknown')].append(s)

    rows = []
    for wing, wseqs in by_wing.items():
        for qty, obs_key in [('Cl', 'cl_mean'), ('Cd', 'cd_mean')]:
            out = os.path.join(OUT, f'curve_{wing}_BL_{qty}.png')
            bins, t_pts = plot_qty(wing, wseqs, qty, obs_key, params, out, rng)
            cov, tot = coverage(bins, t_pts)
            rows.append((wing, qty, cov, tot))
            print(f"  {wing} {qty}: 覆盖率 {cov}/{tot} "
                  f"({cov/tot*100 if tot else 0:.1f}%) -> {os.path.basename(out)}")

    # test 全局 MAE（B-L 单次预测 vs cl_mean）
    mae_cl, mae_cd = 0.0, 0.0
    nc = nd = 0
    for s in test_seqs:
        m = make_model(params, s['Re'])
        t = np.array(s['wt']) / (2 * np.pi * s['f'])
        alpha = np.deg2rad(np.array(s['alpha']))
        r = m.simulate(t, alpha)
        cl_obs = np.array(s['cl_mean'])
        cd_obs = np.array([x for x in s['cd_mean'] if x is not None])
        mae_cl += np.sum(np.abs(r['CL'] - cl_obs)); nc += len(cl_obs)
        if len(cd_obs):
            mae_cd += np.sum(np.abs(r['CD'][:len(cd_obs)] - cd_obs)); nd += len(cd_obs)
    print(f"\ntest 全局 MAE: Cl={mae_cl/nc:.4f}  Cd={mae_cd/nd if nd else float('nan'):.4f}")

    csv_path = os.path.join(OUT, '95%CI_BL.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('wing,quantity,covered,total,coverage_pct\n')
        for wing, qty, cov, tot in rows:
            pct = f"{cov/tot*100:.1f}%" if tot else "N/A"
            f.write(f"{wing},{qty},{cov},{tot},{pct}\n")
    print(f"\n覆盖率已保存: {csv_path}")


if __name__ == '__main__':
    main()
