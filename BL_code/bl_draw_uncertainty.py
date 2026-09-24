#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B-L 模型对齐 LLM 采样口径的不确定性绘图（与 draw_uncertainty.py 一致）+ bias 修正。

输入：cfd_llm_predict.jsonl（LLM 采样的记录，含 input / aoa / trend / true_cl / true_cd）
对每条记录：
  - 用其 input 解析工况（Re/mean/amp/f），构造该 (翼型×工况) 单周期序列
  - 对序列做蒙特卡洛参数扰动仿真（PERT=0.03），取与该记录同攻角/趋势的点
  - 分趋势 bias 修正：CL += bias(α, 趋势)，bias 由 train 残差按 2° 分箱拟合
    （bias_cl_up/down = train 的 E[CFD - B-L]，线性插值，区间外取端值）
  - 归一化周期相位（与 draw_uncertainty.norm_phase 一致）

输出：
  - curve_{wing}_BL_mc_{Cl|Cd}.png   （黑=CFD，蓝=预测均值+95%CI）
  - 95%CI_BL_mc.csv                  （每翼型 Cl/Cd 覆盖率）
  - pred_bl_mc.jsonl                 （逐条：BL MC 样本 + bias 修正后样本）

用法：python bl_draw_uncertainty.py [--pert 0.03] [--nmc 40] [--seed 0]
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(BASE, "..", "cfd_llm_predict.jsonl")
CST = os.path.join(BASE, "..", "datasets", "cst_params.json")
TRAIN = os.path.join(BASE, "..", "datasets", "train.jsonl")
PARAMS = os.path.join(BASE, "bl_params.json")
OUT = BASE

NU = 1.5e-5
SOUND = 340.0
PARAM_KEYS = ['C_Nalpha', 'alpha0', 'alpha1', 'S1', 'S2', 'Cd0', 'C_N1',
              'T_P', 'T_f', 'T_v', 'T_vl', 'dalpha1', 'eta', 'D_f']
BIAS_BIN = 2.0      # bias 分箱宽度（度）
PERT = 0.03
N_MC = 40

_RE = {
    'Re': re.compile(r"雷诺数为：([\d.]+)\*10\^6"),
    'mean': re.compile(r"平均攻角为：([\d.]+)°"),
    'amp': re.compile(r"攻角振幅为：([\d.]+)°"),
    'f': re.compile(r"振荡频率为：([\d.]+)Hz"),
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


def norm_phase(aoa, trend, a_min, a_max):
    span = a_max - a_min
    if span <= 0:
        return 0.5
    if trend == 'up':
        return 0.5 * (aoa - a_min) / span
    return 0.5 + 0.5 * (a_max - aoa) / span


def make_model(params, Re):
    M = (Re * NU) / SOUND
    p = dict(params)
    p['M'] = M
    from bl_model import BLModel
    return BLModel(**p)


def build_seq(mean, amp, f, n_pts=33):
    """单周期序列（与 bl_data.build_sequences 反推相位一致）。"""
    w = 2 * np.pi * f
    wt = np.linspace(0, 2 * np.pi, n_pts)[:-1]
    alpha = mean + amp * np.sin(wt - np.pi / 2)
    t = wt / w
    return wt, alpha, t


def predict_point(seq_cond, alpha_deg, trend, params, pert, nmc, rng):
    """对 (翼型×工况) 序列做 MC 参数扰动，取同攻角/趋势点的 CL/CD 分布。"""
    wt, alpha, t = seq_cond['wt'], seq_cond['alpha'], seq_cond['t']
    from bl_model import BLModel
    pm_base = dict(params)
    # per-wing 参数文件只有 11 个标定键；补齐默认值（dalpha1/eta/D_f 等）
    pm_base.setdefault('dalpha1', 0.04)
    pm_base.setdefault('eta', 0.965)
    pm_base.setdefault('D_f', 8.0)
    CLs, CDs = [], []
    for _ in range(nmc):
        pm = dict(pm_base)
        for k in PARAM_KEYS:
            pm[k] = pm_base[k] + rng.normal(0, pert * abs(pm_base[k]) + 1e-4)
        m = make_model(pm, seq_cond['Re'])
        r = m.simulate(t, np.deg2rad(alpha))
        CLs.append(r['CL']); CDs.append(r['CD'])
    CLs = np.array(CLs); CDs = np.array(CDs)

    n2 = len(alpha) // 2
    if trend == 'up':
        cand = np.arange(0, n2 + 1)
    else:
        cand = np.arange(n2, len(alpha))
    idx = cand[np.argmin(np.abs(alpha[cand] - alpha_deg))]
    return CLs[:, idx], CDs[:, idx]


def build_bias_cl():
    """训练集残差按 (2° 箱, 趋势) 拟合 bias(α, trend) = E[CFD - B-L]。

    返回 dict[trend] -> (bin_centers, bias_values)，线性插值查表。
    """
    from bl_train import _model as _train_model
    from bl_data import load_records, build_sequences, seq_time, seq_alpha_rad
    params = json.load(open(PARAMS, encoding="utf-8"))
    seqs = build_sequences(load_records(TRAIN))
    res = defaultdict(list)
    for s in seqs:
        m = _train_model([params[k] for k in PARAM_KEYS], s['Re'])
        t = seq_time(s)
        alpha = seq_alpha_rad(s)
        r = m.simulate(t, alpha)
        obs = np.array(s['cl_mean'])
        mask = np.isfinite(r['CL']) & np.isfinite(obs)
        a = np.rad2deg(alpha)[mask]
        n2 = len(a) // 2
        for tr, idx in [('up', np.arange(0, n2 + 1)),
                        ('down', np.arange(n2, len(a)))]:
            idx = idx[idx < len(a)]
            for x, e in zip(a[idx], r['CL'][idx] - obs[idx]):
                res[tr].append((x, e))
    out = {}
    for tr in ('up', 'down'):
        xs = np.array([v[0] for v in res[tr]])
        es = np.array([v[1] for v in res[tr]])
        centers, means = [], []
        for b in np.arange(np.floor(xs.min() / BIAS_BIN) * BIAS_BIN,
                           np.ceil(xs.max() / BIAS_BIN) * BIAS_BIN + BIAS_BIN,
                           BIAS_BIN):
            m = (xs >= b) & (xs < b + BIAS_BIN)
            if m.sum() >= 3:
                centers.append(b + BIAS_BIN / 2)
                means.append(es[m].mean())
        out[tr] = (np.array(centers), np.array(means))
    return out


def apply_bias(alpha_deg, trend, bias):
    """bias 查表（线性插值，区间外取端值）。bias 存的是 E[CFD - B-L]，故加回。"""
    if trend not in bias:
        return 0.0
    xs, ys = bias[trend]
    if len(xs) == 0:
        return 0.0
    return float(np.interp(alpha_deg, xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pert", type=float, default=PERT)
    ap.add_argument("--nmc", type=int, default=N_MC)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nobias", action="store_true", help="关闭 bias 修正（对照）")
    ap.add_argument("--params", type=str, default=None,
                    help="参数文件（默认 bl_params.json；可用 bl_params_{wing}.json 做 per-wing）")
    ap.add_argument("--out", type=str, default=None,
                    help="输出目录（默认脚本所在目录）")
    ap.add_argument("--params-dir", type=str, default=None,
                    help="按翼型自动选择参数文件 bl_params_{wing}.json 的目录 "
                         "（实现真正 per-wing；缺省回退 --params 或全局）")
    args = ap.parse_args()
    out_dir = args.out or OUT
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    fp2name = load_fp2name()

    def load_params_for(wing):
        """按翼型加载参数：--params-dir 优先 bl_params_{wing}.json，否则回退 --params/全局。"""
        if args.params_dir:
            f = os.path.join(args.params_dir, f'bl_params_{wing}.json')
            if os.path.exists(f):
                return json.load(open(f, encoding='utf-8'))
            print(f"[WARN] {wing}: 无 bl_params_{wing}.json，回退全局")
        return json.load(open(args.params or PARAMS, encoding="utf-8"))

    if args.params:
        print(f"使用参数文件: {args.params}")
    if args.params_dir:
        print(f"per-wing 参数目录: {args.params_dir}")
    params_global = json.load(open(args.params or PARAMS, encoding="utf-8"))
    bias = None if args.nobias else build_bias_cl()
    if bias:
        print("bias 修正已启用 (分趋势, 2° 箱):")
        for tr in ('up', 'down'):
            xs, ys = bias[tr]
            print(f"  {tr}: {len(xs)} 箱, α∈[{xs[0]:.0f},{xs[-1]:.0f}], "
                  f"bias 范围 [{ys.min():+.3f}, {ys.max():+.3f}]")
    else:
        print("bias 修正关闭 (对照)")

    recs = [json.loads(l) for l in open(PRED, encoding="utf-8") if l.strip()]
    print(f"记录 {len(recs)} 条 (PERT={args.pert}, N_MC={args.nmc})")

    seq_cache = {}
    wings = defaultdict(list)
    for r in recs:
        inp = r['input']
        wing = fp2name.get(extract_fp(inp), 'unknown')
        key = (wing, float(_RE['Re'].search(inp).group(1)) * 1e6,
               float(_RE['mean'].search(inp).group(1)),
               float(_RE['amp'].search(inp).group(1)),
               float(_RE['f'].search(inp).group(1)))
        if key not in seq_cache:
            wt, alpha, t = build_seq(key[2], key[3], key[4])
            seq_cache[key] = dict(Re=key[1], wt=wt, alpha=alpha, t=t)
        r['_wing'] = wing
        r['_key'] = key
        wings[wing].append(r)

    rows = []
    out_recs = []
    for wing, wrecs in wings.items():
        a_min = min(r['aoa'] for r in wrecs)
        a_max = max(r['aoa'] for r in wrecs)
        bins_cl = defaultdict(list)
        bins_cd = defaultdict(list)
        t_pts = []
        per_rec = []
        for r in wrecs:
            seq = seq_cache[r['_key']]
            params = load_params_for(wing)
            cl_s, cd_s = predict_point(seq, r['aoa'], r['trend'],
                                       params, args.pert, args.nmc, rng)
            if bias is not None:
                b = apply_bias(r['aoa'], r['trend'], bias)
                cl_s = cl_s + b
            ph = norm_phase(r['aoa'], r['trend'], a_min, a_max)
            b_ph = round(ph * 20) / 20
            if np.isfinite(cl_s).all():
                bins_cl[b_ph].extend(cl_s.tolist())
            if np.isfinite(cd_s).all():
                bins_cd[b_ph].extend(cd_s.tolist())
            t_pts.append((ph, r.get('true_cl'), r.get('true_cd')))
            per_rec.append(dict(input=r['input'], aoa=r['aoa'], trend=r['trend'],
                                phase=ph, true_cl=r.get('true_cl'),
                                true_cd=r.get('true_cd'),
                                cl_samples=cl_s.tolist(),
                                cd_samples=cd_s.tolist()))
        out_recs.extend(per_rec)

        for qty, bins, t_key in [('Cl', bins_cl, 1), ('Cd', bins_cd, 2)]:
            suffix = '_nobias' if args.nobias else ''
            out = os.path.join(out_dir, f'curve_{wing}_BL_mc{suffix}_{qty}.png')
            fig, ax = plt.subplots(figsize=(10, 6))
            t_ok = [(x[0], x[t_key]) for x in t_pts if x[t_key] is not None]
            if t_ok:
                ax.scatter([p[0] for p in t_ok], [p[1] for p in t_ok],
                           s=6, c='black', alpha=0.5, label='CFD data')
            xs = [p for p in sorted(bins) if len(bins[p]) >= 2]
            if xs:
                ax.plot(xs, [np.mean(bins[p]) for p in xs], 'b-o', lw=2, ms=4,
                        label='Prediction mean')
                ax.fill_between(xs,
                                [np.percentile(bins[p], 2.5) for p in xs],
                                [np.percentile(bins[p], 97.5) for p in xs],
                                color='blue', alpha=0.25,
                                label='Prediction 95% CI')
            ax.axvline(0.5, color='gray', ls=':', lw=1)
            ax.set_xlabel('Normalized Cycle Phase', fontsize=13)
            ax.set_ylabel(qty, fontsize=13)
            ax.set_xticks(np.arange(0, 1.01, 0.1))
            ax.set_title(f'{wing}  Mean AoA=14 deg, Amplitude=10 deg - {qty}', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)

            covered = total = 0
            for x in t_pts:
                b_ph = round(x[0] * 20) / 20
                v = bins.get(b_ph)
                if v is None or len(v) < 2:
                    continue
                lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
                total += 1
                if lo <= x[t_key] <= hi:
                    covered += 1
            pct = covered / total * 100 if total else 0.0
            rows.append((wing, qty, covered, total, pct))
            print(f"  {wing} {qty}: 覆盖率 {pct:.1f}% ({covered}/{total}) -> {os.path.basename(out)}")

    csv_path = os.path.join(out_dir, f'95%CI_BL_mc{"_nobias" if args.nobias else ""}.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('wing,quantity,covered,total,coverage_pct\n')
        for wing, qty, cov, tot, pct in rows:
            f.write(f"{wing},{qty},{cov},{tot},{pct:.1f}%\n")
    print(f"\n覆盖率已保存: {csv_path}")
    if not args.nobias:
        with open(os.path.join(out_dir, 'pred_bl_mc.jsonl'), 'w', encoding='utf-8') as f:
            for r in out_recs:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")
        print(f"逐条样本已保存: {os.path.join(out_dir, 'pred_bl_mc.jsonl')}")


if __name__ == '__main__':
    main()
