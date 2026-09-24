#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 pred_temp_0.5.jsonl 画预测不确定性曲线（Cl 和 Cd 各自单独一张），
并计算每张图的 95% 区间覆盖率，保存到 95%CI.csv。

- 真实 CFD：true_cl / true_cd 散点（多周期，按归一化周期相位展开）
- 预测：cl_samples / cd_samples 采样分布的均值 + 95% CI 带
- 覆盖率：每个 CFD 真实值点落在其相位 bin 的预测 [P2.5, P97.5] 区间内的比例
- 字体：Times New Roman
- 横轴：归一化周期相位（上升 0->0.5，下降 0.5->1）

用法：
    python draw_uncertainty.py [--data pred_temp_0.5.jsonl] [--cst cst_params.json] [--out .]
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 字体：Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def load_fp2name(cst_path):
    """从 cst_params.json 构建 CST 指纹(6位小数)->翼型名 映射。"""
    if not os.path.exists(cst_path):
        print(f"[WARN] cst_params.json 不存在: {cst_path}，翼型将标记为 unknown")
        return {}
    with open(cst_path, encoding='utf-8') as f:
        cst = json.load(f)
    out = {}
    for name, p in cst.items():
        out[','.join(f"{a:.6f}" for a in p['A_u']) + '|'
            + ','.join(f"{a:.6f}" for a in p['A_l'])] = name
    return out


def extract_fp(inp):
    """从 prompt 中提取 A_u/A_l 并格式化为与 cst_params.json 一致的指纹。"""
    au = re.search(r'A_u = \[(.*?)\]', inp)
    al = re.search(r'A_l = \[(.*?)\]', inp)
    if not (au and al):
        return None
    au_vals = [float(x) for x in au.group(1).split(',')]
    al_vals = [float(x) for x in al.group(1).split(',')]
    if len(au_vals) < 8 or len(al_vals) < 8:
        return None
    return (','.join(f"{x:.6f}" for x in au_vals) + '|'
            + ','.join(f"{x:.6f}" for x in al_vals))


def norm_phase(aoa, trend, a_min, a_max):
    span = a_max - a_min
    if span <= 0:
        return 0.5
    if trend == 'up':
        return 0.5 * (aoa - a_min) / span
    return 0.5 + 0.5 * (a_max - aoa) / span


def plot_qty(wing, recs, qty, t_key, s_key, a_min, a_max, out_path):
    """单个量一张图 + 95% 区间覆盖率。

    返回 (covered, total)：真实值点落在其相位 bin 的预测 [P2.5,P97.5]
    闭区间内的点数 / 可统计点数（该 bin 采样数 >= 2）。
    """
    bins = defaultdict(list)          # phase bin -> 采样值
    t_pts = []                        # (phase, bin, true)
    for r in recs:
        ph = norm_phase(r['aoa'], r['trend'], a_min, a_max)
        b = round(ph * 20) / 20
        if r.get(s_key):
            bins[b].extend(r[s_key])
        t = r.get(t_key)
        if t is not None:
            t_pts.append((ph, b, t))

    # 覆盖率：每个真实值点是否落在其所在 bin 的预测区间内
    covered = total = 0
    for _, b, t in t_pts:
        if len(bins[b]) >= 2:
            lo, hi = np.percentile(bins[b], 2.5), np.percentile(bins[b], 97.5)
            total += 1
            if lo <= t <= hi:
                covered += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    if t_pts:
        ax.scatter([p[0] for p in t_pts], [p[2] for p in t_pts],
                   s=6, c='black', alpha=0.5, label='CFD data')
    phases = sorted(bins)
    xs = [p for p in phases if len(bins[p]) >= 2]
    if xs:
        ax.plot(xs, [np.mean(bins[p]) for p in xs], 'b-o', lw=2, ms=4,
                label='Prediction mean')
        ax.fill_between(xs, [np.percentile(bins[p], 2.5) for p in xs],
                        [np.percentile(bins[p], 97.5) for p in xs],
                        color='blue', alpha=0.25, label='Prediction 95% CI')
    ax.axvline(0.5, color='gray', ls=':', lw=1)
    ax.set_xlabel('Normalized Cycle Phase', fontsize=13)
    ax.set_ylabel(qty, fontsize=13)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_title(f'{wing}  Mean AoA=14 deg, Amplitude=10 deg - {qty}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    cov_pct = covered / total * 100 if total else 0.0
    print(f"已保存: {out_path}  ({len(recs)} 条, {len(xs)} 相位档)  "
          f"95% 覆盖率: {cov_pct:.1f}% ({covered}/{total})")
    return covered, total


def main():
    ap = argparse.ArgumentParser(description="预测不确定性曲线（Cl/Cd 各一张）+ 95%% 覆盖率")
    ap.add_argument("--data", default="pred_temp_0.5.jsonl",
                    help="采样预测结果文件（plot_uncertainty.py 的产物）")
    ap.add_argument("--cst", default="cst_params.json",
                    help="CST 指纹->翼型名映射（cst_params.json），默认同目录")
    ap.add_argument("--out", default=".", help="输出目录（图 + 95%%CI.csv）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fp2name = load_fp2name(args.cst)
    recs = [json.loads(ln) for ln in open(args.data, encoding='utf-8') if ln.strip()]
    print(f"采样记录: {len(recs)}")

    wings = defaultdict(list)
    unknown = 0
    for r in recs:
        wing = fp2name.get(extract_fp(r['input']), 'unknown')
        if wing == 'unknown':
            unknown += 1
        wings[wing].append(r)
    if unknown:
        print(f"[WARN] {unknown}/{len(recs)} 条未能匹配翼型（标记为 unknown）")

    rows = []   # (wing, qty, covered, total)
    for wing, wrecs in wings.items():
        a_min = min(r['aoa'] for r in wrecs)
        a_max = max(r['aoa'] for r in wrecs)
        for qty, t_key, s_key in [('Cl', 'true_cl', 'cl_samples'),
                                  ('Cd', 'true_cd', 'cd_samples')]:
            out = os.path.join(args.out, f'curve_{wing}_{qty}.png')
            covered, total = plot_qty(wing, wrecs, qty, t_key, s_key,
                                      a_min, a_max, out)
            rows.append((wing, qty, covered, total))

    # 保存覆盖率 CSV
    csv_path = os.path.join(args.out, '95%CI.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['wing', 'quantity', 'covered', 'total', 'coverage_pct'])
        for wing, qty, covered, total in rows:
            pct = f"{covered/total*100:.1f}%" if total else "N/A"
            writer.writerow([wing, qty, covered, total, pct])
    print(f"\n覆盖率已保存: {csv_path}")
    print(f"完成 -> {args.out}")


if __name__ == '__main__':
    main()
