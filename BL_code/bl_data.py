#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B-L 训练/验证数据准备：解析 train/test jsonl → (翼型×工况) 攻角时间序列。

每条记录是 (翼型, 工况, 攻角, 趋势) 的瞬时点。按 (翼型, Re, 粗糙度, mean, amp, f)
分组，用正弦运动反推每个点的相位，重建一个俯仰周期的攻角时间序列；
同一 (攻角, 趋势) 的多周期观测取均值。
"""
import json
import re
import numpy as np
from collections import defaultdict

_RE = {
    'Re': re.compile(r"雷诺数为：([\d.]+)\*10\^6"),
    'rough': re.compile(r"粗糙度为：(\S+?)，"),
    'mean': re.compile(r"平均攻角为：([\d.]+)°"),
    'amp': re.compile(r"攻角振幅为：([\d.]+)°"),
    'f': re.compile(r"振荡频率为：([\d.]+)Hz"),
    'aoa': re.compile(r"(?<!均)攻角为：\s*([-+]?\d*\.?\d+)°"),
    'trend': re.compile(r"变化趋势为：(\S+?)。"),
    'au': re.compile(r"A_u = \[(.*?)\]"),
    'al': re.compile(r"A_l = \[(.*?)\]"),
    'cl': re.compile(r"升力系数为：\s*([-+]?\d*\.?\d+)"),
    'cd': re.compile(r"阻力系数为：\s*([-+]?\d*\.?\d+)"),
}


def parse_record(r):
    inp, out = r["input"], r["output"]
    au = _RE['au'].search(inp)
    al = _RE['al'].search(inp)
    cst = (au.group(1) + "|" + al.group(1)) if au and al else "?"
    return dict(
        cst=cst,
        Re=float(_RE['Re'].search(inp).group(1)) * 1e6 if _RE['Re'].search(inp) else None,
        rough=_RE['rough'].search(inp).group(1) if _RE['rough'].search(inp) else None,
        mean=float(_RE['mean'].search(inp).group(1)) if _RE['mean'].search(inp) else None,
        amp=float(_RE['amp'].search(inp).group(1)) if _RE['amp'].search(inp) else None,
        f=float(_RE['f'].search(inp).group(1)) if _RE['f'].search(inp) else None,
        aoa=float(_RE['aoa'].search(inp).group(1)) if _RE['aoa'].search(inp) else None,
        trend=_RE['trend'].search(inp).group(1) if _RE['trend'].search(inp) else None,
        cl=float(_RE['cl'].search(out).group(1)) if _RE['cl'].search(out) else None,
        cd=float(_RE['cd'].search(out).group(1)) if _RE['cd'].search(out) else None,
    )


def load_records(path):
    return [parse_record(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


def build_sequences(records):
    """按 (翼型, 工况) 分组，重建单周期攻角时间序列。

    返回 list[dict]，每项：
      cst/Re/rough/mean/amp/f : 工况标识
      alpha [] (deg), trend [], wt [] (rad 相位), cl_mean [], cd_mean []
      n_records, aoa_min, aoa_max
    """
    groups = defaultdict(list)
    for r in records:
        key = (r['cst'], r['Re'], r['rough'], r['mean'], r['amp'], r['f'])
        groups[key].append(r)

    seqs = []
    for key, recs in groups.items():
        cst, Re, rough, mean, amp, f = key
        w = 2 * np.pi * f
        items = []
        for r in recs:
            if r['aoa'] is None or r['trend'] is None or r['cl'] is None:
                continue
            y = (r['aoa'] - mean) / amp if amp > 0 else 0.0
            y = float(np.clip(y, -1, 1))
            if r['trend'] == '上升':
                wt = np.arcsin(y) + np.pi / 2
            else:
                wt = 3 * np.pi / 2 - np.arcsin(y)
            items.append((wt, r['aoa'], r['trend'], r['cl'], r['cd']))
        if not items:
            continue
        items.sort(key=lambda x: x[0])

        # 聚合多周期：同一 (攻角, 趋势) 取均值
        uni = {}
        for wt, aoa, tr, cl, cd in items:
            k = (round(aoa, 1), tr)
            u = uni.setdefault(k, {'wt': wt, 'cl': [], 'cd': []})
            if cl is not None:
                u['cl'].append(cl)
            if cd is not None:
                u['cd'].append(cd)
        order = sorted(uni.items(), key=lambda kv: kv[1]['wt'])

        alpha = [kv[0][0] for kv in order]
        trend = [kv[0][1] for kv in order]
        wt = [kv[1]['wt'] for kv in order]
        cl_mean = [float(np.mean(kv[1]['cl'])) for kv in order]
        cd_mean = [float(np.mean(kv[1]['cd'])) if kv[1]['cd'] else None for kv in order]
        seqs.append(dict(
            cst=cst, Re=Re, rough=rough, mean=mean, amp=amp, f=f,
            alpha=alpha, trend=trend, wt=wt,
            cl_mean=cl_mean, cd_mean=cd_mean,
            n_records=len(items),
            aoa_min=min(alpha), aoa_max=max(alpha),
        ))
    return seqs


def seq_alpha_rad(seq):
    return np.deg2rad(np.array(seq['alpha']))


def seq_time(seq):
    """物理时间 [s]：wt 序列 / ω。"""
    w = 2 * np.pi * seq['f']
    return np.array(seq['wt']) / w


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "datasets", "train.jsonl")
    recs = load_records(p)
    seqs = build_sequences(recs)
    print(f"记录 {len(recs)} 条 → {len(seqs)} 个 (翼型×工况) 序列")
    for s in seqs[:5]:
        print(f"  Re={s['Re']/1e6:.2f}M {s['rough']} mean={s['mean']}° amp={s['amp']}° "
              f"f={s['f']}Hz | {len(s['alpha'])}点 攻角[{s['aoa_min']},{s['aoa_max']}] "
              f"cl均值范围[{min(s['cl_mean']):.2f},{max(s['cl_mean']):.2f}] 记录{s['n_records']}")
