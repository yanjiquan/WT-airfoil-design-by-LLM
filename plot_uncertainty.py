#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""采样预测画图 + 保存原始采样 + 覆盖率（Cl + Cd 兼容）。

- 对目标工况（三个翼型 14°/10°）每个相位 bin 内所有记录各采样 N 次
- 保存原始采样到 pred_temp_0.5.jsonl（含 cl_samples + cd_samples）
- 预测 95% 区间 [P2.5, P97.5] 对 CFD 真实值的覆盖率（Cl 和 Cd 分别）
- 画图：黑散点 = CFD，蓝线/带 = 预测均值 + 采样 95% CI（Cl/Cd 双子图）

环境变量：VLLM_BASE_URL / PRED_FILE / CST_FILE / OUT_DIR
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
PRED_FILE = os.environ.get("PRED_FILE",
                           "/mnt/workspace/LLM/datasets/CFD/predictions.jsonl")
CST_FILE = os.environ.get("CST_FILE",
                          "/mnt/workspace/LLM/datasets/CFD/cst_params.json")
OUT_DIR = os.environ.get("OUT_DIR", "/mnt/workspace/LLM/WEIGHTS")
UP_BOUND, DN_BOUND = 17, 15
N_SAMPLES = 10
CONCURRENCY = 48

# 只预测这三个 14°/10° 工况（每个翼型的 D4 工况）
TARGET_CONDITIONS = [
    (0.99, '光滑', 1.79, 0.084),   # LS-0421
    (1.00, '光滑', 1.85, 0.088),   # NACA4415
    (0.97, '光滑', 1.79, 0.081),   # S825
]


def chat(model, prompt, temperature=0.5, max_tokens=120, retries=3):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None


def parse_cl(text):
    m = re.search(r'升力系数为：(-?[\d.]+)', text)
    return float(m.group(1)) if m else None


def parse_cd(text):
    m = re.search(r'阻力系数为：(-?[\d.]+)', text)
    return float(m.group(1)) if m else None


def extract_aoa(inp):
    return float(re.search(r'(?<!平均)攻角为：(-?[\d.]+)°', inp).group(1))


def extract_trend(inp):
    return 'up' if '趋势为：上升' in inp else 'dn'


def extract_fp(inp):
    au = re.search(r'A_u = \[(.*?)\]', inp)
    al = re.search(r'A_l = \[(.*?)\]', inp)
    return (','.join(f"{float(x):.6f}" for x in au.group(1).split(',')) + '|'
            + ','.join(f"{float(x):.6f}" for x in al.group(1).split(',')))


def norm_phase(aoa, trend, a_min, a_max):
    span = a_max - a_min
    if span <= 0:
        return 0.5
    if trend == 'up':
        return 0.5 * (aoa - a_min) / span
    return 0.5 + 0.5 * (a_max - aoa) / span


def select_model(aoa, trend):
    if trend == 'up':
        return 'high' if aoa >= UP_BOUND else 'low'
    return 'high' if aoa >= DN_BOUND else 'low'


def match_condition(inp):
    """是否属于目标工况（精确匹配 Re/粗糙度/振荡/衰减 + 14°/10°）。"""
    mean = float(re.search(r'平均攻角为：([\d.]+)°', inp).group(1))
    amp = float(re.search(r'攻角振幅为：([\d.]+)°', inp).group(1))
    if abs(mean - 14) > 1e-6 or amp < 7.5:
        return False
    rey = float(re.search(r'雷诺数为：([\d.]+)\*10\^6', inp).group(1))
    rough = '粗糙' if '粗糙度为：粗糙' in inp else '光滑'
    osc = float(re.search(r'振荡频率为：([\d.]+)Hz', inp).group(1))
    dec = float(re.search(r'衰减频率为：([\d.]+)Hz', inp).group(1))
    return any(abs(rey - c[0]) < 1e-6 and rough == c[1]
               and abs(osc - c[2]) < 1e-6 and abs(dec - c[3]) < 1e-6
               for c in TARGET_CONDITIONS)


def sample_one(rec):
    """对单条记录采样 N_SAMPLES 次，返回 (rec, Cl分布, Cd分布)。"""
    inp = rec['input']
    aoa = extract_aoa(inp)
    trend = extract_trend(inp)
    model = select_model(aoa, trend)
    cls, cds = [], []
    for _ in range(N_SAMPLES):
        content = chat(model, inp, temperature=0.5)
        c = parse_cl(content)
        if c is not None:
            cls.append(c)
        d = parse_cd(content)
        if d is not None:
            cds.append(d)
    return rec, cls, cds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cst = json.load(open(CST_FILE, encoding='utf-8'))
    fp2name = {}
    for name, p in cst.items():
        fp2name[','.join(f"{a:.6f}" for a in p['A_u']) + '|'
                + ','.join(f"{a:.6f}" for a in p['A_l'])] = name

    recs = [json.loads(ln) for ln in open(PRED_FILE, encoding='utf-8') if ln.strip()]
    sel = [r for r in recs if match_condition(r['input'])]
    print(f"目标工况记录: {len(sel)}，采样 {N_SAMPLES} 次/条 → {len(sel)*N_SAMPLES} 请求")

    wings = defaultdict(list)
    for r in sel:
        wings[fp2name.get(extract_fp(r['input']), 'unknown')].append(r)

    all_raw = []                     # (rec, cls, cds)
    all_cov = {'Cl': [], 'Cd': []}   # [(true, in_interval)]
    for wing, wrecs in wings.items():
        a_min = min(extract_aoa(r['input']) for r in wrecs)
        a_max = max(extract_aoa(r['input']) for r in wrecs)
        bins = defaultdict(lambda: {'t_cl': [], 't_cd': [], 'recs': []})
        for r in wrecs:
            aoa = extract_aoa(r['input'])
            ph = norm_phase(aoa, extract_trend(r['input']), a_min, a_max)
            b = round(ph * 20) / 20
            bins[b]['t_cl'].append(r['true_cl'])
            bins[b]['t_cd'].append(r.get('true_cd'))
            bins[b]['recs'].append(r)

        tasks = [(b, r) for b, info in bins.items() for r in info['recs']]
        s_cl = defaultdict(list)
        s_cd = defaultdict(list)
        raw = []
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = {ex.submit(sample_one, r): (b, r) for b, r in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{wing} sampling"):
                b, _ = futures[fut]
                rec, cls, cds = fut.result()
                s_cl[b].extend(cls)
                s_cd[b].extend(cds)
                raw.append((rec, cls, cds))
        all_raw.extend(raw)

        # 覆盖率（Cl / Cd 分别）：bin 预测区间覆盖真实值的比例
        cov = {'Cl': [], 'Cd': []}
        for p in sorted(s_cl):
            cl_lo, cl_hi = np.percentile(s_cl[p], 2.5), np.percentile(s_cl[p], 97.5)
            for t in bins[p]['t_cl']:
                cov['Cl'].append((t, cl_lo <= t <= cl_hi))
            if s_cd[p]:
                cd_lo, cd_hi = np.percentile(s_cd[p], 2.5), np.percentile(s_cd[p], 97.5)
                for t in bins[p]['t_cd']:
                    if t is not None:
                        cov['Cd'].append((t, cd_lo <= t <= cd_hi))
        for qty in ['Cl', 'Cd']:
            all_cov[qty].extend(cov[qty])
            if cov[qty]:
                c = np.mean([1.0 if x else 0.0 for _, x in cov[qty]])
                print(f"[{wing}] {qty} 覆盖率 = {c*100:.1f}%  "
                      f"({sum(x for _, x in cov[qty])}/{len(cov[qty])})")

        # 绘图（Cl / Cd 双子图，横轴归一化周期相位）
        phases = sorted(s_cl)
        if len(phases) < 4:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
        for ax, qty, s_map, t_list in [
                (axes[0], 'Cl', s_cl, 't_cl'), (axes[1], 'Cd', s_cd, 't_cd')]:
            all_ph = [norm_phase(extract_aoa(r['input']),
                                 extract_trend(r['input']), a_min, a_max)
                      for r in wrecs]
            ax.scatter(all_ph, [r['true_cl'] if qty == 'Cl' else r.get('true_cd')
                                for r in wrecs],
                       s=6, c='black', alpha=0.5, label='CFD data')
            xs = [p for p in phases if len(s_map[p]) >= 2]
            if xs:
                pm = [np.mean(s_map[p]) for p in xs]
                pl = [np.percentile(s_map[p], 2.5) for p in xs]
                ph = [np.percentile(s_map[p], 97.5) for p in xs]
                ax.plot(xs, pm, 'b-o', lw=2, ms=4, label='Prediction mean')
                ax.fill_between(xs, pl, ph, color='blue', alpha=0.25,
                                label='Prediction 95% CI')
            ax.axvline(0.5, color='gray', ls=':', lw=1)
            ax.set_xlabel('Normalized Cycle Phase', fontsize=13)
            ax.set_ylabel(qty, fontsize=13)
            ax.set_xticks(np.arange(0, 1.01, 0.1))
            ax.set_title(f'{wing}  14°/10° — {qty}', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'curve_14_10_{wing}_sampled.png')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {out}")

    # 保存原始采样预测（Cl + Cd）
    if all_raw:
        pred_temp = os.path.join(OUT_DIR, "pred_temp_0.5.jsonl")
        with open(pred_temp, "w", encoding="utf-8") as f:
            for rec, cls, cds in all_raw:
                inp = rec['input']
                aoa = extract_aoa(inp)
                trend = extract_trend(inp)
                json.dump({"input": inp, "aoa": aoa, "true_cl": rec['true_cl'],
                           "true_cd": rec.get('true_cd'), "trend": trend,
                           "adapter": select_model(aoa, trend),
                           "cl_samples": cls, "cd_samples": cds},
                          f, ensure_ascii=False)
                f.write("\n")
        print(f"原始采样预测已保存: {pred_temp}  ({len(all_raw)} 条)")

    # 全局覆盖率
    for qty in ['Cl', 'Cd']:
        if all_cov[qty]:
            total = len(all_cov[qty])
            covered = sum(1 for _, c in all_cov[qty] if c)
            print(f"全局 {qty} 95% 区间覆盖率: {covered/total*100:.1f}%  "
                  f"({covered}/{total})")


if __name__ == '__main__':
    main()
