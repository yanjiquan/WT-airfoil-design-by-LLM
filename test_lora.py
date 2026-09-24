#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vLLM 后端 LoRA Cl 预测精度评估（并发预测 + 实时 jsonl + 每翼型曲线）。

路由：攻角 ≥14° → "high" 适配器，<14° → "low"（vLLM 已加载 LoRA）。
并发：ThreadPoolExecutor 并发发请求（默认 48，匹配 vLLM 吞吐）。
输出：predictions.jsonl（逐 batch 实时 flush）+ 精度指标 + curve_14_10_<翼型>.png

用法：
    python test_lora.py --data /mnt/workspace/LLM/datasets/CFD/test.jsonl \
                        --limit 0 --concurrency 48 [--plot]
环境变量：VLLM_BASE_URL（默认 http://127.0.0.1:8000/v1）
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_HIGH, MODEL_LOW = "high", "low"
MAX_TOKENS = 120
TIMEOUT = 120
# 与训练数据划分一致：上升≥17° 或 下降≥15° → high（高不确定区），其余 → low
UP_BOUND, DN_BOUND = 17, 15


def extract_trend(rec):
    return "up" if "趋势为：上升" in rec["input"] else "dn"


def select_model(aoa, trend):
    if trend == "up":
        return MODEL_HIGH if aoa >= UP_BOUND else MODEL_LOW
    return MODEL_HIGH if aoa >= DN_BOUND else MODEL_LOW


def load_jsonl(path, limit):
    recs = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                recs.append(json.loads(ln))
                if limit and len(recs) >= limit:
                    break
    return recs


def extract_aoa(rec):
    return float(re.search(r'(?<!平均)攻角为：(-?[\d.]+)°', rec["input"]).group(1))


def extract_true_cl(rec):
    m = re.search(r'升力系数为：(-?[\d.]+)', rec["output"])
    return float(m.group(1)) if m else None


def extract_true_cd(rec):
    m = re.search(r'阻力系数为：(-?[\d.]+)', rec["output"])
    return float(m.group(1)) if m else None


def parse_cl(text):
    m = re.search(r'升力系数为：(-?[\d.]+)', text)
    return float(m.group(1)) if m else None


def parse_cd(text):
    m = re.search(r'阻力系数为：(-?[\d.]+)', text)
    return float(m.group(1)) if m else None


def chat(model, prompt, temperature=0.0, max_tokens=MAX_TOKENS, retries=3):
    """vLLM chat/completions 请求，返回 content；失败重试后返回 None。"""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                print(f"[WARN] 请求失败: {e}")
                return None


def predict_one(rec):
    aoa = extract_aoa(rec)
    content = chat(select_model(aoa, extract_trend(rec)), rec["input"],
                   temperature=0.0)
    return rec, aoa, parse_cl(content), parse_cd(content)


def sample_predict(rec, n_samples=15, temperature=0.5):
    """对单条记录采样预测多次，返回 (aoa, Cl分布, Cd分布)。"""
    aoa = extract_aoa(rec)
    model = select_model(aoa, extract_trend(rec))
    cls, cds = [], []
    for _ in range(n_samples):
        content = chat(model, rec["input"], temperature=temperature)
        c = parse_cl(content)
        if c is not None:
            cls.append(c)
        d = parse_cd(content)
        if d is not None:
            cds.append(d)
    return aoa, cls, cds


def metrics(preds, trues):
    if len(preds) < 2:
        return None
    preds, trues = np.array(preds), np.array(trues)
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    pa, ta = preds - preds.mean(), trues - trues.mean()
    denom = np.sqrt((pa ** 2).sum() * (ta ** 2).sum())
    corr = float((pa * ta).sum() / denom) if denom > 0 else float("nan")
    mape = float(np.mean(np.abs((preds - trues) / (trues + 1e-12))) * 100)
    return {"n": len(preds), "MAE": mae, "RMSE": rmse, "R2": r2,
            "corr": corr, "MAPE%": mape}


def fmt(m):
    if m is None:
        return "  样本不足"
    return (f"  n={m['n']:>4}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
            f"R²={m['R2']:.4f}  corr={m['corr']:.4f}  MAPE={m['MAPE%']:.1f}%")


def extract_fp(rec):
    inp = rec["input"]
    au = re.search(r'A_u = \[(.*?)\]', inp)
    al = re.search(r'A_l = \[(.*?)\]', inp)
    if not (au and al):
        return None
    return (','.join(f"{float(x):.6f}" for x in au.group(1).split(',')) + '|'
            + ','.join(f"{float(x):.6f}" for x in al.group(1).split(',')))


def plot_single_curve(wing, wrecs, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过曲线绘制")
        return

    # 模型预测：每攻角选代表记录，并发采样预测（Cl + Cd）
    aoas = sorted({round(extract_aoa(r)) for r in wrecs})
    reps = {a: next(r for r in wrecs if round(extract_aoa(r)) == a) for a in aoas}
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(tqdm(ex.map(lambda a: sample_predict(reps[a]), aoas),
                            total=len(aoas), desc=f"{wing} sampling"))
    pred = {"Cl": {}, "Cd": {}}
    for a, cls, cds in zip(aoas, results):
        if len(cls) >= 2:
            pred["Cl"][a] = (np.mean(cls), np.percentile(cls, 2.5),
                             np.percentile(cls, 97.5))
        if len(cds) >= 2:
            pred["Cd"][a] = (np.mean(cds), np.percentile(cds, 2.5),
                             np.percentile(cds, 97.5))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
    for ax, qty, getter in [(axes[0], "Cl", extract_true_cl),
                            (axes[1], "Cd", extract_true_cd)]:
        pairs = [(round(extract_aoa(r)), getter(r)) for r in wrecs]
        pairs = [p for p in pairs if p[1] is not None]
        ax.scatter([p[0] for p in pairs], [p[1] for p in pairs],
                   s=6, c="black", alpha=0.5, label="CFD data")
        xs = [a for a in aoas if a in pred[qty]]
        if xs:
            ax.plot(xs, [pred[qty][a][0] for a in xs], "b-o", lw=2, ms=4,
                    label="Prediction mean")
            ax.fill_between(xs, [pred[qty][a][1] for a in xs],
                            [pred[qty][a][2] for a in xs],
                            color="blue", alpha=0.25, label="Prediction 95% CI")
        ax.set_xlabel("AoA (deg)", fontsize=13)
        ax.set_ylabel(qty, fontsize=13)
        ax.set_title(f"{wing}  Mean AoA=14°, Amplitude=10° — {qty}", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"曲线已保存: {out_path}")


def plot_14_10(recs, fp2name, out_dir):
    sel = [r for r in recs
           if abs(float(re.search(r'平均攻角为：([\d.]+)°', r["input"]).group(1)) - 14) < 1e-6
           and float(re.search(r'攻角振幅为：([\d.]+)°', r["input"]).group(1)) >= 7.5]
    print(f"14°/10° 工况记录: {len(sel)}")
    wings = defaultdict(list)
    for r in sel:
        wings[fp2name.get(extract_fp(r), "unknown")].append(r)
    for wing, wrecs in wings.items():
        plot_single_curve(wing, wrecs,
                          os.path.join(out_dir, f"curve_14_10_{wing}.png"))


def main():
    ap = argparse.ArgumentParser(description="vLLM LoRA Cl 预测精度评估")
    ap.add_argument("--data", default="/mnt/workspace/LLM/datasets/CFD/test.jsonl")
    ap.add_argument("--limit", type=int, default=200, help="限制样本数（0=全部）")
    ap.add_argument("--concurrency", type=int, default=48, help="并发请求数")
    ap.add_argument("--out", default=".", help="输出目录（predictions.jsonl / 曲线图）")
    ap.add_argument("--plot", action="store_true", help="绘制 14°/10° 曲线")
    args = ap.parse_args()

    print(f"vLLM: {BASE_URL}  并发: {args.concurrency}")

    recs = load_jsonl(args.data, args.limit)
    print(f"评估样本: {len(recs)}")
    os.makedirs(args.out, exist_ok=True)

    # ---- 并发预测（tqdm + 实时写 jsonl，Cl + Cd） ----
    all_pc, all_tc, all_pd, all_td, all_m = [], [], [], [], []
    pred_path = os.path.join(args.out, "predictions.jsonl")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(predict_one, r) for r in recs]
        with open(pred_path, "w", encoding="utf-8") as f:
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="Predicting", unit="sample"):
                rec, aoa, pred_cl, pred_cd = fut.result()
                true_cl = extract_true_cl(rec)
                true_cd = extract_true_cd(rec)
                if pred_cl is None or true_cl is None:
                    continue
                model = select_model(aoa, extract_trend(rec))
                all_pc.append(pred_cl)
                all_tc.append(true_cl)
                all_pd.append(pred_cd)
                all_td.append(true_cd)
                all_m.append((model, aoa))
                json.dump({"input": rec["input"], "true_cl": true_cl,
                           "pred_cl": pred_cl, "true_cd": true_cd,
                           "pred_cd": pred_cd, "adapter": model, "aoa": aoa},
                          f, ensure_ascii=False)
                f.write("\n")
                f.flush()
    print(f"预测记录实时保存: {pred_path}  ({len(all_pc)} 条)")

    # ---- 精度指标（Cl 和 Cd 分别） ----
    def report(label, preds, trues):
        print(f"\n{label}")
        print(fmt(metrics(preds, trues)))

    report("=== 总体 Cl ===", all_pc, all_tc)
    report("=== 总体 Cd ===", all_pd, all_td)
    for model in [MODEL_HIGH, MODEL_LOW]:
        idx = [i for i, (m, _) in enumerate(all_m) if m == model]
        if idx:
            report(f"=== 适配器 {model} Cl (n={len(idx)}) ===",
                   [all_pc[i] for i in idx], [all_tc[i] for i in idx])
            report(f"=== 适配器 {model} Cd (n={len(idx)}) ===",
                   [all_pd[i] for i in idx], [all_td[i] for i in idx])
    for mean in [8, 14, 20]:
        idx = [i for i, (_, a) in enumerate(all_m) if int(a) == mean]
        if idx:
            report(f"=== mean={mean}° Cl (n={len(idx)}) ===",
                   [all_pc[i] for i in idx], [all_tc[i] for i in idx])
            report(f"=== mean={mean}° Cd (n={len(idx)}) ===",
                   [all_pd[i] for i in idx], [all_td[i] for i in idx])

    # ---- 曲线（用全部数据） ----
    if args.plot:
        all_recs = load_jsonl(args.data, 0)
        cst_path = os.path.join(os.path.dirname(os.path.abspath(args.data)),
                                "cst_params.json")
        fp2name = {}
        if os.path.exists(cst_path):
            cst = json.load(open(cst_path, encoding="utf-8"))
            for name, p in cst.items():
                fp2name[','.join(f"{a:.6f}" for a in p["A_u"]) + '|'
                        + ','.join(f"{a:.6f}" for a in p["A_l"])] = name
        plot_14_10(all_recs, fp2name, args.out)


if __name__ == "__main__":
    main()
