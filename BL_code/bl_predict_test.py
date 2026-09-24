#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地 FP8 + CFD 权重：用 predictions.jsonl 的 input（test 预测 prompt）重跑，
与旧预测对比（pred_cl/pred_cd 一致性 + 各自 vs true 的 MAE）。

用法: python bl_predict_test.py [--limit N] [--concurrency C]
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

PRED = r"D:\LLM Optimize data\CFD Predict\predictions.jsonl"
OUT = r"D:\LLM Optimize data\CFD Predict\local_fp8_predictions.jsonl"
BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"


def chat(model, content, max_tokens=200, retries=3):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": True},
    }).encode("utf-8")
    req = urllib.request.Request(BASE_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    last = None
    for _ in range(retries):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=180).read().decode("utf-8"))
            return r["choices"][0]["message"].get("content", "")
        except Exception as e:
            last = e
            time.sleep(1)
    return f"__ERR__{last}"


def parse(resp):
    resp = re.sub(r"<think>.*?</think>", "", resp or "", flags=re.DOTALL)
    cl = re.search(r"升力系数为[:,：]\s*([-+]?\d*\.?\d+)", resp)
    cd = re.search(r"阻力系数为[:,：]\s*([-+]?\d*\.?\d+)", resp)
    return (float(cl.group(1)) if cl else None,
            float(cd.group(1)) if cd else None)


def predict_one(rec):
    aoa = rec.get("aoa", 0)
    model = "low" if aoa < 14 else "high"
    resp = chat(model, rec["input"])
    cl, cd = parse(resp)
    rec["local_pred_cl"] = cl
    rec["local_pred_cd"] = cd
    rec["local_raw"] = resp[:120]
    return rec


def mae(gt, pred):
    g = np.array([x for x in gt if x is not None])
    p = np.array([x for x in pred if x is not None])
    if len(g) != len(p) or len(g) == 0:
        return float("nan")
    return float(np.mean(np.abs(p - g)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="抽样条数，0=全部")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(PRED, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        recs = recs[:args.limit]
    print(f"待预测 {len(recs)} 条 (并发 {args.concurrency})")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(predict_one, r) for r in recs]
        done = 0
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print("ERR", e)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(recs)}")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"已保存 {len(results)} 条 -> {OUT}")

    # ---- 对比 ----
    n_cl = sum(1 for r in results if r.get("local_pred_cl") is not None)
    n_cd = sum(1 for r in results if r.get("local_pred_cd") is not None)
    print(f"\n=== 对比 (本地 FP8+CFD 权重 vs predictions.jsonl) ===")
    print(f"解析成功: cl={n_cl}/{len(results)}, cd={n_cd}/{len(results)}")

    # 新 vs 旧 pred 一致性
    diff_cl = [r["local_pred_cl"] - r["pred_cl"] for r in results
               if r.get("local_pred_cl") is not None and r.get("pred_cl") is not None]
    diff_cd = [r["local_pred_cd"] - r["pred_cd"] for r in results
               if r.get("local_pred_cd") is not None and r.get("pred_cd") is not None]
    if diff_cl:
        d = np.array(diff_cl)
        print(f"pred_cl 新旧差异: mean={d.mean():+.4f} std={d.std():.4f} "
              f"|Δ|max={np.abs(d).max():.3f}")
    if diff_cd:
        d = np.array(diff_cd)
        print(f"pred_cd 新旧差异: mean={d.mean():+.5f} std={d.std():.5f} "
              f"|Δ|max={np.abs(d).max():.4f}")

    # 各自 vs true
    for name, ck, dk in [("旧 predictions", "pred_cl", "pred_cd"),
                         ("本地 FP8", "local_pred_cl", "local_pred_cd")]:
        mcl = mae([r["true_cl"] for r in results],
                  [r.get(ck) for r in results])
        mcd = mae([r["true_cd"] for r in results],
                  [r.get(dk) for r in results])
        print(f"{name}:  MAE_cl={mcl:.4f}  MAE_cd={mcd:.4f}")


if __name__ == '__main__':
    main()
