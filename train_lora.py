#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qwen3-8B LoRA 微调（HIGH / LOW 两个适配器，支持 sequence packing）。

数据划分（按不确定性）：HIGH ← high.jsonl，LOW ← low.jsonl。
LoRA 超参与现有 WEIGHTS 一致：r=8, lora_alpha=16, lora_dropout=0，
target_modules 覆盖 q/k/v/o/gate/up/down 全部 7 个线性模块。
训练使用 Qwen3 thinking 模板（enable_thinking=True），数据集无需补 think。

--pack 启用 sequence packing（默认开启）：
    多个短样本贪心拼接成 target_len 长序列，消除 padding 浪费 → 吞吐提升 2-3x。
    （Qwen3 新版 masking 不支持 block-diagonal attention mask，故采用朴素拼接：
    labels 掩码只算各样本 assistant 段，跨样本 attention 存在但影响轻微。）

用法（NGC pytorch2.10.0-py312-cu128 容器内）：
    pip install peft datasets accelerate matplotlib
    python train_lora.py --adapter all --batch 4 --epochs 3   # packing 默认开
    python train_lora.py --no-pack                            # 关闭 packing

环境变量（可选）：MODEL_PATH / DATA_DIR / OUT_DIR
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainerCallback,
                          TrainingArguments)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/mnt/workspace/.cache/modelscope/hub/models/Qwen--Qwen3-8B/snapshots/master/")
DATA_DIR = os.environ.get("DATA_DIR", "/mnt/workspace/LLM/datasets/CFD")
OUT_DIR = os.environ.get("OUT_DIR", "/mnt/workspace/LLM/WEIGHTS")

ADAPTERS = {"HIGH": "high.jsonl", "LOW": "low.jsonl"}
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


class LossLogCallback(TrainerCallback):
    """训练过程中实时把 loss 追加到文件（不依赖 Trainer 内部日志机制）。"""

    def __init__(self, path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            with open(self.path, "a", encoding="utf-8") as f:
                json.dump({"step": state.global_step,
                           "epoch": round(float(state.epoch), 4),
                           "loss": float(logs["loss"]),
                           "learning_rate": float(logs.get("learning_rate", 0.0))},
                          f, ensure_ascii=False)
                f.write("\n")


def tokenize_sample(tokenizer, rec, max_len):
    """单样本 tokenize：instruction+input → user，output → assistant，labels 掩码 user。"""
    msgs = [
        {"role": "user", "content": rec["input"]},
        {"role": "assistant", "content": rec["output"]},
    ]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
        chat_template_kwargs={"enable_thinking": True})
    marker = "<|im_start|>assistant"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    prefix_ids = tokenizer(text[:idx], add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_len:
        asst = full_ids[len(prefix_ids):]
        if len(asst) >= max_len:
            return None
        keep = max_len - len(asst)
        full_ids = full_ids[:keep] + asst
        return full_ids, [-100] * keep + asst
    return full_ids, [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]


def pack_samples(tokenized, target_len):
    """贪心把多个样本拼进 target_len 长序列（朴素 packing，不生成 block mask）。

    说明：Qwen3 新版 masking_utils 不支持 block-diagonal attention mask（会 expand
    崩溃），故采用朴素拼接。labels 已掩码各样本的 user 部分（-100），loss 只算各
    样本 assistant 段；跨样本 attention 存在但通常影响轻微。
    """
    packed = []
    cur_ids, cur_labels = [], []
    for ids, labels in tokenized:
        if cur_ids and len(cur_ids) + len(ids) > target_len:
            packed.append({"input_ids": cur_ids, "labels": cur_labels})
            cur_ids, cur_labels = [], []
        cur_ids.extend(ids)
        cur_labels.extend(labels)
    if cur_ids:
        packed.append({"input_ids": cur_ids, "labels": cur_labels})
    return packed


def draw_loss_curve(name, log_history, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过 loss 曲线")
        return
    pts = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    if not pts:
        return
    steps = [p[0] for p in pts]
    losses = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, "b-o", ms=3, lw=1.5, label="train loss")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Training Loss — {name}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{name}] loss 曲线已保存: {out_path}")


def train_adapter(name, tokenizer, base_model, epochs, lr, max_len, batch, pack):
    data_path = os.path.join(DATA_DIR, ADAPTERS[name])
    recs = [r for r in load_jsonl(data_path)]
    print(f"\n[{name}] 数据: {data_path}  样本 {len(recs)}")

    tokenized = [t for r in recs if (t := tokenize_sample(tokenizer, r, max_len)) is not None]
    print(f"[{name}] 有效样本 {len(tokenized)}")

    if pack:
        packed = pack_samples(tokenized, max_len)
        ds = Dataset.from_list(packed)
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
        print(f"[{name}] packing: {len(packed)} 个长序列 (原 {len(tokenized)} 样本)")
    else:
        ds = Dataset.from_list([{"input_ids": ids, "labels": lb}
                                for ids, lb in tokenized])
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    model = get_peft_model(base_model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=max(1, 8 // batch),
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        logging_dir=os.path.join(out_dir, "logs"),
        logging_strategy="steps",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    loss_log_path = os.path.join(out_dir, "loss_log.jsonl")
    kwargs = dict(model=model, args=args, train_dataset=ds,
                  data_collator=collator,
                  callbacks=[LossLogCallback(loss_log_path)])
    try:  # 新版 transformers 用 processing_class
        trainer = Trainer(**kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**kwargs, tokenizer=tokenizer)
    trainer.train()
    print(f"[{name}] 实时 loss 日志: {loss_log_path}")

    loss_path = os.path.join(out_dir, "loss_history.jsonl")
    with open(loss_path, "w", encoding="utf-8") as f:
        for entry in trainer.state.log_history:
            if "loss" in entry:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
    print(f"[{name}] loss 历史已保存: {loss_path}")
    draw_loss_curve(name, trainer.state.log_history,
                    os.path.join(out_dir, "loss_curve.png"))

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[{name}] ✅ 已保存到 {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Qwen3-8B LoRA 微调 (HIGH/LOW)")
    ap.add_argument("--adapter", choices=["HIGH", "LOW", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pack", dest="pack", action="store_true", default=True,
                    help="启用 sequence packing（默认开）")
    ap.add_argument("--no-pack", dest="pack", action="store_false")
    args = ap.parse_args()

    print(f"基座模型: {MODEL_PATH}")
    print(f"数据目录: {DATA_DIR}  输出目录: {OUT_DIR}")
    print(f"packing: {'开启' if args.pack else '关闭'}  max_len={args.max_len}  batch={args.batch}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "right"

    # packing 时用默认 attn（sdpa，torch 2.10 已很快）；不 pack 时可用 flash 加速。
    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto",
                       trust_remote_code=True)
    if not args.pack:
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, attn_implementation="flash_attention_2", **load_kwargs)
            print("[OK] 已启用 flash attention")
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
    base_model.config.use_cache = False

    names = ["HIGH", "LOW"] if args.adapter == "all" else [args.adapter]
    for name in names:
        if isinstance(base_model, PeftModel):
            base_model = base_model.get_base_model()
        train_adapter(name, tokenizer, base_model,
                      args.epochs, args.lr, args.max_len, args.batch, args.pack)


if __name__ == "__main__":
    main()
