#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立展示脚本：计算并绘制给定 case 目录的升阻力系数随攻角变化（Cl/Cd vs AoA 迟滞环）。

用法:
    python plot_forces.py <case_dir>                                   # 自动猜测工况（从 d_table）
    python plot_forces.py <case_dir> --mean 8 --amp 5.2 --freq 0.6 --rho 1.23 --vel 23.07
    python plot_forces.py <case_dir> --wt "s810/C5l75_s810.txt"        # 叠加风洞迟滞环

参数:
    <case_dir>    case 输出目录（如 output/case_S810_1787207918）
    --mean --amp --freq --rho --vel --mu   工况参数（默认从 d_table 猜测，可覆盖）
    --dt         采样时间步（默认 0.05）
    --skip-cycles 跳过前 N 周期
    --wt         风洞数据文件（可选，叠加对比）
    --out        输出 PNG 路径（默认 <case_dir>/forces_<翼型>_<ts>.png）

输出:
    PNG：左 Cl vs AoA、右 Cd vs AoA（红=上行程, 蓝=下行程），攻角用理论值。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "fluent_auto"))
import compare_sim_auto as csa

D_TABLE = Path(r"E:\my data\My Workspace\Simulation\Fluent\dynamic wind tunnel data\d_table.json")


def guess_params(airfoil: str) -> dict | None:
    """从 d_table 猜测该翼型第一个工况的参数；无 d_table 时返回 None。"""
    if not D_TABLE.exists():
        return None
    dtab = json.loads(D_TABLE.read_text(encoding="utf-8"))
    recs = [r for r in dtab if r["af"].lower() == airfoil.lower()]
    if not recs:
        return None
    r = recs[0]
    v = r["Re"] * 1e6 * r["mu"] / (r["rho"] * 0.4572)
    return dict(mean=r["mean"], amp=r["amp"], freq=r["f"],
                rho=r["rho"], mu=r["mu"], vel=v)


def main() -> int:
    ap = argparse.ArgumentParser(description="展示 case 目录的 Cl/Cd vs AoA")
    ap.add_argument("case_dir", help="case 输出目录（如 output/case_S810_xxx）")
    ap.add_argument("--mean", type=float, default=None, help="平均攻角 [deg]")
    ap.add_argument("--amp", type=float, default=None, help="攻角振幅 [deg]")
    ap.add_argument("--freq", type=float, default=None, help="振荡频率 [Hz]")
    ap.add_argument("--rho", type=float, default=None, help="密度 [kg/m3]")
    ap.add_argument("--vel", type=float, default=None, help="来流速度 [m/s]")
    ap.add_argument("--mu", type=float, default=None, help="动力粘度 [Pa*s]")
    ap.add_argument("--dt", type=float, default=0.05, help="采样时间步")
    ap.add_argument("--skip-cycles", type=int, default=0, help="跳过前 N 周期")
    ap.add_argument("--wt", default=None, help="风洞数据文件（叠加迟滞环）")
    ap.add_argument("--out", default=None, help="输出 PNG 路径")
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.is_dir():
        print(f"错误: 目录不存在 {case_dir}")
        return 1

    # 翼型名（case_S810_1787... → S810）
    name = case_dir.name
    parts = name.split("_")
    airfoil = parts[1] if len(parts) > 1 and parts[0] == "case" else parts[0]

    # 工况参数：命令行 > case_params.json（工具生成）> d_table 猜测
    gp = guess_params(airfoil)
    cp = case_dir / "case_params.json"
    cp_params = None
    if cp.exists():
        try:
            cp_params = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            cp_params = None

    def _pick(arg_val, cp_key, gp_key, default):
        if arg_val is not None:
            return arg_val
        if cp_params and cp_params.get(cp_key) is not None:
            return cp_params[cp_key]
        if gp and gp.get(gp_key) is not None:
            return gp[gp_key]
        return default

    mean = _pick(args.mean, "mean_aoa", "mean", 10.0)
    amp = _pick(args.amp, "amp_aoa", "amp", 5.0)
    freq = _pick(args.freq, "freq", "freq", 1.0)
    rho = _pick(args.rho, "rho", "rho", 1.225)
    vel = _pick(args.vel, "V", "vel", 30.0)
    mu = _pick(args.mu, "mu", "mu", 1.7894e-5)

    print(f"翼型: {airfoil} | 工况: mean={mean}°, amp={amp}°, f={freq} Hz, "
          f"rho={rho:.4f}, V={vel:.2f} m/s, mu={mu:.2e}")
    src = "case_params.json" if cp_params else ("d_table" if gp else "默认")
    print(f"（工况参数来源: {src}；如不符请用 --mean/--amp/--freq/--rho/--vel 覆盖）")

    csa.set_case_params(rho=rho, vel=vel, mu=mu, freq=freq, ao0=mean, ao1=amp)
    sim = csa.load_simulation(str(case_dir), dt=args.dt, skip_cycles=args.skip_cycles,
                              cas_arg=None, use_mesh_aoa=False)
    if sim is None:
        print("后处理失败：未提取到仿真数据")
        return 1
    t, aoa, cl, cd = sim["times"], sim["aoas"], sim["cl"], sim["cd"]

    # 上下行程（理论攻角导数）
    dth = np.gradient(aoa, t)
    up = dth > 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, yl, y in [(ax1, "Cl", cl), (ax2, "Cd", cd)]:
        ax.plot(aoa[up], y[up], ".", color="#e41a1c", ms=3, alpha=0.6, label="upstroke")
        ax.plot(aoa[~up], y[~up], ".", color="#377eb8", ms=3, alpha=0.6, label="downstroke")
        ax.set_xlabel("AoA [deg]"); ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        ax.set_title(f"{yl} vs AoA")

    # 可选叠加风洞
    if args.wt:
        try:
            wt_path = Path(args.wt)
            if not wt_path.is_absolute():
                wt_path = D_TABLE.parent / wt_path
            runs = csa.parse_dynamic_file(str(wt_path))
            # 选 mean 最接近的段
            run = min(runs, key=lambda r: abs(r["mean"] - mean))
            a_wt, cl_wt = run["aoa"], run["cl"]
            cdp_wt = run["cdp"]
            ax1.plot(a_wt, cl_wt, "o", color="#ff7f00", ms=3, alpha=0.5, mfc="none",
                     label=f"WT mean≈{run['mean']}")
            ax2.plot(a_wt, cdp_wt, "o", color="#ff7f00", ms=3, alpha=0.5, mfc="none",
                     label="WT Cdp")
            for ax in (ax1, ax2):
                ax.legend(fontsize=9)
        except Exception as e:
            print(f"风洞叠加失败: {e}")

    fig.suptitle(f"{case_dir.name}: Cl/Cd vs AoA (theoretical AoA, {len(aoa)} pts)")
    fig.tight_layout()

    out = Path(args.out) if args.out else \
        case_dir / f"forces_{airfoil}_{int(time.time())}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"\n已保存: {out}")
    print(f"点数: {len(aoa)}")
    print(f"攻角范围: [{aoa.min():.2f}, {aoa.max():.2f}]°")
    print(f"Cl 范围: [{cl.min():.3f}, {cl.max():.3f}]")
    print(f"Cd 范围: [{cd.min():.4f}, {cd.max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
