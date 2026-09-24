# -*- coding: utf-8 -*-
"""气动性能图（Cl/Cd vs AoA 迟滞环，不含风洞数据）。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .solver_interface import SimResult


def plot_aero(result: SimResult, out_path=None) -> Path:
    """绘制气动性能图：左 Cl-AoA，右 Cd-AoA 迟滞环。

    out_path: 若给出则保存 PNG，否则返回 None（用于 GUI 内嵌）。
    """
    # 与 GUI 实时图风格一致（子图布局/标题/颜色/线型）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Dynamic Stall - {result.mesh_name}\n"
                 f"V={result.params.V:.2f} m/s, f={result.params.freq:.2f} Hz, "
                 f"alpha={result.params.mean_aoa:.1f}+-{result.params.amp_aoa:.1f} deg",
                 fontsize=13)

    # Cl vs AoA
    ax1.plot(result.aoa, result.cl, ".-", color="#2ca02c", markersize=3, lw=1)
    ax1.set_xlabel("AoA [deg]"); ax1.set_ylabel("Cl")
    ax1.set_title("Cl vs AoA"); ax1.grid(True, alpha=0.3)

    # Cd vs AoA
    ax2.plot(result.aoa, result.cd, ".-", color="#d62728", markersize=3, lw=1)
    ax2.set_xlabel("AoA [deg]"); ax2.set_ylabel("Cd")
    ax2.set_title("Cd vs AoA"); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    return fig
