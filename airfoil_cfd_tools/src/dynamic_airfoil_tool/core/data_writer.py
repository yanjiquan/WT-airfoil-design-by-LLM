# -*- coding: utf-8 -*-
"""数据导出：纯英文数据清单 + 原始数据文件。"""

import csv
from pathlib import Path

import numpy as np

from .solver_interface import SimResult


def write_datatable(result: SimResult, out_path) -> Path:
    """写出纯英文数据清单（CSV，表头 time/AoA/Cl/Cd）。

    out_path: 目标 .csv 路径
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["mesh", "condition", "n_points"])
        w.writerow([result.mesh_name, _condition_str(result.params), result.n_points])
        w.writerow([])
        w.writerow(["time", "AoA", "Cl", "Cd"])
        for i in range(result.n_points):
            w.writerow([f"{result.time[i]:.6f}", f"{result.aoa[i]:.4f}",
                        f"{result.cl[i]:.6f}", f"{result.cd[i]:.6f}"])
    return out_path


def write_rawdata(result: SimResult, out_dir, base_name: str) -> Path:
    """写出原始数据文件（NPZ，含全部数组）。

    out_dir: 输出目录
    base_name: 基础文件名（不含扩展名）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{base_name}_raw.npz"
    np.savez(path, time=result.time, aoa=result.aoa, cl=result.cl,
             cd=result.cd, mesh_name=result.mesh_name, source=result.source)
    return path


def _condition_str(params) -> str:
    """工况描述（纯英文）。"""
    if params is None:
        return "unknown"
    return (f"V={params.V:.2f}m/s f={params.freq:.2f}Hz "
            f"mean={params.mean_aoa:.1f}deg amp={params.amp_aoa:.1f}deg "
            f"TI={params.TI*100:.1f}% Lt/c={params.Lt_over_c:.1f}")
