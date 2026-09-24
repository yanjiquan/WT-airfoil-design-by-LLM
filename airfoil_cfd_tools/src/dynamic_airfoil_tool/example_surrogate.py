# -*- coding: utf-8 -*-
"""示例代理模型：用户自建模型的参考实现。

只需实现 `predict(params) -> SimResult`，输入输出与 Fluent 求解器一致。
本示例用解析模型近似（气动升力线 + 拖曳极曲线），仅作接口演示。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.solver_interface import SimResult


def predict(params):
    """根据输入参数生成 (time, AoA, Cl, Cd)。"""
    # 俯仰运动：alpha(t) = mean + amp*sin(2*pi*f*t - pi/2)
    f = params.freq
    T = 1.0 / f
    dt = 0.05
    cycles = 4
    t = np.arange(0, cycles * T, dt)
    aoa = params.mean_aoa + params.amp_aoa * np.sin(
        2 * np.pi * f * t - np.pi / 2)

    # 解析近似：升力线斜率随攻角，失速后下降
    aoa_rad = np.deg2rad(aoa)
    aoa_stall = 14.0
    cl = 0.11 * (aoa + 0.0)
    cl = np.where(aoa < aoa_stall, cl,
                  cl - 0.5 * (aoa - aoa_stall))  # 失速后下降

    # 拖曳：极曲线 + 攻角阻力
    cd = 0.01 + 0.01 * (aoa / 10.0) ** 2 + 0.05 * np.maximum(0, (aoa - 12) / 10)

    return SimResult(time=t, aoa=aoa, cl=cl, cd=cd, source="surrogate")
