# -*- coding: utf-8 -*-
"""求解器统一接口。

定义了输入参数与输出数据结构，Fluent 求解器和用户自建代理模型
都实现该接口（输入输出一致），上层 GUI / 数据导出不感知具体求解器。
"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class SimParams:
    """仿真/预测输入参数。"""
    mesh_path: str                # 翼型网格文件路径（必须 c=0.4572）
    chord: float = 0.4572         # 弦长 [m]（18 inch）
    rho: float = 1.225            # 空气密度 [kg/m^3]
    mu: float = 1.7894e-5         # 动力粘度 [Pa*s]
    V: float = 30.0               # 来流速度 [m/s]
    freq: float = 1.0             # 振荡频率 [Hz]
    mean_aoa: float = 10.0        # 平均攻角 [deg]
    amp_aoa: float = 5.0          # 攻角振幅 [deg]
    TI: float = 0.05              # 湍流强度（0~1）
    Lt_over_c: float = 1.0        # 积分长度尺度 / 弦长

    def validate(self) -> None:
        """校验输入。"""
        if abs(self.chord - 0.4572) > 1e-6:
            raise ValueError(
                f"弦长必须为 0.4572 m（18 inch），当前 {self.chord}。"
                "请使用 c=0.4572 的翼型网格。")
        if self.rho <= 0 or self.mu <= 0 or self.V <= 0:
            raise ValueError("密度/粘度/风速必须为正数。")
        if self.freq <= 0:
            raise ValueError("振荡频率必须为正数。")
        if self.amp_aoa <= 0:
            raise ValueError("攻角振幅必须为正数。")
        p = Path(self.mesh_path)
        if not p.exists():
            raise FileNotFoundError(f"网格文件不存在: {self.mesh_path}")


@dataclass
class SimResult:
    """仿真/预测输出。"""
    # 数据表（time, AoA, Cl, Cd）
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    aoa: np.ndarray = field(default_factory=lambda: np.array([]))
    cl: np.ndarray = field(default_factory=lambda: np.array([]))
    cd: np.ndarray = field(default_factory=lambda: np.array([]))
    # 元数据
    mesh_name: str = ""
    params: SimParams = None
    source: str = ""              # "fluent" / "surrogate"

    @property
    def n_points(self) -> int:
        return len(self.time)


class SolverInterface:
    """求解器抽象基类。子类实现 run() 与 stop()。"""

    name: str = "base"

    def run(self, params: SimParams, progress_cb=None,
            stop_event=None, data_cb=None) -> SimResult:
        """运行仿真/预测，返回气动数据。

        params: 输入参数
        progress_cb: 可选进度回调 func(fraction: float)
        stop_event: 可选停止事件（threading.Event）
        data_cb: 可选实时数据回调 func(points)，运行过程中增量推送
                新解析出的 [(time, AoA, Cl, Cd), ...] 列表，供 GUI 实时绘图
        """
        raise NotImplementedError

    def stop(self):
        """立即终止当前运行（如 Ctrl+C：杀掉所有相关进程/线程）。"""
        pass
