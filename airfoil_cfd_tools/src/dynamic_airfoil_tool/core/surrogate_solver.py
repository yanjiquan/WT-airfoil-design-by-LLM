# -*- coding: utf-8 -*-
"""代理模型求解器。

用户自建的代理模型只需实现一个函数/类，输入输出与 SolverInterface 一致：

    def predict(params: SimParams) -> SimResult:
        # params: 输入参数（密度/粘度/风速/频率/攻角振幅/平均攻角/湍流/尺度）
        # 返回: SimResult(time, aoa, cl, cd)

或实现一个类（callable）：
    class MySurrogate:
        def __call__(self, params) -> SimResult: ...

SurrogateSolver 通过 load(path, entry="predict") 加载用户模型：
    - path 指向 .py 文件
    - entry 是模块内可调用对象名（函数或类实例）
"""

import importlib.util
from pathlib import Path

import numpy as np

from .solver_interface import SolverInterface, SimParams, SimResult


class SurrogateSolver(SolverInterface):
    name = "surrogate"

    def __init__(self, model_path: str, entry: str = "predict"):
        self.model_path = Path(model_path)
        self.entry = entry
        self._stop = None
        if not self.model_path.exists():
            raise FileNotFoundError(f"代理模型文件不存在: {self.model_path}")
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("_surrogate_mod", self.model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, entry):
            raise AttributeError(f"模块 {self.model_path} 缺少可调用对象 '{entry}'")
        self.model = getattr(mod, entry)

    def stop(self):
        """请求终止当前预测（同 Ctrl+C：置位事件，run 立即抛出）。"""
        if self._stop is not None:
            self._stop.set()

    def run(self, params: SimParams, progress_cb=None,
            stop_event=None, data_cb=None) -> SimResult:
        # 代理模型为一次性预测，无实时数据流；data_cb 忽略
        params.validate()
        self._stop = stop_event
        if progress_cb:
            progress_cb(0.2)
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("已停止")
        # 调用用户模型
        result = self.model(params)
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("已停止")
        if not isinstance(result, SimResult):
            raise TypeError("代理模型必须返回 SimResult")
        if progress_cb:
            progress_cb(1.0)
        # 补充元数据
        result.mesh_name = Path(params.mesh_path).name
        result.params = params
        result.source = "surrogate"
        return result
