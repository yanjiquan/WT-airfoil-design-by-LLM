# -*- coding: utf-8 -*-
"""主界面：参数/网格输入 + 求解器选择 + 实时气动性能图（30Hz）。

无总进度条 / 无风洞数据 / 无对比图查看等模块。
运行由后台线程执行，GUI 以 30Hz 刷新实时曲线。
"""

import importlib.util
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.font_manager as _fm
for _f in ["Microsoft YaHei", "SimHei"]:
    try:
        matplotlib.rcParams["font.family"] = _f
        if _fm.findfont(_fm.FontProperties(family=_f), fallback_to_default=False):
            break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.solver_interface import SimParams
from core.fluent_solver import FluentSolver
from core.surrogate_solver import SurrogateSolver
from core.data_writer import write_datatable, write_rawdata
from core.plotter import plot_aero

# 工具 config.py（按文件路径加载，避免与 auto_dynamic 的 config 模块重名冲突）
_cfg_spec = importlib.util.spec_from_file_location(
    "dyn_airfoil_config",
    Path(__file__).resolve().parent.parent / "config.py")
_tool_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_tool_cfg)

REFRESH_MS = 33          # ~30Hz


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dynamic Airfoil Tool")
        self.geometry("1180x760")

        self._solver = None
        self._thread = None
        self._stop = threading.Event()
        self._result = None
        self._live_data = dict(time=[], aoa=[], cl=[], cd=[])
        self._live_a = None
        self._live_cd = None
        self._drawn_n = 0          # 已绘制实时点数（避免无新数据时重复重绘）
        self._drawn_result = None  # 已绘制的结果对象

        self._build_ui()
        # 默认 fluent.exe 路径（来自 config.py，无需手动 Browse）
        self._solver_path = None
        if _tool_cfg.FLUENT_EXE and Path(_tool_cfg.FLUENT_EXE).exists():
            self._solver_path = _tool_cfg.FLUENT_EXE
            self.lbl_solver_path.config(text=_tool_cfg.FLUENT_EXE)
        self.after(REFRESH_MS, self._refresh)

    # ------------------------------------------------------------------
    def _build_ui(self):
        # 顶部：输入区
        top = ttk.LabelFrame(self, text="Input")
        top.pack(fill=tk.X, padx=8, pady=4)

        # 求解器选择
        row = ttk.Frame(top); row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text="Solver:").pack(side=tk.LEFT)
        self.solver_var = tk.StringVar(value="fluent")
        self.solver_combo = ttk.Combobox(row, textvariable=self.solver_var,
                                         values=["fluent", "surrogate"],
                                         state="readonly", width=14)
        self.solver_combo.pack(side=tk.LEFT, padx=4)
        self.solver_combo.bind("<<ComboboxSelected>>", self._on_solver_change)
        self.btn_solver_path = ttk.Button(row, text="Set Solver Path",
                                          command=self._set_solver_path)
        self.btn_solver_path.pack(side=tk.LEFT, padx=4)
        self.lbl_solver_path = ttk.Label(row, text="", foreground="#335")
        self.lbl_solver_path.pack(side=tk.LEFT, padx=4)

        # 网格
        row = ttk.Frame(top); row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text="Mesh (c=0.4572):").pack(side=tk.LEFT)
        self.mesh_var = tk.StringVar()
        self.ent_mesh = ttk.Entry(row, textvariable=self.mesh_var, width=50)
        self.ent_mesh.pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Browse", command=self._browse_mesh).pack(side=tk.LEFT)

        # 工况参数
        params = [
            ("Density rho [kg/m3]", "rho", "1.225"),
            ("Viscosity mu [Pa*s]", "mu", "1.7894e-05"),
            ("Velocity V [m/s]", "V", "30.0"),
            ("Frequency f [Hz]", "freq", "1.0"),
            ("Mean AoA [deg]", "mean", "10.0"),
            ("AoA Amp [deg]", "amp", "5.0"),
            ("Turbulence TI", "TI", "0.05"),
            ("Length Scale Lt/c", "Lt_c", "1.0"),
        ]
        self.param_vars = {}
        grid = ttk.Frame(top); grid.pack(fill=tk.X, padx=6, pady=2)
        for i, (label, key, default) in enumerate(params):
            r, c = i // 4, i % 4
            frame = ttk.Frame(grid); frame.grid(row=r, column=c, sticky="w",
                                                padx=8, pady=2)
            ttk.Label(frame, text=label).pack(anchor="w")
            v = tk.StringVar(value=default)
            ttk.Entry(frame, textvariable=v, width=16).pack(anchor="w")
            self.param_vars[key] = v

        # 工作目录 + 输出（默认用当前工作目录下的 outputs，避免硬编码绝对路径）
        row = ttk.Frame(top); row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text="Output Dir:").pack(side=tk.LEFT)
        default_out = str(Path(__file__).resolve().parent.parent / "outputs")
        self.out_var = tk.StringVar(value=default_out)
        ttk.Entry(row, textvariable=self.out_var, width=50).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Browse", command=self._browse_out).pack(side=tk.LEFT)

        # 运行控制
        ctrl = ttk.Frame(self); ctrl.pack(fill=tk.X, padx=8, pady=4)
        self.btn_run = ttk.Button(ctrl, text="Run", command=self._run)
        self.btn_run.pack(side=tk.LEFT)
        ttk.Label(ctrl, text="  Cores:").pack(side=tk.LEFT)
        self.nproc_var = tk.StringVar(value=str(getattr(_tool_cfg, "NPROC", 4)))
        ttk.Entry(ctrl, textvariable=self.nproc_var, width=4).pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self._stop_request,
                                   state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.lbl_status = ttk.Label(ctrl, text="Ready", foreground="#335")
        self.lbl_status.pack(side=tk.LEFT, padx=8)

        # 实时气动性能图（左 Cl-AoA，右 Cd-AoA）
        fig_frame = ttk.LabelFrame(self, text="Aerodynamic Performance (Live)")
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.fig = Figure(figsize=(11, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_cl = self.fig.add_subplot(121)
        self.ax_cd = self.fig.add_subplot(122)
        for ax, yl in [(self.ax_cl, "Cl"), (self.ax_cd, "Cd")]:
            ax.set_xlabel("AoA [deg]"); ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
            ax.set_title(f"{yl} vs AoA")
        (self._line_cl,) = self.ax_cl.plot([], [], ".-", color="#2ca02c",
                                           markersize=3, lw=1)
        (self._line_cd,) = self.ax_cd.plot([], [], ".-", color="#d62728",
                                           markersize=3, lw=1)
        self.fig.tight_layout()

        # Fluent 运行日志（图下方独立区域，不与数据图重叠）
        log_frame = ttk.LabelFrame(self, text="Fluent Log (.trn)")
        log_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._log_text = tk.Text(log_frame, height=9, wrap="none",
                                 font=("Consolas", 8), background="#f5f5f5")
        _sb = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=_sb.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._trn_path = None
        self._trn_pos = 0

    # ------------------------------------------------------------------
    def _on_solver_change(self, _evt=None):
        if self.solver_var.get() == "surrogate":
            self.btn_solver_path.config(text="Load Surrogate .py")
        else:
            self.btn_solver_path.config(text="Set Solver Path")

    def _set_solver_path(self):
        if self.solver_var.get() == "surrogate":
            p = filedialog.askopenfilename(
                title="Select surrogate model .py",
                filetypes=[("Python", "*.py")])
            if p:
                self.lbl_solver_path.config(text=p)
                self._solver_path = p
        else:
            p = filedialog.askopenfilename(
                title="Select fluent.exe",
                filetypes=[("Executable", "*.exe")])
            if p:
                self.lbl_solver_path.config(text=p)
                self._solver_path = p

    def _browse_mesh(self):
        p = filedialog.askopenfilename(title="Select airfoil mesh (c=0.4572)")
        if p:
            self.mesh_var.set(p)

    def _browse_out(self):
        d = filedialog.askdirectory(title="Select output dir")
        if d:
            self.out_var.set(d)

    # ------------------------------------------------------------------
    def _collect_params(self) -> SimParams:
        try:
            return SimParams(
                mesh_path=self.mesh_var.get().strip(),
                rho=float(self.param_vars["rho"].get()),
                mu=float(self.param_vars["mu"].get()),
                V=float(self.param_vars["V"].get()),
                freq=float(self.param_vars["freq"].get()),
                mean_aoa=float(self.param_vars["mean"].get()),
                amp_aoa=float(self.param_vars["amp"].get()),
                TI=float(self.param_vars["TI"].get()),
                Lt_over_c=float(self.param_vars["Lt_c"].get()),
            )
        except ValueError as ex:
            raise ValueError(f"Invalid parameter: {ex}")

    def _run(self):
        if self._thread and self._thread.is_alive():
            return
        try:
            params = self._collect_params()
            params.validate()
        except Exception as ex:
            self.lbl_status.config(text=f"Error: {ex}")
            return
        # 构建求解器
        solver_type = self.solver_var.get()
        try:
            if solver_type == "fluent":
                if not getattr(self, "_solver_path", None):
                    raise ValueError("请先设置 fluent.exe 路径")
                try:
                    nproc = int(self.nproc_var.get())
                except ValueError:
                    nproc = getattr(_tool_cfg, "NPROC", 4)
                self._solver = FluentSolver(self._solver_path,
                                            Path(self.out_var.get()),
                                            nproc=nproc)
            else:
                if not getattr(self, "_solver_path", None):
                    raise ValueError("请先加载代理模型 .py")
                self._solver = SurrogateSolver(self._solver_path)
        except Exception as ex:
            self.lbl_status.config(text=f"Error: {ex}")
            return
        self._stop.clear()
        self._live_data = dict(time=[], aoa=[], cl=[], cd=[])
        self._drawn_n = 0
        self._drawn_result = None
        # 清空并重置日志区
        self._log_text.delete("1.0", tk.END)
        self._trn_path = None
        self._trn_pos = 0
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text="Running...")
        self._thread = threading.Thread(target=self._worker, args=(params,),
                                        daemon=True)
        self._thread.start()

    def _worker(self, params):
        try:
            result = self._solver.run(
                params, stop_event=self._stop, data_cb=self._on_live_data)
            self._result = result
            # 输出
            self._save_outputs(result)
            self.after(0, lambda: self.lbl_status.config(
                text=f"Done ({result.n_points} pts, {result.source})"))
        except InterruptedError:
            self.after(0, lambda: self.lbl_status.config(text="Stopped"))
        except Exception as ex:
            self.after(0, lambda: self.lbl_status.config(text=f"Error: {ex}"))
        finally:
            self.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            self.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

    def _save_outputs(self, result):
        out = Path(self.out_var.get())
        out.mkdir(parents=True, exist_ok=True)
        base = Path(result.mesh_name).stem + f"_{int(time.time())}"
        write_datatable(result, out / f"{base}_datatable.csv")
        write_rawdata(result, out, base)
        plot_aero(result, out / f"{base}_aero.png")
        self.after(0, lambda: self.lbl_status.config(
            text=f"Done - saved to {out}"))

    def _on_live_data(self, points):
        """后台线程实时数据回调：追加到实时曲线缓冲（30Hz 刷新显示）。"""
        for t, a, cl, cd in points:
            self._live_data["time"].append(t)
            self._live_data["aoa"].append(a)
            self._live_data["cl"].append(cl)
            self._live_data["cd"].append(cd)

    def _stop_request(self):
        self._stop.set()
        self.lbl_status.config(text="Stopping...")
        # 同 Ctrl+C：通知求解器杀进程/终止预测
        if self._solver is not None:
            try:
                self._solver.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _refresh(self):
        """30Hz 调度；仅在数据发生变化时才重绘，避免持续绘制卡住窗口拖动/缩放。"""
        if self._thread and self._thread.is_alive():
            # 后台运行中：有新点才重绘（Fluent 每 0.5s 一批，绘制频率随数据）
            n = len(self._live_data["time"])
            if n > 0 and n != self._drawn_n:
                self._drawn_n = n
                self.lbl_status.config(text=f"Running... {n} pts live")
                # 用快照副本，避免后台线程 append 与 set_data 转换的竞态
                aoa = list(self._live_data["aoa"])
                self._line_cl.set_data(aoa, list(self._live_data["cl"]))
                self._line_cd.set_data(aoa, list(self._live_data["cd"]))
                for ax in (self.ax_cl, self.ax_cd):
                    ax.relim(); ax.autoscale_view()
                self.canvas.draw_idle()
        elif self._result is not None and self._result is not self._drawn_result:
            # 完成后显示完整结果（仅一次）
            self._drawn_result = self._result
            r = self._result
            self._line_cl.set_data(r.aoa, r.cl)
            self._line_cd.set_data(r.aoa, r.cd)
            for ax in (self.ax_cl, self.ax_cd):
                ax.relim(); ax.autoscale_view()
            self.canvas.draw_idle()
        self._poll_log()
        self.after(REFRESH_MS, self._refresh)

    def _poll_log(self):
        """增量读取 case 目录最新 .trn（Fluent transcript），追加到日志区。"""
        if not (self._thread and self._thread.is_alive()):
            return
        out = Path(self.out_var.get())
        try:
            trns = sorted(out.glob("case_*/*.trn"))
        except Exception:
            return
        if not trns:
            return
        trn = trns[-1]
        if trn != self._trn_path:
            # 新的 .trn：清空日志区，从头读
            self._trn_path = trn
            self._trn_pos = 0
            self._log_text.delete("1.0", tk.END)
        try:
            with open(trn, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._trn_pos)
                new = f.read()
                self._trn_pos = f.tell()
        except Exception:
            return
        if new:
            self._log_text.insert(tk.END, new)
            # 限制日志行数，避免无限增长
            n = int(self._log_text.index("end-1c").split(".")[0])
            if n > 600:
                self._log_text.delete("1.0", f"{n - 600}.0")
            self._log_text.see(tk.END)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
