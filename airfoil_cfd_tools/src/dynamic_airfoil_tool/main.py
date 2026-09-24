# -*- coding: utf-8 -*-
"""Dynamic Airfoil Tool 入口。

用法:
    python main.py            # 启动 GUI
    python main.py --headless # 命令行模式（测试/批处理）
"""

import argparse
import sys
from pathlib import Path


def main_gui():
    from gui.main_window import main
    main()


def main_headless(params: dict):
    """命令行模式：一次求解，输出数据清单/原始数据/图。"""
    from core.solver_interface import SimParams
    from core.fluent_solver import FluentSolver
    from core.surrogate_solver import SurrogateSolver
    from core.data_writer import write_datatable, write_rawdata
    from core.plotter import plot_aero
    import time

    p = SimParams(**params)
    p.validate()

    solver_type = params.get("solver", "fluent")
    if solver_type == "fluent":
        solver = FluentSolver(params["fluent_exe"], Path(params["work_dir"]))
    else:
        solver = SurrogateSolver(params["model_path"])

    result = solver.run(p)
    out = Path(params.get("out_dir", "outputs"))
    out.mkdir(parents=True, exist_ok=True)
    base = f"run_{int(time.time())}"
    write_datatable(result, out / f"{base}_datatable.csv")
    write_rawdata(result, out, base)
    plot_aero(result, out / f"{base}_aero.png")
    print(f"Saved to {out}: {result.n_points} pts, source={result.source}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dynamic Airfoil Tool")
    ap.add_argument("--headless", action="store_true",
                    help="命令行模式（不启动 GUI）")
    args = ap.parse_args()
    if args.headless:
        # 命令行示例参数（可用 --params 覆盖）
        demo = dict(
            solver="fluent",
            mesh_path="",
            rho=1.225, mu=1.7894e-5, V=30.0,
            freq=1.0, mean_aoa=10.0, amp_aoa=5.0,
            TI=0.05, Lt_over_c=1.0,
            fluent_exe="", work_dir="", out_dir="outputs",
        )
        print("Headless mode requires full params. Use GUI instead.")
        print("GUI: python main.py")
    else:
        main_gui()
