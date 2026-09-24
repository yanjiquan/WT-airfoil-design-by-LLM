# -*- coding: utf-8 -*-
"""Static Airfoil Aerodynamic Tools —— 配置。

独立的静态气动极曲线扫掠工程：对目标翼型（llm / naca4415）做固定攻角
0-30° 的稳态气动仿真，获取 Cl(α) / Cd(α) 极曲线。

核心方法（用户提出，已验证）: 静态气动不需要旋转翼型——固定网格不动，
只把入流速度方向从水平改为倾角 β，等效攻角 = β − 网格几何初始角。
因此可纯稳态求解（无 UDF / 无 zone motion / 无滑移网格运动），最快最稳。

换机器只需改 DYNAMIC_TOOL_ROOT / FLUENT_EXE。数据/网格/参考 case 均从
现有 dynamic_airfoil_tool 与数据目录按路径引用（不复制）。
"""

from pathlib import Path

# ----------------------------------------------------------------------
# 现有 dynamic_airfoil_tool 工程根（提供参考 case + fluent_auto 后处理）
# ----------------------------------------------------------------------
DYNAMIC_TOOL_ROOT = Path(
    r"E:\my data\My Workspace\Code\CFD Simulation"
    r"\dynamic_airfoil_tool_transfer_20260901_1036\dynamic_airfoil_tool")

# 网格目录（llm_0p4572.msh / naca4415_0p4572.msh）
MESH_DIR = Path(r"D:\LLM CFD Data\meshfiles")

# 参考 NACA4415 case（含全部模型/BC/网格拓扑，作为文本模板）
REF_CAS = DYNAMIC_TOOL_ROOT / "resources" / "NACA4415_0p4572_0p01_inflat.cas"

# fluent_auto 目录（后处理力提取 compare_sim_auto 所在）
FLUENT_AUTO_DIR = DYNAMIC_TOOL_ROOT / "fluent_auto"

# batch_auto_run.py（提供 wt_air_properties / AF_INITIAL_AOA / 锁辅助）
BATCH_AUTO_RUN_PY = DYNAMIC_TOOL_ROOT / "batch_auto_run.py"

# ----------------------------------------------------------------------
# Fluent 可执行文件 / ANSYS（与 dynamic 工程 config.py 相同）
# ----------------------------------------------------------------------
FLUENT_EXE = r"E:\software\ANSYS 2025R1\v251\fluent\ntbin\win64\fluent.exe"
ANSYS_ROOT = Path(FLUENT_EXE).resolve().parents[3] if FLUENT_EXE else Path("")
FLUENT_RELEASE = "fluent25.1.0"

# Fluent 并行核数（-t N）。许可证并行上限 4-way 时保持 <=4。
NPROC = 4

# ----------------------------------------------------------------------
# 几何 / 湍流（固定）
# ----------------------------------------------------------------------
CHORD = 0.4572            # 弦长 [m] (18 inch)
TI = 0.05                 # 自由流湍流强度 5%
LT_OVER_C = 0.20          # 积分长度尺度 / 弦长

# ----------------------------------------------------------------------
# Re 档 -> 对应 wind-tunnel 工况（物性经 batch_auto_run.wt_air_properties 取，
# 勿硬编码数值）。静态扫掠无频率，用各 Re 最低频档（R1/R4/R10, f=0.6）代表。
# ----------------------------------------------------------------------
RE_TAGS = {"0p75": "R1", "1p0": "R4", "1p25": "R7", "1p5": "R10"}
AIRFOILS = ["llm", "naca4415"]

# 输出根目录
OUT_DIR = Path(r"D:\LLM CFD Data\static_airfoil_tools\outputs")
