# -*- coding: utf-8 -*-
"""D1T1 工况及路径配置。

所有物理/动态/湍流参数来自风洞数据换算（NACA4415, C5l75_n4415.txt, RUN 407）：
    T=35.8 degF, P=14.3121 psia, V=75.1 ft/s, Re=0.75e6, f=0.60Hz, k=0.038
换算结果：
    T=275.26 K, P=98678.5 Pa, rho=1.24888 kg/m3, mu=1.72643e-5 Pa*s, V=22.89 m/s
"""

import math
from pathlib import Path

# ----------------------------------------------------------------------
# 路径。
# FLUENT_EXE / ANSYS_ROOT 由 dynamic_airfoil_tool/config.py 提供，
# 每次运行前 core/fluent_solver.py 会覆盖；这里给软件包内自带资源
# 提供相对默认，保证 fluent_auto 整体自包含、可整体分发。
# ----------------------------------------------------------------------
# Fluent 可执行文件（运行前由工具 config.py 覆盖）
FLUENT_EXE = ""

# ANSYS 安装根目录（用于 FLUENT_INC / UDF 编译，运行前由工具 config.py 覆盖）
ANSYS_ROOT = Path("")
FLUENT_INC = str(ANSYS_ROOT / "fluent")
PYTHON_DIR = ANSYS_ROOT / "commonfiles" / "CPython" / "3_10" / "winx64" / "Release" / "python"

# UDF 自动编译所需环境变量
# 注意: 只需 FLUENT_INC。设置 PYTHONHOME/PYTHONPATH 会干扰 Fluent 的
# MPI 并行初始化（实测导致 "Fatal failure" / 返回码 -1）。
ANSYS_ENV = {
    "FLUENT_INC": FLUENT_INC,
}

# 软件包内自带资源目录（参考 case + UDF 构建骨架）
_RESOURCE_ROOT = Path(__file__).resolve().parent.parent / "resources"

# 参考工程目录（含 NACA4415 参考 .cas）
REF_PROJECT = _RESOURCE_ROOT

# UDF 参考构建目录（scons.bat/SConstruct，随软件分发）
UDF_REF_DIR = _RESOURCE_ROOT / "udf_build"

# 工作目录
BASE_DIR = Path(__file__).resolve().parent          # fluent_auto/
MESH_DIR = Path("")                 # 每次运行由用户网格路径覆盖
UDF_SRC_DIR = _RESOURCE_ROOT        # 预留（本软件核心链路未使用）
OUT_ROOT = Path("")                 # 每次运行由工作目录覆盖

# Fluent 发布版本（UDF 编译 sconstruct.udf 目录名）。运行前由工具 config.py 覆盖
FLUENT_RELEASE = "fluent25.1.0"

# ----------------------------------------------------------------------
# 翼型 / 网格
# ----------------------------------------------------------------------
AIRFOIL = "NACA4415"
MESH_NAME = f"{AIRFOIL}_0p4572_0p01_inflat.msh"
CHORD = 0.4572          # 18 inch -> 0.4572 m
ROT_X = 0.25 * CHORD    # 0.1143 m，绕 0.25 弦转
ROT_Y = 0.0

# 动态网格区域（从参考 .cas.h5 读取确认）
ZONE_ROTATE = "rotate"        # 旋转体区域
ZONE_AIRFOIL = "airfoil"      # 翼型壁面
ZONE_INLET = "inlet"          # 速度入口
ZONE_OUTLET = "outlet"        # 压力出口
ZONE_INTERFACE_IN = "interface-in"
ZONE_INTERFACE_OUT = "interface-out"

# ----------------------------------------------------------------------
# D1 动态工况（风洞实测）
# ----------------------------------------------------------------------
D_CASE = "D1"
AOA_MEAN = 8.0        # 平均攻角 [deg]
AOA_AMP = 5.5         # 攻角振幅 [deg]
FREQ = 0.60           # 振荡频率 [Hz]
K = 0.038             # 减缩频率（风洞报告值）
AOA_START = AOA_MEAN - AOA_AMP   # 正弦相位起点 = 2.5 deg（同参考 14-10=4）

# 网格初始攻角（mesh 预置角度）。D1T1 的 case 继承参考工程网格，
# 初始网格实测 4.036 deg（参考 Setup）。复位 UDF 把翼型从该初始角
# 转到 AOA_START。若换网格，请按实测更新。
INITIAL_AOA = 4.036

# ----------------------------------------------------------------------
# T1 湍流状态
# ----------------------------------------------------------------------
T_CASE = "T1"
TI = 0.05             # 湍流强度 5%
LT_OVER_C = 0.20      # 积分长度尺度/弦长
LT = LT_OVER_C * CHORD  # = 0.09144 m

# ----------------------------------------------------------------------
# 空气物性（风洞环境换算）
# ----------------------------------------------------------------------
T_F = 35.8            # 环境温度 [degF]
P_PSIA = 14.3121      # 环境压力 [psia]
V_FTS = 75.1          # 来流速度 [ft/s]

# 换算
T_K = (T_F + 459.67) / 1.8
P_PA = P_PSIA * 6894.757
RHO = P_PA / (287.05 * T_K)                      # 1.24888
MU = 1.716e-5 * (T_K / 273.15) ** 1.5 * (273.15 + 110.4) / (T_K + 110.4)  # 1.72643e-5
V_INF = V_FTS * 0.3048                           # 22.8905
RE = RHO * V_INF * CHORD / MU

# ----------------------------------------------------------------------
# 时间推进方案（用户敲定）
# ----------------------------------------------------------------------
# 阶段1: reset_ang 转到 AOA_START，dt=0.05s（旋转完成需 1s -> 20 步）
RESET_DT = 0.05
RESET_DURATION = 1.0
RESET_STEPS = int(round(RESET_DURATION / RESET_DT))     # 20

# 阶段2: 稳定流场，dt=1e-6s 跑 50 步
STAB_DT = 1e-6
STAB_STEPS = 50

# 阶段3: rotate_sine 俯仰，dt=0.05s 跑 6 周期（含第1个导入周期，
# 即 5 个稳定周期——用户提速决定 2026-09-01：11->6，省约 43% 时长，
# 对动态失速迟滞环分析足够）
PITCH_DT = 0.05
PITCH_CYCLES = 6
PERIOD = 1.0 / FREQ                    # 1.6667 s
STEPS_PER_CYCLE = int(round(PERIOD / PITCH_DT))   # 33
PITCH_STEPS = PITCH_CYCLES * STEPS_PER_CYCLE      # 198

# 每时间步最大内迭代次数（dual-time-iterate 第二个参数）。
# 参考工程每步约 10-30 次内迭代收敛；30 兼顾收敛与速度，
# 100 会导致每步满迭代、仿真显著变慢。
ITER_PER_STEP = 30

# ----------------------------------------------------------------------
# 输出命名（对齐参考工程 `{base}-10-00001.cas.h5`）
# ----------------------------------------------------------------------
CASE_ID = f"{AIRFOIL}_{D_CASE}{T_CASE}"            # NACA4415_D1T1
BASE_NAME = f"{AIRFOIL}_0p4572_0p01_inflat"        # 网格/文件基名
OUT_DIR = OUT_ROOT / CASE_ID
JOB_DIR = BASE_DIR / "jobs"
UDF_OUT_DIR = BASE_DIR / "udf"

# 湍流入口参数（transition SST 需设自由流湍流强度）
TURBULENCE_SPEC = "intensity-and-length-scale"
TURB_INTENSITY_PCT = TI * 100.0     # 5.0
TURB_LENGTH_SCALE = LT             # 0.09144


def sanity_check() -> None:
    """打印并校验关键换算结果。"""
    print("=" * 64)
    print(f"工况: {CASE_ID}  翼型: {AIRFOIL}  弦长: {CHORD} m")
    print(f"动态工况 {D_CASE}: 平均攻角={AOA_MEAN} deg, 振幅={AOA_AMP} deg, "
          f"频率={FREQ} Hz, 起始角={AOA_START} deg")
    print(f"湍流 {T_CASE}: TI={TI*100:.1f}%, Lt/c={LT_OVER_C}, Lt={LT:.5f} m")
    print("-" * 64)
    print(f"环境: T={T_F} F -> {T_K:.2f} K")
    print(f"环境: P={P_PSIA} psia -> {P_PA:.1f} Pa")
    print(f"密度  : rho = {RHO:.5f} kg/m3")
    print(f"粘度  : mu  = {MU:.6e} Pa*s")
    print(f"来流  : V   = {V_FTS} ft/s = {V_INF:.4f} m/s")
    print(f"Re    :     = {RE:.3e}  (目标 0.75e6)")
    print(f"k     :     = {math.pi*FREQ*CHORD/V_INF:.4f}  (风洞报告 0.038)")
    print("-" * 64)
    print(f"阶段1 reset: dt={RESET_DT}, {RESET_STEPS} 步 ({RESET_DURATION}s) 转到 {AOA_START} deg")
    print(f"阶段2 稳定 : dt={STAB_DT}, {STAB_STEPS} 步")
    print(f"阶段3 俯仰 : dt={PITCH_DT}, {PITCH_CYCLES} 周期 x {STEPS_PER_CYCLE} 步 = {PITCH_STEPS} 步")
    print(f"输出目录   : {OUT_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    sanity_check()
