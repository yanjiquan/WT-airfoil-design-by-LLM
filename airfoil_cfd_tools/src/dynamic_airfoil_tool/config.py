# -*- coding: utf-8 -*-
"""Dynamic Airfoil Tool 配置。

换机器 / 换 Fluent 安装位置时，只需修改下面的 FLUENT_EXE。
软件其余部分（Fluent 自动化核心 fluent_auto/、参考 case 与 UDF 构建
骨架 resources/）均已随软件打包，不依赖外部目录。
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Fluent 可执行文件（粘贴你的 fluent.exe 完整路径）
# 例如: r"E:\software\ANSYS 2025R1\v251\fluent\ntbin\win64\fluent.exe"
# ----------------------------------------------------------------------
FLUENT_EXE = r"E:\software\ANSYS 2025R1\v251\fluent\ntbin\win64\fluent.exe"

# ANSYS 安装根目录（UDF 编译 / FLUENT_INC 环境变量）。
# 默认按 fluent.exe 路径自动推断：fluent/ntbin/win64/fluent.exe 向上 4 级。
# 若你的 Fluent 目录结构特殊，可在此手动覆盖。
ANSYS_ROOT = Path(FLUENT_EXE).resolve().parents[3] if FLUENT_EXE else Path("")

# Fluent 发布版本（UDF 编译 sconstruct.udf 的目录名）。Fluent 2025R1 = fluent25.1.0
FLUENT_RELEASE = "fluent25.1.0"

# Fluent 并行核数（启动参数 -t N）。机器核数多且许可证支持时提高可加速；
# 注意 Fluent 许可证有并行核数上限（如 4-way），超出可能报错或降级。
NPROC = 4
