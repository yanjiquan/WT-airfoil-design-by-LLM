# Dynamic Airfoil Tool

动态俯仰翼型气动性能仿真/预测软件。

- 输入：翼型网格（**必须 c=0.4572 m, 18 inch**）+ 工况参数
- 输出：气动性能图、纯英文数据清单、仿真原始数据
- 双求解器：Fluent（用户指定路径）或 用户自建代理模型（接口一致）
- Fluent 运行期间实时显示 Cl/Cd vs AoA 曲线（30Hz）

## 配置（config.py）

软件**完全自包含**：Fluent 自动化核心（`fluent_auto/`）、参考 case 与 UDF 构建骨架（`resources/`）均随软件分发，不依赖外部目录。换机器只需修改根目录 `config.py` 中的 `FLUENT_EXE`：

```python
# 粘贴你的 fluent.exe 完整路径
FLUENT_EXE = r"E:\software\ANSYS 2025R1\v251\fluent\ntbin\win64\fluent.exe"

# ANSYS 根目录 / Fluent 发布版本按上述路径自动推断，一般无需修改
ANSYS_ROOT = ...      # 自动 = FLUENT_EXE 向上 4 级
FLUENT_RELEASE = "fluent25.1.0"
```

GUI 启动时自动读取 `FLUENT_EXE` 作为默认路径，无需手动 Browse。

## 换电脑（解包即用）

新电脑只需：安装 Fluent + Python 及依赖，把整个 `dynamic_airfoil_tool/` 目录拷贝过去，修改 `config.py` 中的 `FLUENT_EXE`（UDF 编译会自动用该路径推断的 ANSYS 安装目录）。无需拷贝任何其它目录。仿真数据写入 GUI 所选 **Output Dir** 的 `case_*` 子目录。

## 安装/依赖

- Python 3.10+
- 依赖：`numpy`, `matplotlib`, `h5py`（GUI 用 tkinter，Python 自带）
- Fluent 通道还需：Fluent 2025R1（`fluent.exe`，含自带 scons 编译环境）

## 快速开始

```bash
cd E:\my data\Code Agent Workspace\dynamic_airfoil_tool
python main.py                # 启动 GUI
```

### GUI 使用

1. **Solver**：选择 `fluent` 或 `surrogate`
   - fluent：点 "Set Solver Path" 选 `fluent.exe`
   - surrogate：点 "Load Surrogate .py" 选代理模型文件
2. **Mesh**：选择翼型网格文件（自动校验 c=0.4572）
3. **工况参数**：密度/粘度/风速/振荡频率/平均攻角/攻角振幅/湍流强度/长度尺度
4. **Output Dir**：输出目录
5. 点 **Run** 运行（可 **Stop** 终止），右侧实时显示 Cl/Cd vs AoA 曲线（30Hz）
6. 完成后自动生成：
   - `{mesh}_{ts}_aero.png`（气动性能图，无风洞数据）
   - `{mesh}_{ts}_datatable.csv`（纯英文数据清单：mesh/condition/time/AoA/Cl/Cd）
   - `{mesh}_{ts}_raw.npz`（原始数据）

## 复现论文全量批次（5 翼型 × 12 工况）

本工程随复现包自带 5 个翼型网格（`mesh/` 目录）：`naca4415_0p4572.msh`、`llm_0p4572.msh`、`ga_0p4572.msh`、`cmaes_0p4572.msh`、`pso_0p4572.msh`，以及工况配置示例 `config_14deg_amp10.json`（12 个 R 工况）。用 `batch_auto_run.py` **一条命令**即可复现论文全部动态俯仰算例。

```bash
# 1) 修改 dynamic_airfoil_tool/config.py 中的 FLUENT_EXE 为本机路径

# 2) 全量复现（5 翼型 × R1-R12 = 60 case）
python batch_auto_run.py \
    --mesh-dir <复现包>/mesh \
    --config  <复现包>/example_config/config_14deg_amp10.json \
    --out-dir batch_results \
    --parallel 8 --nproc 4 --retry 2
```

要点：
- **网格**：目录里所有 `*.msh` 都会自动入列（本包 5 个）；如只想跑部分翼型用 `--meshes llm_0p4572.msh naca4415_0p4572.msh ...`
- **工况**：`config_14deg_amp10.json` 定义 12 个 R 工况（Re 0.75-1.5M × 频率 0.6/1.2/1.8 Hz，mean=14° amp=10°）；每工况空气物性由风洞数据自动换算（勿改 JSON 内 T/P 除非有意复现其它条件）
- **断点续跑**：同命令重跑即续 —— 已完成 case 自动跳过、失败重试；`--force` 全量重来
- **并发**：`--parallel 8 --nproc 4` = 8 个 Fluent 并行 × 每 case 4 核（32 线程机满配）；机器核少可降 `--parallel`
- **优先级**：未完成翼型中 llm、naca4415 会优先跑（脚本内固定）
- **UDF 编译**：每 case 独立 libudf，自动经串行编译相处理，无需手动干预

输出（`batch_results/`）：
```
case_{af}_{Rxx}/            # 每 case：udf/jobs/*.cas/*.h5/datatable.csv/aero.png/result_summary.json
batch_plan.json             # 任务清单 + 实时状态
batch_report.json           # 最终每 case 状态汇总
```
每 case 保留全部原始 `.dat.h5/.cas.h5/.trn`（不自动清理），可随时用 `plot_forces.py` 复现图。

## 代理模型接口

用户自建代理模型只需实现一个函数/类（输入输出与 Fluent 一致）：

```python
# my_surrogate.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.solver_interface import SimResult

def predict(params):
    # params: 含 rho, mu, V, freq, mean_aoa, amp_aoa, TI, Lt_over_c
    # 返回 SimResult(time, aoa, cl, cd)
    ...
    return SimResult(time=t, aoa=aoa, cl=cl, cd=cd)
```

参考 `example_surrogate.py`。

## Fluent 通道

- `fluent.exe` 路径在 `config.py` 配置（GUI 自动读取为默认）
- 复用随软件打包的 Fluent 自动化逻辑（UDF 生成 / case 构建 / DES transition SST / 滑移网格），见 `fluent_auto/`
- 参考 case（NACA4415）与 UDF 构建骨架随软件分发于 `resources/`，构建时自动将参考 case 内残留的旧机器绝对路径重定位到当前工作目录
- 输入网格必须 c=0.4572，其他翼型通过 replace-mesh 加载
- 运行期间实时增量读取 h5 并推送 GUI 显示（攻角为理论值预览，完成后全量 mesh-aoa 校正）
- 仿真数据写入所选 **Output Dir** 的 `case_*` 子目录

## 目录结构

```
dynamic_airfoil_tool/
├── main.py                 # 入口
├── config.py               # 配置（只改 FLUENT_EXE 即可换机器）
├── example_surrogate.py    # 代理模型示例
├── core/                   # 求解器接口 / Fluent / 代理模型 / 数据导出 / 绘图
│   ├── solver_interface.py
│   ├── fluent_solver.py
│   ├── surrogate_solver.py
│   ├── data_writer.py
│   └── plotter.py
├── fluent_auto/            # 打包的 Fluent 自动化核心（UDF/case/journal/编译/后处理）
│   ├── config.py  case_runner.py  gen_udf.py  build_case.py
│   ├── gen_journal.py  compare_sim_auto.py
├── resources/              # 随软件分发：参考 case + UDF 构建骨架
│   ├── NACA4415_0p4572_0p01_inflat.cas
│   └── udf_build/2ddp_{host,node}/{SConstruct,scons.bat}
└── gui/
    └── main_window.py      # 主界面（参数输入 + 实时图）
```
