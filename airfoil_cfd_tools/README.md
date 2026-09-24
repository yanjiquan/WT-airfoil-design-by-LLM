# Airfoil CFD Tools — 复现包总说明

动态失速 + 静态极曲线两套 Fluent 翼型气动仿真代码，供研究者复现论文中的 CFD 结果。

## 包内容

```
airfoil_cfd_tools/
├── README.md                     ← 本文件
├── requirements.txt              ← Python 依赖（numpy/matplotlib/h5py）
├── mesh/                         ← 复现所需 5 个翼型网格（c=0.4572 m）
│   ├── naca4415_0p4572.msh       ← 参考翼型（基准）
│   ├── llm_0p4572.msh            ← 优化翼型
│   ├── ga_0p4572.msh             ← 优化翼型
│   ├── cmaes_0p4572.msh          ← 优化翼型
│   └── pso_0p4572.msh            ← 优化翼型
├── example_config/
│   └── config_14deg_amp10.json   ← 动态工况（R1-R12，mean=14° amp=10°）
└── src/
    ├── dynamic_airfoil_tool/     ← 动态失速俯仰仿真工程（复现论文动态算例）
    │   ├── README.md             ← 使用手册 + 一键复现全量批次命令
    │   ├── config.py             ← 换机器改 FLUENT_EXE
    │   ├── batch_auto_run.py     ← 批量扫掠驱动（5翼型×R1-R12 一口气跑完）
    │   ├── main.py / gui/        ← GUI 入口
    │   ├── core/                 ← 求解器接口/数据导出/绘图
    │   ├── fluent_auto/          ← Fluent 自动化（UDF/journal/case/后处理）
    │   └── resources/            ← 参考 NACA4415 case + UDF 构建骨架
    └── static_airfoil_tools/     ← 静态气动极曲线工程（复现论文静态算例）
        ├── README.md             ← 使用手册 + 完整仿真设置表
        ├── config.py             ← 换机器改路径/FLUENT_EXE
        ├── static_case.py        ← 稳态 case 文本生成
        ├── static_journal.py     ← 稳态 journal（SST + TUI 设入流方向）
        ├── static_polar_batch.py ← 批量扫掠驱动（并行+断点续跑）
        ├── static_post.py        ← 后处理（来流系力分解求 Cl/Cd）
        └── plot_*.py             ← 绘图脚本
```

## 复现依赖

1. **ANSYS Fluent 2025 R1**（或兼容版本，见各 config `FLUENT_RELEASE`）——需单独安装
2. **Python 3.10+** + `pip install -r requirements.txt`
3. 每台机器**改两个 config.py 的路径**：
   - `src/dynamic_airfoil_tool/config.py` → `FLUENT_EXE`
   - `src/static_airfoil_tools/config.py` → `FLUENT_EXE`、`DYNAMIC_TOOL_ROOT`、`MESH_DIR`（可指到本包 `mesh/`）

## 两套仿真简介

| | 动态失速（dynamic） | 静态极曲线（static） |
|---|---|---|
| 目的 | 俯仰振荡迟滞环 Cl(α) | 固定攻角静态 Cl(α)/Cd(α) |
| 求解 | 瞬态 dual-time + 滑移网格 | **稳态** + 网格静止 |
| 攻角实现 | DEFINE_ZONE_MOTION UDF 俯仰 | **来流方向倾 β**（不转网格） |
| 湍流 | transition SST（转捩模型，用于瞬态） | 标准 SST k-ω（稳态） |
| 工况 | mean±amp 正弦俯仰，R1-R12 | 0-30°，Re 0.75/1.0/1.25/1.5M |
| 翼型 | llm/naca4415/cmaes/ga/pso | llm / naca4415 |

**关键设计要点（详细见两工程各自 README）**：
- 动态：动态网格 case 一律瞬态微步稳定，**不要切稳态**（会把 zone-motion 变 Frame Motion 使翼型不动）
- 静态：入口方向**必须 journal TUI 设**（改 case 文本无效）；斜来流的 Cl/Cd 须旋转到来流系

## 快速验证

```bash
# 动态：单个 case（naca4415 @ R1）
python src/dynamic_airfoil_tool/batch_auto_run.py \
  --mesh-dir <本包>/mesh --config <本包>/example_config/config_14deg_amp10.json \
  --meshes naca4415_0p4572.msh --conditions R1 --parallel 1 --nproc 4

# 静态：llm 单点 8° Re1.0M
python src/static_airfoil_tools/static_polar_batch.py \
  --airfoils llm --re 1p0 --angles 8 --n-iter 400 --parallel 1 --nproc 4
```

**一键复现论文结果**（完整命令见两工程 README 内「复现论文」节）：
- 动态全量：`dynamic_airfoil_tool/batch_auto_run.py --mesh-dir <包>/mesh --config <包>/example_config/config_14deg_amp10.json --out-dir batch_results --parallel 8 --nproc 4`
- 静态全量：`static_airfoil_tools/static_polar_batch.py --airfoils llm naca4415 --re 0p75 1p0 1p25 1p5 --angles 0 2 ... 30 --parallel 8 --nproc 4`

## 结果说明

两套代码保留每个 case 的完整原始数据（.cas.h5/.dat.h5/.trn 等，不自动清理）。论文中使用的**汇总结果**（极曲线 CSV、平均曲线、对比图）见本工程作者处结果包（结果包不含 .h5 原始 case，仅 CSV/图/汇总表）。

## 论文引用信息

（作者在此补充论文标题/DOI/引用格式）
