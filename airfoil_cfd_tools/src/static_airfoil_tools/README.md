# Static Airfoil Aerodynamic Tools

独立的翼型**静态气动极曲线**（static polar）Fluent 自动化工程。对目标翼型（`llm`、`naca4415`）做固定攻角 0-30° 的**稳态**气动仿真，获取 Cl(α)、Cd(α) 极曲线，供优化翼型与参考翼型的静态性能对比。

## 核心方法（关键设计）

**不旋转翼型 —— 旋转入流方向。** 静态气动性能只取决于翼型相对来流的夹角。让网格完全静止，仅把入口来流方向从水平改为倾角 β，即等效获得攻角：

```
β = α_target + α_mesh_initial
```

其中 `α_mesh_initial` 是网格几何初始角（翼型弦线相对 x 轴的固有倾斜，llm=0.0°、naca4415=+0.236°）。

由此带来的好处：
- **纯稳态求解**（`/solve/iterate`），无需 UDF / 无 zone-motion / 无滑移网格运动
- 规避动态工程的已知坑：切稳态会把 zone motion 变 Frame Motion（本方案 case 本就无运动）
- 比"瞬态保持"方案快 **~20×**：单点从 ~45-60 min 降到 ~5 min

## 湍流模型

本工程静态仿真采用**标准 SST k-ω 湍流模型**（两方程，全湍流假设），由 journal 在运行时通过 TUI 启用并关闭转捩模型：

```
/define/models/viscous/kw-sst
yes
/define/models/viscous/transition-sst
no
```

> 说明：标准 SST k-ω 适用于本工程稳态气动求解的设置。参考 case 原生为 transition SST（带 γ-Reθ 转捩变量，用于动态瞬态转捩预测），稳态求解时按上述命令切换到标准 SST k-ω。若学者希望自行对比其它湍流模型，在 journal 的该段替换对应 TUI 命令即可。

## 仿真设置总览（完整）

下表汇总本工程每个静态 case 的全部 Fluent 设置。参考 case（`NACA4415_0p4572_0p01_inflat.cas`）本身是 **2D 双精度、压力基、coupled 求解器 + 伪瞬态（coupled-pseudo-transient）**的动网格 case，本工程在其上做最小化文本编辑 + journal TUI 覆盖，得到纯稳态配置。

### 求解器（Solvers）

| 项 | 设置 | 说明 |
|---|---|---|
| 维度/精度 | **2D 双精度**（`fluent 2ddp`） | 启动参数 `-t{nproc}` 并行 |
| 求解类型 | **稳态 Steady** | `case-config` 块文本改 `rp-unsteady? #t→#f`、`rp-dual-time? 2→0` |
| 压力基 / 密度基 | **压力基 Pressure-based** | 参考 case 原生（低速不可压外流） |
| 速度-压力耦合 | **Coupled + 伪瞬态**（coupled-pseudo-transient） | 参考 case 原生设置，稳态下继续用；收敛快、对大攻角分离稳健 |
| 空间离散（动量） | **二阶迎风 Second Order Upwind** | 读入时 Fluent 自动切换（trn: "Changing Discretization Scheme for Momentum to Second Order Upwind"） |
| 湍流模型 | **标准 SST k-ω** | journal TUI `/define/models/viscous/kw-sst → yes`，随后**关闭转捩模型选项** |
| 能量方程 | 关闭 | 低温不可压缩，无热效应 |

> **时间推进**：稳态无时间步，迭代用 `/solve/iterate N`（瞬态才用 `/solve/dual-time-iterate`）。

### 流动模型与网格

| 项 | 设置 |
|---|---|
| 网格 | GAMBIT 格式 `.msh`，2D 混合单元，~94k 节点 / ~135k 单元（c=0.4572 m，18 in） |
| 旋转子域 | **无运动**（去掉 zone-motion UDF 绑定名 → `"none"`；保留滑移界面拓扑但稳态下网格静止） |
| 湍流入口 | intensity-and-length-scale 法：**TI = 5%**，**Lt = 0.2×c = 0.09144 m** |

### 材料与边界条件

| 项 | 设置 |
|---|---|
| 流体 | 空气 ideal-gas 关闭、常密度 `rho`（按 Re 档） |
| 粘度 | 常粘度 `mu`（按 Re 档，Sutherland 换算） |
| 入口 inlet（velocity-inlet） | **Magnitude and Direction** 模式：`vmag = V`（按 Re 档），方向 `(cosβ, sinβ, 0)` |
| 出口 outlet（pressure-outlet） | 参考 case 默认（gauge 0，回流容差默认） |
| 翼型壁面 | 无滑移 wall（参考 case 原生） |
| 湍流入口值 | `k/ω` 由 TI 与 Lt 换算（Fluent 自动） |

> **入口方向设置方式（关键）**：velocity-inlet 默认 `velocity-spec . 0`（Magnitude-Normal to Boundary），此时来流垂直于入口面（水平），**忽略任何 flow-direction**。必须切到 **Magnitude and Direction**（`velocity-spec . 1`）并显式给方向分量。实测 **case 文本里改这两个字段无效**（读入后流场仍水平），必须在 journal 用 TUI：
> ```
> /define/boundary-conditions/set/velocity-inlet
> inlet
> <空行>
> velocity-spec → yes        # 切 Magnitude and Direction
> vmag → no → <V>
> direction-0 → no → <cosβ>
> direction-1 → no → <sinβ>
> q
> ```

### 攻角施加

来流倾角 β = α_target + α_mesh_initial（α_mesh_initial 见 `_af_initial_aoa` 表：naca4415 +0.236°，llm 0°）。翼型不转、网格不动，等效攻角 = β − α_mesh_initial = α_target。后处理用 mesh 实测（`mesh_aoa_from_cas`）校验：静态 case 中翼型几何角 ≈ α_mesh_initial（0.03°@naca4415，与 0.236° 表值有 ~0.2° 出入，来自表值是别批网格实测）。

### 数值控制与收敛

| 项 | 设置 |
|---|---|
| 迭代命令 | `/solve/iterate {n_iter}`（默认 400 上限） |
| 收敛判据 | Fluent 默认 scaled residual（各变量 1e-3 量级自动报 `solution is converged`） |
| 附着流（≤~16°） | 通常 **20-100 迭代收敛** |
| 深失速（≥~20°） | 分离非定常，常跑满 400 上限，continuity 停在 ~5e-3 不再降——取末次解作为该角静态值（工程近似，深失速本身物理上即非定常） |
| 初始化 | `/solve/initialize/hyb-initialization`（混合初始化） |
| 松弛 | 求解器默认（coupled 伪瞬态自带） |

### 输出与文件

| 项 | 设置 |
|---|---|
| autosave | **全部关闭**（data/case 频率 0）——稳态不需要逐步 dat，防占盘 |
| 结果保存 | journal 末尾单次 `/file/write-case-data <base>-static.cas.h5`（自动连带 .dat.h5） |
| 数据保留 | 每个 case 目录保留 .cas/.jou/.trn/结果 .h5（不自动清理） |

### 单 case 完整 journal 示例

```
; ============================================================
; Fluent journal - naca4415_S1p0_a08  (静态攻角稳态求解)
; ============================================================
/file/read-case "D:/.../naca4415_S1p0_a08/naca4415_S1p0_a08.cas"
/file/confirm-overwrite?
no
/file/replace-mesh "D:/.../naca4415_S1p0_a08/naca4415_0p4572.msh"

; ---- 湍流: 标准 SST k-omega ----
/define/models/viscous/kw-sst
yes
/define/models/viscous/transition-sst
no

; ---- TUI 设入口方向 β=8.236° ----
/define/boundary-conditions/set/velocity-inlet
inlet
<blank>
velocity-spec
yes
vmag
no
29.9402
direction-0
no
0.989686
direction-1
no
0.143251
q

/solve/initialize/hyb-initialization
/solve/iterate 400
/file/write-case-data "D:/.../naca4415_S1p0_a08/naca4415_S1p0_a08-static.cas.h5"
/exit
yes
```

## 目录结构

```
static_airfoil_tools\
├── config.py              # 所有路径 / Re 档 / 翼型 / 物性配置
├── static_case.py         # case 文本生成（烘焙物性/BC/稳态，去掉 UDF 绑定）
├── static_journal.py      # Fluent journal（含 SST 切换 + TUI 设来流方向）
├── static_polar_batch.py  # 批量扫掠驱动（多进程并行 + resume）
├── static_post.py         # 后处理：读 .h5 → Cl/Cd（来流系力分解）
└── sst_run\               # 稳态 SST 全量结果（case 目录 + 汇总 CSV + 图）
```

## 依赖与路径

工程**自包含但不复制数据**，通过路径引用动态工程与网格目录。换机器只需改 `config.py` 顶部：

| 配置项 | 含义 | 默认值 |
|---|---|---|
| `FLUENT_EXE` | Fluent 可执行文件 | `E:\software\ANSYS 2025R1\...\fluent.exe` |
| `DYNAMIC_TOOL_ROOT` | 动态工程根（参考 case + fluent_auto 后处理） | `E:\my data\...\dynamic_airfoil_tool` |
| `MESH_DIR` | 翼型网格目录（`llm_0p4572.msh` / `naca4415_0p4572.msh`） | `D:\LLM CFD Data\meshfiles` |
| `OUT_DIR` | 输出根目录 | `static_airfoil_tools\outputs` |

需要 Python + `h5py`、`numpy`、`matplotlib`。

## Re 档配置

`config.RE_TAGS` 把 Re 档映射到动态工程的风洞工况（`batch_auto_run.wt_air_properties` 取物性，勿硬编码数值）：

| Re 标签 | 对应工况 | rho (kg/m³) | mu (Pa·s) | V (m/s) |
|---|---|---|---|---|
| `0p75` | R1 (0.75M) | ~1.2528 | ~1.715e-5 | 22.46 |
| `1p0` | R4 (1.0M) | ~1.2528 | ~1.715e-5 | 29.94 |
| `1p25` | R7 (1.25M) | ~1.2529 | ~1.715e-5 | 37.42 |
| `1p5` | R10 (1.5M) | ~1.2534 | ~1.714e-5 | 44.87 |

湍流入口固定：**TI=5%，Lt/c=0.20**（与动态批次一致）。

## 单个静态 case 的完整流程

每个 (翼型, Re, 攻角) 是一个独立 case 目录 `case_{af}_S{re}_a{aoa:02d}/`：

1. **`static_case.build_static_case()`** —— 从参考 NACA4415 `.cas` 文本生成稳态 case：
   - 烘焙物性（密度/粘度/入口 vmag 按 Re）
   - 入口湍流强度/长度尺度 → TI / Lt
   - 去掉 zone-motion UDF 绑定名（`airfoil_pitch::libudf` → `"none"`，稳态下网格静止）
   - 切稳态：`case-config` 块 `rp-unsteady? #t→#f`、`rp-dual-time? 2→0`
   - autosave 频率置 0（稳态不需要逐步 dat）
2. **网格副本**：把目标翼型 `.msh` 复制进 case 目录（并发 replace-mesh 读同文件会竞争）
3. **`static_journal.gen_static_journal()`** 生成 journal：
   ```
   /file/read-case <case.cas>
   /file/replace-mesh <local.msh>          ; 换入目标翼型网格
   /define/models/viscous/kw-sst           ; 标准 SST k-ω
   yes
   /define/models/viscous/transition-sst   ; 关闭转捩
   no
   ; ---- TUI 设入口方向（Magnitude and Direction）----
   /define/boundary-conditions/set/velocity-inlet
   inlet
   <blank>
   velocity-spec → yes → vmag → no → <V> → direction-0 → no → <cosβ>
   → direction-1 → no → <sinβ> → q
   /solve/initialize/hyb-initialization
   /solve/iterate 400                      ; 稳态迭代（多数提前收敛）
   /file/write-case-data <base>-static.cas.h5
   /exit
   ```
   **关键教训**：velocity-inlet 的方向在 Fluent 25 中**必须用 TUI 设置**（case 文本里改 `velocity-spec`/`flow-direction` 实测无效，流场仍水平）。
4. **运行**：`fluent 2ddp -g -t{nproc} -i <journal>`（独立 TEMP 目录防并发竞争；stdout 进 .trn）
5. **`static_post.static_polar_read()`** 后处理：
   - `set_case_params(rho, V, mu, ...)` 设 Q_REF
   - 从 `*-static.cas.h5` 解析翼型壁面几何（`parse_zone_topology` / `build_static_geometry` / `find_wall_offset`）
   - `compute_forces()` 得 global 系合力
   - **来流系力分解**：`Cl = -cd_g·sinβ + cl_g·cosβ`，`Cd = cd_g·cosβ + cl_g·sinβ`
     （斜来流下升力/阻力须相对来流方向定义，即旋转到"阻力//来流、升力⊥来流"的坐标系）
6. 写 `case_params.json` + `result_summary.json`（含 cl/cd/elapsed_s 等）

## 复现论文静态极曲线（llm + naca4415，4 Re × 0-30°）

本工程随复现包自带网格（`mesh/`）与动态工程代码（参考 case / 物性换算来源）。复现论文静态结果只需两步：

```bash
# 1) 改本工程 config.py：FLUENT_EXE 本机路径；DYNAMIC_TOOL_ROOT 指向复现包
#    src/dynamic_airfoil_tool；MESH_DIR 指向复现包 mesh/
# 2) 全量复现（llm + naca4415 × 4 Re × 16 角 = 128 case，断点续跑）
python static_polar_batch.py \
    --airfoils llm naca4415 \
    --re 0p75 1p0 1p25 1p5 \
    --angles 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 \
    --n-iter 400 \
    --parallel 8 --nproc 4 \
    --out-dir sst_run
```

- `--airfoils`/`--re`/`--angles` 可任意裁剪（如只跑某翼型某 Re 的 1-2 角快速验证）
- 同命令重跑即断点续跑（已完成 case 跳过）
- 结果聚合：`--out-dir` 下生成 `polar_{af}_Re{tag}.csv` 与图，用 `plot_*.py` 可重画论文对比图

## 批量运行

```bash
# llm 和 naca4415, 4 个 Re, 0-30° 每 2°（16 点/翼型/Re）
python static_polar_batch.py \
    --airfoils llm naca4415 \
    --re 0p75 1p0 1p25 1p5 \
    --angles 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 \
    --n-iter 400 \
    --parallel 8 --nproc 4 \
    --out-dir sst_run
```

CLI 参数：
| 参数 | 说明 |
|---|---|
| `--airfoils` | 翼型列表（默认 `llm naca4415`） |
| `--re` | Re 标签列表（默认全部 4 档） |
| `--angles` | 攻角列表（如 `0 2 4 ...`） |
| `--angles-range lo hi` | 连续攻角范围（每 1°） |
| `--n-iter` | 稳态迭代上限（默认 400；SST 收敛快，多数 <100 提前停） |
| `--parallel N` | 并发 worker 数（8 = 8 个 Fluent 并行，配 `--nproc 4` 吃满 32 线程） |
| `--nproc` | 每个 Fluent 核数（默认 4） |
| `--out-dir` | 输出目录 |
| `--force` | 忽略已完成全部重跑 |
| `--dry-run` | 只打印任务清单不求解 |
| `--run-one <json>` | worker 单点模式（内部用） |

- **断点续跑**：已完成 case（有 `result_summary.json` 且 status=done）自动跳过，重跑同命令只补缺失点
- **并行**：multiprocessing spawn 池，每 worker 内跑 Fluent 子进程，互不干扰

## 输出

每个 case 目录：`*.cas`（稳态 case 文本）、`*.jou`、`*-static.cas.h5/.dat.h5`（最终结果）、`case_params.json`、`result_summary.json`（含 cl/cd/cd_pres/mesh_aoa/elapsed_s）。

`assemble()` 聚合生成（在 `--out-dir`）：
- `polar_{af}_Re{tag}.csv` —— 每翼型每 Re 极曲线（aoa, cl, cd, cd_pres）
- 汇总 CSV：`{af}_4Re_mean_polar_clean.csv`（4-Re 平均极曲线 aoa_deg, Cl, Cd）、`llm_vs_naca4415_4Re_mean.csv`（对比）
- 图：`llm_Re0p75_SST.png`、`llm_multiRe_summary.png`、`llm_vs_naca4415_4Re_mean.png` 等

## 常见坑（实测）

1. **入口方向必须在 journal 用 TUI 设**，改 case 文本的 velocity-spec/flow-direction 无效
2. **case 文本不能动 motion-spec 或清空 udf/compile/files** —— 会导致 journal 中途 "error occurred while reading the journal" 中断；只需把 UDF 绑定名改 `"none"`
3. **稳态 Cd 需来流系分解**（见 static_post），否则斜来流下 Cd 出负值
4. **并发 replace-mesh**：每 case 必须复制自己的网格副本（同文件并发读会 CAR 崩溃）
5. **Flutter 僵尸进程**：中断后需清理 `cx2510`/`fl_mpi2510`（`taskkill /F /T`），否则残留 .trn 句柄锁死文件
