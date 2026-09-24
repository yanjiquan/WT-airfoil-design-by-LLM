# BL_code — 源码目录说明

Beddoes-Leishman 动态失速模型（Leishman-Crouse 1989）的标定 / 预测 / 不确定性区间生成源码。
本目录需与 `../datasets/`、`../cfd_llm_predict.jsonl` 并列；从包根目录调用（见上级 README.md）。

## 脚本

| 文件 | 功能 | 主要输入 | 主要输出 |
|---|---|---|---|
| `bl_model.py` | B-L 状态空间模型（附着流 indicial / 失速起始 / 后缘分离 Kirchhoff / 前缘涡 / 弦向力 → CL,CD）。`python bl_model.py` 自测 | 攻角时间序列 α(t), dα/dt | CL/CD/CN/Cc + 内部状态（Cnprime, fpp, Cnv） |
| `bl_data.py` | 解析 jsonl → (翼型×工况) 序列：正则提取工况/攻角/趋势，正弦反推相位，多周期同(攻角,趋势)取均值 | train/test/low/high.jsonl | `build_sequences()` → 序列 dict |
| `bl_train.py` | 全局参数标定（least_squares 多起点，bounds） | train.jsonl | `bl_params.json` |
| `bl_train_perwing.py` | per-wing 标定（每翼型用自身 low+high 数据） | low/high.jsonl | `bl_params_{翼型}.json`（11 套） |
| `bl_draw_uncertainty.py` | MC 不确定性图 + 覆盖率（对齐 cfd_llm_predict.jsonl 口径；支持分趋势 CL bias 修正、per-wing/全局参数） | cfd_llm_predict.jsonl + 参数文件 | 6 张 PNG + `95%CI_BL_mc.csv` + `pred_bl_mc.jsonl` |
| `bl_pert_justify.py` | 扰动幅度（PERT）多角度数据论证（覆盖率-扰动扫描 / Jacobian 诊断 / 噪声下限） | PW_15pct/pred_bl_mc.jsonl | 控制台证据表 |
| `bl_cst2param.py` | CST 几何 → B-L 参数映射实验（Ridge + Leave-one-out，需 scikit-learn） | cst_params.json + bl_params_{翼型}.json | 控制台诊断 |
| `bl_eval.py` | （历史）test 自工况覆盖率评估 | test.jsonl | curve_*_BL_*.png / 95%CI_BL.csv |
| `bl_predict_test.py` | （历史）本地 FP8 LLM 预测复跑对比（与 B-L 无关，需本地推理服务） | predictions.jsonl | local_fp8_predictions.jsonl |

## 参数文件

| 文件 | 说明 |
|---|---|
| `bl_params.json` | 全局标定（11 键 + dalpha1=0.04 / eta=0.965 / D_f=8.0） |
| `bl_params_{LS-0417,LS-0421,NACA4415,S801,S809,S810,S812,S813,S814,S815,S825}.json` | per-wing 标定（11 键） |

## 不确定性区间机制（实现口径）

- 区间生成：参数扰动蒙特卡洛 `θ' = θ + N(0, PERT·|θ| + 1e-4)`，对 11 个标定参数各采样 N_MC=40 次，
  每次完整跑一次 `BLModel.simulate`，取同攻角/趋势点样本 → `[P2.5, P97.5]`；SEED=0 可复现。
- PERT 由命令行 `--pert` 传入（README 示例为 0.15）；`bl_pert_justify.py` 提供该取值的多角度数据论证。
- 覆盖率口径：CFD 真值落入对应相位 bin 预测区间 [P2.5, P97.5] 的比例（逐 bin / 逐点两套实现，
  分别见 `bl_eval.py` / `bl_draw_uncertainty.py`）。
- bias 修正（可选，`--nobias` 关闭）：分趋势按攻角 2° 分箱拟合 `E[CFD − B-L]`（数据来自 train 序列），
  线性插值，仅作用于 CL。

## 标定参数（bl_params.json 当前值）

| 参数 | 值 | 物理含义 |
|---|---|---|
| C_Nalpha | 4.082 | 法向力线斜率 1/rad |
| alpha0 | −0.050 | 零升攻角 rad |
| alpha1 | 0.275 | 分离起始攻角 rad |
| S1 / S2 | 0.0425 / 0.0600 | 失速特性系数 |
| Cd0 | 0.0177 | 零升阻力 |
| C_N1 | 1.381 | 临界法向力系数（涡起始） |
| T_P / T_f / T_v / T_vl | 0.53 / 1.22 / 2.12 / 3.24 | 动态时间常数 |
| dalpha1 / eta / D_f | 0.04 / 0.965 / 8.0 | 固定默认（不参与标定） |

## 依赖与运行

- numpy / scipy / matplotlib（bl_cst2param.py 另需 scikit-learn）
- 路径相对本目录：数据 = `../datasets/`，对比基准 = `../cfd_llm_predict.jsonl`
- Windows 控制台中文乱码时可设 `PYTHONIOENCODING=utf-8`
