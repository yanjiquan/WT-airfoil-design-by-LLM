# -*- coding: utf-8 -*-
"""生成 Fluent TUI journal，完成 NACA4415 D1T1 完整仿真流程。

方案: 直接读取 build_case.py 生成的 D1T1 case（含网格+transition SST+
滑移界面+zone motion 合并 UDF 绑定 airfoil_motion+修改后的物性/BC）。
journal 只需编译 UDF + 初始化 + 按时间推进。

流程（用户敲定，由合并 UDF 的 time 门控实现）:
    阶段1: 复位到 AOA_START (dt=0.05s, 20 步, t:0->1s)  -> 保存 -9-00000
    阶段2: 稳定流场 (dt=1e-6s, 50 步, t:1->1.00005s)
    阶段3: 正弦俯仰 11 周期 (dt=0.05s, 363 步, 每周期保存 -10-0000N)

运行: fluent 2ddp -g -i <journal>
"""

from pathlib import Path

import config


def p(path: Path | str) -> str:
    """转成正斜杠路径供 Fluent 使用。"""
    return str(path).replace("\\", "/")


def gen_journal() -> str:
    case_file = config.OUT_DIR / f"{config.BASE_NAME}_{config.D_CASE}{config.T_CASE}.cas"
    out_dir = config.OUT_DIR
    base = config.BASE_NAME
    # 目标翼型网格（build_case 生成的 case 内嵌 NACA4415 网格，
    # 需替换为当前翼型网格）
    mesh_file = config.MESH_DIR / config.MESH_NAME
    need_replace = config.AIRFOIL not in ("NACA4415", "N4415")

    # 阶段3 每周期迭代与保存（与 S801 一致：每周期连续推进 + 显式保存 1 次；
    # 每个时间步的 dat 由 autosave data-frequency=1 自动保存，后处理取密集数据）。
    # 注意: 瞬态推进必须用 /solve/dual-time-iterate N M（N 个时间步，
    # 每步最多 M 次内迭代）；/solve/iterate 只迭代不推进时间步。
    pitch_loop = []
    for i in range(1, config.PITCH_CYCLES + 1):
        pitch_loop.append(f"/solve/dual-time-iterate {config.STEPS_PER_CYCLE} {config.ITER_PER_STEP}")
        pitch_loop.append(f'/file/write-case-data "{p(out_dir)}/{base}-10-{i:05d}.cas.h5"')

    # 设置时间步的固定命令（v251 需先 fixed-user-specified yes 再 time-step-size）
    def set_dt(dt: float) -> str:
        return (f"/solve/set/transient-controls/fixed-user-specified yes\n"
                f"/solve/set/transient-controls/time-step-size {dt}")

    # 稳定阶段：瞬态微步 ramp 建立初始流场（不切稳态）。
    # 重要：切稳态求解会把 zone motion 替换成 Frame Motion（网格不动），切回
    # 瞬态后不恢复 Mesh Motion，导致翼型不做俯仰运动（实测 S810 攻角恒 0.036°）。
    # 故对动态网格 case 一律用瞬态微步稳定；S801 用此方案俯仰攻角变化正常。
    # 微步共 50 步、内迭代 20 即足够（微步物理时间极短，主要是数值预热）。
    stab_ramp = [(1e-9, 10), (1e-8, 15), (1e-7, 25)]   # 共 50 步
    stab_lines = [set_dt(dt_i) + f"\n/solve/dual-time-iterate {n_i} 20"
                  for dt_i, n_i in stab_ramp]
    stab_block = "\n".join(stab_lines)

    # 复位阶段：dt=0.05 x 20 步，总 1.0s（UDF 门控 RESET_TIME=1.0），与 S801 一致。
    reset_block = (set_dt(config.RESET_DT) +
                   f"\n/solve/dual-time-iterate {config.RESET_STEPS} {config.ITER_PER_STEP}")

    return f"""\
; ============================================================
; Fluent TUI journal - {config.CASE_ID} ({config.AIRFOIL})
; 动态失速俯仰 CFD: transition SST + 滑移网格 + DEFINE_ZONE_MOTION
; 基于参考工程 case 修改生成, 合并 UDF 完成 复位->俯仰
; 运行: fluent 2ddp -g -i {config.CASE_ID}.jou
; ============================================================

; ------------------------------------------------------------
; 1. 读 case（含 NACA4415 网格+模型+BC+zone motion 绑定）
; ------------------------------------------------------------
/file/read-case "{p(case_file)}"
; 禁用覆盖确认：write-case-data 遇已存在文件时不提示，避免 journal 中断
/file/confirm-overwrite?
no
; （autosave 已由 build_case 在 case 文本中禁用：mmp/autosave/frequency/case,data 置 0，
;   否则稳态 iterate 每迭代写一个 dat 文件，极慢且占满磁盘）
; ------------------------------------------------------------
; 1b. 替换网格为当前翼型（保留全部求解设置）
;     所有 11 个翼型网格同批生成、拓扑一致，replace-mesh 可行。
; ------------------------------------------------------------
{"/file/replace-mesh \"%s\"" % p(mesh_file) if need_replace else "; NACA4415 无需替换网格"}

; ------------------------------------------------------------
; 2. UDF 编译/加载
;    case 读入时检测到 udf/compile/files 会自动编译并加载 libudf
;    （含 airfoil_motion / airfoil_reset / airfoil_pitch）。
;    注意: 不要在 journal 里手动 compile（交互式多行会中断 journal）。

; ------------------------------------------------------------
; 3. 初始化（hybrid；FMG 仅稳态可用，动态网格 case 不切稳态故不用）
; ------------------------------------------------------------
/solve/initialize/hyb-initialization

; （湍流限制器暂不启用：Fluent 25 的 /solve/set/limits/ 路径无效会中断 journal，
;   正确路径为 /solution/controls/limits/，待需要时再启用）

; ------------------------------------------------------------
; 4. 时间推进
; ------------------------------------------------------------
; 阶段1: 稳定起始流场。瞬态微步 ramp（1e-9 -> 1e-8 -> 1e-7）预热初始流场，
;        不切稳态（切稳态会把 zone motion 变 Frame Motion 导致翼型不动）。
;        若 hyb 初始化后湍流变量异常（k/omega<=0）致湍流求解 log2f 崩溃，
;        可先调小首段步长或增大内迭代让湍流先稳定。
{stab_block}

; 阶段2: 复位到 AOA_START={config.AOA_START} deg（dt=0.05 x 20 步，总 1.0s，与 S801 一致）
{reset_block}
/file/write-case-data "{p(out_dir)}/{base}-9-00000.cas.h5"

; 阶段3: 正弦俯仰 {config.PITCH_CYCLES} 周期 (dt={config.PITCH_DT}s, 每周期 {config.STEPS_PER_CYCLE} 步)
{set_dt(config.PITCH_DT)}
{chr(10).join(pitch_loop)}

; ------------------------------------------------------------
; 5. 保存最终 case/data
; ------------------------------------------------------------
/file/write-case-data "{p(out_dir)}/{base}-final.cas.h5"

; 结束
/exit
yes
"""


def main() -> None:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.JOB_DIR.mkdir(parents=True, exist_ok=True)
    config.UDF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    jou = gen_journal()
    path = config.JOB_DIR / f"{config.CASE_ID}.jou"
    path.write_text(jou, encoding="utf-8")
    print(f"已生成 journal: {path}")
    print(f"journal 长度: {len(jou.splitlines())} 行")


if __name__ == "__main__":
    main()
