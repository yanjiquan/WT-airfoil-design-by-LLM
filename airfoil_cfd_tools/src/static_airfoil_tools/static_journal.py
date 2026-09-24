# -*- coding: utf-8 -*-
"""纯稳态静态攻角仿真的 Fluent journal 生成（TUI 设置入口方向）。

流程（每攻角一个 case）:
  1. /file/read-case 读入 static_case.py 生成的稳态 .cas
     （已烘焙目标 Re 物性/BC、无 zone-motion 绑定、rp-unsteady #f、
       autosave 关；vmag 已设为目标 V）
  2. /file/replace-mesh 换入目标翼型网格（llm/naca4415，拓扑同参考）
  3. TUI 设置 velocity-inlet 为 "Magnitude and Direction" 并把方向设为
     β（cosβ, sinβ）—— 用 Fluent 标准 TUI 命令，可靠（实测文本编辑
     velocity-spec/flow-direction 无效，TUI 生效）。
  4. /solve/initialize/hyb-initialization 初始化
  5. 稳态迭代 /solve/iterate N（收敛）
  6. /file/write-case-data 保存 *-static.cas.h5
  7. /exit

实测 TUI 序列（Fluent 25.1, 2026-09-03）:
  /define/boundary-conditions/set/velocity-inlet
  inlet                    # zone 名
                           # (空行 = 完成 zone 选择, 进入 setting)
  velocity-spec            # 选 Velocity Specification Method
  yes                      # 切到 "Magnitude and Direction"
  vmag
  no                       # Use Profile? no
  <V>                      # 速度量值
  direction-0
  no
  <cosβ>
  direction-1
  no
  <sinβ>
  q                        # 退出 BC 设置

β = α_target + initial_aoa（几何补偿：naca4415 +0.236°, llm 0°）。
"""

import math
from pathlib import Path


def p(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _tui_inflow_block(beta_deg: float, V: float) -> str:
    """生成 velocity-inlet TUI 设置块（Magnitude and Direction + β 方向）。"""
    b = math.radians(beta_deg)
    cx, cy = math.cos(b), math.sin(b)
    return f"""; ---- TUI: 设置入口为 Magnitude-and-Direction, 方向 β={beta_deg:.4f}° ----
/define/boundary-conditions/set/velocity-inlet
inlet

velocity-spec
yes
vmag
no
{V:.4f}
direction-0
no
{cx:.6f}
direction-1
no
{cy:.6f}
q
"""


def gen_static_journal(case_file: Path, mesh_file: Path, out_dir: Path,
                       base: str, n_iter: int = 800, V: float = 45.0,
                       beta_deg: float = 8.0,
                       need_replace: bool = True) -> str:
    """生成静态稳态 journal 文本。

    case_file   : static_case.py 生成的 .cas 绝对路径
    mesh_file   : 目标翼型网格 .msh 绝对路径（本地副本）
    out_dir     : 输出目录（写 *-static.cas.h5）
    base        : 文件名基（如 naca4415_S1p5_a08）
    n_iter      : 稳态迭代次数
    V           : 来流速度量值 [m/s]
    beta_deg    : 来流倾角 β = α_target + initial_aoa [deg]
    need_replace: 是否 replace-mesh
    """
    inflow = _tui_inflow_block(beta_deg, V)
    turb = """\
; ---- 湍流模型: 切到标准 SST k-omega（稳态正确做法, 关 transition SST）----
/define/models/viscous/kw-sst
yes
/define/models/viscous/transition-sst
no
"""
    return f"""\
; ============================================================
; Fluent journal - {base}  (静态攻角稳态求解)
; 方法: 网格静止 + 来流倾 β 等效攻角 (TUI 设入口方向)
; 湍流: 标准 SST k-omega (稳态), 非 transition SST
; ============================================================

/file/read-case "{p(case_file)}"
/file/confirm-overwrite?
no
{"/file/replace-mesh \"%s\"" % p(mesh_file) if need_replace else "; 无需替换网格"}

{turb}
{inflow}
/solve/initialize/hyb-initialization

; 稳态迭代（case 已 rp-unsteady #f, /solve/iterate 不推进时间）
/solve/iterate {n_iter}

/file/write-case-data "{p(out_dir)}/{base}-static.cas.h5"

/exit
yes
"""


def main() -> None:
    """CLI 演示: 打印一个 journal（不求解）。"""
    print(gen_static_journal(
        Path("D:/x/out.cas"), Path("D:/x/llm_0p4572.msh"),
        Path("D:/x"), "demo_S1p5_a08", n_iter=800, V=44.87, beta_deg=8.236))


if __name__ == "__main__":
    main()
