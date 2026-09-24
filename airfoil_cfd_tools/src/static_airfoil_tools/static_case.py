# -*- coding: utf-8 -*-
"""从参考 NACA4415 case 文本生成"静态攻角" case。

核心方法: 不旋转翼型，只把入流速度方向改为倾角 β，等效攻角 α = β −
网格几何初始角（弦线相对 x 轴的固有倾斜）。网格/UDF/zone-motion 全部
静止，纯稳态求解。

文本变换（确定性锚点替换，仿 dynamic 工程 build_case.py 思路）:
  1. 材料 密度 / 粘度  -> 目标 Re 的 rho / mu
  2. 入口 vmag         -> 目标 Re 的 V
  3. 入口湍流强度/长度尺度 -> TI / LT_OVER_C
  4. inlet 的 flow-direction-component (1,0,0) -> (cos β, sin β, 0)，共 2 处
     （velocity-inlet 方向在 Fluent 25 case 里出现于两处，都须替换）
  5. 中和 zone-motion: 去掉 "rotate" 区 UDF 绑定（udf-zmotion-name /
     mgrid-udf-zmotion-name 的 ::libudf 引用）并置 motion-spec 0，
     使 case 无任何网格运动（规避切稳态 Frame-Motion 风险）
  6. autosave data-frequency -> 0（稳态不需要每步 dat，防占盘）
  7. 清理外部绝对路径（species.isat / dynamesh history）重定位到输出目录

生成的 .cas 仍内嵌 NACA4415 网格；目标翼型网格由 journal 用
/file/replace-mesh 换入（与动态批次一致，网格拓扑须同参考）。

β 角约定: β = α_target + initial_aoa。
  - 来流相对 x 轴倾 β 后，翼型（弦线本身在 x 轴上方 initial_aoa 处）
    看到的相对来流角 = β − initial_aoa = α_target。
  - naca4415 initial_aoa = +0.236°, llm = 0.0°。
"""

import math
import re
from pathlib import Path

import config


def p(path: Path | str) -> str:
    """转正斜杠路径供 Fluent 文本使用。"""
    return str(path).replace("\\", "/")


def _replace_anchored(text: str, pattern: str, new_value: str,
                      anchor: str, occurrence: int = 0) -> str:
    """在 anchor 之后第 occurrence 处替换 pattern（仅替换匹配本身）。"""
    a_idx = text.find(anchor)
    if a_idx == -1:
        raise ValueError(f"锚点未找到: {anchor}")
    pat = re.compile(pattern)
    found = list(pat.finditer(text[a_idx:]))
    if not found:
        raise ValueError(f"锚点 '{anchor}' 后未找到模式: {pattern}")
    if occurrence >= len(found):
        raise ValueError(f"锚点 '{anchor}' 后模式只出现 {len(found)} 次, 无法取第 {occurrence} 次")
    m = found[occurrence]
    s = a_idx + m.start()
    e = a_idx + m.end()
    return text[:s] + new_value + text[e:]


def _neutralize_motion(text: str, changes: list) -> str:
    """去掉 zone-motion UDF 绑定名，使 case 在稳态下纯静止。

    Fluent 25 case 中 rotate 区有两处 UDF 引用（udf-zmotion-name 与
    mgrid-udf-zmotion-name），值形如 "airfoil_pitch::libudf"。把它们
    改为 "none"。

    注意（实测 2026-09-03）: 切稳态后 Fluent 自动把 zone motion 当
    Frame Motion（网格固定，正是静态所需），UDF 绑定无需额外处理即可。
    但把绑定名置 none 更干净；**不要**动 motion-spec 或清空 compile/files
    —— 实测这两项会让 case 读取后 journal 中断（"error occurred while
    reading the journal"）。
    """
    n_bind = text.count('"airfoil_pitch::libudf"')
    text = text.replace('"airfoil_pitch::libudf"', '"none"')
    changes.append(f"zone-motion UDF 绑定名: -> none ({n_bind} 处, 稳态下网格静止)")
    return text


def build_static_case(airfoil: str, re_tag: str, aoa_deg: float,
                      wtp: dict, out_dir: Path) -> tuple[Path, list[str]]:
    """生成一个静态攻角 case 文本，返回 (输出路径, 修改日志)。

    airfoil: 'llm' / 'naca4415'
    re_tag : '0p75' / '1p0' / '1p5'
    aoa_deg: 目标攻角 [deg]（0-30）
    wtp    : wt_air_properties 返回的 dict（rho/mu/V/T_K/P_PA）
    out_dir: 本 case 的输出目录（含网格副本，journal 读此 .cas）
    """
    if not config.REF_CAS.exists():
        raise FileNotFoundError(f"参考 case 不存在: {config.REF_CAS}")

    # 初始角补偿: β = α_target + initial_aoa
    initial_aoa = _initial_aoa(airfoil)
    beta = aoa_deg + initial_aoa
    bx = math.cos(math.radians(beta))
    by = math.sin(math.radians(beta))

    text = config.REF_CAS.read_text(encoding="latin-1", newline="")
    changes: list[str] = []

    # 1. 材料密度
    text = _replace_anchored(text, r"\(density \(constant \. 1\.2508\)",
                             f"(density (constant . {wtp['rho']:.5f})",
                             "air fluid", occurrence=0)
    changes.append(f"密度 -> {wtp['rho']:.5f}")

    # 2. 材料粘度
    text = _replace_anchored(text, r"\(viscosity \(constant \. 1\.717e-05\)",
                             f"(viscosity (constant . {wtp['mu']:.6e})",
                             "air fluid", occurrence=0)
    changes.append(f"粘度 -> {wtp['mu']:.6e}")

    # 3. 入口速度
    text = _replace_anchored(text, r"\(vmag \(constant \. 30\.358\)",
                             f"(vmag (constant . {wtp['V']:.4f})",
                             "velocity-inlet inlet", occurrence=0)
    changes.append(f"入口速度 -> {wtp['V']:.4f} m/s")

    # 4. 湍流强度 / 长度尺度
    text = _replace_anchored(text, r"\(turb-intensity \. 0\.025\)",
                             f"(turb-intensity . {config.TI:.3f})",
                             "velocity-inlet inlet", occurrence=0)
    changes.append(f"入口湍流强度 -> {config.TI:.3f}")
    text = _replace_anchored(text, r"\(turb-length-scale \. 0\.5\)",
                             f"(turb-length-scale . {config.LT_OVER_C*config.CHORD:.5f})",
                             "velocity-inlet inlet", occurrence=0)
    changes.append(f"入口长度尺度 -> {config.LT_OVER_C*config.CHORD:.5f}")

    # 4b-5. 来流方向: 不在 case 文本里改（velocity-spec/flow-direction 的
    #   文本编辑实测不可靠，改后流场仍水平）。改为在 journal 里用 TUI
    #   命令设置（/define/boundary-conditions/set/velocity-inlet），见
    #   static_journal.py 的 inflow TUI 段。此处仅保留 vmag（量值）。
    #   β = α_target + initial_aoa（几何补偿），由 journal 计算。
    changes.append(f"来流方向: 由 journal TUI 设置 β={beta:.4f}° "
                   f"(cos={bx:.6f}, sin={by:.6f})")

    # 6. 中和 zone motion（仅 UDF 绑定名置 none，稳态下网格静止）
    text = _neutralize_motion(text, changes)

    # 6b. 切稳态: case-config 块中 rp-unsteady? #t -> #f, rp-dual-time? 2 -> 0
    #      （否则 /solve/iterate 仍按瞬态推进时间步）。只替换 case-config 块内
    #      的一次出现（它在文件后部，前面那些是 scheme 预设无关）。
    cc = text.find("(case-config ((")
    if cc == -1:
        raise ValueError("未找到 (case-config 块，无法切稳态")
    # 在 case-config 块内（到其闭合前）替换
    seg = text[cc:]
    seg2 = re.sub(r"\(rp-unsteady\? \. #t\)", "(rp-unsteady? . #f)", seg, count=1)
    seg2 = re.sub(r"\(rp-dual-time\? \. 2\)", "(rp-dual-time? . 0)", seg2, count=1)
    text = text[:cc] + seg2
    changes.append("case-config: rp-unsteady? #t->#f, rp-dual-time? 2->0 (切稳态)")

    # 7. autosave: case 与 data 频率都置 0（稳态无时间步，也不需要每步 dat）
    for key in ("case", "data"):
        text = re.sub(rf"\(autosave/frequency/{key}\s+\d+\)",
                      f"(autosave/frequency/{key} 0)", text)
        text = re.sub(rf"\(mmp/autosave/frequency/{key}\s+\d+\)",
                      f"(mmp/autosave/frequency/{key} 0)", text)
    changes.append("autosave: case & data 频率 -> 0 (稳态)")

    # 8. 清理外部绝对路径
    text = re.sub(r'(\(species/isat-file ")[^"]*(")',
                  lambda m: m.group(1) + p(out_dir / "species.isat") + m.group(2),
                  text, count=1)
    text = re.sub(r'(\(dynamesh/motion-history/basename ")[^"]*(")',
                  lambda m: m.group(1) + p(out_dir / "motion_history") + m.group(2),
                  text, count=1)
    changes.append("清理外部绝对路径 -> 输出目录")

    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{airfoil}_S{re_tag}_a{int(aoa_deg):02d}"
    out_path = out_dir / f"{base}.cas"
    out_path.write_text(text, encoding="latin-1", newline="")
    return out_path, changes


def _initial_aoa(airfoil: str) -> float:
    """网格几何初始攻角（弦线相对 x 轴）。从 dynamic 工程 AF_INITIAL_AOA 取。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dyn_batch_auto", config.BATCH_AUTO_RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AF_INITIAL_AOA.get(airfoil, 0.0)


def main() -> None:
    """CLI 演示: 生成一个 case 并打印修改日志（不求解）。"""
    import argparse
    import json
    import importlib.util
    ap = argparse.ArgumentParser(description="生成静态攻角 case 文本")
    ap.add_argument("--airfoil", default="naca4415")
    ap.add_argument("--re", default="1p5")
    ap.add_argument("--aoa", type=float, default=8.0)
    ap.add_argument("--out", default=str(config.OUT_DIR))
    args = ap.parse_args()

    # 取物性
    spec = importlib.util.spec_from_file_location(
        "dyn_batch_auto", config.BATCH_AUTO_RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wtp = mod.wt_air_properties(config.RE_TAGS[args.re])

    case_dir = Path(args.out) / f"case_{args.airfoil}_S{args.re}_a{int(args.aoa):02d}"
    path, changes = build_static_case(args.airfoil, args.re, args.aoa,
                                      wtp, case_dir)
    print(f"已生成 case: {path}")
    for c in changes:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
