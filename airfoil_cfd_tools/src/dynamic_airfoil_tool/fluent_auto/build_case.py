# -*- coding: utf-8 -*-
"""从参考工程 .cas 文本生成 D1T1 专属 case。

策略: 参考 .cas 是验证过的完整 Fluent case（网格+transition SST+
滑移界面+动态网格 UDF 绑定+BC）。直接修改文本中的少量字段，
得到 NACA4415 D1T1 的可运行 case，避免脆弱的交互式 TUI。

修改点（均为位置锚定，避免误替换）:
  1. air 材料 密度 1.2508 -> config.RHO
  2. air 材料 粘度 1.717e-05 -> config.MU
  3. inlet  速度 vmag 30.358 -> config.V_INF
  4. inlet  湍流强度 0.025 -> config.TURB_INTENSITY_PCT/100
  5. inlet  长度尺度 0.5 -> config.TURB_LENGTH_SCALE
  6. 操作压力/时间步保持不变（阶段2起于 1e-6s）
"""

import re
from pathlib import Path

import config

REF_CAS = config.REF_PROJECT / f"{config.BASE_NAME}.cas"


def p(path: Path | str) -> str:
    """转成正斜杠路径供 Fluent case 文本使用。"""
    return str(path).replace("\\", "/")


def _replace_anchored(text: str, pattern: str, new_value: str,
                      anchor: str, occurrence: int = 0) -> str:
    """在 anchor 之后第 occurrence 处替换 pattern 匹配（仅替换模式本身）。"""
    a_idx = text.find(anchor)
    if a_idx == -1:
        raise ValueError(f"锚点未找到: {anchor}")
    pat = re.compile(pattern)
    segment = text[a_idx:]
    found = list(pat.finditer(segment))
    if not found:
        raise ValueError(f"在锚点 '{anchor}' 后未找到模式: {pattern}")
    if occurrence >= len(found):
        raise ValueError(f"锚点 '{anchor}' 后模式只出现 {len(found)} 次, 无法取第 {occurrence} 次")
    m = found[occurrence]
    s = a_idx + m.start()
    e = a_idx + m.end()
    return text[:s] + new_value + text[e:]


def build_case() -> tuple[Path, list[str]]:
    """生成 D1T1 case 文件，返回 (输出路径, 修改日志)。"""
    if not REF_CAS.exists():
        raise FileNotFoundError(f"参考 case 不存在: {REF_CAS}")
    # 注意: 必须禁用换行转换 (newline='')，否则 Windows 下 \n -> \r\n，
    # Fluent 文本 case 解析会失败（zone 6 读取错误）。
    text = REF_CAS.read_text(encoding="latin-1", newline="")

    changes: list[str] = []

    # 1. 材料密度：只替换数值，保留原括号结构 (constant . X) (compressible-liquid ...)
    old = r"\(density \(constant \. 1\.2508\)"
    new = f"(density (constant . {config.RHO:.5f})"
    text = _replace_anchored(text, old, new, "air fluid", occurrence=0)
    changes.append(f"密度: 1.2508 -> {config.RHO:.5f}")

    # 2. 材料粘度：只替换数值
    old = r"\(viscosity \(constant \. 1\.717e-05\)"
    new = f"(viscosity (constant . {config.MU:.6e})"
    text = _replace_anchored(text, old, new, "air fluid", occurrence=0)
    changes.append(f"粘度: 1.717e-05 -> {config.MU:.6e}")

    # 3. 入口速度：只替换数值
    old = r"\(vmag \(constant \. 30\.358\)"
    new = f"(vmag (constant . {config.V_INF:.4f})"
    text = _replace_anchored(text, old, new, "velocity-inlet inlet", occurrence=0)
    changes.append(f"入口速度: 30.358 -> {config.V_INF:.4f} m/s")

    # 4. 入口湍流强度：只替换数值
    old = r"\(turb-intensity \. 0\.025\)"
    new = f"(turb-intensity . {config.TURB_INTENSITY_PCT / 100.0:.3f})"
    text = _replace_anchored(text, old, new, "velocity-inlet inlet", occurrence=0)
    changes.append(f"入口湍流强度: 0.025 -> {config.TURB_INTENSITY_PCT / 100.0:.3f}")

    # 5. 入口长度尺度：只替换数值
    old = r"\(turb-length-scale \. 0\.5\)"
    new = f"(turb-length-scale . {config.TURB_LENGTH_SCALE:.5f})"
    text = _replace_anchored(text, old, new, "velocity-inlet inlet", occurrence=0)
    changes.append(f"入口长度尺度: 0.5 -> {config.TURB_LENGTH_SCALE:.5f}")

    # 6. 区域运动 UDF 绑定 -> 合并 UDF airfoil_motion（含复位+俯仰）
    #    参考绑定 "airfoil_pitch::libudf" 出现于 udf-zmotion-name 和
    #    mgrid-udf-zmotion-name 两处，均需替换。
    n_old = text.count('"airfoil_pitch::libudf"')
    text = text.replace('"airfoil_pitch::libudf"', '"airfoil_motion::libudf"')
    changes.append(f"区域运动 UDF: airfoil_pitch -> airfoil_motion (替换 {n_old} 处, 合并复位+俯仰)")

    # 7. UDF 编译源文件 -> 指向 auto_dynamic/udf 下的源文件
    #    参考: (udf/compile/files (("libudf" (source "E:/.../rotate_sine_k0p087.c" "E:/.../reset_ang.c") (header))))
    srcs = ' '.join(f'"{p(u)}"' for u in
                    [config.UDF_OUT_DIR / "airfoil_motion.c",
                     config.UDF_OUT_DIR / "reset_ang.c",
                     config.UDF_OUT_DIR / "rotate_sine.c"])
    new_compile = f'(udf/compile/files (("libudf" (source {srcs}) (header))))'
    old_compile = r'\(udf/compile/files \(\(\"libudf\" \(source [^\n]*?\) \(header\)\)\)\)'
    text = re.sub(old_compile, new_compile, text, count=1)
    changes.append(f"UDF 编译源: 指向 auto_dynamic/udf (airfoil_motion+reset+rotate)")

    # 8. autosave filename -> 输出目录（否则 Fluent 自动保存写到参考工程目录）
    #    参考: (autosave/filename "E:\\...\\Fluent\\NACA4415_0p4572_0p01_inflat")
    autosave_base = p(config.OUT_DIR / config.BASE_NAME)
    old_auto = r'\(autosave/filename "([^"]*)"\)'
    new_auto = f'(autosave/filename "{autosave_base}")'
    text = re.sub(old_auto, new_auto, text, count=1)
    changes.append(f"autosave 路径: -> {autosave_base}")

    # 8b. autosave：保持 data-frequency=1（每时间步保存 dat，与 S801 一致，
    #     后处理需要密集数据点）；case-frequency 置 0（不每步写 case，省磁盘）。
    #     注意 Fluent 25 case 有两套字段：autosave/... 与 mmp/autosave/...，都要处理。
    for key in ("case",):
        text = re.sub(rf"\(autosave/frequency/{key}\s+\d+\)",
                      f"(autosave/frequency/{key} 0)", text)
        text = re.sub(rf"\(mmp/autosave/frequency/{key}\s+\d+\)",
                      f"(mmp/autosave/frequency/{key} 0)", text)
    changes.append("autosave: data-frequency 保留 1（每步保存 dat），case-frequency -> 0")

    # 9. 清理参考 case 中残留的外部绝对路径（species/isat、dynamesh 运动历史），
    #    这些指向原始机器的文件，换机器后失效；重定位到当前工作目录。
    text = re.sub(r'(\(species/isat-file ")[^"]*(")',
                  lambda m: m.group(1) + p(config.OUT_DIR / "species.isat") + m.group(2),
                  text, count=1)
    text = re.sub(r'(\(dynamesh/motion-history/basename ")[^"]*(")',
                  lambda m: m.group(1) + p(config.OUT_DIR / "motion_history") + m.group(2),
                  text, count=1)
    changes.append("清理外部绝对路径: species/isat 与 dynamesh history -> 当前工作目录")

    # 输出
    out_dir = config.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config.BASE_NAME}_{config.D_CASE}{config.T_CASE}.cas"
    out_path.write_text(text, encoding="latin-1", newline="")
    return out_path, changes


def main() -> None:
    path, changes = build_case()
    print(f"已生成 D1T1 case: {path}")
    print("修改项:")
    for c in changes:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
