# -*- coding: utf-8 -*-
"""生成 D1T1 工况的 Fluent DEFINE_ZONE_MOTION UDF。

参考模板: UDF/reset_ang.c 与 UDF/rotate_sine_k0p087.c
生成的 UDF 绑定到旋转区域 (rotate)，通过滑移界面与静止域连接。
"""

from pathlib import Path

import config

PI = "3.141592654"


def gen_reset_ang(out: Path) -> Path:
    """静态转动 UDF：把翼型从 0 deg 转到 AOA_START，保持 1s。"""
    # 负号对齐参考工程：负 omega 对应攻角增大
    src = f"""\
#include "udf.h"
#define PI           {PI}
#define ROT_X        {config.ROT_X}
#define ROT_Y        {config.ROT_Y}
#define AOA_START    {config.AOA_START}

DEFINE_ZONE_MOTION(airfoil_reset, omega, axis, origin, velocity, time, dtime)
{{
    origin[0] = ROT_X;
    origin[1] = ROT_Y;
    origin[2] = 0.0;

    axis[0] = 0.0;
    axis[1] = 0.0;
    axis[2] = 1.0;

    velocity[0] = 0.0;
    velocity[1] = 0.0;
    velocity[2] = 0.0;

    if (time < 1.0)
    {{
        *omega = - AOA_START * PI / 180.0;
    }}
    else
    {{
        *omega = 0.0;
    }}
}}
"""
    out.write_text(src, encoding="utf-8")
    return out


def gen_rotate_sine(out: Path) -> Path:
    """正弦俯仰 UDF：alpha(t) = AOA_MEAN + AOA_AMP*sin(2*pi*f*t - pi/2)。"""
    src = f"""\
#include "udf.h"

#define PI           {PI}
#define AOA_MEAN     {config.AOA_MEAN}
#define AOA_AMP      {config.AOA_AMP}
#define FREQ         {config.FREQ}
#define PHASE        (-PI/2)
#define ROT_X        {config.ROT_X}
#define ROT_Y        {config.ROT_Y}

DEFINE_ZONE_MOTION(airfoil_pitch, omega, axis, origin, velocity, time, dtime)
{{
    origin[0] = ROT_X;
    origin[1] = ROT_Y;
    origin[2] = 0.0;

    axis[0] = 0.0;
    axis[1] = 0.0;
    axis[2] = 1.0;

    velocity[0] = 0.0;
    velocity[1] = 0.0;
    velocity[2] = 0.0;

    *omega = - AOA_AMP * 2.0 * PI * FREQ
             * cos(2.0 * PI * FREQ * time + PHASE)
             * (PI / 180.0);
}}
"""
    out.write_text(src, encoding="utf-8")
    return out


def gen_airfoil_motion(out: Path) -> Path:
    """合并 UDF：先复位到 AOA_START，再正弦俯仰（time 门控）。

    t < 1.0s : 从网格初始角 INITIAL_AOA 匀速转到 AOA_START。
               因为 d(alpha)/dt = -omega（正 omega -> alpha 减小），
               复位 omega = (INITIAL_AOA - AOA_START)/1s。
    t >= 1.0s: 正弦俯仰 alpha = AOA_MEAN + AOA_AMP*sin(2πf(t-1)-π/2)
               t=1 处角度 = AOA_MEAN - AOA_AMP = AOA_START，平滑衔接。

    该 UDF 把"先复位后俯仰"合并为一个运动，避免 journal 中途切换
    zone-motion 绑定的脆弱性。
    """
    reset_omega = (config.INITIAL_AOA - config.AOA_START) / 1.0  # deg/s
    src = f"""\
#include "udf.h"

#define PI           {PI}
#define AOA_MEAN     {config.AOA_MEAN}
#define AOA_AMP      {config.AOA_AMP}
#define FREQ         {config.FREQ}
#define PHASE        (-PI/2)
#define AOA_START    {config.AOA_START}
#define RESET_OMEGA  {reset_omega:.6f}
#define RESET_TIME   1.0
#define ROT_X        {config.ROT_X}
#define ROT_Y        {config.ROT_Y}

DEFINE_ZONE_MOTION(airfoil_motion, omega, axis, origin, velocity, time, dtime)
{{
    origin[0] = ROT_X;
    origin[1] = ROT_Y;
    origin[2] = 0.0;

    axis[0] = 0.0;
    axis[1] = 0.0;
    axis[2] = 1.0;

    velocity[0] = 0.0;
    velocity[1] = 0.0;
    velocity[2] = 0.0;

    if (time < RESET_TIME)
    {{
        /* Reset phase: rotate at constant rate from INITIAL_AOA to AOA_START */
        *omega = RESET_OMEGA * PI / 180.0;
    }}
    else
    {{
        /* Pitching phase: sine motion, phase continuing from RESET_TIME */
        *omega = - AOA_AMP * 2.0 * PI * FREQ
                 * cos(2.0 * PI * FREQ * (time - RESET_TIME) + PHASE)
                 * (PI / 180.0);
    }}
}}
"""
    out.write_text(src, encoding="utf-8")
    return out


def main() -> None:
    out_dir = config.UDF_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    reset = gen_reset_ang(out_dir / "reset_ang.c")
    pitch = gen_rotate_sine(out_dir / "rotate_sine.c")
    motion = gen_airfoil_motion(out_dir / "airfoil_motion.c")
    print(f"已生成: {reset}")
    print(f"已生成: {pitch}")
    print(f"已生成: {motion}")


if __name__ == "__main__":
    main()
