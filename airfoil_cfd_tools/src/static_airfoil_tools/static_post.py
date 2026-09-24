# -*- coding: utf-8 -*-
"""静态攻角 case 的后处理：读取最终 *-static.cas.h5 / .dat.h5 求 Cl/Cd。

复用 dynamic 工程 fluent_auto/compare_sim_auto.py 的力提取：
  set_case_params(rho, vel, mu, freq, ao0, ao1)   # 设 Q_REF
  parse_zone_topology(cas_h5)                     # 翼型壁面 face 范围 (zmin,zmax)
  build_static_geometry(cas_h5, zmin, zmax)       # 当前网格的 face 形心/法向/长度
  find_wall_offset(cas_h5, zmin)
  compute_forces(dat_h5, aoa, zmin,zmax, ...)     # -> (flow_time, aoa, cl, cd_tot, cd_pres)

静态 case 几何固定，几何只解析一次（用 *-static.cas.h5）。aoa_deg 仅作标注，
不影响力（法向已携带几何朝向）。
"""

import glob
import math
import os
import sys
from pathlib import Path

import config

# 复用 dynamic 工程的 fluent_auto
if str(config.FLUENT_AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(config.FLUENT_AUTO_DIR))


def static_polar_read(case_dir: Path, target_aoa: float, rho: float,
                      V: float, mu: float, cas_h5: Path | None = None,
                      freq: float = 1.0, initial_aoa: float = 0.0) -> dict:
    """读取单个静态 case 的 Cl/Cd。

    静态 case 来流相对 x 轴倾斜 β = target_aoa + initial_aoa，而
    compare_sim_auto.compute_forces 把 global -x 当阻力、global +y 当升力
    （对水平来流的动态 case 成立）。斜来流必须把合力旋转回**来流坐标系**：
      x' = 来流方向（阻力）, y' = 垂直来流（升力）
    否则会把升力的水平分量错记成阻力（实测 α=8° 时 Cd 错误为负）。
    返回 dict(aoa, cl, cd, cd_pres, mesh_aoa?)。若数据缺失抛错。
    """
    import compare_sim_auto as csa
    import numpy as np

    # 1. 设 Q_REF 等（freq 需为正占位；ao1=0 静态）
    csa.set_case_params(rho=rho, vel=V, mu=mu, freq=freq,
                        ao0=target_aoa, ao1=0.0)
    beta = math.radians(target_aoa + initial_aoa)   # 来流倾角

    case_dir = Path(case_dir)

    # 2. 选几何参考 cas：显式 cas_h5 > 首 glob 的 *-static.cas.h5
    if cas_h5 is None:
        candidates = sorted(glob.glob(str(case_dir / "*-static.cas.h5")))
        if not candidates:
            # 回退: 任一 *.cas.h5
            candidates = sorted(glob.glob(str(case_dir / "*.cas.h5")))
        if not candidates:
            raise FileNotFoundError(f"{case_dir} 无 *-static.cas.h5")
        cas_ref = candidates[0]
    else:
        cas_ref = str(cas_h5)
    # 对应 dat
    dat_ref = cas_ref.replace(".cas.h5", ".dat.h5")
    if not Path(dat_ref).exists():
        # dat 可能命名不同，取同基名 .dat.h5
        dats = sorted(glob.glob(str(case_dir / "*.dat.h5")))
        if not dats:
            raise FileNotFoundError(f"{case_dir} 无 .dat.h5")
        dat_ref = dats[0]

    # 3. 几何（静态，解析一次）
    zmin, zmax = csa.parse_zone_topology(cas_ref)
    centroids, normals, lengths = csa.build_static_geometry(cas_ref, zmin, zmax)
    wall_offset = csa.find_wall_offset(cas_ref, zmin)

    # 4. 力（compute_forces 返回 global 系系数: cl=F_y/Q, cd=F_x/Q）
    r = csa.compute_forces(dat_ref, target_aoa, zmin, zmax,
                           centroids, normals, lengths, wall_offset)
    if r is None:
        raise RuntimeError(f"compute_forces 失败: {dat_ref}")
    _ft, _aoa, cl_g, cd_g, cdp_g = r   # global 轴系数

    # 4b. 旋转到来流系（阻力//来流, 升力⊥来流）。
    #    compute_forces: F = (cd_g*Q, cl_g*Q) 是 global (x,y) 分量。
    #    来流方向单位向量 e_x'=(cosβ, sinβ), 垂直 e_y'=(-sinβ, cosβ)。
    #    L = F·e_y', D = F·e_x'（忽略剪应力在 D 的微小差异用同一旋转）。
    cd_tot =  cd_g * math.cos(beta) + cl_g * math.sin(beta)   # D = Fx cosβ + Fy sinβ
    cl     = -cd_g * math.sin(beta) + cl_g * math.cos(beta)   # L = -Fx sinβ + Fy cosβ
    cd_pres = cdp_g * math.cos(beta) + cl_g * math.sin(beta)
    # （验证: β=8.236°, cl_g=0.86(实为Fy主导), cd_g=-0.061(负因 Fx<0) →
    #   cd_tot = -0.061*0.99 + 0.86*0.143 = +0.063 ✓ 正阻力; cl 基本不变 ✓）

    # 5. (可选) mesh 实测攻角校验
    mesh_aoa = None
    try:
        le_id, te_id = csa.find_airfoil_le_te_node_ids(cas_ref, zmin, zmax)
        mesh_aoa = float(csa.mesh_aoa_from_cas(cas_ref, le_id, te_id))
    except Exception:
        pass

    return dict(aoa=float(target_aoa), cl=float(cl), cd=float(cd_tot),
                cd_pres=float(cd_pres), mesh_aoa=mesh_aoa,
                cas_h5=cas_ref, dat_h5=dat_ref)


def main() -> None:
    """CLI: 重跑某 case 的后处理。"""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="静态 case 后处理")
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--aoa", type=float, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--v", type=float, required=True)
    ap.add_argument("--mu", type=float, required=True)
    args = ap.parse_args()
    res = static_polar_read(Path(args.case_dir), args.aoa,
                            args.rho, args.v, args.mu)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
