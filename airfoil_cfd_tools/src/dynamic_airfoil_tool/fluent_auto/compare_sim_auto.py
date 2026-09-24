#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
compare_sim_dynamic_n4415_k0087.py - Compare DES simulation
            (multi-cycle, reduced frequency k=0.087)
            with NACA4415 dynamic-stall wind-tunnel data
====================================================================
This is the k=0.087 variant of compare_sim_dynamic_n4415.py.

The simulation pitching frequency has been set so that the reduced
frequency matches the wind-tunnel C10h100 (Run 436, mean=14 deg):

    k = pi * f * c / V = 0.087
      -> f = k * V / (pi * c) = 0.087 * 14.6 / pi = 0.4043 Hz
      -> period T = 1 / f = 2.4733 s

This makes the simulation kinematic conditionally-equivalent to the
tunnel run (same k, same Re~1.0e6, same mean=14 deg, same amplitude
10 deg, same surface = clean), so the dynamic-stall hysteresis loops
are directly comparable.

Simulation motion (must match the UDF with FREQ=0.4043):
    alpha(t) = 14 + 10 * sin(2*pi*0.4043*t - pi/2)   [deg]
  -> starts at 4 deg, pitches to 24 deg, period 2.4733 s.

Wind-tunnel data format (n4415 folder, e.g. C10h100_n4415.txt):
  * Each file contains 3 "RUN nnn <mean> degree mean angle" blocks
    (mean AoA = 20 / 14 / 8 deg).
  * Each Run block has ~120 time samples
        Sample No, Time (sec), AOA (deg), Cl, Cdp, Cm
    covering one oscillation cycle.
  * File-name code: [C|G][5|10][l|m|h][75|100|125|150]
  * Drag is Cdp (pressure drag only); use --add-cd0 to add an
    estimated skin-friction Cd0 for total-drag comparison.

This script:
  * Parses the n4415 dynamic file, picks the Run with mean AoA closest
    to 14 deg.
  * Plots the wind-tunnel dynamic hysteresis loop as SCATTER
    (upstroke red triangles, downstroke blue triangles).
  * Overlays the simulation multi-cycle data (all cycles kept).

Usage:
  python compare_sim_dynamic_n4415_k0087.py --sim-dir <k0087_sim_folder> --wt-file "n4415/C10h100_n4415.txt"
  python compare_sim_dynamic_n4415_k0087.py --sim-dir <k0087_sim_folder> --wt-file "n4415/C10h100_n4415.txt" --mean 14 --dt 0.1 --skip-cycles 1
  python compare_sim_dynamic_n4415_k0087.py --sim-dir <k0087_sim_folder> --wt-file "n4415/G10h100_n4415.txt" --add-cd0

Outputs:
  compare_simdyn_<sim>_<wt>.png

Dependencies: h5py numpy matplotlib
====================================================================
"""

import os
import re
import sys
import glob
import math
import argparse

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ==================================================================
# Simulation constants.
# 这些值可在 main() 里根据风洞数据 run 自动覆盖（见 set_case_params），
# 使脚本对任意工况自适应，无需手动修改顶部常量。
# ==================================================================
RHO    = 1.251        # freestream density           [kg/m^3]  (可被 --rho 覆盖)
V_INF  = 30.358         # freestream velocity          [m/s]  (可被 --vel 或风洞 vel 覆盖)
C_REF  = 0.4572          # chord length (2D)            [m]
MU_AIR = 1.717e-5    # dynamic viscosity            [Pa*s]  (可被 --mu 覆盖)

FREQ   = 1.85       # pitching frequency [Hz]  (可被 --freq 或风洞 osc_f 覆盖)
AOA0   = 14.0         # mean AoA                     [deg]  (仅理论攻角用，默认 mesh-measured)
AOA1   = 10.0         # pitching amplitude           [deg]
PHASE  = -np.pi / 2   # phase offset                 [rad]

PERIOD = 1.0 / FREQ
Q_REF  = 0.5 * RHO * V_INF**2 * C_REF   # dynamic pressure * reference area
RE_SIM = RHO * V_INF * C_REF / MU_AIR
K_SIM  = math.pi * FREQ * C_REF / V_INF


def set_case_params(rho=None, vel=None, mu=None, freq=None, ao0=None, ao1=None):
    """覆盖工况参数并更新派生态量（PERIOD/Q_REF/K_SIM 等）。"""
    global RHO, V_INF, MU_AIR, FREQ, AOA0, AOA1, PERIOD, Q_REF, RE_SIM, K_SIM
    if rho  is not None: RHO = rho
    if vel  is not None: V_INF = vel
    if mu   is not None: MU_AIR = mu
    if freq is not None: FREQ = freq
    if ao0  is not None: AOA0 = ao0
    if ao1  is not None: AOA1 = ao1
    PERIOD = 1.0 / FREQ
    Q_REF  = 0.5 * RHO * V_INF**2 * C_REF
    RE_SIM = RHO * V_INF * C_REF / MU_AIR
    K_SIM  = math.pi * FREQ * C_REF / V_INF


def auto_freq_from_aoa(times, aoas):
    """从 mesh-measured AoA 时间序列自动推算俯仰频率（找连续峰值间距中位数）。"""
    if len(times) < 5:
        return None
    aoas = np.asarray(aoas)
    times = np.asarray(times)
    peaks = []
    for i in range(1, len(aoas) - 1):
        if aoas[i] > aoas[i - 1] and aoas[i] >= aoas[i + 1]:
            peaks.append(i)
    if len(peaks) >= 2:
        period = np.median(np.diff(times[peaks]))
        if period > 0:
            return 1.0 / period
    return None


# ==================================================================
# Angle of attack history (theoretical sine, only with --no-mesh-aoa)
# ==================================================================
def aoa_at(t):
    # 匹配 UDF 相位：alpha = AOA0 + AOA1*sin(2πf(t - 复位1s) + PHASE)。
    # 若用 t（漏 -1），攻角与翼型实际运动错位 2πf 相位（D1 f=0.6 → 216°），
    # 导致 Cl/Cd 对攻角错乱（攻角增大 Cl 反降、Cd 反）。
    return AOA0 + AOA1 * np.sin(2.0 * np.pi * FREQ * (t - 1.0) + PHASE)


# ==================================================================
# 1. Zone topology (HDF5)
# ==================================================================
def parse_zone_topology(cas_path):
    with h5py.File(cas_path, "r") as f:
        names_raw = f["meshes/1/faces/zoneTopology/name"][()][0]
        if isinstance(names_raw, bytes):
            names_raw = names_raw.decode("utf-8", errors="replace")
        else:
            names_raw = str(names_raw)
        n_zones = len(f["meshes/1/faces/zoneTopology/id"])
        zone_names = None
        for delim in [";", "|", "\x00"]:
            parts = [n.strip() for n in names_raw.split(delim) if n.strip()]
            if len(parts) == n_zones:
                zone_names = parts
                break
            elif len(parts) > n_zones:
                zone_names = parts[:n_zones]
                break
        if zone_names is None:
            zone_names = [n.strip() for n in names_raw.replace("\x00", ";").split(";")
                          if n.strip()]
            if len(zone_names) < n_zones:
                zone_names += [f"zone_{i}" for i in range(len(zone_names), n_zones)]
            zone_names = zone_names[:n_zones]

        airfoil_idx = None
        for i, name in enumerate(zone_names):
            if "airfoil" in name.lower() or "blade" in name.lower():
                airfoil_idx = i
                break
        if airfoil_idx is None:
            for i in range(n_zones):
                if f["meshes/1/faces/zoneTopology/zoneType"][i] == 3:
                    airfoil_idx = i
                    break
        if airfoil_idx is None:
            print(f"  ERROR: cannot find airfoil zone.")
            sys.exit(1)
        zmin = int(f["meshes/1/faces/zoneTopology/minId"][airfoil_idx])
        zmax = int(f["meshes/1/faces/zoneTopology/maxId"][airfoil_idx])
        return zmin, zmax


# ==================================================================
# 2. Static face geometry (HDF5)
# ==================================================================
def _node_coords(f):
    """读取节点坐标，自动适配不同网格的 coords dataset 编号（如 5 / 31）。

    完整网格(ICEM/Pointwise)的坐标在 coords/5，Prime mesh 生成的网格可能在
    coords/31 等。硬编码 /5 会导致 Prime mesh 数据读不到（GUI 实时图无数据）。
    """
    grp = f["meshes/1/nodes/coords"]
    for key in grp:
        ds = grp[key]
        # 坐标可能是 3D (N,3) 或 2D (N,2)，后续只用前两列 (x,y)
        if isinstance(ds, h5py.Dataset) and ds.ndim == 2 and ds.shape[1] in (2, 3):
            return ds[:]
    raise KeyError("no node coords dataset under meshes/1/nodes/coords")


def build_static_geometry(cas_path, zmin, zmax):
    with h5py.File(cas_path, "r") as f:
        coords   = _node_coords(f)
        f2n_ptr  = f["meshes/1/faces/nodes/1/nnodes"][:]
        f2n_data = f["meshes/1/faces/nodes/1/nodes"][:]
    offsets = np.zeros(len(f2n_ptr) + 1, dtype=np.int64)
    np.cumsum(f2n_ptr, out=offsets[1:])
    n_zone = zmax - zmin + 1
    centroids = np.zeros((n_zone, 2))
    normals   = np.zeros((n_zone, 2))
    lengths   = np.zeros(n_zone)
    for i_local, i_global in enumerate(range(zmin - 1, zmax)):
        start = offsets[i_global]
        end   = offsets[i_global + 1]
        node_ids = f2n_data[start:end] - 1
        pts = coords[node_ids]
        if len(pts) < 2:
            continue
        edge_vec = pts[-1] - pts[0]
        length   = np.linalg.norm(edge_vec)
        centroid = pts.mean(axis=0)
        normal   = np.array([edge_vec[1], -edge_vec[0]])
        if length > 1e-12:
            normal /= length
        centroids[i_local] = centroid
        normals[i_local]   = normal
        lengths[i_local]   = length
    body_center = centroids.mean(axis=0)
    for i in range(n_zone):
        if np.dot(normals[i], centroids[i] - body_center) > 0:
            normals[i] = -normals[i]
    return centroids, normals, lengths


# ==================================================================
# 2b. Mesh-measured angle of attack from a .cas.h5 mesh
# ==================================================================
def find_airfoil_le_te_node_ids(cas_path, zmin, zmax):
    """Locate the leading-edge and trailing-edge node ids of the airfoil
    wall from the REFERENCE cas file (which is at a small angle).

    The airfoil is meshed once, so the same two node ids identify LE and
    TE in every saved .cas.h5 of the run. Reusing them avoids relying on
    per-file argmin/argmax, which can mis-identify LE/TE when the airfoil
    is pitched to large angles.
    """
    with h5py.File(cas_path, "r") as f:
        coords   = _node_coords(f)
        f2n_ptr  = f["meshes/1/faces/nodes/1/nnodes"][:]
        f2n_data = f["meshes/1/faces/nodes/1/nodes"][:]
    offsets = np.zeros(len(f2n_ptr) + 1, dtype=np.int64)
    np.cumsum(f2n_ptr, out=offsets[1:])

    nodes = set()
    for i in range(zmin - 1, zmax):
        nodes.update((f2n_data[offsets[i]:offsets[i + 1]] - 1).tolist())
    node_ids = np.array(sorted(nodes))
    pts = coords[node_ids]

    # Leading edge = node nearest the far end of the chord, trailing edge
    # = the other far end. Use x-extreme in the reference (small-angle) mesh.
    i_le = int(np.argmin(pts[:, 0]))   # min x  -> LE
    i_te = int(np.argmax(pts[:, 0]))   # max x  -> TE
    return int(node_ids[i_le]), int(node_ids[i_te])


def mesh_aoa_from_cas(cas_path, le_node_id, te_node_id):
    """Extract the actual grid angle of attack from the LE/TE node
    coordinates of a .cas.h5 file.

    The dynamic-mesh case stores the airfoil at its instantaneous
    rotated position, so the leading-edge / trailing-edge segment
    orientation gives the true instantaneous AoA. This is more reliable
    than the theoretical sine formula when the grid is pre-rotated
    (reset_ang.c) or the UDF phase differs.
    """
    with h5py.File(cas_path, "r") as f:
        coords = _node_coords(f)
    le = coords[le_node_id]
    te = coords[te_node_id]
    return np.degrees(np.arctan2(le[1] - te[1], te[0] - le[0]))


# ==================================================================
# 3. Flow time / wall offset / force integration
# ==================================================================
def parse_flow_time(dat_path):
    with h5py.File(dat_path, "r") as f:
        dv = f["settings/Data Variables"][()][0]
        if isinstance(dv, bytes):
            dv = dv.decode("utf-8", errors="replace")
    m = re.search(r"\(flow-time\s+([\d.eE+\-]+)\)", dv)
    return float(m.group(1)) if m else 0.0


def find_wall_offset(cas_path, zmin):
    with h5py.File(cas_path, "r") as f:
        ztypes = f["meshes/1/faces/zoneTopology/zoneType"][:]
        zmins  = f["meshes/1/faces/zoneTopology/minId"][:]
        zmaxs  = f["meshes/1/faces/zoneTopology/maxId"][:]
    offset = 0
    for i in range(len(ztypes)):
        if ztypes[i] == 3:
            if int(zmins[i]) < zmin:
                offset += int(zmaxs[i] - zmins[i] + 1)
            elif int(zmins[i]) == zmin:
                break
    return offset


def compute_forces(dat_path, aoa_deg, zmin, zmax, centroids, normals,
                   lengths, wall_offset):
    """Integrate forces over the airfoil.

    Returns (flow_time, aoa_deg, CL, CD_total, CD_pressure) where
      CD_total   = (F_pressure + F_shear) in the freestream direction
      CD_pressure = F_pressure only (the "Cdp" equivalent to the
                   wind-tunnel data, which reports pressure drag only).
    """
    n_zone = zmax - zmin + 1
    flow_time = parse_flow_time(dat_path)
    # 注意: 法线必须是**当前文件姿态**下的 freestream 系法线（由调用方用
    # build_static_geometry 对该文件网格重算）。这里不做额外旋转——攻角只
    # 用于画图/攻角轴。若用参考 case 的静态法线并旋转，动态网格旋转姿态下
    # 会把升力投影到阻力方向，导致 Cd 虚高或出现负阻力（实测 D3T4 Cd 从
    # 合理 +0.02~0.09 变成 -0.12~0.21）。
    with h5py.File(dat_path, "r") as f:
        p_all = f["results/1/phase-1/faces/SV_P/1"][:]
    if p_all.shape[0] < zmax:
        return None
    p_zone = p_all[zmin - 1 : zmax]
    with h5py.File(dat_path, "r") as f:
        shear_all = f["results/1/phase-1/faces/SV_WALL_SHEAR/1"][:]
    if shear_all.ndim == 2:
        shear_zone = shear_all[wall_offset : wall_offset + n_zone, :2]
    else:
        n_val = min(shear_all.shape[0], (wall_offset + n_zone) * 2)
        flat = shear_all[wall_offset * 2 : n_val].reshape(-1, 2)
        shear_zone = flat[:n_zone]
    F_p = np.zeros(2)
    F_s = np.zeros(2)
    for i in range(n_zone):
        L = lengths[i]
        F_p += p_zone[i] * normals[i] * L
        F_s += shear_zone[i] * L
    F_total = F_p + F_s
    cl  = F_total[1] / Q_REF
    cd_tot = F_total[0] / Q_REF
    cd_pres = F_p[0] / Q_REF
    return flow_time, aoa_deg, cl, cd_tot, cd_pres


# ==================================================================
# 4. Load simulation data (multi-cycle KEPT)
# ==================================================================
def load_simulation(sim_dir, dt, skip_cycles, cas_arg=None, use_mesh_aoa=True):
    sim_dir = os.path.abspath(sim_dir)
    if not os.path.isdir(sim_dir):
        print(f"ERROR: simulation folder not found: {sim_dir}")
        return None
    label = os.path.basename(sim_dir.rstrip("/\\"))
    print("\n" + "=" * 68)
    print(f"  Simulation folder: {label}")
    print(f"  Motion: alpha = {AOA0:.1f} + {AOA1:.1f}*sin(2*pi*{FREQ}*t - pi/2) deg, "
          f"period {PERIOD:.4f} s, k = {K_SIM:.4f}")
    print(f"  AoA source      : {'mesh-measured' if use_mesh_aoa else 'theoretical'}")
    print("=" * 68)

    if cas_arg:
        cas_ref = cas_arg if os.path.isabs(cas_arg) else os.path.join(sim_dir, cas_arg)
    else:
        cas_candidates = sorted(glob.glob(os.path.join(sim_dir, "*.cas.h5")))
        if not cas_candidates:
            print(f"ERROR: no .cas.h5 files in {sim_dir}")
            return None
        cas_ref = cas_candidates[0]
    if not os.path.exists(cas_ref):
        print(f"ERROR: case file not found: {cas_ref}")
        return None
    print(f"  Reference case : {os.path.basename(cas_ref)}")

    dat_files = sorted(glob.glob(os.path.join(sim_dir, "*.dat.h5")))
    if not dat_files:
        print(f"ERROR: no .dat.h5 files in {sim_dir}")
        return None
    records = []
    for dp in dat_files:
        try:
            ft = parse_flow_time(dp)
        except Exception:
            ft = 0.0
        records.append((dp, ft))
    records.sort(key=lambda r: r[1])
    records = [(p, t) for p, t in records if t > 0.0]
    if not records:
        print(f"ERROR: no valid timesteps in {sim_dir}")
        return None
    t_all = np.array([t for _, t in records])
    p_all = [p for p, _ in records]
    t_end = float(t_all.max())
    print(f"  Found {len(records)} timesteps, flow-time "
          f"[{t_all.min():.4f}, {t_end:.4f}] s  "
          f"(~{t_end / PERIOD:.1f} cycles of {PERIOD:.4f} s)")

    zmin, zmax = parse_zone_topology(cas_ref)
    centroids, normals, lengths = build_static_geometry(cas_ref, zmin, zmax)
    print(f"  Airfoil perimeter = {lengths.sum():.4f} m")
    wall_offset = find_wall_offset(cas_ref, zmin)

    # LE/TE node ids (fixed across the run) for mesh-measured AoA
    if use_mesh_aoa:
        le_id, te_id = find_airfoil_le_te_node_ids(cas_ref, zmin, zmax)
        print(f"  LE node id = {le_id}, TE node id = {te_id}")

    # 复位/稳定阶段阈值：复位在 t<1s（UDF 门控 RESET_TIME=1.0），稳定微步也
    # 在 t<1s。这些阶段不是俯仰运动，混入会污染 AoA/力系数统计。
    # 采样从 max(t_min, 1.0s) 开始：保留完整第一俯仰周期（含 t=1.0~1.5 段）。
    t_targets = []
    t_min = float(t_all.min())
    t_start = max(t_min, 1.0)
    t_samp = dt * max(1.0, math.ceil(t_start / dt - 1e-12))
    while t_samp <= t_end + 1e-9:
        t_targets.append(t_samp)
        t_samp += dt
    if not t_targets:
        t_targets = [t_min]
    sampled_idx = [int(np.argmin(np.abs(t_all - tt))) for tt in t_targets]
    sampled_idx = sorted(set(sampled_idx))

    if skip_cycles > 0:
        t_cut = t_start + skip_cycles * PERIOD
        sampled_idx = [idx for idx in sampled_idx if t_all[idx] >= t_cut - 1e-9]
        print(f"  --skip-cycles {skip_cycles}: t >= {t_cut:.3f} s "
              f"({len(sampled_idx)} points)")
    else:
        print(f"  采样自 t >= {t_start:.1f} s (跳过复位/稳定阶段, "
              f"{len(sampled_idx)} points)")

    times, aoas, cl_list, cd_tot_list, cd_pres_list = [], [], [], [], []
    for idx in sampled_idx:
        dp = p_all[idx]
        cas_t = dp.replace(".dat.h5", ".cas.h5")
        # 动态网格: 每个 .cas.h5 的网格是当前攻角姿态，且 airfoil zone 的
        # minId/maxId 可能随文件不同（实测差 1）。必须对该文件解析 zone
        # 并重算法线，否则压力/法线错位，导致 Cl/Cd 错误。
        if os.path.exists(cas_t):
            try:
                zmin_f, zmax_f = parse_zone_topology(cas_t)
                centroids_f, normals_f, lengths_f = build_static_geometry(
                    cas_t, zmin_f, zmax_f)
                aoa_deg = mesh_aoa_from_cas(cas_t, le_id, te_id) if use_mesh_aoa \
                    else aoa_at(t_all[idx])
            except Exception:
                zmin_f, zmax_f = zmin, zmax
                centroids_f, normals_f, lengths_f = centroids, normals, lengths
                aoa_deg = aoa_at(t_all[idx]) if not use_mesh_aoa \
                    else mesh_aoa_from_cas(cas_t, le_id, te_id)
        else:
            zmin_f, zmax_f = zmin, zmax
            centroids_f, normals_f, lengths_f = centroids, normals, lengths
            aoa_deg = aoa_at(t_all[idx])
        result = compute_forces(dp, aoa_deg, zmin_f, zmax_f, centroids_f,
                                normals_f, lengths_f, wall_offset)
        if result is None:
            continue
        ft, aoa, cl, cd_tot, cd_pres = result
        times.append(ft); aoas.append(aoa); cl_list.append(cl)
        cd_tot_list.append(cd_tot); cd_pres_list.append(cd_pres)
    if not times:
        print("ERROR: no valid force data.")
        return None

    # ---- 攻角系统偏移校正（随攻角变化，分箱中位数插值）----
    # find_airfoil_le_te_node_ids 用 x 极值找 LE/TE，对圆前缘翼型（S 系列、
    # NACA、LS）有系统性偏移，且偏移**随攻角非线性变化**（如 S801 D4：
    # 0-4° 约 -1.0°，12-16° 约 -0.2°，20-24° 约 +0.03°）。
    # 用常数中位差会导致低攻角校正不足、高攻角过度校正（附着区对不上）。
    # 用分箱中位数 + 线性插值 offset(alpha)，稳健且随攻角变化。
    if use_mesh_aoa and AOA0 is not None and AOA1 > 0:
        t_arr = np.array(times)
        theory_arr = np.array([aoa_at(t) for t in t_arr])
        aoa_arr = np.array(aoas)
        diff = aoa_arr - theory_arr
        # 只用俯仰阶段（排除复位/稳定，t >= 2*PERIOD）
        mask = t_arr >= 2.0 * PERIOD
        if mask.sum() >= 30:
            a_fit, d_fit = aoa_arr[mask], diff[mask]
            # 剔除异常点（|diff-中位| > 3*MAD 剔除）
            med = np.median(d_fit)
            mad = np.median(np.abs(d_fit - med)) + 1e-9
            good = np.abs(d_fit - med) < 6.0 * mad
            if good.sum() >= 20:
                a_g, d_g = a_fit[good], d_fit[good]
                # 分箱：每 4° 一箱，取中位偏移
                bin_edges = np.arange(np.floor(a_g.min()), np.ceil(a_g.max()) + 1, 4.0)
                centers, meds = [], []
                for b in range(len(bin_edges) - 1):
                    m = (a_g >= bin_edges[b]) & (a_g < bin_edges[b + 1])
                    if m.sum() >= 2:
                        med = float(np.median(d_g[m]))
                        # 剔除异常箱：偏移量过大说明该攻角区 mesh_aoa 失效
                        # （深失速回流时 LE/TE 识别异常），用相邻正常箱代替
                        if abs(med) > 1.5:
                            continue
                        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                        meds.append(med)
                if len(centers) >= 3:
                    # 线性插值偏移
                    off_interp = np.interp(aoa_arr, centers, meds,
                                           left=meds[0], right=meds[-1])
                    new_aoas = aoa_arr - off_interp
                    corr_mag = np.abs(off_interp).max()
                    if corr_mag < 3.0:
                        print(f"  AoA 偏移随攻角校正 (分箱插值, {len(centers)}箱): "
                              f"偏移范围 {off_interp.min():+.2f}~{off_interp.max():+.2f} deg")
                        aoas = new_aoas.tolist()
                    else:
                        offset = float(np.median(d_fit))
                        if abs(offset) > 0.3:
                            print(f"  AoA 常数偏移校正(插值异常): {offset:+.3f} deg")
                            aoas = (aoa_arr - offset).tolist()
                else:
                    offset = float(np.median(d_fit))
                    if abs(offset) > 0.3:
                        print(f"  AoA 常数偏移校正(箱数少): {offset:+.3f} deg")
                        aoas = (aoa_arr - offset).tolist()
            else:
                offset = float(np.median(d_fit))
                if abs(offset) > 0.3:
                    print(f"  AoA 常数偏移校正(有效点少): {offset:+.3f} deg")
                    aoas = (aoa_arr - offset).tolist()
        else:
            offset = float(np.median(diff))
            if abs(offset) > 0.3:
                print(f"  AoA 常数偏移校正(数据少): {offset:+.3f} deg")
                aoas = (aoa_arr - offset).tolist()

    print(f"  -> {len(times)} points kept (multi-cycle)")
    return dict(label=label, times=np.array(times), aoas=np.array(aoas),
                cl=np.array(cl_list),
                cd=np.array(cd_tot_list),       # total drag (pressure + skin)
                cd_pres=np.array(cd_pres_list))  # pressure drag only (Cdp)


# ==================================================================
# 5. Parse a NACA4415 dynamic (unsteady) wind-tunnel file
# ==================================================================
def parse_dynamic_file(path):
    """Parse an n4415 dynamic file into a list of run dicts.

    Each run dict: mean, run_id, Re, k, f, vel, and arrays
    (time, aoa, cl, cdp, cm).
    """
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    runs = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "RUN" in line and "degree mean angle" in line:
            m_run = re.search(r"RUN\s+(\d+)\s+([\d.]+)\s+degree mean angle", line)
            run_id = m_run.group(1)
            mean = float(m_run.group(2))
            n_pts, sample_rate, vel, re_v, osc_f, red_k = None, None, None, None, None, None
            for j in range(i, min(i + 8, len(lines))):
                m_n = re.search(r"NUMBER OF DATA POINTS = (\d+)\s+SAMPLE RATE = ([\d.]+)", lines[j])
                if m_n:
                    n_pts, sample_rate = int(m_n.group(1)), float(m_n.group(2))
                m_v = re.search(r"TUNNEL\s+AIRSPEED\s*=\s*([\d.]+)\s*FT/SEC\s*,\s*REYNOLDS\s+NUMBER\s*=\s*([\d.]+)\s*MILLION", lines[j], re.I)
                if m_v:
                    vel, re_v = float(m_v.group(1)), float(m_v.group(2))
                m_f = re.search(r"OSCILLATOR\s+FREQUENCY\s*=\s*([\d.]+)\s*Hz\s*,\s*REDUCED\s+FREQUENCY\s*=\s*([\d.]+)", lines[j], re.I)
                if m_f:
                    osc_f, red_k = float(m_f.group(1)), float(m_f.group(2))
            time, aoa, cl, cdp, cm = [], [], [], [], []
            for k in range(i, min(i + 300, len(lines))):
                if "Unsteady Integrated Data" in lines[k]:
                    for row in lines[k + 1: k + 1 + n_pts]:
                        parts = [p.strip() for p in row.strip().split(",")]
                        if len(parts) == 6:
                            try:
                                vals = [float(p) for p in parts]
                            except ValueError:
                                continue
                            time.append(vals[1]); aoa.append(vals[2])
                            cl.append(vals[3]); cdp.append(vals[4]); cm.append(vals[5])
                    break
            if time:
                runs.append(dict(run_id=run_id, mean=mean, n_pts=n_pts,
                                 sample_rate=sample_rate, vel=vel, re_v=re_v,
                                 osc_f=osc_f, red_k=red_k,
                                 time=np.array(time), aoa=np.array(aoa),
                                 cl=np.array(cl), cdp=np.array(cdp),
                                 cm=np.array(cm)))
        i += 1
    return runs


def pick_dynamic_run(runs, target_mean=14.0):
    """Pick the run whose mean AoA is closest to the target."""
    if not runs:
        return None
    best = min(runs, key=lambda r: abs(r["mean"] - target_mean))
    return best


# ==================================================================
# 6. Plot comparison (dynamic loop scatter + simulation multi-cycle)
# ==================================================================
def plot_compare(sim, run, dt, out_path, add_cd0, cd0, use_mesh_aoa=True,
                 sim_cdp=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   constrained_layout=True)

    wt_label = (f"Wind tunnel dyn. (mean={run['mean']:.0f}deg, "
                f"k={run['red_k']:.3f}, Re={run['re_v']:.2f}M)")

    # --- Wind-tunnel dynamic hysteresis loop (SCATTER) ---
    aoas = run["aoa"]
    cds = run["cdp"] + (cd0 if add_cd0 else 0.0)

    # Upstroke / downstroke by AoA derivative
    d_aoa = np.diff(aoas)
    phase = np.ones(len(aoas), dtype=int)  # 1=up, 0=down
    for i in range(1, len(aoas)):
        if d_aoa[i - 1] < -0.01:
            phase[i] = 0

    up_aoas = aoas[phase == 1]; up_cl = run["cl"][phase == 1]; up_cd = cds[phase == 1]
    dn_aoas = aoas[phase == 0]; dn_cl = run["cl"][phase == 0]; dn_cd = cds[phase == 0]

    ax1.scatter(up_aoas, up_cl, color="#e41a1c", marker="^", s=28,
                alpha=0.85, edgecolors="none",
                label=f"{wt_label} - upstroke")
    ax1.scatter(dn_aoas, dn_cl, color="#377eb8", marker="v", s=28,
                alpha=0.85, edgecolors="none",
                label=f"{wt_label} - downstroke")
    ax2.scatter(up_aoas, up_cd, color="#e41a1c", marker="^", s=28,
                alpha=0.85, edgecolors="none",
                label=f"{wt_label} - upstroke")
    ax2.scatter(dn_aoas, dn_cd, color="#377eb8", marker="v", s=28,
                alpha=0.85, edgecolors="none",
                label=f"{wt_label} - downstroke")

    # --- Simulation: all cycles kept, SCATTER ONLY (no connecting lines) ---
    # CD can be either the total drag or the pressure-only drag (Cdp), to
    # match the wind-tunnel data which reports pressure drag only.
    if sim_cdp and "cd_pres" in sim:
        sim_cd = sim["cd_pres"]
        sim_cd_note = ", CD=Cdp"
    else:
        sim_cd = sim["cd"]
        sim_cd_note = ", CD=total"

    ax1.scatter(sim["aoas"], sim["cl"], color="#2ca02c", marker="o", s=16,
                alpha=0.6, edgecolors="none",
                label=f"DES {sim['label']} ({len(sim['aoas'])} pts)")
    ax2.scatter(sim["aoas"], sim_cd, color="#2ca02c", marker="o", s=16,
                alpha=0.6, edgecolors="none",
                label=f"DES {sim['label']} ({len(sim['aoas'])} pts)")

    aoa_lo = math.floor(min(sim["aoas"].min(), aoas.min()) - 1)
    aoa_hi = math.ceil(max(sim["aoas"].max(), aoas.max()) + 1)
    for ax in (ax1, ax2):
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_xlim(aoa_lo, aoa_hi)
        ax.grid(True, alpha=0.3, linestyle="--")
    ax1.set_ylabel("Lift Coefficient $C_L$")
    ax1.set_title("$C_L$ vs AoA - Dynamic Stall Loop")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.set_ylabel("Drag Coefficient $C_D$")
    ax2.set_title("$C_D$ vs AoA - Dynamic Stall Loop")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    cd_note = " (Cdp + Cd0)" if add_cd0 else " (Cdp)"
    aoa_src = "mesh-measured AoA" if use_mesh_aoa else "theoretical AoA"
    fig.suptitle(
        f"NACA4415 DES dynamic stall (multi-cycle, k={K_SIM:.3f}) vs "
        f"Dynamic wind-tunnel loop (k={run['red_k']:.3f})\n"
        f"Sim: {sim['label']} (f={FREQ}Hz, T={PERIOD:.3f}s, "
        f"Re~1.0e6, dt={dt:g}s, {len(sim['aoas'])} pts, {aoa_src}"
        f"{sim_cd_note})  |  "
        f"Test: {wt_label}{cd_note}",
        fontsize=12, fontweight="bold")

    fig.savefig(out_path, dpi=150)
    print(f"\n  Saved figure: {out_path}")
    plt.close(fig)


# ==================================================================
# 7. Main
# ==================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Compare DES simulation (multi-cycle, k=0.087) with "
                    "NACA4415 dynamic-stall (unsteady) wind-tunnel data.")
    ap.add_argument("--sim-dir", type=str, required=True,
                    help="Simulation data folder (e.g. the new k=0.087 run).")
    ap.add_argument("--wt-file", type=str, required=True,
                    help="Dynamic wind-tunnel data file "
                         "(e.g. 'n4415/C10h100_n4415.txt').")
    ap.add_argument("--mean", type=float, default=None,
                    help="Target mean AoA of the tunnel run to use. "
                         "Default: auto-detected from simulation AoA.")
    ap.add_argument("--dt", type=float, default=0.05,
                    help="Sampling time step for the simulation. Default: 0.05")
    ap.add_argument("--vel", type=float, default=None,
                    help="Freestream velocity [m/s]. Default: from tunnel run vel (ft/s).")
    ap.add_argument("--freq", type=float, default=None,
                    help="Pitching frequency [Hz]. Default: from tunnel run osc_f, "
                         "or auto-detected from simulation AoA.")
    ap.add_argument("--rho", type=float, default=None,
                    help="Freestream density [kg/m3]. Default: auto from tunnel Re.")
    ap.add_argument("--mu", type=float, default=None,
                    help="Dynamic viscosity [Pa*s]. Default: auto from tunnel Re.")
    ap.add_argument("--skip-cycles", type=int, default=0,
                    help="Drop the first N pitching cycles of the simulation.")
    ap.add_argument("--cas", type=str, default=None,
                    help="Optional explicit .cas.h5 filename in the sim folder.")
    ap.add_argument("--no-mesh-aoa", action="store_true",
                    help="Use the theoretical sine AoA instead of the "
                         "mesh-measured AoA from the .cas.h5 files "
                         "(default: use mesh-measured AoA).")
    ap.add_argument("--sim-cdp", action="store_true",
                    help="Plot the simulation PRESSURE-ONLY drag (Cdp) "
                         "instead of the total drag, to match the "
                         "wind-tunnel data which reports pressure drag only.")
    ap.add_argument("--add-cd0", action="store_true",
                    help="Add flat-plate skin-friction Cd0 to tunnel Cdp for "
                         "total-drag comparison.")
    ap.add_argument("--cd0", type=float, default=0.004,
                    help="Skin-friction coefficient to add when --add-cd0. "
                         "Default: 0.004")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path (default: "
                         "compare_simdyn_<sim>_<wt>.png).")
    args = ap.parse_args()

    # ---- 1) 先解析风洞数据（不依赖仿真参数）----
    runs = parse_dynamic_file(args.wt_file)
    if not runs:
        print(f"ERROR: no valid runs parsed from {args.wt_file}")
        sys.exit(1)
    # 若未指定 --mean，从仿真 AoA 自动检测（见后）。先取第一个 run 作为占位，
    # 待 load_simulation 后用仿真均值校正。若指定则直接用。
    run = pick_dynamic_run(runs, args.mean if args.mean is not None else runs[0]["mean"])
    if run is None:
        print("ERROR: no matching wind-tunnel run.")
        sys.exit(1)

    # ---- 2) 用风洞 run 自动设置 V_INF / FREQ / RHO / MU（未显式指定时）----
    #     vel 单位为 ft/s，转 m/s
    wt_vel = run["vel"] * 0.3048 if run.get("vel") else None
    v_inf   = args.vel if args.vel is not None else wt_vel
    freq    = args.freq if args.freq is not None else run.get("osc_f")
    # RHO/MU：未指定时由风洞 Re 与弦长反推（取标准空气密度 1.225 作为参考）
    re_t    = (run["re_v"] * 1e6) if run.get("re_v") else None
    rho = args.rho if args.rho is not None else 1.225
    if args.mu is not None:
        mu = args.mu
    elif re_t and v_inf:
        mu = rho * v_inf * C_REF / re_t
    else:
        mu = None

    set_case_params(rho=rho, vel=v_inf, mu=mu, freq=freq)

    # ---- 3) 读取仿真数据（此时 PERIOD/Q_REF 已按工况正确）----
    use_mesh_aoa = not args.no_mesh_aoa
    sim = load_simulation(args.sim_dir, args.dt, args.skip_cycles, args.cas,
                          use_mesh_aoa=use_mesh_aoa)
    if sim is None:
        sys.exit(1)

    # ---- 4) 若未指定 --mean，用仿真 mesh-measured AoA 选段 ----
    if args.mean is None and len(sim["aoas"]):
        # 排除复位/稳定阶段（flow-time 小于 2 倍周期），只统计俯仰周期
        t_cut = 2.0 * PERIOD
        mask = sim["times"] >= t_cut
        if mask.sum() >= 10:
            auto_mean = float(np.median(sim["aoas"][mask]))
        else:
            auto_mean = float(np.median(sim["aoas"]))
        run = pick_dynamic_run(runs, auto_mean)
        if run is None:
            run = pick_dynamic_run(runs, runs[0]["mean"])
        print(f"  Auto mean AoA from sim = {auto_mean:.1f} deg -> "
              f"tunnel run mean={run['mean']:.1f} deg")
        # run 已选，重新用该 run 的参数覆盖（若用户未显式指定）
        if args.vel is None and run.get("vel"):
            set_case_params(vel=run["vel"] * 0.3048)
        if args.freq is None and run.get("osc_f"):
            set_case_params(freq=run["osc_f"])
    else:
        run = pick_dynamic_run(runs, args.mean)
        if run is None:
            print(f"ERROR: no run with mean AoA near {args.mean} deg")
            sys.exit(1)

    # ---- 5) 打印工况汇总 ----
    print(f"\n  Wind-tunnel file : {os.path.basename(args.wt_file)}")
    print(f"  Runs parsed      : {len(runs)} "
          f"(means {[r['mean'] for r in runs]} deg)")
    print(f"  Using run        : #{run['run_id']} mean={run['mean']:.0f} deg, "
          f"k={run['red_k']}, f={run['osc_f']} Hz, Re={run['re_v']:.2f}M, "
          f"{len(run['time'])} samples")
    print(f"  Simulation       : k={K_SIM:.4f}, f={FREQ} Hz, "
          f"V={V_INF:.3f} m/s, rho={RHO:.4f}, mu={MU_AIR:.3e}, "
          f"period={PERIOD:.4f} s")

    if args.out:
        out_path = args.out
    else:
        out_path = f"compare_simdyn_{sim['label']}_{os.path.splitext(os.path.basename(args.wt_file))[0]}.png"
    if not out_path.lower().endswith(".png"):
        out_path += ".png"

    plot_compare(sim, run, args.dt, out_path, args.add_cd0, args.cd0,
                 use_mesh_aoa=use_mesh_aoa, sim_cdp=args.sim_cdp)
    print("Done.")


if __name__ == "__main__":
    main()
