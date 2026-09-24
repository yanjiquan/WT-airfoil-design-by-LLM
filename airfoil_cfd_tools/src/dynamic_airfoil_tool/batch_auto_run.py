# -*- coding: utf-8 -*-
"""
batch_auto_run.py - Batch dynamic-stall CFD automation for the
Dynamic Airfoil Tool (dynamic_airfoil_tool).

Purpose:
    Automate the dynamic-stall pitching simulations defined by a
    user-supplied condition JSON (e.g. config_14deg_amp10.json) across
    a set of airfoil meshes. Each (airfoil, condition) pair is a single
    Fluent case:  UDF generation -> case build -> journal -> libudf
    compile -> Fluent solve -> post-processing (Cl/Cd/AoA).

Usage:
    python batch_auto_run.py --mesh-dir <mesh_dir> \
                             --config  <conditions.json> \
                             --out-dir <out_dir> \
                             [--meshes cmaes_0p4572.msh ga_0p4572.msh ...] \
                             [--conditions R1 R2 ...] \
                             [--start N] [--stop N] \
                             [--nproc 4] [--parallel 1] \
                             [--retry 1] [--force] [--dry-run] [--debug]

    Example:
    python batch_auto_run.py --mesh-dir ../meshfiles \
                             --config  ../config_14deg_amp10.json \
                             --out-dir ../batch_results \
                             --parallel 4 --nproc 4

Resume (断点续跑):
    Re-running the SAME command is a resume: cases whose
    result_summary.json has status="done" are skipped automatically;
    cases marked "failed" or left "running" (crashed/interrupted) are
    re-run; failed cases are retried up to --retry times. Use --force
    to ignore existing results and re-run everything from scratch.
    Progress is written live to batch_plan.json (per-task status) and
    batch_progress.json (rolling summary), so an interrupted batch can
    always be continued. If interrupted, re-run the same command and it
    will resume (done cases skipped, failed cases retried, crashed
    cases re-run).

Workflow:
    1. Plan phase  : build the full (airfoil x condition) task list,
                     scan existing case dirs for already-finished runs,
                     write a manifest (batch_plan.json).
    2. Run phase   : execute tasks in worker processes (default 1 worker =
                     serial). Each worker isolates itself in its own
                     process so the Fluent auto core (global config module
                     state) is never shared between concurrent cases.
    3. Report phase: after all tasks finish, write batch_report.json with
                     per-case status and result summaries.

Environment / configuration:
    * FLUENT_EXE in dynamic_airfoil_tool/config.py must point to the
      local fluent.exe (updated automatically at install time).
    * Each airfoil mesh must be c = 0.4572 m (18 inch) and topologically
      identical to the reference NACA4415 mesh (the tool uses
      /file/replace-mesh).
    * The condition JSON (config_14deg_amp10.json) carries:
        conditions.<Rxx>.reynolds / mean_aoa / amplitude / osc_freq
        roughness: list of surface states (only "光滑"/clean is simulated)
      Freestream density / viscosity come from the wind-tunnel
      conversion in fluent_auto/config.py (T=275.26 K, P=98678.5 Pa).

Output layout:
    <out-dir>/
        batch_plan.json      # full task manifest + status
        batch_report.json    # final per-case results
        case_<airfoil>_<Rxx>/  # one Fluent case per (airfoil, condition)
            udf/ jobs/ *.cas *.h5 ...   # Fluent work dir
            case_params.json            # SimParams dump
            result_summary.json         # per-case statistics
            forces_*.png / *_datatable.csv / *_raw.npz  # deliverables
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Constants (mirror the wind-tunnel conversion in fluent_auto/config.py)
# ----------------------------------------------------------------------
C_CHORD = 0.4572            # chord [m] (18 inch)
# Wind-tunnel environment (RUN 407): T=35.8 degF, P=14.3121 psia
# NOTE: per-condition T/P are now taken from the actual wind-tunnel
# runs (C10l/m/h + 75/100/125/150, mean=14 deg), see WT_CONDITIONS.
T_AMB = 275.26              # [K]  (fallback only)
P_AMB = 98678.5             # [Pa] (fallback only)

# ----------------------------------------------------------------------
# Per-condition wind-tunnel environment (NACA4415 dynamic files under
# D:/My Data/Graduate/SIMULATION/12_files/dp0/FLU-4/n4415, mean=14 deg
# runs). Each condition maps to its closest tunnel run:
#   R* l -> f=0.6 Hz  (C10l75/100/125/150)
#   R* m -> f=1.2 Hz  (C10m75/100/125/150)
#   R* h -> f=1.8 Hz  (C10h75/100/125/150)
# Values are the actual AMBIENT TEMPERATURE / AMBIENT PRESSURE recorded
# in each run header. density/viscosity are computed per condition.
# ----------------------------------------------------------------------
WT_CONDITIONS = {
    "R1":  dict(Re=0.75e6, f=0.6, wt_file="C10l75",  run=431, T_F=31.6, P_psia=14.2352),
    "R2":  dict(Re=0.75e6, f=1.2, wt_file="C10m75",  run=432, T_F=32.3, P_psia=14.2352),
    "R3":  dict(Re=0.75e6, f=1.8, wt_file="C10h75",  run=433, T_F=32.4, P_psia=14.2352),
    "R4":  dict(Re=1.0e6,  f=0.6, wt_file="C10l100", run=434, T_F=31.6, P_psia=14.2352),
    "R5":  dict(Re=1.0e6,  f=1.2, wt_file="C10m100", run=435, T_F=32.1, P_psia=14.2352),
    "R6":  dict(Re=1.0e6,  f=1.8, wt_file="C10h100", run=436, T_F=32.4, P_psia=14.2352),
    "R7":  dict(Re=1.25e6, f=0.6, wt_file="C10l125", run=437, T_F=31.5, P_psia=14.2338),
    "R8":  dict(Re=1.25e6, f=1.2, wt_file="C10m125", run=438, T_F=32.1, P_psia=14.2338),
    "R9":  dict(Re=1.25e6, f=1.8, wt_file="C10h125", run=439, T_F=32.1, P_psia=14.2338),
    "R10": dict(Re=1.5e6,  f=0.6, wt_file="C10l150", run=440, T_F=31.3, P_psia=14.2338),
    "R11": dict(Re=1.5e6,  f=1.2, wt_file="C10m150", run=441, T_F=32.2, P_psia=14.2338),
    "R12": dict(Re=1.5e6,  f=1.8, wt_file="C10h150", run=442, T_F=32.1, P_psia=14.2338),
}


def wt_air_properties(condition: str) -> dict:
    """Return per-condition air properties from the matching tunnel run.

    T_K / P_PA are read from the wind-tunnel run header; density and
    viscosity are computed with the same ideal-gas / Sutherland laws
    used by fluent_auto/config.py. `V` is solved so that the target
    Reynolds number is reproduced exactly.
    """
    w = WT_CONDITIONS[condition]
    T_K = (w["T_F"] + 459.67) / 1.8
    P_PA = w["P_psia"] * 6894.757
    rho = P_PA / (287.05 * T_K)
    mu = 1.716e-5 * (T_K / 273.15) ** 1.5 * (273.15 + 110.4) / (T_K + 110.4)
    V = float(w["Re"]) * mu / (rho * C_CHORD)
    return dict(rho=rho, mu=mu, V=V, T_K=T_K, P_PA=P_PA,
                wt_file=w["wt_file"], wt_run=w["run"])



TOOL_ROOT = Path(__file__).resolve().parent
FLUENT_EXE = None           # filled lazily from TOOL_ROOT/config.py

# Airfoil mesh initial angle of attack (measured from the .msh LE/TE
# line; the reset UDF pitches from this angle to AOA_START). Extend this
# table when new meshes are introduced.
# 注: 用户新增的 naca4415_0p4572.msh 文件名是小写，airfoil_name() 取
# _ 前首段得到 "naca4415"，故加小写键（与 core/fluent_solver.py 一致）。
AF_INITIAL_AOA = {
    "NACA4415": 0.236, "N4415": 0.236, "naca4415": 0.236,
    "S801": 0.113, "S809": 0.239, "S810": 0.036, "S812": -0.546,
    "S813": 0.129, "S814": -0.613, "S815": -0.819, "S825": -0.785,
    "LS0417": -0.108, "LS0421": -0.188,
}

# Free-stream turbulence (the reference case / wind tunnel D1T1 uses 5%
# intensity, 0.2 chord integral length scale). Fixed for all cases.
TI_DEFAULT = 0.05
LT_OVER_C_DEFAULT = 0.20

# Roughness mapping: the config lists surface states in Chinese; the
# current Fluent chain only simulates the clean ("光滑") configuration.
ROUGHNESS_CLEAN = "光滑"

# Each case runs 11 pitching cycles (first cycle is the transient
# import cycle); post-processing keeps every cycle (compare_sim_auto).
PITCH_CYCLES = 11


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def log(msg: str) -> None:
    """Timestamped console log (flushed so background capture works)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_fluent_exe() -> str:
    """Load FLUENT_EXE from the tool's config.py."""
    global FLUENT_EXE
    if FLUENT_EXE:
        return FLUENT_EXE
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dyn_batch_cfg", TOOL_ROOT / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    FLUENT_EXE = str(mod.FLUENT_EXE)
    if not Path(FLUENT_EXE).exists():
        raise RuntimeError(
            f"FLUENT_EXE 不存在: {FLUENT_EXE}\n"
            "请修改 dynamic_airfoil_tool/config.py 中的 FLUENT_EXE。")
    return FLUENT_EXE


def airfoil_name(mesh_stem: str) -> str:
    """Airfoil name from mesh stem (first token before '_')."""
    return mesh_stem.split("_")[0]


def _priority(t: dict) -> tuple:
    """Run priority key for pending tasks (user requirement): among the
    unfinished airfoils, finish llm and naca4415 first.

    Returns (group, condition_index) so the sort is stable and keeps the
    natural R1..R12 order within each airfoil:
      group 0 = llm       (最高优先)
      group 1 = naca4415
      group 2 = 其余 (cmaes/ga/pso)
    Applied to the selected slice (sel) before dispatch, so both the
    serial and the parallel pool pick priority cases first.
    """
    af = t.get("airfoil", "")
    group = 0 if af == "llm" else 1 if af == "naca4415" else 2
    cid = t.get("condition", "R0")
    idx = int(cid[1:]) if cid[:1] == "R" and cid[1:].isdigit() else 0
    return (group, idx)


def summarize(case_dir: Path) -> dict:
    """Compute per-case statistics from the post-processed result.

    Returns dict with mean/max/min Cl, Cd over the sampled pitching
    cycles, plus cycle/hysteresis width (max diff between upstroke and
    downstroke Cl at the same AoA). Falls back to NaN when no data.
    """
    import numpy as np
    try:
        sim = _load_sim(case_dir)
        if sim is None or not len(sim["cl"]):
            return {}
        t, aoa, cl, cd = sim["times"], sim["aoas"], sim["cl"], sim["cd"]
        dth = np.gradient(aoa, t)
        up = dth > 0
        out = dict(
            n_points=int(len(t)),
            cl_mean=float(np.mean(cl)), cl_max=float(np.max(cl)),
            cl_min=float(np.min(cl)),
            cd_mean=float(np.mean(cd)), cd_max=float(np.max(cd)),
            cd_min=float(np.min(cd)),
        )
        # hysteresis width: median over AoA bins of |Cl_up - Cl_down|
        hw = []
        for lo in np.arange(np.floor(aoa.min()), np.ceil(aoa.max()), 1.0):
            m_up = up & (aoa >= lo) & (aoa < lo + 1.0)
            m_dn = (~up) & (aoa >= lo) & (aoa < lo + 1.0)
            if m_up.sum() >= 3 and m_dn.sum() >= 3:
                hw.append(abs(float(np.mean(cl[m_up]) - np.mean(cl[m_dn]))))
        out["hyst_width"] = float(np.median(hw)) if hw else None
        return out
    except Exception:
        return {}


def _load_sim(case_dir: Path):
    """Load post-processed simulation via compare_sim_auto (same as
    plot_forces.py). Returns None if the case dir has no data."""
    try:
        sys.path.insert(0, str(TOOL_ROOT / "fluent_auto"))
        import compare_sim_auto as csa
        cp = case_dir / "case_params.json"
        if not cp.exists():
            return None
        p = json.loads(cp.read_text(encoding="utf-8"))
        csa.set_case_params(rho=p["rho"], vel=p["V"], mu=p["mu"],
                            freq=p["freq"], ao0=p["mean_aoa"], ao1=p["amp_aoa"])
        sim = csa.load_simulation(str(case_dir), dt=0.05, skip_cycles=0,
                                  cas_arg=None, use_mesh_aoa=False)
        return sim
    except Exception:
        return None


# ----------------------------------------------------------------------
# Single case runner (one Fluent case in its own process)
# ----------------------------------------------------------------------
def _setup_paths():
    """Ensure fluent_auto/ and tool root are importable (worker process)."""
    sys.path.insert(0, str(TOOL_ROOT))
    for d in (str(TOOL_ROOT / "fluent_auto"), str(TOOL_ROOT)):
        if d not in sys.path:
            sys.path.insert(0, d)


def compile_case(args: dict) -> dict:
    """Pre-compile the UDF for a single case (serial phase).

    Generates UDF sources + case + journal, then compiles libudf
    WITHOUT running Fluent. The compiled libudf stays under
    case_dir/udf/libudf so the later parallel solve phase reuses it.
    Returns a status dict; on success marks the case 'compiled'.
    """
    t0 = time.time()
    case_dir = Path(args["case_dir"])
    condition = args["condition"]
    airfoil = args["airfoil"]

    # Resume: a case already compiled (compiled libudf present) is skipped.
    if (case_dir / "udf" / "libudf" / "win64" / "2ddp_host"
            / "libudf.dll").exists():
        return dict(status="compiled", condition=condition,
                    airfoil=airfoil, case_dir=str(case_dir),
                    elapsed_s=0.0)

    log(f"[{airfoil} {condition}] compile -> {case_dir.name}")
    try:
        _setup_paths()
        from core.fluent_solver import FluentSolver
        from core.solver_interface import SimParams

        params = SimParams(
            mesh_path=args["mesh_path"],
            chord=C_CHORD,
            rho=args["rho"], mu=args["mu"], V=args["V"],
            freq=args["freq"], mean_aoa=args["mean_aoa"],
            amp_aoa=args["amp_aoa"],
            TI=TI_DEFAULT, Lt_over_c=LT_OVER_C_DEFAULT,
        )
        params.validate()
        # Prepare case config WITHOUT solving; then compile UDF in place.
        # Use the batch's stable friendly case dir so compile and solve
        # land in the SAME directory (case_{af}_{cid}).
        friendly_id = f"{airfoil}_{condition}"
        solver = FluentSolver(get_fluent_exe(), Path(args["out_dir"]),
                              nproc=args["nproc"], skip_udf_compile=True,
                              case_id=friendly_id)
        # Use the solver's internal prep to generate UDF/case/journal.
        # (FluentSolver.run() does this; here we only want the compile step.)
        case_cfg = solver._prepare_config(params)
        import case_runner, gen_udf, build_case, gen_journal
        import config as cfg
        for _f in cfg.OUT_DIR.glob("*.h5"):
            try:
                _f.unlink()
            except OSError:
                pass
        gen_udf.main()
        build_case.build_case()
        gen_journal.main()
        ok = case_runner.compile_udf()
        if not ok:
            raise RuntimeError("UDF 编译失败")
        log(f"[{airfoil} {condition}] compiled in {time.time()-t0:.1f}s")
        return dict(status="compiled", condition=condition, airfoil=airfoil,
                    case_dir=str(case_dir), elapsed_s=round(time.time() - t0, 1))
    except Exception as e:
        log(f"[{airfoil} {condition}] COMPILE FAILED: {e}")
        (case_dir / "result_summary.json").write_text(
            json.dumps(dict(status="failed", condition=condition,
                            airfoil=airfoil, error=str(e)),
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(status="failed", condition=condition, airfoil=airfoil,
                    case_dir=str(case_dir), error=str(e))


def run_case(args: dict) -> dict:
    """Run a single (airfoil, condition) Fluent case and return status.

    args: dict produced by build_tasks() with all SimParams fields.
    This function is executed in a worker process (never shares the
    global config module state with other cases).
    """
    t0 = time.time()
    case_dir = Path(args["case_dir"])
    out_dir = Path(args["out_dir"])
    condition = args["condition"]
    airfoil = args["airfoil"]

    # Resume support: a case whose result_summary says "done" is skipped.
    # "compiled" (UDF pre-compiled but not solved) must NOT be skipped —
    # the solver reuses the cached libudf and runs the solve.
    sd = case_dir / "result_summary.json"
    if sd.exists():
        try:
            prev = json.loads(sd.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        if prev.get("status") == "done":
            return dict(status="skipped", condition=condition,
                        airfoil=airfoil, case_dir=str(case_dir),
                        reason="already-complete")

    log(f"[{airfoil} {condition}] start -> {case_dir.name}")
    solver = None
    try:
        _setup_paths()
        from core.fluent_solver import FluentSolver
        from core.solver_interface import SimParams
        from core.data_writer import write_datatable, write_rawdata
        from core.plotter import plot_aero

        params = SimParams(
            mesh_path=args["mesh_path"],
            chord=C_CHORD,
            rho=args["rho"], mu=args["mu"], V=args["V"],
            freq=args["freq"], mean_aoa=args["mean_aoa"],
            amp_aoa=args["amp_aoa"],
            TI=TI_DEFAULT, Lt_over_c=LT_OVER_C_DEFAULT,
        )
        params.validate()

        # In the batched flow the UDF was already pre-compiled by the
        # serial compile phase; skip re-compiling to avoid the
        # concurrent-scons conflict entirely.
        compiled = (case_dir / "udf" / "libudf" / "win64" / "2ddp_host"
                    / "libudf.dll").exists()
        friendly_id = f"{airfoil}_{condition}"
        solver = FluentSolver(get_fluent_exe(), out_dir, nproc=args["nproc"],
                              skip_udf_compile=compiled, case_id=friendly_id)
        result = solver.run(params)

        # Write deliverables into the case dir (mirrors GUI behavior)
        case_dir.mkdir(parents=True, exist_ok=True)
        base = f"{airfoil}_{condition}"
        write_datatable(result, case_dir / f"{base}_datatable.csv")
        write_rawdata(result, case_dir, base)
        plot_aero(result, case_dir / f"{base}_aero.png")

        summ = summarize(case_dir)
        summ.update(condition=condition, airfoil=airfoil,
                    elapsed_s=round(time.time() - t0, 1),
                    n_points=result.n_points)
        (case_dir / "result_summary.json").write_text(
            json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
        # 保留原始数据：不删除 case 目录下的 *.dat.h5 / *.cas.h5 等中间产物
        # （用户要求保留完整 Fluent 原始数据，禁止自动清理）
        log(f"[{airfoil} {condition}] DONE in {time.time()-t0:.1f}s, "
            f"{result.n_points} pts")
        return dict(status="done", condition=condition, airfoil=airfoil,
                    case_dir=str(case_dir), elapsed_s=summ["elapsed_s"],
                    n_points=result.n_points)
    except Exception as e:
        import traceback
        log(f"[{airfoil} {condition}] FAILED: {e}")
        if args.get("debug"):
            traceback.print_exc()
        (case_dir / "result_summary.json").write_text(
            json.dumps(dict(status="failed", condition=condition,
                            airfoil=airfoil, error=str(e)),
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(status="failed", condition=condition, airfoil=airfoil,
                    case_dir=str(case_dir), error=str(e))
    finally:
        # Always kill the Fluent process tree this worker started, so a
        # cancelled/aborted worker never leaves orphan MPI processes
        # holding the case files (which would block directory cleanup).
        if solver is not None:
            try:
                solver.stop()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Task planning / manifest
# ----------------------------------------------------------------------
def build_tasks(mesh_dir: Path, config_path: Path, out_dir: Path,
                mesh_filter=None, cond_filter=None, nproc: int = 4):
    """Build the full (airfoil x condition) task list.

    Returns (tasks, meshes, conditions, cfg) where each task dict
    carries everything run_case() needs.
    """
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    conditions = cfg["conditions"]
    meshes = sorted(Path(mesh_dir).glob("*.msh"))

    if mesh_filter:
        names = set(mesh_filter)
        meshes = [m for m in meshes if m.name in names or m.stem in names]
    if not meshes:
        raise RuntimeError(f"未在 {mesh_dir} 找到任何 .msh 网格")

    if cond_filter:
        conditions = {k: v for k, v in conditions.items() if k in cond_filter}
    if not conditions:
        raise RuntimeError("--conditions 过滤后没有剩余工况")

    tasks = []
    for m in meshes:
        af = airfoil_name(m.stem)
        for cid, c in conditions.items():
            # Per-condition wind-tunnel air properties (T/P from the
            # closest NACA4415 tunnel run, V solved for the target Re).
            wtp = wt_air_properties(cid)
            case_id = f"case_{af}_{cid}"
            tasks.append(dict(
                airfoil=af,
                mesh_path=str(m),
                condition=cid,
                case_dir=str(out_dir / case_id),
                out_dir=str(out_dir),
                reynolds=float(c["reynolds"]),
                mean_aoa=float(c["mean_aoa"]),
                amp_aoa=float(c["amplitude"]),
                freq=float(c["osc_freq"]),
                rho=wtp["rho"],
                mu=wtp["mu"],
                V=wtp["V"],
                T_K=wtp["T_K"],
                P_PA=wtp["P_PA"],
                wt_file=wtp["wt_file"],
                wt_run=wtp["wt_run"],
                nproc=nproc,
                debug=False,
            ))
    return tasks, meshes, conditions, cfg


def write_manifest(tasks, meshes, conditions, cfg, out_dir: Path,
                   start: int = 0, stop: int = None) -> Path:
    """Persist the plan + status manifest to batch_plan.json."""
    manifest = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        meshes=[m.name for m in meshes],
        conditions=list(conditions.keys()),
        roughness=list(cfg.get("roughness", [])),
        nproc=cfg.get("concurrency", 4),
        total=len(tasks),
        start=start,
        stop=stop,
        tasks=tasks,
    )
    path = out_dir / "batch_plan.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def scan_finished(tasks, force=False, retry=1):
    """Classify tasks for the next run based on their case dir state.

    A case's durable state lives in result_summary.json inside its dir:
      {"status": "done",     ...} -> completed; skipped on resume
      {"status": "compiled", ...} -> UDF compiled, not solved; resume
                                     goes straight to the solve phase
      {"status": "failed",   ...} -> failed; re-run if n_retry < retry,
                                     else left failed and surfaced later
      no summary file              -> never started -> pending
    A stale result_running.lock (left by a crashed/interrupted worker)
    is treated as interrupted: lock removed, case re-run.

    Each task dict gains: done(bool), status, n_retry (times failed),
    reason, needs_compile(bool). With force=True every case is treated
    as pending (and stale locks are cleared so the case can start
    cleanly).

    并发 replace-mesh 的 Settings 临时文件竞争问题（CAR: invalid
    argument 中断 journal）已通过给每个 Fluent 进程独立 TEMP 目录解决
    （见 fluent_solver._run_fluent_async / case_runner.run_fluent），
    因此此前因该问题 "failed" 的 case 不必再受重试次数限制：重跑时
    failed 一律进入待跑（n_retry 归零），让同一次 resume 就把它们
    重新跑起来。
    """
    import json as _json
    for t in tasks:
        sd = Path(t["case_dir"]) / "result_summary.json"
        lock = Path(t["case_dir"]) / "result_running.lock"
        t["done"] = False
        t["status"] = "pending"
        t["n_retry"] = 0
        t["reason"] = None
        t["needs_compile"] = True
        if force:
            # Full re-run: clear any leftover running lock so the case
            # can be restarted from scratch.
            try:
                lock.unlink()
            except OSError:
                pass
            continue
        if sd.exists():
            try:
                st = _json.loads(sd.read_text(encoding="utf-8"))
            except Exception:
                st = {}
            if st.get("status") == "done":
                t["done"] = True
                t["status"] = "done"
                t["reason"] = "already-complete"
            elif st.get("status") == "compiled":
                # UDF already compiled -> skip the compile phase, solve only
                t["status"] = "compiled"
                t["needs_compile"] = False
                t["reason"] = "compiled-resume"
            elif st.get("status") == "failed":
                # 竞争问题已修复：failed 一律重新入队（不再受 --retry 限制）
                t["reason"] = "retry-after-fix"
        elif lock.exists():
            # Worker crashed / batch interrupted mid-case: the lock is
            # stale now. Clean it so the case can restart cleanly.
            t["reason"] = "interrupted-resume"
            try:
                lock.unlink()
            except OSError:
                pass
    return tasks


def _update_plan(tasks, out_dir: Path) -> None:
    """Persist the current per-task status to batch_plan.json (live)."""
    path = out_dir / "batch_plan.json"
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        old = {}
    old["tasks"] = tasks
    old["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _write_progress(out_dir: Path, sel, results, running: int) -> None:
    """Write a rolling progress snapshot to batch_progress.json.

    Called after every task completes so an interrupted batch can always
    be inspected and resumed. `results` holds per-case status dicts.
    """
    n_done = sum(1 for r in results if r.get("status") == "done")
    n_fail = sum(1 for r in results if r.get("status") == "failed")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    prog = dict(
        updated=time.strftime("%Y-%m-%d %H:%M:%S"),
        total=len(sel),
        done=n_done, failed=n_fail, skipped=n_skip,
        running=running,
        remaining=len(sel) - n_done - n_fail - n_skip - running,
    )
    try:
        (out_dir / "batch_progress.json").write_text(
            json.dumps(prog, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch dynamic-stall Fluent automation "
                    "(airfoil meshes x conditions)")
    ap.add_argument("--mesh-dir", required=True, help="目录：含 .msh 网格")
    ap.add_argument("--config", required=True, help="工况 JSON（如 config_14deg_amp10.json）")
    ap.add_argument("--out-dir", default=None,
                    help="输出根目录（默认: <mesh-dir>/../batch_results_<ts>）")
    ap.add_argument("--meshes", nargs="*", default=None,
                    help="只跑指定网格（文件名或 stem，默认全部）")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="只跑指定工况 R1..R12（默认全部）")
    ap.add_argument("--start", type=int, default=0,
                    help="从任务列表第 N 个开始（断点续跑）")
    ap.add_argument("--stop", type=int, default=None,
                    help="跑到任务列表第 N 个结束（不含）")
    ap.add_argument("--nproc", type=int, default=4,
                    help="每个 Fluent case 使用核数（默认 4）")
    ap.add_argument("--parallel", type=int, default=1,
                    help="并发 worker 数（默认 1=串行；>1 时同跑多个 Fluent）")
    ap.add_argument("--retry", type=int, default=1,
                    help="失败 case 的重试次数（默认 1；0=不重试）")
    ap.add_argument("--force", action="store_true",
                    help="忽略已有 result_summary.json，全部重新运行")
    ap.add_argument("--dry-run", action="store_true",
                    help="只生成计划清单，不运行 Fluent")
    ap.add_argument("--debug", action="store_true",
                    help="出错时打印完整 traceback")
    args = ap.parse_args()

    mesh_dir = Path(args.mesh_dir).resolve()
    config_path = Path(args.config).resolve()
    if not mesh_dir.is_dir():
        log(f"错误: mesh 目录不存在 {mesh_dir}")
        return 2
    if not config_path.exists():
        log(f"错误: 配置不存在 {config_path}")
        return 2

    # Verify the Fluent executable early so a wrong path fails fast.
    try:
        fex = get_fluent_exe()
        log(f"FLUENT_EXE = {fex}")
    except RuntimeError as e:
        log(str(e))
        return 2

    out_dir = Path(args.out_dir).resolve() if args.out_dir else \
        mesh_dir.parent / f"batch_results_{time.strftime('%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, meshes, conditions, cfg = build_tasks(
        mesh_dir, config_path, out_dir,
        mesh_filter=args.meshes, cond_filter=args.conditions,
        nproc=args.nproc)
    tasks = scan_finished(tasks, force=args.force, retry=args.retry)

    total = len(tasks)
    stop = args.stop if args.stop is not None else total
    stop = min(stop, total)
    sel = tasks[args.start:stop]
    # 用户要求：未完成翼型中优先完成 llm / naca4415（稳定排序，组内保 R1..R12）
    sel = sorted(sel, key=_priority)
    done = sum(1 for t in sel if t["done"])

    manifest_path = write_manifest(tasks, meshes, conditions, cfg, out_dir,
                                   start=args.start, stop=stop)
    log(f"计划清单: {manifest_path}")
    log(f"网格 {len(meshes)} 个 x 工况 {len(conditions)} 个 = 共 {total} case；"
        f"本次执行 [{args.start}, {stop}) = {len(sel)} 个，"
        f"已完成跳过 {done} 个"
        + (f"，失败重试 {sum(1 for t in sel if t['status'] == 'failed')} 个"
           if not args.force else "（--force 全量重跑）"))

    if args.dry_run:
        for t in sel:
            log(f"  [{'SKIP' if t['done'] else t['status'].upper()}] "
                f"{t['condition']:>4} {Path(t['mesh_path']).name} -> {t['case_dir']}"
                + (f" ({t['reason']})" if t["reason"] else ""))
        log(f"DRY-RUN 结束（未运行 Fluent）")
        return 0

    # ---- Run phase --------------------------------------------------
    if args.parallel <= 1:
        # Serial: run each case in a child process so the fluent_auto
        # global-config state is pristine per case.
        results = []
        for i, t in enumerate(sel):
            if t["done"]:
                results.append(dict(status="skipped",
                                    condition=t["condition"],
                                    airfoil=t["airfoil"],
                                    case_dir=t["case_dir"],
                                    reason="already-complete"))
                continue
            # Live progress persistence (resume bookkeeping)
            _touch_lock(t)
            r = _run_case_isolated(t)
            _finish_lock(t, r)
            results.append(r)
            _write_progress(out_dir, sel, results, running=0)
            _update_plan(tasks, out_dir)
    else:
        # Parallel: launch N workers via subprocess on the same script.
        results = _run_parallel(args, sel, out_dir)
    _write_progress(out_dir, sel, results, running=0)
    _update_plan(tasks, out_dir)

    # ---- Report phase -----------------------------------------------
    n_done = sum(1 for r in results if r["status"] == "done")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    # Count cases whose retry budget is exhausted (surfaced in report).
    n_exhausted = sum(1 for t in sel if t["status"] == "failed"
                      and (t.get("n_retry", 0) >= args.retry or args.retry == 0))
    report = dict(
        finished=time.strftime("%Y-%m-%d %H:%M:%S"),
        out_dir=str(out_dir),
        total=len(sel), done=n_done, failed=n_fail, skipped=n_skip,
        retry_budget=args.retry,
        retry_exhausted=n_exhausted,
        results=results,
    )
    report_path = out_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    log(f"报告: {report_path}")
    log(f"完成 {n_done} / 失败 {n_fail} / 跳过 {n_skip} / 共 {len(sel)}"
        + (f"，重试耗尽 {n_exhausted}" if n_exhausted else ""))
    return 0 if n_fail == 0 and n_exhausted == 0 else 1


def _compile_case_isolated(t: dict) -> dict:
    """Run compile_case() in a subprocess (one at a time, no concurrency).

    Compilation must never run concurrently with another process touching
    the ANSYS scons/clang shared caches, so this is only ever called
    serially (from the parent orchestrator's compile phase).
    """
    code = ("import json,sys;sys.path.insert(0,%r);"
            "from batch_auto_run import compile_case;"
            "r=compile_case(json.loads(sys.argv[1]));"
            "print('BATCH_RESULT '+json.dumps(r))") % str(TOOL_ROOT)
    r = subprocess.run([sys.executable, "-c", code, json.dumps(t)],
                       capture_output=True, text=True,
                       cwd=str(TOOL_ROOT), timeout=None)
    out = r.stdout or ""
    marker = "BATCH_RESULT "
    idx = out.find(marker)
    if idx >= 0:
        try:
            return json.loads(out[idx + len(marker):].strip().splitlines()[0])
        except Exception:
            pass
    tail = (out + r.stderr).strip()[-2000:]
    return dict(status="failed", condition=t["condition"], airfoil=t["airfoil"],
                case_dir=t["case_dir"], error=f"compile worker exit={r.returncode}",
                output_tail=tail)


def _run_case_isolated(t: dict) -> dict:
    """Run run_case() in a subprocess with a compact wrapper that prints
    a single JSON status line on stdout.

    The child process is spawned as a new process group (CREATE_NEW_PROCESS_GROUP
    on Windows) so that if this batch is interrupted, the child (and the
    Fluent/MPI tree it spawns) can be killed as a group instead of being
    left as orphans.
    """
    import subprocess as _sp
    creationflags = 0
    if os.name == "nt":
        creationflags = _sp.CREATE_NEW_PROCESS_GROUP
    code = ("import json,sys;sys.path.insert(0,%r);"
            "from batch_auto_run import run_case;"
            "r=run_case(json.loads(sys.argv[1]));"
            "print('BATCH_RESULT '+json.dumps(r))") % str(TOOL_ROOT)
    proc = _sp.Popen(
        [sys.executable, "-c", code, json.dumps(t)],
        stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
        cwd=str(TOOL_ROOT), creationflags=creationflags)
    try:
        out, _ = proc.communicate(timeout=None)
    except KeyboardInterrupt:
        # Batch interrupted: kill the child tree so no Fluent is orphaned.
        _kill_process_tree(proc.pid)
        proc.wait()
        raise
    r = proc
    out = out or ""
    marker = "BATCH_RESULT "
    idx = out.find(marker)
    if idx >= 0:
        try:
            return json.loads(out[idx + len(marker):].strip().splitlines()[0])
        except Exception:
            pass
    # Fallback: reconstruct a failed status from the process output
    tail = out.strip()[-2000:]
    return dict(status="failed", condition=t["condition"], airfoil=t["airfoil"],
                case_dir=t["case_dir"], error=f"worker exit={proc.returncode}",
                output_tail=tail)


def _kill_process_tree(pid: int) -> None:
    """Kill a process tree by PID (Windows: taskkill /T /F; POSIX: killpg)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=15)
        else:
            os.killpg(pid, 9)
    except Exception:
        try:
            os.kill(pid, 9)
        except Exception:
            pass


def _run_parallel(args, sel, out_dir: Path):
    """Parallel execution: serial UDF-compile phase, then parallel solve.

    Phase 1 (COMPILE, serial): every pending case's UDF is compiled one
    at a time (module-level compile_case runs in its own subprocess, so
    the ANSYS scons / clang shared caches are never touched by two
    processes at once). Each compiled libudf is cached under
    case_dir/udf/libudf/.

    Phase 2 (SOLVE, parallel): the fixed pool runs cases concurrently;
    each case skips UDF compilation (FluentSolver(skip_udf_compile=True)
    because the pre-compiled libudf exists), so concurrent Fluent
    instances never race on shared compile artifacts. A crash in one
    Fluent cannot take down a pool worker.

    NOTE: worker callables/task dicts must be module-level / picklable
    for the Windows spawn start method (no nested functions here).
    """
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")

    pending = [t for t in sel if not t.get("done")]
    to_compile = [t for t in pending if t.get("needs_compile", True)]
    already_compiled = [t for t in pending if not t.get("needs_compile", True)]
    log(f"  编译阶段: {len(to_compile)} 个 case 需要预编译 UDF（串行）"
        + (f"，{len(already_compiled)} 个已编译直接求解" if already_compiled else ""))

    # Phase 1 - serial compile (one subprocess at a time, no concurrency)
    compiled_ok = {}
    for t in to_compile:
        _touch_lock(t)
        r = _compile_case_isolated(t)
        compiled_ok[t["case_dir"]] = (r.get("status") == "compiled")
        _finish_lock(t, r)
        if r.get("status") != "compiled":
            log(f"  编译失败: {t['condition']} {Path(t['mesh_path']).name}: "
                f"{r.get('error')}")
        else:
            log(f"  已编译: {t['condition']} {Path(t['mesh_path']).name}")

    # Phase 2 - parallel solve (compiled this run + pre-compiled on resume)
    solve_list = [t for t in to_compile if compiled_ok.get(t["case_dir"])]
    solve_list += already_compiled
    skip_list = [t for t in to_compile if not compiled_ok.get(t["case_dir"])]
    log(f"  求解阶段: {len(solve_list)} 个 case 并行求解"
        + (f"，{len(skip_list)} 个编译失败跳过" if skip_list else ""))

    results = []
    if solve_list:
        with ctx.Pool(processes=args.parallel) as pool:
            for r in pool.imap_unordered(_worker_run, solve_list, chunksize=1):
                results.append(r)
                _write_progress(out_dir, sel, results, running=0)
    for t in skip_list:
        results.append(dict(status="failed", condition=t["condition"],
                            airfoil=t["airfoil"], case_dir=t["case_dir"],
                            error="UDF compile failed (see result_summary.json)"))
    return results


def _cleanup_intermediate_h5(case_dir: Path) -> None:
    """(已禁用) 曾用于删除已完成 case 的逐时间步 HDF5 中间产物以释放磁盘。

    用户要求保留全部 Fluent 原始数据（*.dat.h5 / *.cas.h5 / *.trn 等），
    禁止自动删除，故本函数不再被调用（run_case 中已移除调用点）。
    保留定义仅为兼容性，未在任何路径执行。
    """
    try:
        for ext in ("*.dat.h5", "*.cas.h5", "*.trn", "*.flprj", "*.bat"):
            for p in case_dir.glob(ext):
                try:
                    p.unlink()
                except OSError:
                    pass
        jobs = case_dir / "jobs"
        if jobs.is_dir():
            import shutil as _sh
            _sh.rmtree(jobs, ignore_errors=True)
    except Exception:
        pass


def _touch_lock(t: dict) -> None:
    """Mark a case as currently-running (resume bookkeeping)."""
    try:
        Path(t["case_dir"]).mkdir(parents=True, exist_ok=True)
        lock = Path(t["case_dir"]) / "result_running.lock"
        lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    except OSError:
        pass


def _finish_lock(t: dict, r: dict) -> None:
    """Record status/n_retry in result_summary.json and clear the running
    lock, so resume sees a clean state.

    Statuses written:
      - "done"    : solve completed (final result written)
      - "compiled": UDF pre-compiled, not yet solved (resume continues
                    straight to the solve phase)
      - "failed"  : compile or solve failed; n_retry incremented
    """
    sd = Path(t["case_dir"]) / "result_summary.json"
    try:
        if sd.exists():
            st = json.loads(sd.read_text(encoding="utf-8"))
        else:
            st = {}
        st.setdefault("condition", t["condition"])
        st.setdefault("airfoil", t["airfoil"])
        sts = r.get("status")
        if sts in ("done", "compiled"):
            st["status"] = sts
            st["n_retry"] = t.get("n_retry", 0)
            st["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            st["status"] = "failed"
            st["n_retry"] = t.get("n_retry", 0) + 1
            if r.get("error"):
                st["error"] = str(r["error"])[:500]
        sd.write_text(json.dumps(st, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    finally:
        try:
            lock = Path(t["case_dir"]) / "result_running.lock"
            if lock.exists():
                lock.unlink()
        except OSError:
            pass


def _worker_run(t):
    """Pool worker: run one case, return its result dict."""
    # The pool task dict may already carry a done/status from a previous
    # run; only execute cases that still need work.
    if t.get("done"):
        return dict(status="skipped", condition=t["condition"],
                    airfoil=t["airfoil"], case_dir=t["case_dir"],
                    reason="already-complete")
    _touch_lock(t)
    r = _run_case_isolated(t)
    _finish_lock(t, r)
    return r


if __name__ == "__main__":
    sys.exit(main())
