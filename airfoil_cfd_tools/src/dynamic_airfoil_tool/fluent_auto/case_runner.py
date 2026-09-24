# -*- coding: utf-8 -*-
"""单 case 执行器（方案A：每个 case 独立 UDF/libudf）。

用法:
    python case_runner.py --job <case_config.json> [--run] [-t 4]

由 batch_run.py 在独立子进程中调用。每个 case 有独立工作目录:
    cas/{case_id}/
        ├── udf/          # 该 case 的 UDF 源 + libudf（独立编译）
        ├── jobs/         # 该 case 的 journal
        └── {base}*.h5    # 输出数据（含 -9-00000 / -10-0000N / -final）
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

# 必须在设置 config 前 import config（后续用 setattr 覆盖）
import config


def set_case_config(cfg: dict) -> None:
    """根据 case 配置动态覆盖 config 模块全局。"""
    # 路径
    config.AIRFOIL = cfg["airfoil"]
    config.D_CASE = cfg["condition"]
    config.T_CASE = cfg["turbulence"]
    config.CASE_ID = cfg["case_id"]
    # 网格前缀（如 N4415 -> NACA4415）；BASE_NAME 用于网格与输出文件名
    mesh_prefix = cfg.get("mesh_name", cfg["airfoil"])
    config.BASE_NAME = f"{mesh_prefix}_0p4572_0p01_inflat"

    # 每个 case 独立工作目录（方案A）
    config.OUT_DIR = config.OUT_ROOT / config.CASE_ID
    config.UDF_OUT_DIR = config.OUT_DIR / "udf"
    config.JOB_DIR = config.OUT_DIR / "jobs"

    # 网格（不同翼型用不同网格文件）
    config.MESH_NAME = f"{config.BASE_NAME}.msh"

    # 动态工况
    config.AOA_MEAN = cfg["mean"]
    config.AOA_AMP = cfg["amp"]
    config.FREQ = cfg["freq"]
    config.K = cfg["k"]
    config.AOA_START = config.AOA_MEAN - config.AOA_AMP

    # 物性（来自 d_table 换算）
    config.RHO = cfg["rho"]
    config.MU = cfg["mu"]
    config.V_INF = cfg["v_inf"]
    config.RE = config.RHO * config.V_INF * config.CHORD / config.MU
    config.T_F = cfg.get("T", 288.16)
    config.P_PSIA = cfg.get("P", 14.696)

    # 湍流
    config.TI = cfg["ti"]
    config.LT_OVER_C = cfg["lt_c"]
    config.LT = config.LT_OVER_C * config.CHORD
    config.TURB_INTENSITY_PCT = config.TI * 100.0
    config.TURB_LENGTH_SCALE = config.LT

    # 时间推进（按工况自适应）
    # 大振幅深失速工况（振幅>=10°，D2/D4/D6）用更小时间步（每周期 54 步，
    # D4/D6 dt≈0.01s）保证分离流稳定；其余工况 33 步/周期（dt≈0.05s）。
    # 深失速时出口回流/分离剧烈，0.05s 步长会导致浮点发散（实测 D4T6）。
    config.PERIOD = 1.0 / config.FREQ
    if config.AOA_AMP >= 10.0:
        config.STEPS_PER_CYCLE = 54
    else:
        config.STEPS_PER_CYCLE = 33
    config.PITCH_DT = config.PERIOD / config.STEPS_PER_CYCLE
    config.PITCH_STEPS = config.PITCH_CYCLES * config.STEPS_PER_CYCLE

    # 网格初始攻角：各翼型实测（读 .msh 测 LE/TE），默认约 0°
    config.INITIAL_AOA = cfg.get("initial_aoa", 0.0)


def reset_module_state() -> None:
    """build_case.py 的 REF_CAS 固定指向 NACA4415 参考工程。

    参考工程只有 NACA4415 的 .cas（含全部模型/BC/动态网格/滑移界面设置）。
    其他翼型网格拓扑一致（同批生成），在 journal 里用 /file/replace-mesh
    把 case 中的 NACA4415 网格替换为目标翼型网格即可（保留设置）。
    """
    import build_case
    build_case.REF_CAS = config.REF_PROJECT / "NACA4415_0p4572_0p01_inflat.cas"


def generate_files() -> dict:
    """生成 UDF、case、journal（复用现有脚本，已设好 config）。"""
    import gen_udf, build_case, gen_journal

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.JOB_DIR.mkdir(parents=True, exist_ok=True)
    config.UDF_OUT_DIR.mkdir(parents=True, exist_ok=True)

    gen_udf.main()
    case_path, _ = build_case.build_case()
    gen_journal.main()
    jou_path = config.JOB_DIR / f"{config.CASE_ID}.jou"
    return {"case": case_path, "journal": jou_path}


def find_fluent() -> str | None:
    if config.FLUENT_EXE and Path(config.FLUENT_EXE).exists():
        return config.FLUENT_EXE
    found = shutil.which("fluent")
    return found


def find_scons() -> Path:
    """Locate the ANSYS-bundled scons.exe, auto-detecting the CPython
    version subdirectory (3_10 / 3_11 / ...) so the package works on any
    ANSYS install without hardcoding a Python version."""
    cpython_root = config.ANSYS_ROOT / "commonfiles" / "CPython"
    if cpython_root.is_dir():
        # Prefer the newest CPython version folder found
        vers = sorted([d for d in cpython_root.iterdir() if d.is_dir()],
                      reverse=True)
        for v in vers:
            cand = v / "winx64" / "Release" / "python" / "Scripts" / "scons.exe"
            if cand.exists():
                return cand
    # Fallback to the config-provided PYTHON_DIR
    return (config.PYTHON_DIR / "Scripts" / "scons.exe")


def compile_udf() -> bool:
    """手动编译 libudf（host+node），方案A：每 case 独立。

    UDF 编译是 CPU/磁盘密集且**跨 case 共享全局缓存**（scons 的
    $HOME/.sconsign.dblite 以及 clang 的模块缓存），并发多个 case 同时
    编译会互相踩踏（WinError 32/145 目录占用、UDF 编译失败）。
    这里加一个进程间互斥锁：同一时刻只允许一个进程编译 UDF，
    其余 worker 串行等待（用 waitpid 语义轮询锁文件），
    编译完成后释放，Fluent 求解仍可完全并行。
    """
    scons = find_scons()
    if not scons.exists():
        print(f"!! scons 未找到: {scons}")
        return False

    # ---- inter-process mutex: serialize UDF compilation ----
    import msvcrt
    lock_path = Path(os.environ.get("TEMP", ".")) / "dynamic_airfoil_udf_compile.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    except OSError:
        lock_fd = None
    try:
        if lock_fd is not None:
            # Non-blocking acquire; if held, wait (poll 0.5s) until free.
            waited = 0.0
            while True:
                try:
                    msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                    break  # acquired
                except OSError:
                    if waited > 3600:
                        print("!! UDF 编译锁等待超时(>1h)")
                        return False
                    time.sleep(0.5)
                    waited += 0.5

        try:
            return _compile_udf_locked(scons)
        finally:
            if lock_fd is not None:
                try:
                    msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


def _compile_udf_locked(scons) -> bool:
    """实际编译逻辑（已在互斥锁保护下执行）。"""
    libudf = config.UDF_OUT_DIR / "libudf"
    libudf.mkdir(parents=True, exist_ok=True)
    src_dir = libudf / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    for name in ["airfoil_motion.c", "reset_ang.c", "rotate_sine.c"]:
        shutil.copy2(config.UDF_OUT_DIR / name, src_dir / name)
    sconstruct_src = config.ANSYS_ROOT / "fluent" / config.FLUENT_RELEASE / "src" / "udf" / "sconstruct.udf"
    if sconstruct_src.exists():
        shutil.copy2(sconstruct_src, libudf / "sconstruct")

    env = {**os.environ, **config.ANSYS_ENV,
           "FLUENT_UDF_COMPILER": "clang", "FLUENT_UDF_CLANG": "builtin",
           "PYTHONHOME": str(config.PYTHON_DIR), "PYTHONPATH": str(config.PYTHON_DIR)}

    ok = True
    for ver in ["2ddp_host", "2ddp_node"]:
        build_dir = libudf / "win64" / ver
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        ref_build = config.UDF_REF_DIR / ver
        if ref_build.exists():
            shutil.copy2(ref_build / "scons.bat", build_dir / "scons.bat")
            shutil.copy2(ref_build / "SConstruct", build_dir / "SConstruct")
        host_flags = "-DUSE_UDF_HOST_DUMMY" if ver == "2ddp_host" else ""
        user_txt = (f"VERSION='{ver}'\nFLUENT_ARCH='win64'\n"
                    f"FLUENT_RELEASE='{config.FLUENT_RELEASE}'\n"
                    f"CSOURCES=' $(SRC)airfoil_motion.c $(SRC)reset_ang.c $(SRC)rotate_sine.c'\n"
                    f"HSOURCES=''\nHOST_FLAGS='{host_flags}'\nGPU_SUPPORT='off'\n")
        (build_dir / "user.txt").write_text(user_txt, encoding="ascii")
        r = subprocess.run([str(scons), "-s"], cwd=str(build_dir), env=env,
                           capture_output=True, text=True)
        if not (build_dir / "libudf.dll").exists():
            print(f"!! UDF 编译失败 ({ver}):\n{r.stderr[-500:]}")
            ok = False
        else:
            print(f"  UDF 编译成功 ({ver})")
    return ok


def run_fluent(jou_path: Path, nproc: int) -> int:
    fluent = find_fluent()
    if not fluent:
        print("!! 未找到 Fluent 可执行文件。")
        return 2
    cmd = [fluent, "2ddp", "-g", f"-t{nproc}", "-i", str(jou_path)]
    print(">> 运行 Fluent:", subprocess.list2cmdline(cmd))
    env = {**os.environ, **config.ANSYS_ENV}
    # 与 core/fluent_solver.py 的 _run_fluent_async 保持一致：给每个
    # Fluent 进程独立 TEMP 目录，避免并发 replace-mesh 写同名 Settings
    # 临时文件互相覆盖导致 "Reading Settings file" 后 CAR 错误中断。
    tmp_dir = config.OUT_DIR / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env["TEMP"] = str(tmp_dir)
        env["TMP"] = str(tmp_dir)
    except OSError:
        pass
    result = subprocess.run(cmd, cwd=str(config.OUT_DIR), env=env)
    if result.returncode == 0:
        print(">> Fluent 运行完成。")
    else:
        print(f">> Fluent 返回码 {result.returncode}")
    return result.returncode


def is_complete(case_id: str) -> bool:
    """检查 case 是否已完成（存在 -final.dat.h5 且 flow-time>0）。"""
    out_dir = config.OUT_ROOT / case_id
    final = out_dir / f"{config.BASE_NAME}-final.dat.h5"
    return final.exists()


def main() -> int:
    ap = argparse.ArgumentParser(description="单 case 仿真执行器")
    ap.add_argument("--job", required=True, help="case 配置 JSON 路径")
    ap.add_argument("--run", action="store_true", help="生成后运行 Fluent")
    ap.add_argument("-t", "--nproc", type=int, default=4, help="Fluent 核数 (默认 4)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.job).read_text(encoding="utf-8"))
    set_case_config(cfg)
    reset_module_state()

    print("=" * 68)
    print(f"  Case: {config.CASE_ID}  ({config.AIRFOIL} {config.D_CASE}{config.T_CASE})")
    print(f"  motion: alpha={config.AOA_MEAN}±{config.AOA_AMP} deg, "
          f"f={config.FREQ} Hz, k={config.K}, V={config.V_INF:.3f} m/s")
    print(f"  rho={config.RHO:.5f}, mu={config.MU:.3e}, "
          f"TI={config.TI*100:.1f}%, Lt/c={config.LT_OVER_C}")
    print("=" * 68)

    # 断点续跑：已完整生成且无 --force 时跳过
    if not args.run:
        # 仅生成模式
        generate_files()
        print(f"已生成文件: {config.OUT_DIR}")
        return 0

    # 运行模式
    paths = generate_files()
    ok = compile_udf()
    if not ok:
        print("!! UDF 编译失败，中止")
        return 1
    ret = run_fluent(paths["journal"], args.nproc)
    return ret


if __name__ == "__main__":
    sys.exit(main())
