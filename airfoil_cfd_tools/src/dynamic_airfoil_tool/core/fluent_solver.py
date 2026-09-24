# -*- coding: utf-8 -*-
"""Fluent 求解器实现。

复用现有 auto_dynamic 的核心逻辑（UDF 生成 / case 构建 / journal /
编译 / 运行 / 后处理），但路径由本工具 config.py 指定（换电脑/换 Fluent
安装位置只需改 config.py）。输入输出遵循 SolverInterface。

工作目录:
    <user_specified_workdir>/case_{timestamp}/
        ├── udf/          # UDF + libudf
        ├── jobs/         # journal
        └── *.h5          # Fluent 原始数据

运行 Fluent 期间支持 data_cb 实时增量推送 (time, AoA, Cl, Cd)。
"""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .solver_interface import SolverInterface, SimParams, SimResult


def _load_tool_config():
    """按文件路径加载本工具 config.py（唯一模块名，避免与 auto_dynamic 的 config 冲突）。"""
    cfg_path = Path(__file__).resolve().parent.parent / "config.py"
    spec = importlib.util.spec_from_file_location("dyn_airfoil_config", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tool_config = _load_tool_config()

# 打包进软件内部的 Fluent 自动化核心（自包含，随软件整体分发）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AUTO_DIR = _PROJECT_ROOT / "fluent_auto"
_PY_DIR = _PROJECT_ROOT / "fluent_auto"


class FluentSolver(SolverInterface):
    name = "fluent"

    def __init__(self, fluent_exe: str, work_root: Path, nproc: int = None,
                 skip_udf_compile: bool = False, case_id: str = None):
        self.fluent_exe = fluent_exe
        self.work_root = Path(work_root)
        self.nproc = nproc if nproc else getattr(_tool_config, "NPROC", 4)
        self.skip_udf_compile = skip_udf_compile
        # Optional fixed case-id override (batch mode uses a stable
        # friendly dir name, e.g. case_cmaes_R1, instead of the default
        # timestamped one) so compile + solve land in the SAME directory.
        self.case_id_override = case_id
        self._proc = None   # 当前运行的 Fluent 进程句柄
        # 复用 auto_dynamic 逻辑（放 sys.path 最前，确保其内部 `import config`
        # 命中 auto_dynamic/config.py，而非本工具根目录的 config.py）
        for d in (str(_AUTO_DIR), str(_PY_DIR)):
            if d not in sys.path:
                sys.path.insert(0, d)

    def _prepare_config(self, params: SimParams):
        """构造 case 配置 dict 并设置 auto_dynamic 的 config 全局。"""
        import config as cfg
        import case_runner

        # 解析翼型名（网格文件名前缀，如 LS0417_0p4572_0p01_inflat.msh -> LS0417）
        # 注意: case_runner.set_case_config 会用 mesh_name 拼 _0p4572_0p01_inflat，
        # 因此这里只传翼型名（不传完整前缀，否则文件名重复）。
        mesh_stem = Path(params.mesh_path).stem
        airfoil = mesh_stem.split("_")[0]

        # 各翼型网格初始攻角（实测，读 .msh 后 replace-mesh 测 LE/TE 连线）。
        # 复位 UDF 从该初始角转到 AOA_START；固定 0 会导致翼型从 0° 开始俯仰。
        # 注: 用户新增的 naca4415_0p4572.msh 文件名是小写，而 airfoil_name()
        # 取 _ 前首段得到 "naca4415"（非 "NACA4415"），故需小写键。
        _af_initial_aoa = {
            "NACA4415": 0.236, "N4415": 0.236, "naca4415": 0.236,
            "S801": 0.113, "S809": 0.239, "S810": 0.036, "S812": -0.546,
            "S813": 0.129, "S814": -0.613, "S815": -0.819, "S825": -0.785,
            "LS0417": -0.108, "LS0421": -0.188,
        }

        case_cfg = dict(
            case_id=self.case_id_override or f"{airfoil}_{int(time.time())}",
            airfoil=airfoil,
            mesh_name=airfoil,
            initial_aoa=_af_initial_aoa.get(airfoil, 0.0),
            condition="D0",
            turbulence="T0",
            mean=params.mean_aoa,
            amp=params.amp_aoa,
            freq=params.freq,
            k=0.0,
            rho=params.rho,
            mu=params.mu,
            v_inf=params.V,
            T=288.16,
            P=101325.0,
            ti=params.TI,
            lt_c=params.Lt_over_c,
        )
        # 网格必须 c=0.4572
        if abs(params.chord - 0.4572) > 1e-6:
            raise ValueError("网格必须为 c=0.4572 (18 inch)")

        # Fluent / ANSYS 路径（来自本工具 config.py，换电脑只需改那里）
        cfg.FLUENT_EXE = self.fluent_exe
        cfg.ANSYS_ROOT = _tool_config.ANSYS_ROOT
        cfg.FLUENT_INC = str(_tool_config.ANSYS_ROOT / "fluent")
        # 自动探测 ANSYS 自带的 CPython（版本号随 ANSYS 版本变化，避免硬编码 3_10）
        _cpython_root = _tool_config.ANSYS_ROOT / "commonfiles" / "CPython"
        _py_dir = None
        if _cpython_root.is_dir():
            for _v in sorted([d for d in _cpython_root.iterdir() if d.is_dir()],
                             reverse=True):
                _cand = _v / "winx64" / "Release" / "python"
                if _cand.exists():
                    _py_dir = _cand
                    break
        cfg.PYTHON_DIR = _py_dir if _py_dir is not None else (
            _tool_config.ANSYS_ROOT / "commonfiles" / "CPython" / "3_10"
            / "winx64" / "Release" / "python")
        cfg.ANSYS_ENV = {"FLUENT_INC": cfg.FLUENT_INC}
        # 软件包内自带资源（参考 case + UDF 构建骨架）
        cfg.REF_PROJECT = _PROJECT_ROOT / "resources"
        cfg.UDF_REF_DIR = _PROJECT_ROOT / "resources" / "udf_build"
        cfg.FLUENT_RELEASE = _tool_config.FLUENT_RELEASE

        case_runner.set_case_config(case_cfg)
        # set_case_config 内部会把 OUT_DIR 重置为 auto_dynamic 的
        # OUT_ROOT/CASE_ID（其 cas 目录）。重新覆盖为本工具指定的工作目录，
        # 数据不再写入 auto_dynamic 的 cas。
        case_dir = self.work_root / f"case_{case_cfg['case_id']}"
        case_dir.mkdir(parents=True, exist_ok=True)
        cfg.OUT_ROOT = self.work_root
        cfg.OUT_DIR = case_dir
        cfg.UDF_OUT_DIR = case_dir / "udf"
        cfg.JOB_DIR = case_dir / "jobs"
        # 网格：复制一份到本 case 工作目录，避免多个并发 Fluent 同时
        # replace-mesh 读同一个 .msh 导致 eof inside list（4 并行实测 R4 系列
        # 反复失败即源于此——同一网格的多个并发 case 争抢读同一文件）。
        src_mesh = Path(params.mesh_path)
        local_mesh = case_dir / src_mesh.name
        if not local_mesh.exists() or local_mesh.stat().st_size != src_mesh.stat().st_size:
            try:
                import shutil as _sh
                _sh.copy2(src_mesh, local_mesh)
            except OSError:
                local_mesh = src_mesh
        cfg.MESH_DIR = local_mesh.parent
        cfg.MESH_NAME = local_mesh.name
        case_runner.reset_module_state()
        # 生成工况参数表（供 plot_forces.py / 外部直接读取画图，避免重复猜测参数）
        import json as _json
        _params_info = {
            "airfoil": airfoil,
            "mesh": Path(params.mesh_path).name,
            "chord": params.chord,
            "mean_aoa": params.mean_aoa,
            "amp_aoa": params.amp_aoa,
            "freq": params.freq,
            "rho": params.rho,
            "mu": params.mu,
            "V": params.V,
            "TI": params.TI,
            "Lt_over_c": params.Lt_over_c,
        }
        (case_dir / "case_params.json").write_text(
            _json.dumps(_params_info, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return case_cfg

    def run(self, params: SimParams, progress_cb=None,
            stop_event=None, data_cb=None) -> SimResult:
        params.validate()
        # 工作目录
        self.work_root.mkdir(parents=True, exist_ok=True)
        case_cfg = self._prepare_config(params)

        import case_runner, gen_udf, build_case, gen_journal
        import config as cfg

        # 清理该 case 目录旧 Fluent 输出（*.h5），避免 write-case-data 遇已存在
        # 文件时提示覆盖确认导致 journal 中断（journal 里也已用 confirm-overwrite? no）
        for _f in cfg.OUT_DIR.glob("*.h5"):
            try:
                _f.unlink()
            except OSError:
                pass

        # 1) 生成 UDF / case / journal
        if progress_cb:
            progress_cb(0.05)
        gen_udf.main()
        build_case.build_case()
        gen_journal.main()

        # 2) 编译 libudf（批处理预编译时跳过，直接用已编译产物）
        if not self.skip_udf_compile:
            if progress_cb:
                progress_cb(0.15)
            ok = case_runner.compile_udf()
            if not ok:
                raise RuntimeError("UDF 编译失败")

        # 3) 运行 Fluent（完整仿真，可停止，运行中实时推送数据）
        jou = cfg.JOB_DIR / f"{cfg.CASE_ID}.jou"
        if progress_cb:
            progress_cb(0.2)
        live = dict(seen=set(), geom=None, wall_offset=None, ready=False,
                    stability={})
        if data_cb:
            import compare_sim_auto as csa
            csa.set_case_params(rho=params.rho, vel=params.V, mu=params.mu,
                                freq=params.freq, ao0=params.mean_aoa,
                                ao1=params.amp_aoa)
            live["csa"] = csa
        self._run_fluent_async(jou, self.nproc, stop_event,
                               data_cb=data_cb, live=live)

        # 4) 后处理提取 Cl/Cd/AoA
        if progress_cb:
            progress_cb(0.9)
        sim = self._postprocess(params, case_cfg)
        if progress_cb:
            progress_cb(1.0)
        return sim

    def stop(self):
        """仅终止由本工具启动的 Fluent 进程树（不误杀其它 Fluent）。

        用 taskkill /T /PID 只杀 self._proc 及其子进程（含 MPI 节点），
        避免按镜像名 taskkill /IM 误杀系统上其它工程/脚本的 Fluent。
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, text=True, timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _run_fluent_async(self, jou_path, nproc, stop_event=None,
                          data_cb=None, live=None):
        """非阻塞运行 Fluent，支持 stop_event 终止 + data_cb 实时数据。"""
        import config as cfg
        import case_runner
        fluent = case_runner.find_fluent()
        if not fluent:
            raise RuntimeError("未找到 Fluent 可执行文件")
        cmd = [fluent, "2ddp", "-g", f"-t{nproc}", "-i", str(jou_path)]
        env = {**os.environ, **cfg.ANSYS_ENV}
        # 每个 Fluent 进程独立 TEMP 目录：多个并发 Fluent 同时执行
        # /file/replace-mesh 时会在系统 %TEMP% 写同名 Settings 临时文件
        # （如 Temp\5、Temp\6）并互相覆盖，导致 "Reading Settings file"
        # 后报 CAR: invalid argument 中断 journal。把 TEMP 指到本 case
        # 目录下唯一的 tmp/ 子目录，物理隔离该竞争（8 并发实测 7 个
        # 失败即源于此；4 并发不触发只是时序巧合）。
        tmp_dir = cfg.OUT_DIR / "tmp"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            env["TEMP"] = str(tmp_dir)
            env["TMP"] = str(tmp_dir)
        except OSError:
            pass
        proc = subprocess.Popen(cmd, cwd=str(cfg.OUT_DIR), env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        self._proc = proc
        last_poll = 0.0
        try:
            # 轮询等待，可停止
            while proc.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    self.stop()
                    raise InterruptedError("仿真已停止")
                now = time.time()
                if data_cb and live and now - last_poll >= 0.5:
                    last_poll = now
                    pts = self._poll_live(live, cfg.OUT_DIR)
                    if pts:
                        data_cb(pts)
                time.sleep(0.2)
            if proc.returncode != 0:
                raise RuntimeError(f"Fluent 返回码 {proc.returncode}")
        finally:
            self._proc = None

    def _poll_live(self, live, out_dir):
        """增量读取 OUT_DIR 中新增 .dat.h5，实时解析 Cl/Cd/AoA。

        关键: Fluent(MPI) 正写 .dat.h5 时若被 Python 的 h5py 打开，会触发
        HDF5 文件锁冲突导致 Fluent 写入 Permission denied。因此只读取
        "文件大小连续两次轮询不变"（Fluent 已写完关闭）的文件。

        live: dict(seen=已处理文件, stability={path: size}, geom, wall_offset,
                 ready)。攻角用理论值（实时预览），完成后 _postprocess 做全量
        mesh-aoa 校正。
        """
        csa = live.get("csa")
        if csa is None:
            return []
        out = Path(out_dir)
        if not out.is_dir():
            return []
        if not live["ready"]:
            # 与 dat 一样，只解析"已稳定"的 cas：Fluent 正写 .cas.h5 时若被
            # h5py 打开会触发 HDF5 文件锁冲突（实测 Fluent 写 Permission denied）。
            for cf in sorted(out.glob("*.cas.h5")):
                fp = str(cf)
                try:
                    sz = cf.stat().st_size
                except OSError:
                    continue
                prev = live["stability"].get(fp)
                if prev is not None and prev == sz and sz > 0:
                    try:
                        zmin, zmax = csa.parse_zone_topology(fp)
                        centroids, normals, lengths = csa.build_static_geometry(
                            fp, zmin, zmax)
                        live["geom"] = (zmin, zmax, centroids, normals, lengths)
                        live["wall_offset"] = csa.find_wall_offset(fp, zmin)
                        live["ready"] = True
                        live["stability"].pop(fp, None)
                        break
                    except Exception:
                        pass
                else:
                    live["stability"][fp] = sz
        if not live["ready"]:
            return []
        pts = []
        stable = live["stability"]
        for f in sorted(out.glob("*.dat.h5")):
            fp = str(f)
            if fp in live["seen"]:
                continue
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            prev = stable.get(fp)
            if prev is not None and prev == sz and sz > 0:
                # 文件已稳定（Fluent 写完关闭），此刻打开读取安全
                stable.pop(fp, None)
                r = self._read_live_point(live, csa, fp)
                live["seen"].add(fp)   # 无论成败都标记，避免反复重试
                if r is not None:
                    pts.append(r)
            else:
                # 正在写入或首次出现：记录大小，下轮再判断是否稳定
                stable[fp] = sz
        return pts

    def _read_live_point(self, live, csa, fp):
        """读取单个已稳定 .dat.h5，返回 (time, AoA, Cl, Cd)；失败返回 None。"""
        try:
            ft = csa.parse_flow_time(fp)
        except Exception:
            return None
        if ft < 1.0:   # 复位阶段(t<1s)后即显示：俯仰数据（UDF 复位门控=1s）
            return None
        try:
            aoa_deg = csa.aoa_at(ft)
            zmin, zmax, centroids, normals, lengths = live["geom"]
            res = csa.compute_forces(fp, aoa_deg, zmin, zmax, centroids,
                                     normals, lengths, live["wall_offset"])
            if res:
                return (float(res[0]), float(res[1]),
                        float(res[2]), float(res[3]))
        except Exception:
            pass
        return None

    def _postprocess(self, params: SimParams, case_cfg) -> SimResult:
        """从 Fluent 输出提取 time/AoA/Cl/Cd。"""
        import compare_sim_auto as csa
        import config as cfg

        sim_dir = str(cfg.OUT_DIR)
        csa.set_case_params(rho=params.rho, vel=params.V, mu=params.mu,
                            freq=params.freq, ao0=params.mean_aoa,
                            ao1=params.amp_aoa)
        # 攻角用理论值（UDF 实际施加攻角）：mesh_aoa 在深失速区测量不可靠，
        # 偏移校正不稳定（S810 实测 -0.44 ~ +1.08° 波动）。
        sim = csa.load_simulation(sim_dir, dt=0.05, skip_cycles=0,
                                  cas_arg=None, use_mesh_aoa=False)
        if sim is None:
            raise RuntimeError("后处理失败：未提取到仿真数据")
        return SimResult(
            time=sim["times"], aoa=sim["aoas"], cl=sim["cl"], cd=sim["cd"],
            mesh_name=Path(params.mesh_path).name, params=params,
            source="fluent")
