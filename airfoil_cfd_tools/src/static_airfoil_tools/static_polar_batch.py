# -*- coding: utf-8 -*-
"""静态气动极曲线扫掠驱动（llm / naca4415 × 0.75/1.0/1.5M × 0-30°）。

流程（每 (airfoil, Re, α) 一个 case）:
  1. static_case.build_static_case  -> 稳态 .cas（来流倾 β=α+initial_aoa,
     无 zone-motion, rp-unsteady #f, autosave 关, 烘焙 Re 物性/BC）
  2. 网格副本入 case 目录
  3. static_journal.gen_static_journal -> 稳态 journal
  4. 隔离子进程跑 Fluent: fluent 2ddp -g -t{nproc} -i journal
  5. static_post.static_polar_read     -> Cl/Cd
  6. 写 case_params.json / result_summary.json
聚合: polar_{af}_Re{tag}.csv / polar_{af}.csv / 图

用法:
  python static_polar_batch.py --re 0p75 1p0 1p5 --angles 0 2 4 ... --airfoils llm naca4415
  python static_polar_batch.py --angles-range 0 30        # 全量 0..30 每 1°
  --nproc 4 --parallel 2 --n-iter 800 --retry 1 --force --dry-run
单点子进程: python static_polar_batch.py --run-one <task.json>   (worker)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import config

# ----------------------------------------------------------------------
# 路径 / 复用 dynamic 工程的 batch_auto_run 辅助
# ----------------------------------------------------------------------
def _setup_paths():
    """把 dynamic 工程 root 与 fluent_auto 加入 sys.path（worker 可 import）。"""
    root = str(config.DYNAMIC_TOOL_ROOT)
    for d in (root, str(config.FLUENT_AUTO_DIR)):
        if d not in sys.path:
            sys.path.insert(0, d)


def _wt_air_properties(re_tag):
    """经 dynamic batch_auto_run.wt_air_properties 取物性（勿硬编码）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dyn_batch_auto", config.BATCH_AUTO_RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.wt_air_properties(config.RE_TAGS[re_tag])


def _initial_aoa(airfoil):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dyn_batch_auto", config.BATCH_AUTO_RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AF_INITIAL_AOA.get(airfoil, 0.0)


def case_id(af, re_tag, aoa):
    return f"case_{af}_S{re_tag}_a{int(aoa):02d}"


# ----------------------------------------------------------------------
# 单点 worker（在隔离子进程运行）
# ----------------------------------------------------------------------
def run_one(t: dict) -> dict:
    """跑一个 (af, re, aoa) 静态点，返回 result dict。"""
    t0 = time.time()
    af, re_tag, aoa = t["airfoil"], t["re_tag"], t["aoa"]
    out_root = Path(t["out_dir"])
    case_dir = out_root / case_id(af, re_tag, aoa)
    case_dir.mkdir(parents=True, exist_ok=True)

    import static_case, static_journal, static_post

    wtp = _wt_air_properties(re_tag)

    # 1. 静态 case 文本
    cas_path, _changes = static_case.build_static_case(
        af, re_tag, aoa, wtp, case_dir)

    # 2. 网格副本（并发 replace-mesh 读同 .msh 会竞争，每 case 独立副本）
    src_mesh = Path(config.MESH_DIR) / f"{af}_0p4572.msh"
    if not src_mesh.exists():
        return dict(status="failed", airfoil=af, re_tag=re_tag, aoa=aoa,
                    case_dir=str(case_dir), error=f"网格不存在 {src_mesh}")
    local_mesh = case_dir / src_mesh.name
    if not local_mesh.exists() or local_mesh.stat().st_size != src_mesh.stat().st_size:
        import shutil
        shutil.copy2(src_mesh, local_mesh)

    # 3. journal（replace-mesh 换入目标网格）
    base = f"{af}_S{re_tag}_a{int(aoa):02d}"
    # 来流倾 β = α + initial_aoa（几何补偿）
    beta = aoa + _initial_aoa(af)
    jou = static_journal.gen_static_journal(
        cas_path, local_mesh, case_dir, base,
        n_iter=t.get("n_iter", 800), V=wtp["V"], beta_deg=beta,
        need_replace=True)
    jou_path = case_dir / f"{base}.jou"
    jou_path.write_text(jou, encoding="utf-8")

    # 4. 清旧 h5（避免残留干扰后处理）
    for f in case_dir.glob("*.h5"):
        try:
            f.unlink()
        except OSError:
            pass

    # 5. 跑 Fluent（独立 TEMP 目录，防并发 replace-mesh Settings 竞争）
    env = {**os.environ}
    tmp_dir = case_dir / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env["TEMP"] = str(tmp_dir)
        env["TMP"] = str(tmp_dir)
    except OSError:
        pass
    cmd = [config.FLUENT_EXE, "2ddp", "-g",
           f"-t{t.get('nproc', config.NPROC)}", "-i", str(jou_path)]
    # Fluent 输出进其 .trn 即可；DEVNULL 避免 GBK 输出混入 worker stdout
    # 导致父进程 capture 解码崩溃（UnicodeDecodeError）。
    try:
        rc = subprocess.run(cmd, cwd=str(case_dir), env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=t.get("timeout_s", None)).returncode
    except subprocess.TimeoutExpired:
        return dict(status="failed", airfoil=af, re_tag=re_tag, aoa=aoa,
                    case_dir=str(case_dir), error="Fluent 超时")
    except Exception as e:
        return dict(status="failed", airfoil=af, re_tag=re_tag, aoa=aoa,
                    case_dir=str(case_dir), error=f"运行异常: {e}")
    if rc != 0:
        return dict(status="failed", airfoil=af, re_tag=re_tag, aoa=aoa,
                    case_dir=str(case_dir), error=f"Fluent 返回码 {rc}")

    # 6. 后处理（传入 initial_aoa 供来流系力分解）
    try:
        res = static_post.static_polar_read(
            case_dir, aoa, wtp["rho"], wtp["V"], wtp["mu"],
            initial_aoa=_initial_aoa(af))
    except Exception as e:
        return dict(status="failed", airfoil=af, re_tag=re_tag, aoa=aoa,
                    case_dir=str(case_dir), error=f"后处理失败: {e}")

    # 7. 写 deliverables
    params = dict(airfoil=af, re_tag=re_tag, re=config.RE_TAGS[re_tag],
                  aoa_deg=aoa, rho=wtp["rho"], mu=wtp["mu"], V=wtp["V"],
                  TI=config.TI, Lt_over_c=config.LT_OVER_C,
                  n_iter=t.get("n_iter", 800), initial_aoa=_initial_aoa(af))
    (case_dir / "case_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    summ = dict(status="done", **params, cl=res["cl"], cd=res["cd"],
                cd_pres=res["cd_pres"], mesh_aoa=res["mesh_aoa"],
                elapsed_s=round(time.time() - t0, 1))
    (case_dir / "result_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"[{af} S{re_tag} a{int(aoa):02d}] done  Cl={res['cl']:.4f} "
        f"Cd={res['cd']:.5f} ({time.time()-t0:.0f}s)")
    return summ


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# 任务构建 + resume + 聚合
# ----------------------------------------------------------------------
def build_tasks(airfoils, re_tags, angles, out_dir, nproc, n_iter):
    tasks = []
    for af in airfoils:
        for re_tag in re_tags:
            for aoa in angles:
                tasks.append(dict(airfoil=af, re_tag=re_tag, aoa=float(aoa),
                                  out_dir=str(out_dir), nproc=nproc,
                                  n_iter=n_iter))
    return tasks


def _is_done(t):
    sd = Path(t["out_dir"]) / case_id(t["airfoil"], t["re_tag"], t["aoa"]) \
        / "result_summary.json"
    if not sd.exists():
        return False
    try:
        return json.loads(sd.read_text(encoding="utf-8")).get("status") == "done"
    except Exception:
        return False


def _touch_lock(t):
    d = Path(t["out_dir"]) / case_id(t["airfoil"], t["re_tag"], t["aoa"])
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "result_running.lock").write_text(
            time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    except OSError:
        pass


def _clear_lock(t):
    lock = Path(t["out_dir"]) / case_id(t["airfoil"], t["re_tag"], t["aoa"]) \
        / "result_running.lock"
    try:
        lock.unlink()
    except OSError:
        pass


def _run_one_worker(task_json):
    """(picklable 顶层) 多进程池 worker: 跑单个静态点并返回 result dict。"""
    import json
    t = json.loads(task_json)
    return run_one(t)


def _run_parallel(args, tasks) -> list:
    """并行执行: 用 multiprocessing 池跑 run_one（每 worker 内 Fluent 子进程）。

    Windows spawn 需顶层可 pickle 的 worker。每 worker 独立进程，模块全局
    态互不串扰；case 间无共享文件（网格已各自复制），Fluent 独立 TEMP。
    """
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    n_worker = max(1, min(args.parallel, len(tasks)))
    log(f"并行求解: {len(tasks)} 个 case, {n_worker} workers")
    results = []
    with ctx.Pool(processes=n_worker) as pool:
        for r in pool.imap_unordered(_run_one_worker,
                                     [json.dumps(t) for t in tasks],
                                     chunksize=1):
            results.append(r)
            n_done = sum(1 for x in results if x.get("status") == "done")
            n_fail = sum(1 for x in results if x.get("status") == "failed")
            if r.get("status") == "failed":
                log(f"  !! 失败: {r.get('airfoil')} {r.get('re_tag')} "
                    f"a{int(r.get('aoa', -1))}: {r.get('error')}")
            else:
                log(f"  进度 {len(results)}/{len(tasks)} "
                    f"(成功 {n_done}, 失败 {n_fail})")
    return results


def run_batch(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    angles = args.angles
    if args.angles_range:
        lo, hi = args.angles_range
        angles = list(range(lo, hi + 1))
    tasks = build_tasks(args.airfoils, args.re, angles, out_dir,
                        args.nproc, args.n_iter)
    if args.dry_run:
        log(f"任务 {len(tasks)} 个 (dry-run, 不求解):")
        for t in tasks[:5] + (tasks[-3:] if len(tasks) > 5 else []):
            log(f"  {case_id(t['airfoil'], t['re_tag'], t['aoa'])}")
        if len(tasks) > 8:
            log(f"  ... 共 {len(tasks)} 个")
        return 0

    pending = [t for t in tasks if not _is_done(t)]
    log(f"任务 {len(tasks)} 个, 已完成 {len(tasks)-len(pending)}, "
        f"待跑 {len(pending)}")
    if args.force:
        pending = tasks

    # 并行: 池内跑 run_one（内含 touch/clear lock 由 run_one 自己管理? 不:
    # 锁在池外不可靠——run_one 内不设锁；resume 靠 result_summary.json）
    if args.parallel and args.parallel > 1:
        results = _run_parallel(args, pending)
        for r in results:
            if r.get("status") == "failed":
                log(f"  !! 失败: {r.get('error')}")
    else:
        results = []
        failed = 0
        for i, t in enumerate(pending):
            log(f"[{i+1}/{len(pending)}] 启动 {case_id(t['airfoil'], t['re_tag'], t['aoa'])}")
            _touch_lock(t)
            code = ("import json,sys;sys.path.insert(0,%r);"
                    "from static_polar_batch import run_one;"
                    "r=run_one(json.loads(sys.argv[1]));"
                    "print('BATCH_RESULT '+json.dumps(r))") % str(Path(__file__).parent)
            try:
                p = subprocess.run([sys.executable, "-c", code, json.dumps(t)],
                                   capture_output=True, text=False,
                                   cwd=str(Path(__file__).parent),
                                   timeout=args.timeout)
                out = (p.stdout or b"").decode("utf-8", errors="replace")
                errs = (p.stderr or b"").decode("utf-8", errors="replace")
                marker = "BATCH_RESULT "
                idx = out.find(marker)
                if idx >= 0:
                    r = json.loads(out[idx + len(marker):].strip().splitlines()[0])
                else:
                    r = dict(status="failed", airfoil=t["airfoil"],
                             re_tag=t["re_tag"], aoa=t["aoa"],
                             case_dir=str(Path(t["out_dir"]) / case_id(
                                 t["airfoil"], t["re_tag"], t["aoa"])),
                             error=f"worker exit={p.returncode} "
                                   f"{(out + errs)[-800:]}")
            except subprocess.TimeoutExpired:
                r = dict(status="failed", airfoil=t["airfoil"], re_tag=t["re_tag"],
                         aoa=t["aoa"], error="worker 超时")
            results.append(r)
            if r.get("status") == "failed":
                failed += 1
                log(f"  !! 失败: {r.get('error')}")
            _clear_lock(t)

    # 聚合
    assemble(out_dir, args.airfoils, args.re)
    n_done = sum(1 for r in results if r.get("status") == "done")
    n_fail = sum(1 for r in results if r.get("status") == "failed")
    log(f"完成: 成功 {n_done} / 失败 {n_fail}")
    return 0 if n_fail == 0 else 1


def assemble(out_dir: Path, airfoils, re_tags):
    """读各 case result_summary.json 聚合 CSV。"""
    for af in airfoils:
        for re_tag in re_tags:
            rows = []
            for aoa in range(0, 31):
                d = out_dir / case_id(af, re_tag, aoa)
                sd = d / "result_summary.json"
                if not sd.exists():
                    continue
                try:
                    s = json.loads(sd.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if s.get("status") != "done":
                    continue
                rows.append((s["aoa_deg"], s["cl"], s["cd"], s["cd_pres"]))
            rows.sort()
            if not rows:
                continue
            csv = out_dir / f"polar_{af}_Re{re_tag}.csv"
            with open(csv, "w", encoding="utf-8") as f:
                f.write("aoa,cl,cd,cd_pres\n")
                for a, cl, cd, cdp in rows:
                    f.write(f"{a:.1f},{cl:.6f},{cd:.6f},{cdp:.6f}\n")
            log(f"聚合 -> {csv} ({len(rows)} 点)")


def main():
    ap = argparse.ArgumentParser(description="静态气动极曲线扫掠")
    ap.add_argument("--airfoils", nargs="*", default=config.AIRFOILS)
    ap.add_argument("--re", nargs="*", default=list(config.RE_TAGS.keys()))
    ap.add_argument("--angles", nargs="*", type=float, default=None,
                    help="具体攻角列表，如 --angles 0 5 8 12")
    ap.add_argument("--angles-range", nargs=2, type=int, default=None,
                    help="攻角范围(含)，如 --angles-range 0 30")
    ap.add_argument("--out-dir", default=str(config.OUT_DIR))
    ap.add_argument("--nproc", type=int, default=config.NPROC)
    ap.add_argument("--parallel", type=int, default=1, help="并发 worker 数")
    ap.add_argument("--n-iter", type=int, default=800, help="稳态迭代次数")
    ap.add_argument("--retry", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="忽略已完成全部重跑")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=None, help="每点超时(秒)")
    ap.add_argument("--run-one", default=None, help="worker: 单点 task json 路径")
    args = ap.parse_args()

    if args.run_one:
        t = json.loads(Path(args.run_one).read_text(encoding="utf-8"))
        r = run_one(t)
        print("BATCH_RESULT " + json.dumps(r))
        return 0 if r.get("status") == "done" else 1

    if args.angles is None and args.angles_range is None:
        ap.error("须给 --angles 或 --angles-range")
    return run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
