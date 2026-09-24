# OpenFAST Airfoil-Replacement Simulation Platform

An automated OpenFAST 4.2.0 workflow that replaces the **tip airfoil** of three
reference wind turbines (IEA 5MW / IEA 10MW / IEA 15MW) with a NACA-4415-class
polar supplied by the user, then runs the full **DATA vs OP** comparison set
(3 turbines x 2 variants = 6 simulations) under an identical **6.533 m/s-mean
turbulent wind field**.

---

## What the package contains

```
OpenFAST_AirfoilReplacement/
├── openfast_runner/          # platform source (pure stdlib + numpy, no pip deps)
│   ├── main.py               # entry point: GUI (default) or CLI (--cli)
│   ├── gui.py                # tkinter UI - 6 live case cards
│   ├── pipeline.py           # stages + runs the 6 cases, collects .out files
│   ├── run_case.py           # single-case process manager (Popen + progress)
│   ├── dependency.py         # .fst dependency-chain resolver / isolated run tree
│   ├── airfoil_io.py         # polar parsing, alpha0/1/2 + C_nalpha, AeroDyn rewrite
│   ├── cfd_to_dat.py         # CFD CSV -> merged polar (pitch-cycle AND static)
│   ├── progress.py           # OpenFAST stdout progress parser
│   └── config.py             # turbine mapping + input/base-polar/wind paths
├── simulations/              # OpenFAST source trees (read-only at runtime)
│   ├── 5MW/  10MW/  15MW/    # per-turbine model trees + openfast_x64.exe
│   └── .../XFOIL6.99/...     # XFOIL base polars used for the merge
├── input_polars/             # bundled CFD static polar inputs (see below)
│   ├── naca4415_4Re_mean_polar_clean.csv
│   └── llm_4Re_mean_polar_clean.csv
├── runner_work/              # created at runtime (run trees / results / logs)
└── README.md                 # this file
```

> `simulations/` contains everything OpenFAST needs to run: the three turbine
> model trees, `openfast_x64.exe`, the airfoil polar base files and the six
> wind files. Each wind file is an OpenFAST "uniform" time-series
> (`WindType=2`, `IEA15MW_Hysteresis_Wind_UD.dat`) carrying a **Kaimal
> turbulence sequence with 6.533 m/s mean** (same seed as the reference run),
> so all six simulations share one identical wind field.

---

## The CFD input files

The two bundled CSVs are **static polars** (one representative lift/drag value
per angle of attack), produced by averaging steady RANS CFD over **four
Reynolds numbers** (Re = 0.75/1.0/1.25/1.5e6):

```
aoa_deg,Cl,Cd          # header
0,0.423082,0.004742    # rows 0..30 deg, step 2 deg
```

| file | airfoil | role |
|---|---|---|
| `input_polars/naca4415_4Re_mean_polar_clean.csv` | NACA 4415 (baseline) | **DATA** |
| `input_polars/llm_4Re_mean_polar_clean.csv`      | NACA-4415-LLM (optimized) | **OP** |

**Replacement semantics** (as agreed for this protocol):

* The static CFD **CL** replaces the base XFOIL polar over the positive range
  **[0, 30] deg** only.
* **CD is not replaced** (the base polar's CDp column is kept) and the
  **negative-AoA side is not mirrored** (base values kept there).
* **alpha0 / alpha1 / alpha2 and C_nalpha are recomputed** from the merged CL
  curve (first down-crossing of the f = 0.7 line), so the dynamic-stall model
  (AeroDyn `UA_Mod=3`) sees CFD-derived parameters:
  * DATA (NACA4415): alpha0 = -4.17 deg, alpha1 = 27.25 deg, C_nalpha = 3.80 /rad
  * OP   (LLM)     : alpha0 = -5.61 deg, alpha1 = 28.66 deg, C_nalpha = 3.49 /rad

The same machinery also accepts legacy **pitch-oscillation cycle** CSVs
(`time,AoA,Cl,Cd`, rising segment over [4,24] deg) - the file kind is
auto-detected from the header.

---

## Requirements

* Windows 10/11, Python 3.8+ on PATH
* `numpy` only (`pip install numpy`)
* (optional post-processing of finished runs uses `matplotlib`, `scipy`,
  `rainflow` - see "Post-processing the results" below)

---

## How to reproduce the results

### Option A - GUI

```
python openfast_runner/main.py
```

1. Input type: **CFD data (CSV)**.
2. Browse **DATA** -> `input_polars/naca4415_4Re_mean_polar_clean.csv`
3. Browse **OP**   -> `input_polars/llm_4Re_mean_polar_clean.csv`
4. Both pickers show green "static polar OK (16 points, 0 deg~30 deg)".
5. Parallelism: 6 (or fewer on a busy machine). Click **Start**.
6. Six case cards progress; finished results land in
   `runner_work/results/<timestamp>/` (`<MW>_DATA.out`, `<MW>_OP.out`,
   `manifest.json`).

### Option B - CLI (same thing, headless)

```
python openfast_runner/main.py --cli ^
    input_polars/naca4415_4Re_mean_polar_clean.csv ^
    input_polars/llm_4Re_mean_polar_clean.csv --parallel 6
```

### What you should see

Six `openfast_x64.exe` runs (5MW ~10-15 min, 10MW/15MW ~2-4 min), all sharing
the 6.533 m/s turbulent wind. With the bundled polars the observed
DATA-vs-OP trends on this protocol were:

| metric (30%-100%, RtFldCp-filtered) | 5MW | 10MW | 15MW |
|---|---|---|---|
| average Cp change (OP vs DATA) | **+2.65%** | **+1.60%** | **+2.64%** |
| root-moment CV change (pp) | -0.82 | -0.02 | -0.59 |

> Note: DEL (fatigue) numbers from the optional post-processing are dominated
> by the single largest gust cycle (m=10 sensitivity). Differences between
> DATA and OP should be interpreted with the seed-dependence caveat (a single
> 600 s realization carries statistical noise of order the observed DEL
> deltas).

---

## Post-processing the results

After a run completes, point the analysis scripts at the produced results
directory (each script reads `<MW>_DATA.out` / `<MW>_OP.out` from its working
directory):

```
python analysis.py          # avg RtFldCp + root-moment CV, DATA vs OP, bar charts
python aep_del_analysis.py  # AEP (Rayleigh at 6.533 m/s) + rainflow DEL (m=10)
```

`analysis.py` needs `matplotlib`; `aep_del_analysis.py` additionally needs
`scipy` and `rainflow`. `aep_del_analysis.py` auto-backfills its AEP `Cp`
parameters from the measured `avg_cp` of the run it is analysing.

---

## How the pipeline works

1. **Polar merge** - the CFD static CL is interpolated onto the 361-row
   (-180..180 deg) XFOIL base polar on [0,30] deg; alpha0/alpha1/alpha2 and
   C_nalpha are recomputed from the merged curve.
2. **Airfoil replacement** - `dependency.py` resolves the .fst dependency
   chain, stages an isolated run tree per case under
   `runner_work/run/<CASE>/`, and `airfoil_io.replace_tip_airfoil` rewrites
   the target AeroDyn polar (5MW -> NACA64_A17, 10MW -> Polar_29,
   15MW -> Polar_49), updating NumAlf to 361 plus alpha0/1/2 and C_nalpha.
3. **Parallel simulation** - up to `--parallel` (default 6) `openfast_x64.exe`
   run concurrently; stdout progress is parsed into per-case percent.
4. **Collection** - `.out` files are copied to
   `runner_work/results/<timestamp>/` with a `manifest.json`
   (status / returncode / size / end-time check).

Source trees are read-only; all runtime copies live under `runner_work/`.
The only third-party import is `numpy`.

---

## Customisation

* **Different CFD polars**: any static polar CSV with an `aoa_*` and a `cl`
  column header, or any pitch-cycle `time,AoA,Cl,Cd` CSV, is accepted.
* **Wind field**: to run a different wind, replace the six
  `IEA15MW_Hysteresis_Wind_UD.dat` files under `simulations/` (and keep the
  `Filename_Uni` reference in each `*_InflowFile.dat` intact). The pipeline
  refuses to run with a constant (non-turbulent) wind field.
* **Base polar**: `config.CFD_BASE_DATA` / `config.CFD_BASE_OP` point at the
  XFOIL base polars used for the merge.
* **Replacement window**: `config.STATIC_AOA_MIN` / `config.STATIC_AOA_MAX`
  (default 0 / 30 deg) and `CFD_AOA_MIN` / `CFD_AOA_MAX` (default 4 / 24 deg
  for legacy pitch-cycle inputs).

## Notes

* The wind files already inside `simulations/` are the 6.533 m/s-mean
  turbulent field used by the reference results. Regenerating them from
  scratch requires the original Kaimal-series generator (not shipped); keep
  the shipped copies if you want bit-identical reproduction.
* Results directories contain raw OpenFAST `.out` (text) files, tens to
  hundreds of MB per turbine - they are intentionally not part of the zip.
