# Airfoil LLM-guided Shape Optimizer

This repository contains the complete **source code** of a framework that couples an
**LLM-based aerodynamic surrogate** (Qwen3-8B + LoRA, served via vLLM) with an
**evolutionary optimizer shell** (Gaussian resampling driven by a per-generation
"mean updater") for wind-turbine airfoil shape optimization under **pitching
(dynamic-stall) operating conditions**.

Four optimization methods are supported through one unified loop:

| Method | Role of the method | Module |
|---|---|---|
| `LLM` | A finetuned LLM proposes the next-generation sampling mean (`x_new`/`y_new`) from an elite-history prompt | `llm_model.py`, `prompts.py` |
| `GA` | Genetic algorithm (elite + roulette + arithmetic crossover + Gaussian mutation) | `ga_updater.py` |
| `PSO` | Particle swarm optimization (Kennedy & Eberhart 1995; inertia Shi & Eberhart 1998) | `pso_updater.py` |
| `CMA-ES` | Covariance-matrix adaptation evolution strategy (Hansen & Ostermeier 2001) | `cmaes_updater.py` |

---

## 1. Design overview

The optimizer maximizes a scalar **fitness** derived from the predicted
aerodynamic performance of a CST-parameterized airfoil over a set of pitching
conditions (Reynolds number × mean angle of attack × amplitude × oscillation
frequency × decay frequency) × surface-roughness regimes.

Per generation the loop performs:

```
sample N individuals ~ N(current_mean, σ²)      (population_manager.py)
       │
       ▼
predict Cl(α) for every (condition × roughness × α × up/down stroke)
       │                                          (aerodynamic_prediction.py → LLM/vLLM)
       ▼
compute per-condition metrics + A3 fitness      (airfoil_optimizer.py)
       │
       ▼
record generation (mean, best individual, all fitness, population)   → pso_final_result.json
       │
       ▼
update current_mean:
    LLM     → prompt from generation history → parse x_new/y_new
    GA/PSO/CMA-ES → single-step numerical updater on the evaluated population
```

**Design decisions (important for correct reproduction):**

- **Surrogate-as-a-function.** The "optimizer" methods never interact with CFD;
  every fitness evaluation is a *batch of LLM queries* against a vLLM server that
  hosts a Qwen3-8B base model plus two rank-8 LoRA adapters (`high`/`low`) trained
  on high-fidelity CFD pitching-airfoil data. The 14° / 10° case set used for the
  experiments corresponds to `config_14deg_amp10.json` (§ 3.1).
- **Gaussian-population mean update.** GA/PSO/CMA-ES do **not** directly carry a
  population across generations in the classical sense. Instead the sampler keeps a
  single `current_mean`; every generation is drawn as a Gaussian cloud around it,
  the numerical optimizer proposes an improved mean from the evaluated cloud, and
  the next generation re-samples around that mean. This single interface lets the
  four methods share one loop, one evaluator and one result format.
- **Deterministic prediction.** All LLM calls use `temperature = 0.0` and greedy
  decoding, both in the aerodynamic predictor and in the LLM mean updater, so that
  reported runs are reproducible up to the numerical optimizers' internal RNG.
- **A3 fitness.** Per condition the normalized mean lift is divided by the
  normalized hysteresis-loop area (with a baseline `c` added to the denominator):
  `obj_i = clip((cl_i - cl_min)/(cl_max - cl_min)) / (clip((hyst_i - hyst_min)/(hyst_max - hyst_min)) + c)`,
  then averaged with equal weights across conditions. Hysteresis area is computed by
  the trapezoid rule on the |Cl_up(α) − Cl_down(α)| gap over the shared α grid.
  The `cl_min…c` constants are per-config calibration values (§ 4).
- **Resume.** `pso_final_result.json` is a checkpoint: re-running
  `main_optimize.py` with the same method detects the last finished generation and
  continues (or prints *already finished*).

---

## 2. Repository layout

```
airfoil_llm_optimizer/
├── README.md
├── requirements.txt
└── src/
    ├── main_optimize.py             # CLI entry point (auto-resume, --method, --config)
    ├── airfoil_optimizer.py         # AirfoilOptimizer: sample → evaluate → record → update
    ├── population_manager.py        # GaussianSampler: mean, bounds, population sampling
    ├── aerodynamic_prediction.py    # per-condition α sweep, LLM batching, hysteresis & stats
    ├── llm_model.py                 # LLM client (vLLM OpenAI-compatible backend, LoRA switch)
    ├── prompts.py                   # prediction prompt + mean-update prompts (few_shot/standard)
    ├── ga_updater.py                # single-step GA mean update
    ├── pso_updater.py               # single-step PSO mean update (module-level state)
    ├── cmaes_updater.py             # single-step CMA-ES mean update (module-level state)
    ├── visualization.py             # convergence / population boxplot / param distribution
    ├── config.json                  # default (8°/14° mixed) operating conditions + vLLM endpoint
    └── config_14deg_amp10.json      # deep-stall 14°/10° config used for the reported runs
```

`src/` is intended to be run as its own working directory (`cd src`), which is where
the scripts write their runtime outputs (`pso_final_result.json`,
`pso_generations.jsonl`, `generated_predictions.jsonl`, `convergence.png`, …).

---

## 3. Quick start

### 3.1 Environment

- Python ≥ 3.9 (developed on 3.13 / Windows 11; Linux recommended for LLM serving).
- Third-party libraries (see `requirements.txt`):
  `numpy`, `scipy`, `matplotlib`, `tqdm`, `jsonlines`.
- Optional (only for the `local` inference backend, § 5.4): `torch`,
  `transformers`, `peft`, `safetensors`, `tiktoken`.

```bash
pip install numpy scipy matplotlib tqdm jsonlines
```

### 3.2 Reproduce a run on the deep-stall 14°/10° condition set

```bash
cd src

# LLM-guided optimization (needs a running vLLM server with LoRA, § 5)
python main_optimize.py --method LLM --population 50 --generations 100 \
    --config config_14deg_amp10.json

# Numerical methods (require the same LLM-based aerodynamic predictor!)
python main_optimize.py --method GA     --population 50 --generations 100 --config config_14deg_amp10.json
python main_optimize.py --method PSO    --population 50 --generations 100 --config config_14deg_amp10.json
python main_optimize.py --method CMA-ES --population 50 --generations 100 --config config_14deg_amp10.json
```

These are the exact invocations whose truncated result files were shared as the
companion *gen17 dataset* (`total_generations = 100`, recorded
generations 1…38/39, `population_size = 50`, objective `hybrid` = A3,
prompt mode `few_shot`). Each of the four runs produces
`pso_final_result.json` in `src/` with the following schema:

```jsonc
{
  "best_individual": { "A_u": [9], "A_l": [9], "index": 0 },
  "best_fitness": 1.8699,
  "current_mean": { "A_u": [9], "A_l": [9] },
  "optimization_objective": "hybrid",
  "method": "CMA-ES",
  "prompt_mode": "few_shot",
  "hybrid_weights": { "w1": 0.5, "w2": 0.5 },
  "generation_history": [ {
      "generation": 1,
      "mean": { "A_u": [9], "A_l": [9] },
      "best_individual": { "A_u": [9], "A_l": [9], "index": i },
      "best_fitness": 1.6169,
      "all_fitness": [50],
      "population_size": 50,
      "population": [ { "A_u": [9], "A_l": [9], "index": i }, ... 50 ],
      "individual_metrics": [ { "obj": ..., "cycle_mean_cl": ..., "hysteresis_mean_area": ... } ],
      "hyst_comparable": true,
      "top_individuals": [ [18], ... ]        // flattened A_u+A_l, top-3
  }, ... ],
  "total_generations": 100,
  "population_size": 50,
  "parameter_bounds": { "A_u": [0.0, 0.7], "A_l": [-0.7, 0.0] },
  "hyst_comparable": true
}
```

> **Note on stored history length.** The history stores **up to a bounded number
> of most recent generations** (the shipped 14°/10° runs contain 38–39 of the 100
> generations; the *global best* fields at the top level are always the true
> best over all 100 generations). Analysis scripts that replay a run therefore see
> the last generations plus the global optimum, not the complete trace.

### 3.3 Smoke test without an LLM server

```bash
cd src
python main_optimize.py --test
```

`--test` injects a `test_mode` block (single condition D1, α = 7–10° up/down,
2 individuals, 5 generations) — it still needs the vLLM endpoint, but issues only a
handful of requests.

---

## 4. Configurations and calibrated normalization constants

### 4.1 `config_14deg_amp10.json` — deep-stall condition set (R1–R12)

12 conditions, all `mean_aoa = 14°`, `amplitude = 10°`, α ∈ [4°, 24°] swept at 1°
intervals (21 points/condition); Reynolds ∈ {0.75, 1.00, 1.25, 1.50}×10⁶ ×
oscillation frequency ∈ {0.6, 1.2, 1.8} Hz. The **decay frequency is derived**
from a single constant `decay_factor = 0.045` as

```
decay_freq = decay_factor · osc_freq / (Re / 10⁶)
```

| | Re = 0.75 M | Re = 1.00 M | Re = 1.25 M | Re = 1.50 M |
|---|---|---|---|---|
| f = 0.6 Hz | R1 (0.036) | R4 (0.027) | R7 (0.022) | R10 (0.018) |
| f = 1.2 Hz | R2 (0.072) | R5 (0.054) | R8 (0.043) | R11 (0.036) |
| f = 1.8 Hz | R3 (0.108) | R6 (0.081) | R9 (0.065) | R12 (0.054) |

(decay frequencies in parentheses) — these values match the decay frequencies of
the same (Re, f) combinations in the LoRA training data.

**A3 objective normalization** (calibrated on the 14°/10° training subset, 233
hysteresis loops; robust P1/P99 percentiles):

```jsonc
"objective": {
  "method": "A3",
  "cl_min": 0.7276,  "cl_max": 1.3160,
  "hyst_min": 6.5,   "hyst_max": 9.5,    // raw hysteresis area (NOT span-divided)
  "hyst_floor": 0.05                      // baseline c (anti-division-by-zero only)
}
```

> In this config `hysteresis_mean_area` fed to the objective is the **raw**
> (not span-divided) trapezoid area. The observed best individuals of the four
> reported runs have cl ≈ 1.16–1.18 and raw hysteresis ≈ 8.3–8.5, inside the
> normalized range above.

**Per-individual request volume**: 12 conditions × 21 α × 2 roughness
(光滑/粗糙, smooth/rough) × 2 strokes (上升/下降) = **1008 LLM queries per
individual** (×50 individuals = 50,400 queries/generation).

### 4.2 `config.json` — mixed 8°/14° condition set (D1–D12)

Default config with conditions D1–D12 spanning mean α ∈ {8°, 14°}, amplitudes
5.5°/10°, Re ∈ {0.75, 1.0, 1.25, 1.5} M and f ∈ {0.6, 1.2, 1.8} Hz, and its own
calibrated normalization (`cl_min 0.4701 … cl_max 1.4088`, `hyst_min 0.0353 … 
hyst_max 0.7705`, `c = 0.05`). Because it mixes different (mean_aoa, amplitude)
combinations, its `hyst_comparable` flag is `false` and the population boxplot
skips the raw-hysteresis panel.

### 4.3 CST parameterization

Fixed CST settings across the whole pipeline (see `config.json`, prompts and
`population_manager.py`):

```
N1 = 0.5,  N2 = 1.0,  z_u_TE = z_l_TE = 0.0,  degree = 8  →  9 coefficients per surface
upper surface:  A_u ∈ [0.0, 0.7]        lower surface:  A_l ∈ [-0.7, 0.0]
```

Initial (baseline, NACA4415-like) coefficient vector used as generation-1 mean:

```
A_u = [0.250922, 0.304949, 0.255768, 0.359807, 0.245305,
       0.312583, 0.323615, 0.274348, 0.372711]
A_l = [-0.167089, -0.125904, -0.119999, -0.016318, -0.171019,
        0.027394, -0.133103,  0.010577, -0.104305]
```

The CST airfoil reconstruction used for plotting/export follows

```
z(ψ) = ψ^N1 (1−ψ)^N2 · Σ_{i=0}^{8} A_i · C(8,i) ψ^i (1−ψ)^{8−i},   ψ = x/c ∈ [0,1]
```

---

## 5. Serving the LLM surrogate (vLLM + LoRA)

The aerodynamic predictor talks to an OpenAI-compatible chat endpoint. The shipped
model/weights are **not** part of this source package (too large); they are assumed
to exist already (see § 7).

### 5.1 Base model and adapters

| Component | Requirement |
|---|---|
| Base model | Qwen3-8B (`Qwen/Qwen3-8B`), decoder-only causal LLM |
| Adapter `high` | LoRA r=8, α=16 trained on high-α CFD points (deep stall) |
| Adapter `low` | LoRA r=8, α=16 trained on low-α CFD points |
| Adapter modules | q/k/v/o/gate/up/down projections of the transformer blocks |

The two adapters were trained on regression-style data: given a text prompt that
fully describes the airfoil (CST coefficients), Re, roughness, mean α, amplitude,
oscillation frequency, decay frequency, instantaneous α and stroke direction
(up/down), the model answers the lift coefficient (and drag for the CFD-trained
weights), in Chinese, e.g. `升力系数为：1.2345，阻力系数为：0.0567`.
Thinking-token mode is enabled at training and inference time.

### 5.2 Starting the server

```bash
# base model dir: path/to/Qwen3-8B
# LoRA dirs:      path/to/WEIGHTS/HIGH , path/to/WEIGHTS/LOW
vllm serve path/to/Qwen3-8B \
    --served-model-name base \
    --quantization fp8 \                        # or fp16/bf16; weights were trained in bf16
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --enable-lora \
    --lora-modules high=path/to/WEIGHTS/HIGH low=path/to/WEIGHTS/LOW \
    --max-lora-rank 16
```

(The `--kernel-config '{"linear_backend":"torch"}'` flag is only needed on the
Windows community build to avoid a CUTLASS FP8 kernel crash; on standard Linux
builds it is unnecessary.)

### 5.3 Endpoint configuration

The client reads, in priority order, environment variables
`VLLM_BASE_URL / VLLM_API_KEY / VLLM_MODEL_HIGH / VLLM_MODEL_LOW /
VLLM_MODEL_BASE / VLLM_TIMEOUT / VLLM_RETRIES` and falls back to the `"vllm"`
section of the active config JSON (default `http://127.0.0.1:8000/v1`, adapter
names `high`/`low`/`base`). **Adapter selection is angle-based**: prediction
prompts at α ≥ 14° are routed to `high`, below 14° to `low`; mean-update prompts
(no α, used only by the `LLM` method) go to `base`.

Set `LLM_BACKEND=local` instead to use an in-process transformers + peft inference
path (paths in `llm_model.py` `MODEL_PATH`/`LORA_*_PATH`; sequential, low-VRAM,
slower).

### 5.4 Concurrency

`aerodynamic_prediction.py` issues the per-individual α/roughness/stroke sweep
with `ThreadPoolExecutor` (concurrency = `"concurrency"` in the config, 48 by
default) against vLLM. Total wall time per generation is dominated by the 1008
queries × population size.

---

## 6. Outputs and reproducibility notes

| File (in `src/`) | Contents |
|---|---|
| `pso_final_result.json` | final checkpoint: global best + bounded generation history (§ 3.2) |
| `pso_generations.jsonl` | detailed per-generation log (numerical methods; prompt/response log for `LLM`) |
| `generated_predictions.jsonl` | every raw LLM prediction prompt/response parsed Cl |
| `convergence.png`, `pop_boxplot.png`, `population_dist.png` | auto-generated by `visualization.analyze_pso_results` at the end of each generation |

Determinism: `temperature = 0` everywhere; GA/PSO/CMA-ES use `numpy.random`
global state (seeded by the caller's environment, not fixed in code), so the
numerical trajectories are not bit-reproducible unless you seed `np.random` before
launching; the reported *shared* experiment used the same condition config and
CLI arguments for all four methods.

---

## 7. Known external artifacts (not in this package)

Reproducing the reported *gen17* result tables/figures additionally requires:

1. The **Qwen3-8B base model** and the two **CFD-trained LoRA adapters**
   (`HIGH`/`LOW`). These are large binary artifacts; contact the authors for
   access. The CFD training data pipeline (CST fitting of 11 airfoils, high/low
   split rule: up-stroke α ≥ 17° / down-stroke α ≥ 15° → `high`, else `low`) is
   separate from this optimizer package.
2. A **vLLM** installation compatible with the model (the authors used
   `vllm 0.27.1` on Python 3.13 with a community Windows wheel; any Linux vLLM ≥
   0.6 with LoRA support is equivalent).
3. **Post-processing/analysis scripts** (truncating history to the first n
   generations, drawing the per-generation-best airfoil overlay, exporting CST
   parameters and scaled coordinate files) that were executed *outside* this
   package against the result JSONs.

## 8. License / citation

This is research code shared for reproducibility. Cite the associated paper when
using it; if the paper DOI is not yet available, contact the authors.
