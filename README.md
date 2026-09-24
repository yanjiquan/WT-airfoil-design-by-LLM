# Code and Data for *A large language model-driven design method for wind turbine blade airfoils*

This archive accompanies the manuscript **“A large language model-driven design method for wind turbine blade airfoils.”** It contains source code, processed data, trained model adapters, and precomputed outputs for the following parts of the study:

1. wind-tunnel-data processing and aerodynamic-coefficient prediction;
2. comparison with MLP and XGBoost baselines;
3. independent training and uncertainty evaluation using stochastic CFD data;
4. Beddoes–Leishman (B–L) uncertainty comparison;
5. LLM-guided airfoil optimization and GA/PSO/CMA-ES comparisons;
6. independent Fluent validation of the optimized airfoils; and
7. OpenFAST evaluation of 5, 10, and 15 MW reference wind turbines.

The release is distributed as compressed archives. Extract only the packages needed for a given analysis.

## Repository layout

```text
Data and Code/
├── code/          # Source-code archives
├── data/          # Datasets, weights, predictions, and simulation outputs
└── README.md
```

Archive-name prefixes have the following meanings:

- `cfd_*`: Fluent airfoil simulations and processed CFD results;
- `openfast_*`: turbine-level OpenFAST simulations and analysis;
- `optimize_*`: LLM, GA, PSO, and CMA-ES airfoil optimization;
- `uq_*`: aerodynamic prediction and uncertainty-quantification workflows;
- `src`, `datasets`, `weights`, and `data`: source code, model-ready datasets, trained parameters, and outputs, respectively.

## Source-code archives

| Archive | Contents | Main entry points |
|---|---|---|
| `code/cfd_src_airfoil_cfd_tools.zip` | Dynamic-pitching and static-polar Fluent workflows, five meshes, a reference case, and detailed internal READMEs | `batch_auto_run.py`, `static_polar_batch.py`, `main.py` |
| `code/openfast_src.zip` | Automated tip-airfoil replacement and six-case OpenFAST workflow for 5/10/15 MW turbines | `openfast_runner/main.py` |
| `code/optimize_src.zip` | Shared optimization loop for LLM, GA, PSO, and CMA-ES using an LLM aerodynamic surrogate | `src/main_optimize.py` |
| `code/uq_src_llm_and_compare_methods.zip` | Wind-tunnel-data conversion/evaluation scripts and MLP/XGBoost comparison models | `Dynamic/*.py`, `Static/*.py`, `Compare/*.py` |
| `code/uq_src_train_llm_by_cfd.zip` | Independent CFD-based LoRA training, inference, sampling, and interval plotting | `train_lora.py`, `test_lora.py`, `plot_uncertainty.py`, `draw_uncertainty.py` |
| `code/uq_src_bl.zip` | B–L model, parameter files, calibration utilities, Monte Carlo interval generation, and evaluation | `BL_code/bl_draw_uncertainty.py` |

Each of the CFD, OpenFAST, optimization, and B–L source archives contains a more specialized README or inline usage documentation. Read those files before running a full experiment.

## Data and model archives

| Archive | Contents |
|---|---|
| `data/cfd_data.zip` | Processed dynamic CFD results for five airfoils and 12 pitching conditions, plus static polar results for NACA 4415 and the LLM-optimized airfoil |
| `data/openfast_data.zip` | Six OpenFAST text outputs (`DATA` and `OP` for 5/10/15 MW), run manifest, and analysis scripts |
| `data/optimize_data.zip` | Final optimization checkpoints, CST parameters, airfoil coordinates, and convergence/geometry plots for LLM, GA, PSO, and CMA-ES |
| `data/uq_datasets_wt.rar` | Model-ready static and dynamic wind-tunnel samples, including CST-history, no-history, and variable-only variants |
| `data/uq_weights_llm.zip` | Final HIGH/LOW LoRA adapters associated with the wind-tunnel model |
| `data/uq_data_predict_by_llm_trained_by_wt.rar` | Archived wind-tunnel-model predictions and interpolated prediction files for NACA 4415 |
| `data/uq_data_compare_methods.zip` | NACA 4415 prediction tables, evaluation reports, and XGBoost models |
| `data/uq_weights_compare_methods.zip` | MLP model weights and preprocessing artifacts for lift and drag |
| `data/uq_datasets_cfd.zip` | Independent CFD model datasets: `train.jsonl`, `test.jsonl`, `low.jsonl`, `high.jsonl`, and `cst_params.json` |
| `data/uq_weights_cfd_llm.zip` | HIGH/LOW LoRA adapters trained only on the stochastic CFD dataset |
| `data/uq_data_predict_by_llm_trained_by_cfd.zip` | CFD-model prediction samples, interval plots, and coverage summary |
| `data/uq_data_bl.zip` | B–L Monte Carlo samples and interval-coverage summary evaluated against the same CFD reference cases |

The wind-tunnel and CFD datasets were used to train **separate models**. Do not pool the datasets or interchange `uq_weights_llm.zip` and `uq_weights_cfd_llm.zip`.

## Recommended starting points

| Goal | Code | Required input or weights | Reference output |
|---|---|---|---|
| Inspect the final results without rerunning models | None | None | All archives named `*_data*` |
| Recompute dynamic/static CFD summaries | `cfd_src_airfoil_cfd_tools.zip` | Included meshes/configuration; ANSYS Fluent required | `cfd_data.zip` |
| Retrain and evaluate the CFD-only LoRA model | `uq_src_train_llm_by_cfd.zip` | `uq_datasets_cfd.zip` and Qwen3-8B | `uq_weights_cfd_llm.zip`, `uq_data_predict_by_llm_trained_by_cfd.zip` |
| Recompute the B–L comparison | `uq_src_bl.zip` | `uq_datasets_cfd.zip` and CFD prediction records | `uq_data_bl.zip` |
| Reproduce airfoil optimization | `optimize_src.zip` | Qwen3-8B plus the CFD HIGH/LOW adapters | `optimize_data.zip` |
| Reproduce the turbine-level comparison | `openfast_src.zip` | Included OpenFAST models and CFD polar CSVs | `openfast_data.zip` |
| Inspect wind-tunnel prediction/baseline processing | `uq_src_llm_and_compare_methods.zip` | `uq_datasets_wt.rar` and the appropriate archived weights | wind-tunnel prediction and comparison archives |

## Software requirements

The repository is an archive release rather than a single installable Python package. Create separate environments for the workflows when practical.

### Common Python packages

Depending on the selected workflow, the scripts use:

```text
numpy, scipy, pandas, matplotlib, tqdm, h5py, jsonlines,
scikit-learn, torch, xgboost, transformers, peft, datasets,
accelerate, safetensors, and rainflow
```

Use the `requirements.txt` included in the CFD, OpenFAST, and optimization source archives as the primary specification for those packages. For the LoRA workflow, begin with the framework versions recorded in the adapter model cards and adjust only if required by the local CUDA/PyTorch environment.

### External software

- **ANSYS Fluent 2025 R1** or a compatible release is required for the supplied CFD automation.
- The OpenFAST archive contains the Windows executables and controller libraries used by its packaged model trees.
- **Qwen3-8B is not included.** Only LoRA adapters and tokenizer/configuration files are supplied.
- A CUDA-capable GPU is strongly recommended for LoRA training and vLLM inference.
- The CFD and OpenFAST workflows were prepared for Windows. Standard Linux vLLM deployments are recommended for model serving.

## Unpacking the release

ZIP files can be extracted with Windows Explorer, PowerShell `Expand-Archive`, or any standard archive manager. A tool with RAR support is required for the two `.rar` files.

Keep each archive in a separate working directory unless a layout is explicitly shown below. Several scripts use relative paths, while the legacy wind-tunnel preprocessing scripts retain absolute paths from the original workstation.

## Reproduction workflows

### 1. Fluent CFD validation

Extract `code/cfd_src_airfoil_cfd_tools.zip`, then follow its top-level README.

Minimum setup:

```powershell
cd airfoil_cfd_tools
python -m pip install -r requirements.txt
```

Update `FLUENT_EXE` in both CFD configuration files. The static workflow also requires valid `DYNAMIC_TOOL_ROOT` and `MESH_DIR` values. A one-case dynamic smoke test is:

```powershell
python src\dynamic_airfoil_tool\batch_auto_run.py --mesh-dir mesh --config example_config\config_14deg_amp10.json --meshes naca4415_0p4572.msh --conditions R1 --parallel 1 --nproc 4
```

The full dynamic matrix contains five airfoils × 12 conditions. Static-polar and full-batch commands, solver settings, output schemas, and restart behavior are documented inside the archive. The provided `data/cfd_data.zip` contains processed CSV/PNG outputs, not the approximately 5 GB of raw Fluent HDF5 case/data files.

### 2. CFD-only LoRA training and uncertainty evaluation

Extract these archives into one workspace:

```text
cfd_uq/
├── src/        # contents of uq_src_train_llm_by_cfd.zip
├── datasets/   # contents of uq_datasets_cfd.zip/datasets
└── weights/    # output directory or contents of uq_weights_cfd_llm.zip
```

Install the model-training dependencies and set `MODEL_PATH`, `DATA_DIR`, and `OUT_DIR` before training:

```bash
pip install torch transformers peft datasets accelerate numpy matplotlib tqdm
export MODEL_PATH=/path/to/Qwen3-8B
export DATA_DIR=/path/to/cfd_uq/datasets
export OUT_DIR=/path/to/cfd_uq/weights
python src/train_lora.py --adapter all --epochs 5 --batch 4
```

`high.jsonl` and `low.jsonl` train separate HIGH/LOW adapters. The adapters predict both lift and drag in one response. `test_lora.py` performs deterministic evaluation; `plot_uncertainty.py` samples the OpenAI-compatible vLLM endpoint and writes per-record prediction samples; `draw_uncertainty.py` converts those samples into plots and `95%CI.csv`.

The scripts default to paths under `/mnt/workspace`; override them with command-line arguments or environment variables. The supplied adapters require the Qwen3-8B base model and are not standalone checkpoints.

### 3. B–L uncertainty comparison

Use the following extraction layout:

```text
bl_reproduction/
├── BL_code/                 # from uq_src_bl.zip
├── datasets/                # from uq_datasets_cfd.zip
└── cfd_llm_predict.jsonl    # from uq_data_predict_by_llm_trained_by_cfd.zip
```

Install `numpy`, `scipy`, `matplotlib`, and optionally `scikit-learn`, then run from `BL_code/`:

```bash
python bl_draw_uncertainty.py --pert 0.15 --nmc 40 --seed 0 --params-dir . --out output
```

The command uses the included per-airfoil parameter JSON files and produces Monte Carlo samples, plots, and a coverage table. `bl_train.py` and `bl_train_perwing.py` are optional refitting utilities implemented with SciPy least-squares optimization; using them will replace the supplied parameter estimates and may not reproduce the archived output exactly.

### 4. Airfoil optimization

Extract `code/optimize_src.zip`, install its requirements, and run from `airfoil_llm_optimizer/src/`:

```bash
python main_optimize.py --method LLM --population 50 --generations 100 --config config_14deg_amp10.json
python main_optimize.py --method GA --population 50 --generations 100 --config config_14deg_amp10.json
python main_optimize.py --method PSO --population 50 --generations 100 --config config_14deg_amp10.json
python main_optimize.py --method CMA-ES --population 50 --generations 100 --config config_14deg_amp10.json
```

All four methods use the same LLM-based aerodynamic evaluator. The LLM method additionally uses the base model to propose the next sampling mean. Start an OpenAI-compatible vLLM server with Qwen3-8B and the HIGH/LOW adapters from `data/uq_weights_cfd_llm.zip`, then set the endpoint/model names in the active JSON configuration or through the documented environment variables.

The optimizer writes `pso_final_result.json` and diagnostic plots in its working directory and supports resume from the existing checkpoint. Numerical optimizer trajectories are not bit-reproducible unless the NumPy random state is fixed.

### 5. OpenFAST turbine-level evaluation

Extract `code/openfast_src.zip`, then run from `OpenFAST_AirfoilReplacement/`:

```powershell
python -m pip install -r requirements.txt
python openfast_runner\main.py --cli input_polars\naca4415_4Re_mean_polar_clean.csv input_polars\llm_4Re_mean_polar_clean.csv --parallel 6
```

The pipeline stages isolated working copies, replaces the specified tip-airfoil polar, runs DATA/OP cases for the 5, 10, and 15 MW reference turbines, and collects six `.out` files plus a manifest. The source model trees remain unchanged.

After extracting `data/openfast_data.zip`, run its post-processing scripts in the directory containing the six `.out` files:

```powershell
python analysis.py
python aep_del_analysis.py
```

The second script additionally requires `scipy` and `rainflow`. Consult the internal OpenFAST README for polar-merging rules, wind-field assumptions, channel definitions, and result caveats.

### 6. Wind-tunnel prediction and comparison scripts

`code/uq_src_llm_and_compare_methods.zip` contains three script groups:

- `Static/`: converts static wind-tunnel data to prompt/response records;
- `Dynamic/`: converts pitching-cycle data, merges HIGH/LOW predictions, and evaluates errors;
- `Compare/`: trains and evaluates MLP and XGBoost models with NACA 4415 held out.

These scripts preserve original absolute paths such as `D:\AirfoilDesign\...`. Update `ROOT_DIR`, `Dataset_Path`, `base_path`, and related constants before execution. There is no single end-to-end launcher or consolidated requirements file for this archive. Its principal dependencies are `numpy`, `pandas`, `scipy`, `h5py`, `scikit-learn`, `torch`, and `xgboost`.

The wind-tunnel adapter archive and archived prediction files support inspection of the final model outputs. They should not be assumed to provide a fully automated from-scratch recreation of the complete two-stage wind-tunnel fine-tuning workflow.

## Data conventions

- Angles of attack are reported in degrees unless explicitly converted to radians inside a physical model.
- Reynolds numbers in text prompts are expressed in millions, for example `1.00*10^6`.
- Lift and drag coefficients are dimensionless.
- Dynamic records retain the instantaneous angle of attack and its trend so that upstroke and downstroke branches remain distinguishable.
- JSONL training records generally contain `instruction`, `input`, and `output`; CFD split files also contain a `region` label.
- CFD uncertainty records contain the true coefficients and repeated model samples (`cl_samples` and `cd_samples`).
- `95%CI.csv` and `95%CI_BL_mc.csv` report empirical coverage of percentile-based 95% intervals; they should not be interpreted as proof of nominal 95% calibration.
- Complete airfoil/condition/trajectory/turbulence groups must remain intact when constructing new train/test splits. Do not randomly split neighboring phase points.

The wind-tunnel-derived data and stochastic-CFD data have different provenance and represent different sources of variability. The former spans sampled operating conditions; the latter uses repeated turbulent-inflow realizations under fixed nominal geometry and operating inputs.

## Important limitations

1. The Qwen3-8B base model is not redistributed in this archive.
2. Raw Fluent HDF5 case/data files are not included; processed CFD tables and figures are provided.
3. Some scripts contain machine-specific absolute paths and require manual configuration.
4. Exact runtime depends strongly on Fluent licenses, CPU parallelism, GPU hardware, CUDA/PyTorch/vLLM versions, and local storage.
5. Task-specific HIGH/LOW data partitions and routing rules are defined in the corresponding scripts/configurations. Do not assume that one threshold applies unchanged to every wind-tunnel, CFD, and optimization workflow.
## Citation

If you use this code or data, please cite the associated manuscript:

> J. Yan, W. Hu, W. Zhang, L. Fan, T. Zhang, J. Fang, and J. Tan, “A large language model-driven design method for wind turbine blade airfoils,” manuscript.

Replace this entry with the final journal citation and DOI after publication.

## Contact

For questions about the code, data provenance, or omitted raw simulation files, contact the corresponding author:

**Weifei Hu** — `weifeihu@zju.edu.cn`

## License and third-party components

No standalone software or data license is included in this release. Contact the authors for reuse terms. Third-party models, solvers, reference turbine models, executables, and datasets remain subject to their respective licenses and terms of use.
