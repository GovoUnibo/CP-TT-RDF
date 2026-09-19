# CP-TT-RDF

```bibtex
@article{GOVONI2026105717,
  title   = {Fast GPU evaluation of differentiable Signed Distance Fields for robotics via tensor decompositions},
  journal = {Robotics and Autonomous Systems},
  volume  = {206},
  pages   = {105717},
  year    = {2026},
  doi     = {10.1016/j.robot.2026.105717},
  author  = {Andrea Govoni and Sylvain Calinon and Gianluca Palli}
}
```

Quick user guide for running RDF experiments with the available parameterizations:

- `Weights`
- `CP`
- `TT`

## Install

```bash
pip install -r requirements.txt
```

If you use CUDA, install the PyTorch build that matches your system first.

## Workspaces

This repository is organized around three runnable workspaces:

- `panda_test/` for the Panda experiments
- `benchmark/` for timing and plotting scripts
- `torch/` for the Torcia workflow

Each workspace follows the same asset layout:

- `Meshes/`
- `Dataset/`
- `Models/`

The scripts create or load files through helpers such as `init_robot_folder(...)`, `add_models(...)`, and `create_model(...)`. In normal use, you only need to edit the top-level constants in each script.

## Quick Start

```bash
python train_rdf_3dcp.py
python visu_rdf_3dcp.py
python banchmark_tempi_panda.py
```

## Panda Training

| Script | Purpose | Run |
| --- | --- | --- |
| `train_rdf_3dcp.py` | Train the Panda CP model | `python train_rdf_3dcp.py` |
| `train_rdf_3dtt.py` | Train the Panda TT model | `python train_rdf_3dtt.py` |
| `train_rdf_weights.py` | Train the Panda weights baseline | `python train_rdf_weights.py` |
| `test_sdf_train.py` | Sanity check for dataset/model creation | `python test_sdf_train.py` |

Common values to tweak:

- `WS_PATH`
- `ROBOT_NAME`
- `BASE_LINK_NAMES`
- `N_FUNC`
- `TRAIN_ITERS`

## Panda Visualization

| Script | Purpose | Run |
| --- | --- | --- |
| `visu_rdf_3dcp.py` | Open the CP scene with two Panda robots | `python visu_rdf_3dcp.py` |
| `visu_rdf_3dtt.py` | Open the TT scene with two Panda robots | `python visu_rdf_3dtt.py` |
| `visu_rdf_weights.py` | Open the weights-based scene | `python visu_rdf_weights.py` |

Useful flags:

- `RUN_GD_VISU`
- `GD_ITERS`
- `GD_EPS`

## Benchmarks

| Script | Purpose | Notes |
| --- | --- | --- |
| `banchmark_tempi_panda.py` | Compare Panda inference times for `Weights`, `CP`, and `TT` | Includes FK in the timed path |
| `benchmark/alpha_beta_pred_vs_estimate.py` | Print theoretical vs empirical timing tables | Terminal output only |
| `benchmark/alpha_beta_delta_model_plot.py` | Save the alpha/beta comparison plot | Writes `benchmark/alpha_beta_delta_model_plot.png` |
| `benchmark/panda_bench_n128.py` | Structured benchmark for the Panda models | Requires the extra `benchmark.panda_benchmark_core` helper |

## Torcia Workflow

| Script | Purpose | Run |
| --- | --- | --- |
| `torch/train_eval_visualize_torcia_3d.py` | Train, evaluate, and visualize the Torcia mesh | `python torch/train_eval_visualize_torcia_3d.py` |
| `torch/torcia_cp_tt_memory_benchmark.py` | Memory and runtime benchmark for Torcia CP/TT | `python torch/torcia_cp_tt_memory_benchmark.py` |

Common values to tweak in `torch/train_eval_visualize_torcia_3d.py`:

- `MESH_INPUT`
- `RUN_TRAIN_WEIGHTS`
- `RUN_TRAIN_CP`
- `RUN_TRAIN_TT`
- `RUN_VISUALIZATION`
- `EVAL_USE_DATASET`
- `GET_GRAD`

## Outputs

- `panda_test/Meshes/`, `panda_test/Dataset/`, `panda_test/Models/`
- `torch/Meshes/`, `torch/Dataset/`, `torch/Models/`
- `benchmark/alpha_beta_delta_model_plot.png`

## Troubleshooting

- If a script cannot find a mesh or model, check the workspace path and file names first.
- If a viewer does not open, confirm that your machine has a graphical session available.
- If `benchmark/panda_bench_n128.py` fails, add the missing helper module or skip that script.
