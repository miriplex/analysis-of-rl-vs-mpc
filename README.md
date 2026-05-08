# control_bench

`control_bench` is the benchmark and experiment framework used for the dissertation study comparing **Model Predictive Control (MPC)** and **Reinforcement Learning (RL)** controllers on:

- first-order plants
- second-order plants
- inverted-pendulum control

The repository is built around a reusable simulation/control package in `src/control_bench` and a set of runnable experiment scripts in `experiments`.

This README explains:

- what the repository contains
- how the code is organized
- which scripts are the main entry points
- how the shared manifest is used
- how to rerun training, evaluation, and figure generation

## 1. What This Repository Is For

The codebase supports a benchmark workflow rather than a single application. It is designed to let you:

- define plant families
- define controller families
- run matched nominal and robustness experiments
- train RL controllers
- compare trained RL controllers against MPC and PID-style baselines
- generate figures, metrics tables, and presentation assets

The codebase is especially strong on the **first-order benchmark track**, where the shared manifest is the live source of truth for the MLP controller registry and robust training setup.

## 2. Repository Layout

At the top level, the main folders and files are:

- [README.md](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/README.md)
  This document.

- [EXPERIMENT_MANIFEST.yaml](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/EXPERIMENT_MANIFEST.yaml)
  Shared declarative benchmark manifest. This is the main configuration layer for the first-order MLP study and for many experiment definitions.

- [src/control_bench](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench)
  Reusable Python package containing plants, controllers, simulation logic, and manifest helpers.

- [experiments](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments)
  Runnable scripts for training, benchmarking, plotting, open-loop demos, and presentation asset generation.

- [experiments/results](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/results)
  Generated outputs: trained weights, figures, metrics, summaries, and videos.

There are also `analysis`, `configs`, and `results` folders at repo root, but the core workflow is concentrated in `src/control_bench`, `experiments`, and `EXPERIMENT_MANIFEST.yaml`.

## 3. Package Structure

The reusable package lives under [src/control_bench](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench).

### 3.1 Core simulation layer

- [core/protocols.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/core/protocols.py)
  Interface-style definitions for plants, controllers, and scenarios.

- [core/types.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/core/types.py)
  Shared type helpers, arrays, bounds, and I/O descriptors.

- [core/sim.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/core/sim.py)
  Generic closed-loop simulation engine. This is the common execution kernel used by the benchmark scripts.

- [core/scenario_impl.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/core/scenario_impl.py)
  Standard scenario implementation for reference generation, disturbance handling, and measurement processing.

### 3.2 Plants

- [plants/first_order_family.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/plants/first_order_family.py)
  Generates the 9 first-order benchmark plants.

- [plants/second_order_family.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/plants/second_order_family.py)
  Defines second-order benchmark plants.

- [plants/inverted_pendulum.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/plants/inverted_pendulum.py)
  Linearized inverted pendulum model.

- [plants/inverted_pendulum_nonlinear.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/plants/inverted_pendulum_nonlinear.py)
  Nonlinear inverted pendulum model.

- [plants/lti_siso.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/plants/lti_siso.py)
  Discrete SISO first-order model utilities.

- [plants/lti_state_space.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/plants/lti_state_space.py)
  Generic LTI state-space helpers.

### 3.3 Controllers

- [controllers/mpc.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/controllers/mpc.py)
  First-order/linear MPC implementation.

- [controllers/inverted_pendulum_mpc.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/controllers/inverted_pendulum_mpc.py)
  Pendulum MPC implementation.

- [controllers/pid.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/controllers/pid.py)
  Conventional PID controller implementation.

- [controllers/inverted_pendulum_rl.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/controllers/inverted_pendulum_rl.py)
  Pendulum RL policy classes, rollout BPTT training, validation, and runtime wrapper.

There is also a nested RL numpy implementation used heavily by the first-order track:

- `controllers/rl_numpy/`
  This directory contains policy classes, feature set definitions, Adam optimizer, data structures, and rollout training logic for the first-order RL experiments.

### 3.4 Shared configuration and manifest access

- [config.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/config.py)
  Global study constants. This is still the live source of truth for many shared plant, cost, and training constants, especially for pendulum and second-order experiments.

- [experiment_manifest.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/experiment_manifest.py)
  Loader and helper accessors for [EXPERIMENT_MANIFEST.yaml](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/EXPERIMENT_MANIFEST.yaml).

## 4. Experiment Script Structure

The main runnable scripts are in [experiments](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments).

### 4.1 First-order experiments

Primary scripts:

- [experiments/first_order/benchmark_first_order_robustness_metrics.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/benchmark_first_order_robustness_metrics.py)
  Main first-order benchmark pipeline used to compute nominal and robustness metrics.

- [experiments/first_order/train_first_order_manifest_controllers.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/train_first_order_manifest_controllers.py)
  Manifest-backed first-order RL training sweep across registered MLP variants and PID-features.

- [experiments/train_rl_first_order_numpy.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/train_rl_first_order_numpy.py)
  Lower-level first-order RL training entry point. Still useful, but the manifest-backed script is the cleaner study interface.

Comparison and analysis scripts:

- [experiments/first_order/compare_first_order_9plants.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/compare_first_order_9plants.py)
- [experiments/first_order/compare_first_order_9plants_disturbed.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/compare_first_order_9plants_disturbed.py)
- [experiments/first_order/compare_first_order_9plants_gain_mismatch.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/compare_first_order_9plants_gain_mismatch.py)
- [experiments/first_order/compare_first_order_9plants_secret_pole_mismatch.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/compare_first_order_9plants_secret_pole_mismatch.py)
- [experiments/first_order/compare_first_order_mlp_variants.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/compare_first_order_mlp_variants.py)
- [experiments/first_order/compare_first_order_mlp_variants_robust_vs_baseline.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/compare_first_order_mlp_variants_robust_vs_baseline.py)

Stress-test scripts:

- [experiments/first_order/robustness_tests/hidden_pole_stress_test.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/robustness_tests/hidden_pole_stress_test.py)
- [experiments/first_order/robustness_tests/measurement_noise_stress_test.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/robustness_tests/measurement_noise_stress_test.py)
- [experiments/first_order/robustness_tests/input_noise_stress_test.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/robustness_tests/input_noise_stress_test.py)
- [experiments/first_order/robustness_tests/step_disturbance_stress_test.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/robustness_tests/step_disturbance_stress_test.py)
- [experiments/first_order/robustness_tests/disturbance_stress_test.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/robustness_tests/disturbance_stress_test.py)

Presentation/demo scripts:

- [experiments/first_order/open_loop_first_order_demo.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/open_loop_first_order_demo.py)
- [experiments/first_order/closed_loop_first_order_suite_all_controllers.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/closed_loop_first_order_suite_all_controllers.py)
- [experiments/first_order/plot_unstable_rhp_zero_hidden_pole_failure_stages.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/plot_unstable_rhp_zero_hidden_pole_failure_stages.py)
- [experiments/first_order/plot_unstable_rhp_zero_measurement_noise_failure_stages.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/plot_unstable_rhp_zero_measurement_noise_failure_stages.py)

### 4.2 Second-order experiments

- [experiments/second_order/compare_second_order_single_plant.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/second_order/compare_second_order_single_plant.py)
- [experiments/train_rl_second_order_single_plant_numpy.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/train_rl_second_order_single_plant_numpy.py)
- [experiments/second_order/open_loop_second_order_demo.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/second_order/open_loop_second_order_demo.py)

### 4.3 Inverted-pendulum experiments

Main nominal and robustness scripts:

- [experiments/inverted_pendulum/benchmark_inverted_pendulum_rl_variants_nominal.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/benchmark_inverted_pendulum_rl_variants_nominal.py)
- [experiments/inverted_pendulum/hidden_pole_stress_test.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/hidden_pole_stress_test.py)
- [experiments/inverted_pendulum/measurement_noise_stress_test.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/measurement_noise_stress_test.py)

RL training:

- [experiments/inverted_pendulum/train_rl_inverted_pendulum_numpy.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/train_rl_inverted_pendulum_numpy.py)
- [experiments/inverted_pendulum/train_rl_inverted_pendulum_variants_numpy.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/train_rl_inverted_pendulum_variants_numpy.py)
- [experiments/inverted_pendulum/inverted_pendulum_rl_variants.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/inverted_pendulum_rl_variants.py)

Open-loop and simulation scripts:

- [experiments/inverted_pendulum/open_loop_inverted_pendulum.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/open_loop_inverted_pendulum.py)
- [experiments/inverted_pendulum/open_loop_nonlinear_inverted_pendulum.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/open_loop_nonlinear_inverted_pendulum.py)
- [experiments/inverted_pendulum/simulate_inverted_pendulum_mpc.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/simulate_inverted_pendulum_mpc.py)
- [experiments/inverted_pendulum/simulate_nonlinear_inverted_pendulum_mpc.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/simulate_nonlinear_inverted_pendulum_mpc.py)

Animation/export scripts:

- [experiments/inverted_pendulum/export_nominal_nonlinear_controller_grid_mp4.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/export_nominal_nonlinear_controller_grid_mp4.py)
- [experiments/inverted_pendulum/export_hidden_pole_stress_progression_mp4.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/export_hidden_pole_stress_progression_mp4.py)
- [experiments/inverted_pendulum/export_measurement_noise_stress_progression_mp4.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/export_measurement_noise_stress_progression_mp4.py)
- [experiments/inverted_pendulum/export_open_loop_nonlinear_inverted_pendulum_mp4.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/export_open_loop_nonlinear_inverted_pendulum_mp4.py)

## 5. The Shared Manifest

The repository has two configuration layers:

1. [src/control_bench/config.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/config.py)
   Global Python constants and dataclasses.

2. [EXPERIMENT_MANIFEST.yaml](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/EXPERIMENT_MANIFEST.yaml)
   Shared declarative experiment manifest.

The manifest is loaded via [experiment_manifest.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/experiment_manifest.py). If `PyYAML` is unavailable, the loader falls back to Ruby YAML parsing, so installing `PyYAML` is recommended.

### 5.1 What the manifest is authoritative for

In the current codebase, the manifest is especially important for the **first-order MLP benchmark track**.

It defines:

- controller registry
- available MLP feature families
- MLP architecture variants
- default MLP variant
- rollout mixtures for robust RL training
- training noise levels
- domain-randomization ranges
- robust validation settings
- robust seed-selection weights and gates
- experiment entry points for nominal and stress-test studies
- output namespaces and frozen output locations

Important constraint:

- for the first-order MLP controllers, the manifest is the **live source of truth**
- if you change feature definitions, hidden layers, training noise, rollout mix, or cost overrides, you must usually **retrain RL weights**

### 5.2 Main top-level manifest sections

The main top-level sections are:

- `manifest`
  Versioning, repo root, project purpose, frozen state notes.

- `shared`
  Shared assumptions such as the closed-loop engine, runtime measurement conventions, and default controller sets.

- `first_order`
  First-order family definition, plant ordering, evaluation scripts, RL training registry, and robustness studies.

- `second_order`
  Second-order benchmark definition and RL training scope.

- `inverted_pendulum`
  Open-loop demos, MPC experiments, RL training definitions, and nominal/robustness benchmark scope.

- `frozen_outputs`
  Canonical result directories used by the dissertation workflow.

### 5.3 What the first-order controller registry defines

Inside the first-order section, the manifest-backed controller registry defines:

- feature sets:
  - `rich11`
  - `compact6`

- MLP training defaults:
  - sampler kind
  - validation sampler kind
  - rollout mixture
  - training noise
  - domain randomization
  - hidden-pole randomization policy
  - robust validation settings
  - robust seed-selection strategy

- MLP variants:
  - architecture id
  - label
  - feature family
  - hidden layers
  - activation
  - number of seeds
  - steps cap
  - learning rate
  - gradient clipping
  - output file naming prefixes
  - cost overrides

### 5.4 How to edit the manifest safely

Typical safe edits:

- adding or removing a first-order MLP variant
- changing a variant label
- changing hidden layer sizes
- switching `feature_set` between `compact6` and `rich11`
- changing training noise levels
- changing robust validation settings
- adjusting seed-selection weights

Typical edits that require retraining RL:

- cost changes
- feature-set changes
- architecture changes
- rollout-mix changes
- noise/domain-randomization changes

## 6. How To Run The Code

There is no packaged CLI wrapper in this repository. The intended usage is to run scripts directly from the repo root.

Work from:

```bash
cd /Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench
```

Use `python3`, not `python`, unless your local environment aliases `python` correctly.

### 6.1 Recommended runtime assumptions

At minimum, you should have:

- Python 3
- `numpy`
- `matplotlib`
- `PyYAML` recommended

For animation/video export scripts, you may also need:

- `ffmpeg`

The scripts modify `sys.path` internally, so they are intended to run directly from the repository without formal package installation.

## 7. Typical Workflows

### 7.1 Train first-order RL controllers from the manifest

```bash
python3 experiments/first_order/train_first_order_manifest_controllers.py
```

What it does:

- loads the first-order MLP registry from the manifest
- trains PID-features and all registered MLP variants
- writes per-seed and canonical selected weights
- writes training summaries and loss plots

### 7.2 Run the main first-order benchmark

```bash
python3 experiments/first_order/benchmark_first_order_robustness_metrics.py
```

What it does:

- loads the canonical controllers
- evaluates nominal and robustness behaviour across the 9 plants
- writes metrics, summaries, and figures under `experiments/results`

### 7.3 Run a specific first-order stress test

Examples:

```bash
python3 experiments/first_order/robustness_tests/hidden_pole_stress_test.py
python3 experiments/first_order/robustness_tests/measurement_noise_stress_test.py
python3 experiments/first_order/robustness_tests/input_noise_stress_test.py
```

### 7.4 Train pendulum RL variants

```bash
python3 experiments/inverted_pendulum/train_rl_inverted_pendulum_variants_numpy.py
```

What it does:

- trains all pendulum RL variants listed in [inverted_pendulum_rl_variants.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/inverted_pendulum_rl_variants.py)
- stores per-seed and canonical variant weights
- writes training summary tables

### 7.5 Run the nominal pendulum benchmark

```bash
python3 experiments/inverted_pendulum/benchmark_inverted_pendulum_rl_variants_nominal.py
```

### 7.6 Run pendulum robustness benchmarks

```bash
python3 experiments/inverted_pendulum/hidden_pole_stress_test.py
python3 experiments/inverted_pendulum/measurement_noise_stress_test.py
```

### 7.7 Generate presentation assets

Examples:

```bash
python3 experiments/first_order/open_loop_first_order_demo.py
python3 experiments/inverted_pendulum/export_nominal_nonlinear_controller_grid_mp4.py
python3 experiments/inverted_pendulum/export_hidden_pole_stress_progression_mp4.py
python3 experiments/plot_rl_architecture_grid.py
python3 experiments/plot_rl_input_family_grid.py
```

## 8. Where Outputs Are Written

Most generated outputs go under:

- [experiments/results/figures](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/results/figures)
- [experiments/results/metrics](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/results/metrics)
- [experiments/results/rl_numpy](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/results/rl_numpy)

Important output classes:

- trained RL weights:
  - first-order per-plant policies
  - second-order per-plant policies
  - pendulum per-variant policies

- metrics:
  - nominal summaries
  - stress-test summaries
  - seed-selection and training summaries

- figures:
  - open-loop plots
  - closed-loop comparison plots
  - robustness progression plots
  - RL architecture/input diagrams
  - presentation-ready exports

- videos:
  - MP4 exports for pendulum demos and robustness animations

The manifest also records some canonical output paths under `frozen_outputs`.

## 9. What Is Modular In This Codebase

The code is modular in four distinct ways.

### 9.1 Plant/controller/simulation separation

Plants, controllers, and the scenario engine are separated by simple interfaces. The generic closed-loop simulation engine does not need to know whether it is running:

- a first-order plant
- a second-order plant
- a linearized pendulum
- a nonlinear pendulum

### 9.2 Reusable controller families

The same high-level RL and MPC ideas are reused across multiple benchmark tracks:

- first-order MPC and RL
- second-order RL
- pendulum MPC and RL

### 9.3 Manifest-driven first-order MLP track

The first-order MLP study is intentionally declarative. The manifest can change:

- which MLP variants exist
- what feature sets they use
- how they are trained
- how they are validated
- how seeds are selected

without rewriting the training loop itself.

### 9.4 Separate experiment orchestration

The package code in `src/control_bench` provides reusable mechanics. The `experiments` scripts define:

- what to run
- how to compare controllers
- what outputs to save

This separation is what makes the benchmark reusable.

## 10. Important Practical Notes

- The first-order MLP training and selection logic is manifest-driven. If you edit the manifest, check whether the stored weights are still valid.
- Many result scripts assume canonical output weights already exist. Training usually needs to happen before benchmarking.
- The pendulum study trains on the **linearized** pendulum but benchmarks on the **nonlinear** pendulum for the main evaluation.
- Actuator bounds are off in the main first-order and pendulum benchmark configurations unless a specific local or diagnostic script overrides them.
- Some older or lower-level scripts still exist alongside the newer manifest-backed workflow. Prefer the manifest-backed scripts when they exist.

## 11. Recommended Entry Points

If you only need the main study workflow, start with these scripts:

1. [experiments/first_order/train_first_order_manifest_controllers.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/train_first_order_manifest_controllers.py)
2. [experiments/first_order/benchmark_first_order_robustness_metrics.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/benchmark_first_order_robustness_metrics.py)
3. [experiments/inverted_pendulum/train_rl_inverted_pendulum_variants_numpy.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/train_rl_inverted_pendulum_variants_numpy.py)
4. [experiments/inverted_pendulum/benchmark_inverted_pendulum_rl_variants_nominal.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/benchmark_inverted_pendulum_rl_variants_nominal.py)
5. [experiments/inverted_pendulum/hidden_pole_stress_test.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/hidden_pole_stress_test.py)
6. [experiments/inverted_pendulum/measurement_noise_stress_test.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/measurement_noise_stress_test.py)

## 12. If You Want To Extend The Study

Typical extension paths:

- add a new first-order MLP variant in the manifest
- add a new stress-test schedule in the manifest
- add a new pendulum RL feature family in [inverted_pendulum_rl.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/src/control_bench/controllers/inverted_pendulum_rl.py) and [inverted_pendulum_rl_variants.py](/Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/inverted_pendulum/inverted_pendulum_rl_variants.py)
- add a new benchmark script under `experiments/first_order` or `experiments/inverted_pendulum`
- add a new figure/video export script under `experiments`

If you extend RL training logic, keep these three pieces aligned:

- training sampler
- validation suite
- seed-selection logic

If those drift apart, the stored “best” policy can stop matching the benchmark objective.
