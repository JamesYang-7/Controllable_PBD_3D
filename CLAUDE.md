# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated Position-Based Dynamics (PBD/XPBD) framework for 3D deformable object simulation with inverse kinematics control. Built on **Taichi** for GPU compute (CUDA/Vulkan) with real-time OpenGL visualization and USD/OBJ output.

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `taichi`, `usd-core`, `meshio`, `scipy`, `glfw`, `PyOpenGL`, `pyvista`

## Running Simulations

Config-driven simulations (triangle or tet mesh):
```bash
python -m test.test_pbd_mesh --cfg config/sphere_simple_tri.yaml
python -m test.test_pbd_mesh --cfg config/simple_sphere_tet.yaml
```

Standalone scripts (hardcoded paths):
```bash
python test/standalone/cube_tet_ik.py
python test/standalone/sphere_drag.py
```

## Asset Preprocessing

Convert OBJ to tetrahedral mesh or control cage:
```bash
python scripts/obj_to_tet_mesh.py
python scripts/obj_to_tgf_converter.py
```

Assets go in `assets/<name>/` with OBJ mesh, optional `.tgf` cage, and `.txt` weight files.

## File Structure

```
├── cons/              # Constraint system (XPBD + VBD + MGPBD solvers) — CORE
├── data/              # Mesh data structures (Taichi fields) — CORE
├── compdyn/           # Deformation control & inverse kinematics — CORE
├── interface/         # Rendering (Taichi UI viewer, USD export, MG visualizer)
├── utils/             # Geometry, I/O, math, graph coloring, benchmarking helpers
├── experiments/       # Experiment scripts (convergence, scaling, stiffness, ablation)
│   └── run_all.py     # Master runner: python -m experiments.run_all --exp 1.1 1.2
├── config/            # YAML simulation configs
│   └── experiments/   # Per-experiment YAML configs
├── test/              # Test & demo scripts
│   ├── standalone/    # Standalone scripts with hardcoded paths (cube, sphere, prostate, spot)
│   └── jacobi/        # Jacobi vs Gauss-Seidel solver comparison tests
├── scripts/           # Asset preprocessing (OBJ→tet, control point gen)
├── assets/            # Mesh assets (19 subdirs: cube, sphere, prostate, spot, cheb)
│   └── <name>/        # Each contains: .obj/.mesh, .tgf (control points), weights.npy/.txt
├── notes/             # Research notes & constraint derivations
├── out/               # Generated simulation output (.obj, .usdc, experiment plots)
│   └── experiments/   # Per-experiment output directories
├── related_work/      # Reference papers (VBD TOG 2024, Augmented VBD SIGGRAPH 2025)
└── dependencies/
    └── TetGen/        # Git submodule — tetrahedral mesh generator (C++)
```

## Architecture

### Simulation Loop Pattern

All test scripts follow this flow:
1. Load mesh data (`data/`) and config (`config/` YAML)
2. Create `pbd_framework` instance with gravity, timestep, damping
3. Add constraints at indexed levels (level 0 = physics, level 1 = control)
4. Call `init_rest_status()` for each level
5. Per frame: `preupdate_cons()` → substep loop of `make_prediction()` + `update_cons()` → `update_vel()`

### Core Modules

**`cons/` — Constraint system (XPBD + VBD + MGPBD solvers)**
- `framework.py`: `pbd_framework` — manages constraint pipeline, Verlet integration, velocity updates
- `length.py`: `LengthCons` — edge distance preservation with multiple solver methods (default/Jacobi/Gauss-Seidel/graph-coloring)
- `deform3d.py`: `Deform3D` — Neo-Hookean material on tet meshes (coupled hydrostatic + deviatoric XPBD). Accepts optional `benchmark=SolverBenchmark` for profiling.
- `bend.py`: `Bend3D` — dihedral angle bending resistance between adjacent faces
- `volume.py`: `Volume` — global volume preservation
- `vbd.py`: `VBDSolver` — Vertex Block Descent solver (alternative to XPBD). Per-vertex local 3x3 Newton solve with Neo-Hookean material, supports serial and graph-coloring methods, optional Chebyshev acceleration. Accepts optional `benchmark=SolverBenchmark`.
- `mgpbd.py`: `MGPBDSolver` — CPU MGPBD solver. Assembles a global sparse system A * dlambda = b per Newton iteration using ARAP constraint energy, then solves with MGPCG (PyAMG smoothed-aggregation hierarchy + Galerkin RAP coarse-matrix updates). Supports optional backtracking line search. Accepts optional `benchmark=SolverBenchmark`.
- `mgpbd_gpu.py`: `MGPBDSolverGPU` — Fully GPU-resident MGPBD solver. Same interface as `MGPBDSolver` but the inner solve loop has zero CPU-GPU transfers. Uses `GPUMultigridSolver` and `tet_aggregation` hierarchy. Accepts `aggregation='tet'` (geometry-aware) or `'pyamg'`.
- `sparse_gpu.py`: `GPUSparseMatrix` — GPU-resident CSR sparse matrix (Taichi fields). Module-level kernels: `gpu_dot`, `gpu_axpy`, `gpu_xpay`, `gpu_copy`, `gpu_fill_zero`, `gpu_norm_sq`, etc. SpMV and weighted Jacobi smoothing.
- `mgpcg_gpu.py`: `GPUMultigridSolver` — GPU-native V-cycle + MGPCG solver. Setup builds contribution maps on CPU (one-time), then coarse matrix values are recomputed on GPU via pre-computed `(k,l)` pair indices (avoids SpMM). `solve(b, x)` runs PCG with V-cycle preconditioner entirely on GPU.
- `tet_aggregation.py`: Tet-specialized multigrid hierarchy. `tet_geometry_aggregation()` does greedy face-adjacent tet matching. `build_prolongation_from_aggregates()` builds piecewise-constant P matrices. `build_tet_hierarchy()` recurses until `max_coarse` reached.
- `adaptive_zone.py`: `AdaptiveZone` — marks tets active/passive by proximity to tool or strain rate threshold. `expand_zone(n_rings)` inflates boundary for stability buffer.
- `mgpbd_adaptive.py`: `MGPBDAdaptiveSolver` — wraps CPU or GPU pipeline; inflates `alpha_tilde` to 1e20 for passive tets to freeze them while retaining a single solve call.

All XPBD constraints implement: `init_rest_status()`, `preupdate_cons()`, `update_cons()`

**`data/` — Mesh data structures (Taichi fields)**
- `cloth_data.py`: `ClothData` — triangle mesh (positions `v_p`, faces `f_i`, edges `e_i`, inverse mass `v_invm`)
- `tet_data.py`: `TetData` — tetrahedral mesh with surface extraction for rendering
- `cage_data.py`: `CageData` — control cage with skinning weights
- `points_data.py`: `PointsData` — control points with rotation matrices and weight matrices
- `lbs_data.py`: `CageLBS3D`, `LBS3D` — linear blend skinning (translational / rotation+translation)
- `motion_data.py`: `MotionData3D` — per-bone per-frame motion capture data

**`compdyn/` — Deformation control & IK**
- `base.py`: `CompDynBase` — translational LBS constraint (XPBD); `CompDynMomentum` — adds momentum conservation
- `point.py`: `CompDynPoint` — rotation-enhanced LBS constraint (6 DOF per control point)
- `IK.py`: `CageIK`, `PointsIK` — scipy-based inverse kinematics (solve cage/point transforms from mesh deformation)
- `IK_taylor.py`: `PointsTaylorIK` — Taylor expansion IK variant with incremental rotation
- `inverse.py`: `CompDynInv` — direct inverse CompDyn with precomputed solver

**`interface/` — Rendering**
- `mesh_render_3d.py`: `MeshRender3D` — real-time Taichi UI viewer with camera, events, wireframe toggle
- `render_funcs.py`: Factory functions for rendering meshes, cages, particles into `ti.ui.Scene`
- `usd_render.py`: `UsdRender` — USD stage creation
- `usd_objs.py`: `SaveMesh`, `SaveLines`, `SavePoints` — USD geometry exporters with per-frame update
- `mg_visualizer.py`: `MultigridVisualizer` — colored tet overlay showing aggregate groups per hierarchy level; `ZoneVisualizer` — colors active/passive tets from `AdaptiveZone`. Both provide `get_render_func()` for `MeshRender3D`. Call `cycle_level()` (bind to 'g') to step through MG levels.

**`utils/` — Helpers**
- `geom2d.py` / `geom3d.py`: Edge extraction, rest lengths, vertex mass, surface extraction from tets
- `io.py`: Load OBJ, TET (.mesh), TGF cage files via meshio
- `mathlib.py`: Vector rotations (axis-angle ↔ matrix), SPH kernels, NaN detection
- `graph_coloring.py`: Greedy graph coloring for parallel constraint solving
- `control_utils.py`: Control point generation (k-means, KDTree weighting, influence radius)
- `anim.py`: Sinusoidal movement for operating points
- `arg_parser.py`: CLI argument parser + YACS config loader
- `objs.py`: `Quad`, `BoundBox3D` — collision objects (planes, AABB)
- `bench.py`: `SolverBenchmark` — per-frame timing (uses `ti.sync()` + `time.perf_counter()`) and per-iteration residual tracking. `save(path)` writes JSON; `summary()` returns avg frame time, iterations, residual. Used by `MGPBDSolver`, `MGPBDSolverGPU`, `VBDSolver`, `Deform3D`.

**`experiments/` — Benchmark & experiment scripts**
- `plot_utils.py`: Matplotlib helpers — `plot_convergence()`, `plot_resolution_scaling()`, `plot_stiffness_scaling()`, `plot_timing_bars()`, `plot_adaptive_tradeoff()`
- `exp_convergence.py`: Exp 1.1 — XPBD vs VBD vs MGPBD-CPU vs MGPBD-GPU convergence curves
- `exp_resolution.py`: Exp 1.2 — wall-clock scaling with mesh resolution
- `exp_stiffness.py`: Exp 1.3 — iteration count vs material stiffness (mu 1e3–1e7)
- `exp_tool_tissue.py`: Exp 2.2 — prostate mesh with displacement-controlled tool interaction
- `exp_adaptive.py`: Exp 2.3 — speedup vs accuracy with spatial adaptivity zones
- `exp_ablation.py`: Exp 3.1 — per-component speedup: generic AMG → tet aggregation → GPU solve → spatial adaptivity
- `run_all.py`: Master runner; `python -m experiments.run_all --exp 1.1 1.2` or `--list`

### Config Format (YAML)

```yaml
dir_name: "asset_folder"
obj_name: "mesh_name.xxx"           # with extension
fixed_id: [0, 11]               # fixed vertex indices
operating_points_id: [11]       # user-controlled vertices
common:
  fps: 60
  substeps: 5
  damp: 0.993
  gravity: [0.0, 0.0, 0.0]
cons:
  length: { enabled: true, alpha: 1e-4, method: 'default' }
  bend: { enabled: false, alpha: 100.0 }
  volume: { enabled: false, alpha: 0.0 }
```

`alpha` is XPBD compliance: higher = softer constraint.

### Running Experiments

```bash
python -m experiments.run_all --list                  # show available experiments
python -m experiments.run_all --exp 1.1               # convergence comparison
python -m experiments.run_all --exp 1.1 1.2 1.3       # multiple experiments
python -m experiments.run_all --n_frames 10           # quick smoke test
```

Output plots are saved to `out/experiments/<exp_name>/`.

### Taichi Conventions

- All physics computation uses `@ti.kernel` and `@ti.func` decorators for GPU execution
- Mesh data stored as `ti.field` and `ti.Vector.field` (GPU-resident)
- Taichi must be initialized before any field allocation (typically `ti.init(arch=ti.vulkan)`)
- Constraint solvers use atomic operations (`ti.atomic_add`) for parallel Jacobi updates
- **`@staticmethod` + `@ti.kernel` with `ti.template()` does NOT work in Taichi 1.7.1.** Use module-level `@ti.kernel` functions instead (see `cons/sparse_gpu.py` pattern).
