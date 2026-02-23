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
├── cons/              # Constraint system (XPBD + VBD solvers) — CORE
├── data/              # Mesh data structures (Taichi fields) — CORE
├── compdyn/           # Deformation control & inverse kinematics — CORE
├── interface/         # Rendering (Taichi UI viewer, USD export)
├── utils/             # Geometry, I/O, math, graph coloring helpers
├── config/            # YAML simulation configs
├── test/              # Test & demo scripts
│   ├── standalone/    # Standalone scripts with hardcoded paths (cube, sphere, prostate, spot)
│   └── jacobi/        # Jacobi vs Gauss-Seidel solver comparison tests
├── scripts/           # Asset preprocessing (OBJ→tet, control point gen)
├── assets/            # Mesh assets (19 subdirs: cube, sphere, prostate, spot, cheb)
│   └── <name>/        # Each contains: .obj/.mesh, .tgf (control points), weights.npy/.txt
├── notes/             # Research notes & constraint derivations
├── out/               # Generated simulation output (.obj, .usdc)
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
- `deform3d.py`: `Deform3D` — Neo-Hookean material on tet meshes (coupled hydrostatic + deviatoric XPBD)
- `bend.py`: `Bend3D` — dihedral angle bending resistance between adjacent faces
- `volume.py`: `Volume` — global volume preservation
- `vbd.py`: `VBDSolver` — Vertex Block Descent solver (alternative to XPBD). Per-vertex local 3x3 Newton solve with Neo-Hookean material, supports serial and graph-coloring methods, optional Chebyshev acceleration
- `mgpbd.py`: `MGPBDSolver` — Multigrid-accelerated Global XPBD solver for tetrahedral elasticity. Assembles a global sparse system A * dlambda = b per Newton iteration using ARAP constraint energy, then solves it with MGPCG (PyAMG smoothed-aggregation hierarchy + Galerkin RAP coarse-matrix updates). Supports optional backtracking line search.

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

**`utils/` — Helpers**
- `geom2d.py` / `geom3d.py`: Edge extraction, rest lengths, vertex mass, surface extraction from tets
- `io.py`: Load OBJ, TET (.mesh), TGF cage files via meshio
- `mathlib.py`: Vector rotations (axis-angle ↔ matrix), SPH kernels, NaN detection
- `graph_coloring.py`: Greedy graph coloring for parallel constraint solving
- `control_utils.py`: Control point generation (k-means, KDTree weighting, influence radius)
- `anim.py`: Sinusoidal movement for operating points
- `arg_parser.py`: CLI argument parser + YACS config loader
- `objs.py`: `Quad`, `BoundBox3D` — collision objects (planes, AABB)

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

### Taichi Conventions

- All physics computation uses `@ti.kernel` and `@ti.func` decorators for GPU execution
- Mesh data stored as `ti.field` and `ti.Vector.field` (GPU-resident)
- Taichi must be initialized before any field allocation (typically `ti.init(arch=ti.vulkan)`)
- Constraint solvers use atomic operations (`ti.atomic_add`) for parallel Jacobi updates
