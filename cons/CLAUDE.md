# cons/
PBD solver and constraints, with other solvers like VBD and MGPBD.

## Code Structure

### Constraint Solvers

All constraint classes follow the same interface (used by `pbd_framework`):
- `init_rest_status()` — compute rest-state quantities (called once)
- `preupdate_cons()` — reset Lagrange multipliers to zero (called each timestep)
- `update_cons()` — solve one iteration of the constraint (called per solver iteration)

Each class is decorated with `@ti.data_oriented` and contains `@ti.kernel` solve methods.

| File | Class | Description |
|------|-------|-------------|
| `framework.py` | `pbd_framework` | Orchestrates the PBD loop: prediction, constraint solving, velocity update. Holds ordered lists of constraints, pre-update hooks, and collision objects. |
| `length.py` | `LengthCons` | Edge length (distance) constraints. Supports 4 solve methods: `default` (parallel Gauss-Seidel with race conditions), `gauss_seidel` (serialized), `jacobi` (two-phase accumulate+apply), `graph_coloring` (parallel conflict-free). |
| `bend.py` | `Bend3D` | Dihedral angle bending constraints for cloth. Uses 4 vertices (2 edge + 2 side). |
| `volume.py` | `Volume` | Global volume preservation constraint for closed meshes. Single scalar multiplier over all faces. |
| `deform3d.py` | `Deform3D` | 3D tetrahedral deformation constraints with split hydrostatic (det(F)-1) and deviatoric (||F||²-3) terms, solved jointly via a 2x2 coupled XPBD update. Supports `default`, `gauss_seidel`, and `graph_coloring` methods. |
| `vbd.py` | `VBDSolver` | Vertex Block Descent solver for tetrahedral elasticity. Self-contained (does not use `pbd_framework`): has its own `step()` → `make_prediction()` → `solve()` → `update_vel()`. Uses Neo-Hookean energy with per-vertex 3x3 local solves, CSR vertex-to-tet adjacency, optional Chebyshev acceleration, and graph-coloring parallelism. |
| `mgpbd.py` | `MGPBDSolver` | Multigrid-accelerated Global XPBD solver for tetrahedral elasticity (SIGGRAPH 2025). Self-contained with its own `step()` → `make_prediction()` → `solve()` → `update_vel()`. Assembles a global sparse system A * dlambda = b each Newton iteration and solves it with MGPCG (PyAMG smoothed-aggregation hierarchy + Galerkin RAP coarse updates). Uses ARAP constraint energy. Supports optional backtracking line search. |