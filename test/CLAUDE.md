# test/

Test and demo scripts for the PBD simulation framework.

## Config-driven scripts (top level)

Run from the repo root with `python -m test.<script> --cfg config/<file>.yaml`:

- `test_pbd_mesh.py` — XPBD mesh simulation
- `test_vbd.py` — VBD solver simulation
- `test_mgpbd_tetmesh.py` — MGPBD solver simulation
- `vedo_mesh.py` — mesh visualization with vedo
- `viewer_weight.py` — visualize skinning weights

## standalone/

Scripts with hardcoded asset paths. Run from the repo root:

```bash
python test/standalone/cube_tet_ik.py
python test/standalone/sphere_drag.py
```

Covers: cube (balloon, drag, IK), sphere (drag, IK), prostate (drag, IK), spot (free, IK, mesh, tet).

## jacobi/

Comparison tests for Jacobi vs Gauss-Seidel constraint solving methods.
