#!/usr/bin/env python3
"""
Side-by-side visualization:
  (Left)  OBJ triangle surface mesh
  (Right) TetGen -g output .mesh (MEDIT) tetrahedral mesh:
          clipped cutaway + edges to reveal interior tets

Dependencies:
  pip install pyvista meshio numpy

Usage:
  python viz_obj_vs_tetgen_mesh.py --obj surface.obj --mesh volume.mesh
  python viz_obj_vs_tetgen_mesh.py --obj surface.obj --mesh volume.mesh --out compare.png
"""

import argparse
import numpy as np
import meshio
import pyvista as pv


VTK_TETRA = 10  # VTK cell type id for tetrahedron


def meshio_to_pyvista_tet_grid(m: meshio.Mesh) -> pv.UnstructuredGrid:
    points = np.asarray(m.points, dtype=np.float64)

    # Collect tetra cells from meshio cell blocks (format-agnostic)
    tets = []
    for block in m.cells:
        if block.type in ("tetra", "tetra4"):
            tets.append(np.asarray(block.data, dtype=np.int64))

    if not tets:
        # Some writers use "tetra10" (quadratic). Handle by taking the first 4 vertices.
        for block in m.cells:
            if block.type == "tetra10":
                data = np.asarray(block.data, dtype=np.int64)[:, :4]
                tets.append(data)

    if not tets:
        available = sorted({b.type for b in m.cells})
        raise ValueError(
            f"No tetrahedral cells found. Available cell types: {available}. "
            f"If this file is not a volume mesh, you may need TetGen '-g' output or a different file."
        )

    tet = np.vstack(tets)  # (n_tet, 4)

    # Build VTK "cells" connectivity array: [4, i0, i1, i2, i3, 4, j0, j1, j2, j3, ...]
    n = tet.shape[0]
    cells = np.empty((n, 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = tet
    cells = cells.ravel()

    celltypes = np.full(n, VTK_TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, help="Path to triangle surface mesh (.obj)")
    ap.add_argument("--mesh", required=True, help="Path to TetGen -g output (.mesh, MEDIT)")
    ap.add_argument("--out", default="", help="Optional output image path (e.g., compare.png)")
    ap.add_argument(
        "--clip-normal",
        default="0,0,-1",
        help="Clip plane normal for tet cutaway, e.g. '1,0,0' or '0,1,0' or '0,0,1'",
    )
    ap.add_argument(
        "--clip-offset",
        type=float,
        default=0.0,
        help="Offset along clip normal (in model units). 0 means clip through the grid center.",
    )
    args = ap.parse_args()

    # Read meshes
    surf = pv.read(args.obj)

    # TetGen -g writes MEDIT .mesh; meshio reads it via medit reader.
    vol_m = meshio.read(args.mesh)
    vol = meshio_to_pyvista_tet_grid(vol_m)

    # Prepare tet visualizations
    normal = np.array([float(x) for x in args.clip_normal.split(",")], dtype=np.float64)
    normal /= (np.linalg.norm(normal) + 1e-12)

    origin = np.array(vol.center, dtype=np.float64) + args.clip_offset * normal

    # Clip the volume, then extract surface for rendering (faster + cleaner)
    clipped = vol.clip(normal=tuple(normal), origin=tuple(origin), invert=False)
    clipped_surf = clipped.extract_surface()

    # Clip the OBJ surface mesh with the same plane for consistent comparison
    surf_clipped = surf.clip(normal=tuple(normal), origin=tuple(origin), invert=False)

    # Plot
    off_screen = bool(args.out)
    pl = pv.Plotter(shape=(1, 2), window_size=(2000, 900), off_screen=off_screen)

    # Left: OBJ surface (cutaway)
    pl.subplot(0, 0)
    pl.add_text("Triangle mesh (cutaway)", font_size=14)
    # pl.add_mesh(surf_clipped, style='wireframe', line_width=1)
    # pl.add_mesh(surf_clipped, smooth_shading=True)
    pl.add_mesh(surf_clipped, show_edges=True, smooth_shading=False)
    pl.add_mesh(surf.outline(), line_width=2)
    pl.add_axes()
    # pl.show_grid(False)

    # Right: Tet mesh cutaway + edges
    pl.subplot(0, 1)
    pl.add_text("Tetrahedral mesh (cutaway)", font_size=14)
    pl.add_mesh(clipped_surf, show_edges=True, smooth_shading=False)
    pl.add_mesh(vol.outline(), line_width=2)
    pl.add_axes()
    # pl.show_grid(False)

    # Link camera so the comparison is fair
    pl.subplot(0, 0)
    pl.view_isometric()
    cam = pl.camera_position
    pl.subplot(0, 1)
    pl.camera_position = cam

    if args.out:
        pl.show(screenshot=args.out, auto_close=True)
        print(f"Saved: {args.out}")
    else:
        pl.show()


if __name__ == "__main__":
    main()
