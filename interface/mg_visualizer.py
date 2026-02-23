"""Multigrid hierarchy visualizer — colored tet overlay showing aggregate groups.

Provides render functions compatible with MeshRender3D.add_render_func().
"""

import taichi as ti
import numpy as np


# Tab20-like colormap (20 distinct colors)
_TAB20 = np.array([
    [0.12, 0.47, 0.71], [0.68, 0.78, 0.91],
    [1.00, 0.50, 0.05], [1.00, 0.73, 0.47],
    [0.17, 0.63, 0.17], [0.60, 0.87, 0.54],
    [0.84, 0.15, 0.16], [1.00, 0.60, 0.59],
    [0.58, 0.40, 0.74], [0.77, 0.69, 0.84],
    [0.55, 0.34, 0.29], [0.77, 0.61, 0.58],
    [0.89, 0.47, 0.76], [0.97, 0.71, 0.85],
    [0.50, 0.50, 0.50], [0.78, 0.78, 0.78],
    [0.74, 0.74, 0.13], [0.86, 0.86, 0.55],
    [0.09, 0.75, 0.81], [0.62, 0.85, 0.90],
], dtype=np.float32)


class MultigridVisualizer:
  """Visualizes multigrid hierarchy levels as colored tet overlays."""

  def __init__(self, Ps, t_i_np, v_p_field, f_i_field, n_tets, n_verts):
    """
    Args:
        Ps: List of prolongation matrices (scipy CSR), from hierarchy builder.
        t_i_np: Tet index array (flat, length 4*n_tets).
        v_p_field: Vertex positions (ti.Vector.field).
        f_i_field: Surface face indices (ti.field) for rendering.
        n_tets: Number of tetrahedra.
        n_verts: Number of vertices.
    """
    self.Ps = Ps
    self.t_i_np = t_i_np
    self.v_p_field = v_p_field
    self.f_i_field = f_i_field
    self.n_tets = n_tets
    self.n_verts = n_verts
    self.n_levels = len(Ps) + 1  # level 0 = original

    # Pre-compute per-level aggregate assignments
    self._agg_ids_per_level = self._compute_all_levels()

    # Per-vertex color field for rendering
    self.vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    self._current_level = 0
    self._set_level_colors(0)

  def _compute_all_levels(self):
    """Compute aggregate IDs for each tet at each hierarchy level."""
    result = []
    # Level 0: each tet is its own group
    result.append(np.arange(self.n_tets, dtype=np.int32))

    # For each subsequent level, compose the P matrices
    cumulative = np.arange(self.n_tets, dtype=np.int32)
    for P in self.Ps:
      # P is (n_fine x n_coarse), piecewise constant.
      # For each fine tet, its coarse aggregate is argmax of P's row.
      P_dense = P.toarray() if hasattr(P, 'toarray') else np.array(P)
      new_ids = np.zeros(self.n_tets, dtype=np.int32)
      for t in range(self.n_tets):
        fine_id = cumulative[t]
        if fine_id < P_dense.shape[0]:
          row = P_dense[fine_id]
          new_ids[t] = int(np.argmax(row))
        else:
          new_ids[t] = 0
      cumulative = new_ids
      result.append(cumulative.copy())

    return result

  def _set_level_colors(self, level: int):
    """Set per-vertex colors based on tet aggregate membership at given level."""
    level = min(level, len(self._agg_ids_per_level) - 1)
    agg_ids = self._agg_ids_per_level[level]
    n_groups = int(np.max(agg_ids)) + 1

    # Assign colors to vertices based on their tet membership
    # If a vertex belongs to multiple tets with different aggregates, use majority
    vert_colors = np.zeros((self.n_verts, 3), dtype=np.float32)
    vert_counts = np.zeros(self.n_verts, dtype=np.float32)

    for t in range(self.n_tets):
      color_idx = agg_ids[t] % len(_TAB20)
      color = _TAB20[color_idx]
      for l in range(4):
        vid = self.t_i_np[t * 4 + l]
        vert_colors[vid] += color
        vert_counts[vid] += 1.0

    # Average
    mask = vert_counts > 0
    vert_colors[mask] /= vert_counts[mask, np.newaxis]

    self.vertex_colors.from_numpy(vert_colors)
    self._current_level = level

  def extract_aggregation_groups(self, level: int) -> np.ndarray:
    """Per-tet aggregate ID at given level."""
    level = min(level, len(self._agg_ids_per_level) - 1)
    return self._agg_ids_per_level[level]

  def get_render_func(self):
    """Returns render function for the current level.

    Compatible with MeshRender3D.add_render_func().
    """
    v_p = self.v_p_field
    f_i = self.f_i_field
    vc = self.vertex_colors

    def render_func(scene: ti.ui.Scene):
      scene.mesh(v_p, f_i, per_vertex_color=vc, show_wireframe=True)

    return render_func

  def cycle_level(self):
    """Advance to next hierarchy level (wraps around)."""
    next_level = (self._current_level + 1) % self.n_levels
    self._set_level_colors(next_level)
    print(f"MG Visualizer: level {next_level}/{self.n_levels - 1}, "
          f"groups={int(np.max(self._agg_ids_per_level[next_level])) + 1}")

  def set_level(self, level: int):
    """Set a specific hierarchy level."""
    self._set_level_colors(level)


class ZoneVisualizer:
  """Visualizes active/passive zones from AdaptiveZone."""

  def __init__(self, zone, t_i_np, v_p_field, f_i_field, n_tets, n_verts):
    self.zone = zone
    self.t_i_np = t_i_np
    self.v_p_field = v_p_field
    self.f_i_field = f_i_field
    self.n_tets = n_tets
    self.n_verts = n_verts

    self.vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

  def update_colors(self):
    """Update vertex colors: active=red/orange, passive=blue/gray."""
    active = self.zone.tet_active.to_numpy()

    vert_colors = np.zeros((self.n_verts, 3), dtype=np.float32)
    vert_counts = np.zeros(self.n_verts, dtype=np.float32)

    active_color = np.array([1.0, 0.3, 0.1], dtype=np.float32)
    passive_color = np.array([0.3, 0.4, 0.7], dtype=np.float32)

    for t in range(self.n_tets):
      color = active_color if active[t] else passive_color
      for l in range(4):
        vid = self.t_i_np[t * 4 + l]
        vert_colors[vid] += color
        vert_counts[vid] += 1.0

    mask = vert_counts > 0
    vert_colors[mask] /= vert_counts[mask, np.newaxis]
    self.vertex_colors.from_numpy(vert_colors)

  def get_render_func(self):
    """Returns render function compatible with MeshRender3D."""
    v_p = self.v_p_field
    f_i = self.f_i_field
    vc = self.vertex_colors
    viz = self

    def render_func(scene: ti.ui.Scene):
      viz.update_colors()
      scene.mesh(v_p, f_i, per_vertex_color=vc, show_wireframe=True)

    return render_func
