"""Spatial zone definition for adaptive MGPBD solving.

Marks tets as active (near tool / high strain) or passive, allowing
the solver to concentrate effort where it matters.
"""

import taichi as ti
import numpy as np

from cons.tet_aggregation import build_tet_face_adjacency


@ti.data_oriented
class AdaptiveZone:

  def __init__(self, n_tets: int, t_i: ti.Field, v_p: ti.MatrixField):
    self.n_tets = n_tets
    self.n_verts = v_p.shape[0]
    self.t_i = t_i
    self.v_p = v_p

    # Per-tet active flag: 0 = passive, 1 = active
    self.tet_active = ti.field(dtype=ti.i32, shape=n_tets)
    # Fill all active by default
    self._fill_all_active()

    # Pre-build face-adjacency for zone expansion
    self._adj = None

  @ti.kernel
  def _fill_all_active(self):
    for t in range(self.n_tets):
      self.tet_active[t] = 1

  @ti.kernel
  def update_by_proximity(self, tool_x: ti.f32, tool_y: ti.f32,
                          tool_z: ti.f32, radius: ti.f32):
    """Mark tets with any vertex within radius of tool position."""
    r2 = radius * radius
    for t in range(self.n_tets):
      active = 0
      for l in ti.static(range(4)):
        p = self.v_p[self.t_i[t * 4 + l]]
        dx = p[0] - tool_x
        dy = p[1] - tool_y
        dz = p[2] - tool_z
        if dx * dx + dy * dy + dz * dz < r2:
          active = 1
      self.tet_active[t] = active

  @ti.kernel
  def update_by_strain(self, v_p_old: ti.template(), threshold: ti.f32,
                       dt: ti.f32):
    """Mark tets exceeding strain-rate threshold."""
    for t in range(self.n_tets):
      max_vel = 0.0
      for l in ti.static(range(4)):
        vid = self.t_i[t * 4 + l]
        dp = self.v_p[vid] - v_p_old[vid]
        vel = dp.norm() / dt
        if vel > max_vel:
          max_vel = vel
      if max_vel > threshold:
        self.tet_active[t] = 1
      else:
        self.tet_active[t] = 0

  def expand_zone(self, n_rings: int):
    """Expand active zone by n face-adjacent rings (CPU-side)."""
    if self._adj is None:
      t_i_np = self.t_i.to_numpy()
      self._adj = build_tet_face_adjacency(t_i_np, self.n_tets)

    active = self.tet_active.to_numpy()
    for _ in range(n_rings):
      new_active = active.copy()
      for t in range(self.n_tets):
        if active[t] == 1:
          for nb in self._adj[t]:
            new_active[nb] = 1
      active = new_active
    self.tet_active.from_numpy(active)

  def get_active_mask(self) -> np.ndarray:
    """Return numpy bool array of active tets."""
    return self.tet_active.to_numpy().astype(bool)

  def get_active_fraction(self) -> float:
    """Fraction of tets that are active."""
    active = self.tet_active.to_numpy()
    return float(np.sum(active)) / self.n_tets
