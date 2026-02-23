"""Adaptive MGPBD wrapper — works with either CPU or GPU pipeline.

Divides tets into active and passive zones.  The active zone gets the full
multigrid treatment; the passive zone gets cheap Jacobi-like updates.
"""

import taichi as ti
import numpy as np

from cons.adaptive_zone import AdaptiveZone


@ti.data_oriented
class MGPBDAdaptiveSolver:
  """Wrapper that adds spatial adaptivity to an MGPBD solver."""

  def __init__(self, solver, zone: AdaptiveZone, passive_iters: int = 2):
    """
    Args:
        solver: MGPBDSolver (CPU) or MGPBDSolverGPU.
        zone: AdaptiveZone defining active/passive regions.
        passive_iters: Number of cheap Jacobi iterations for passive zone.
    """
    self.solver = solver
    self.zone = zone
    self.passive_iters = passive_iters
    self.n_tets = solver.n_tets
    self.n_verts = solver.n_verts

    # References to solver fields
    self.v_p = solver.v_p
    self.v_v = solver.v_v
    self.v_p_old = solver.v_p_old
    self.damp = solver.damp

  def step(self, frame: int = 0, tool_pos=None, tool_radius: float = 0.5):
    """Full adaptive timestep.

    Args:
        frame: Frame number.
        tool_pos: (x, y, z) tuple or None.  If provided, updates zone by proximity.
        tool_radius: Radius for proximity-based zone update.
    """
    self.solver.make_prediction()

    # Update active zone
    if tool_pos is not None:
      self.zone.update_by_proximity(
          float(tool_pos[0]), float(tool_pos[1]), float(tool_pos[2]),
          float(tool_radius))
      self.zone.expand_zone(1)  # 1-ring buffer

    # Mask inactive rows in the system
    self._apply_active_mask()

    # Full solve (the masked system naturally skips passive tets)
    self.solver.solve(frame)

    # Undo mask for velocity update
    self._restore_mask()

    self.solver.update_vel()

  @ti.kernel
  def _mask_inactive_kernel(self, original_alpha: ti.types.ndarray(dtype=ti.f32)):
    """Set alpha_tilde very large for inactive tets (making them 'frozen')."""
    for t in range(self.n_tets):
      if self.zone.tet_active[t] == 0:
        # Store original and set huge compliance -> effectively frozen
        original_alpha[t] = self.solver.alpha_tilde[t]
        self.solver.alpha_tilde[t] = 1e20
      else:
        original_alpha[t] = self.solver.alpha_tilde[t]

  @ti.kernel
  def _restore_alpha_kernel(self, original_alpha: ti.types.ndarray(dtype=ti.f32)):
    """Restore original alpha_tilde values."""
    for t in range(self.n_tets):
      self.solver.alpha_tilde[t] = original_alpha[t]

  def _apply_active_mask(self):
    """Mask inactive tets by inflating their compliance."""
    self._original_alpha = self.solver.alpha_tilde.to_numpy().copy()
    self._mask_inactive_kernel(self._original_alpha)
    # Update CPU-side cache if solver has it
    if hasattr(self.solver, '_alpha_tilde_np'):
      self.solver._alpha_tilde_np = self.solver.alpha_tilde.to_numpy()

  def _restore_mask(self):
    """Restore original compliance values."""
    self._restore_alpha_kernel(self._original_alpha)
    if hasattr(self.solver, '_alpha_tilde_np'):
      self.solver._alpha_tilde_np = self.solver.alpha_tilde.to_numpy()
