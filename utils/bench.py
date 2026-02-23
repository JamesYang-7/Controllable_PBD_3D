"""Reusable timing and convergence tracker for solver benchmarking."""

import time
import json
import numpy as np

try:
  import taichi as ti
  _has_taichi = True
except ImportError:
  _has_taichi = False


class SolverBenchmark:
  """Records per-frame, per-iteration timing and convergence data."""

  def __init__(self, solver_name: str, mesh_name: str, params: dict = None):
    self.solver_name = solver_name
    self.mesh_name = mesh_name
    self.params = params or {}
    self.frames = []
    self._current_frame = None

  def begin_frame(self, frame_id: int):
    if _has_taichi:
      ti.sync()
    self._current_frame = {
        'frame_id': frame_id,
        'iterations': [],
        't_start': time.perf_counter(),
        't_end': None,
    }

  def record_iteration(self, iteration: int, residual: float,
                       cg_iters: int = 0, omega: float = 1.0):
    if self._current_frame is None:
      return
    if _has_taichi:
      ti.sync()
    self._current_frame['iterations'].append({
        'iteration': iteration,
        'residual': float(residual),
        'cg_iters': cg_iters,
        'omega': float(omega),
        't': time.perf_counter(),
    })

  def end_frame(self):
    if self._current_frame is None:
      return
    if _has_taichi:
      ti.sync()
    self._current_frame['t_end'] = time.perf_counter()
    self.frames.append(self._current_frame)
    self._current_frame = None

  def save(self, path: str):
    data = {
        'solver_name': self.solver_name,
        'mesh_name': self.mesh_name,
        'params': self.params,
        'frames': self.frames,
    }
    with open(path, 'w') as f:
      json.dump(data, f, indent=2)

  @staticmethod
  def load(path: str) -> 'SolverBenchmark':
    with open(path, 'r') as f:
      data = json.load(f)
    bench = SolverBenchmark(data['solver_name'], data['mesh_name'],
                            data.get('params', {}))
    bench.frames = data['frames']
    return bench

  def summary(self) -> dict:
    if not self.frames:
      return {}
    frame_times = []
    iter_counts = []
    final_residuals = []
    for fr in self.frames:
      if fr['t_end'] is not None and fr['t_start'] is not None:
        frame_times.append(fr['t_end'] - fr['t_start'])
      iters = fr['iterations']
      iter_counts.append(len(iters))
      if iters:
        final_residuals.append(iters[-1]['residual'])
    return {
        'solver_name': self.solver_name,
        'mesh_name': self.mesh_name,
        'n_frames': len(self.frames),
        'avg_frame_time': float(np.mean(frame_times)) if frame_times else 0.0,
        'avg_iterations': float(np.mean(iter_counts)) if iter_counts else 0.0,
        'avg_final_residual': float(np.mean(final_residuals)) if final_residuals else 0.0,
        'total_time': float(np.sum(frame_times)) if frame_times else 0.0,
    }
