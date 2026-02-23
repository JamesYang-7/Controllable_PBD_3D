"""Matplotlib plotting helpers for MGPBD experiments."""

import json
import numpy as np

try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  _has_mpl = True
except ImportError:
  _has_mpl = False


def load_benchmark(path):
  """Load a SolverBenchmark JSON file."""
  with open(path, 'r') as f:
    return json.load(f)


def plot_convergence(benchmarks, title='Convergence', save_path=None):
  """Plot residual vs iteration count for multiple solvers.

  Args:
      benchmarks: list of dicts, each with 'solver_name' and 'frames' keys
                  (from SolverBenchmark.save()).
      title: Plot title.
      save_path: If provided, save figure to this path.
  """
  if not _has_mpl:
    print("matplotlib not available, skipping plot")
    return

  fig, ax = plt.subplots(1, 1, figsize=(8, 5))

  for bench in benchmarks:
    name = bench['solver_name']
    # Average residual per iteration across frames
    max_iters = 0
    for fr in bench['frames']:
      max_iters = max(max_iters, len(fr['iterations']))

    if max_iters == 0:
      continue

    avg_residuals = np.zeros(max_iters)
    counts = np.zeros(max_iters)
    for fr in bench['frames']:
      for it in fr['iterations']:
        idx = it['iteration']
        if idx < max_iters:
          avg_residuals[idx] += it['residual']
          counts[idx] += 1

    mask = counts > 0
    avg_residuals[mask] /= counts[mask]

    iters = np.arange(max_iters)[mask]
    ax.semilogy(iters, avg_residuals[mask], label=name, linewidth=2)

  ax.set_xlabel('Iteration')
  ax.set_ylabel('Residual (log)')
  ax.set_title(title)
  ax.legend()
  ax.grid(True, alpha=0.3)

  if save_path:
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
  plt.close(fig)


def plot_resolution_scaling(results, title='Resolution Scaling', save_path=None):
  """Plot wall-clock vs mesh resolution for multiple solvers.

  Args:
      results: list of dicts with keys:
          'solver_name', 'n_tets', 'avg_frame_time', 'fps'
  """
  if not _has_mpl:
    return

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  # Group by solver
  solvers = {}
  for r in results:
    name = r['solver_name']
    if name not in solvers:
      solvers[name] = {'n_tets': [], 'time': [], 'fps': []}
    solvers[name]['n_tets'].append(r['n_tets'])
    solvers[name]['time'].append(r['avg_frame_time'])
    solvers[name]['fps'].append(r.get('fps', 1.0 / r['avg_frame_time']
                                       if r['avg_frame_time'] > 0 else 0))

  for name, data in solvers.items():
    order = np.argsort(data['n_tets'])
    n = np.array(data['n_tets'])[order]
    t = np.array(data['time'])[order]
    f = np.array(data['fps'])[order]
    ax1.loglog(n, t, 'o-', label=name, linewidth=2)
    ax2.semilogx(n, f, 'o-', label=name, linewidth=2)

  ax1.set_xlabel('Number of Tets')
  ax1.set_ylabel('Frame Time (s)')
  ax1.set_title(f'{title} — Frame Time')
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  ax2.set_xlabel('Number of Tets')
  ax2.set_ylabel('FPS')
  ax2.set_title(f'{title} — FPS')
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  if save_path:
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
  plt.close(fig)


def plot_stiffness_scaling(results, title='Stiffness Sensitivity', save_path=None):
  """Plot iterations-to-convergence vs material stiffness.

  Args:
      results: list of dicts with keys:
          'solver_name', 'mu', 'avg_iterations', 'avg_frame_time'
  """
  if not _has_mpl:
    return

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  solvers = {}
  for r in results:
    name = r['solver_name']
    if name not in solvers:
      solvers[name] = {'mu': [], 'iters': [], 'time': []}
    solvers[name]['mu'].append(r['mu'])
    solvers[name]['iters'].append(r['avg_iterations'])
    solvers[name]['time'].append(r['avg_frame_time'])

  for name, data in solvers.items():
    order = np.argsort(data['mu'])
    mu = np.array(data['mu'])[order]
    iters = np.array(data['iters'])[order]
    time = np.array(data['time'])[order]
    ax1.semilogx(mu, iters, 'o-', label=name, linewidth=2)
    ax2.semilogx(mu, time, 'o-', label=name, linewidth=2)

  ax1.set_xlabel('Stiffness (μ)')
  ax1.set_ylabel('Avg Iterations')
  ax1.set_title(f'{title} — Iterations')
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  ax2.set_xlabel('Stiffness (μ)')
  ax2.set_ylabel('Frame Time (s)')
  ax2.set_title(f'{title} — Frame Time')
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  if save_path:
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
  plt.close(fig)


def plot_timing_bars(results, title='Solver Comparison', save_path=None):
  """Bar chart comparing solver timings.

  Args:
      results: list of dicts with 'solver_name' and 'avg_frame_time'.
  """
  if not _has_mpl:
    return

  fig, ax = plt.subplots(1, 1, figsize=(8, 5))
  names = [r['solver_name'] for r in results]
  times = [r['avg_frame_time'] for r in results]

  bars = ax.bar(names, times, color=plt.cm.tab10(np.linspace(0, 1, len(names))))
  ax.set_ylabel('Frame Time (s)')
  ax.set_title(title)

  for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
            f'{t:.4f}s', ha='center', va='bottom', fontsize=9)

  if save_path:
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
  plt.close(fig)


def plot_adaptive_tradeoff(results, title='Adaptive Tradeoff', save_path=None):
  """Plot speedup vs accuracy for different active zone fractions.

  Args:
      results: list of dicts with 'zone_fraction', 'speedup', 'error'.
  """
  if not _has_mpl:
    return

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  fracs = [r['zone_fraction'] for r in results]
  speedups = [r['speedup'] for r in results]
  errors = [r['error'] for r in results]

  ax1.plot(fracs, speedups, 'o-', linewidth=2, color='tab:blue')
  ax1.set_xlabel('Active Zone Fraction')
  ax1.set_ylabel('Speedup (×)')
  ax1.set_title(f'{title} — Speedup')
  ax1.grid(True, alpha=0.3)

  ax2.plot(fracs, errors, 'o-', linewidth=2, color='tab:red')
  ax2.set_xlabel('Active Zone Fraction')
  ax2.set_ylabel('Position Error (L2)')
  ax2.set_title(f'{title} — Accuracy')
  ax2.grid(True, alpha=0.3)

  if save_path:
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
  plt.close(fig)
