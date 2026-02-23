"""Experiment 1.2: Resolution scaling.

Same geometry at multiple resolutions.  Measures wall-clock per frame,
iterations to tolerance, and FPS at fixed iteration budget.
Compares CPU vs GPU MGPBD pipelines.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
from data import tet_data
from cons.mgpbd import MGPBDSolver
from cons.mgpbd_gpu import MGPBDSolverGPU
from utils.bench import SolverBenchmark
from experiments.plot_utils import plot_resolution_scaling


# Meshes at increasing resolution
MESH_CONFIGS = [
    {'dir_name': 'simple_sphere', 'obj_name': 'simple_sphere.mesh', 'fixed_id': [0, 11]},
    {'dir_name': 'sphere_600',    'obj_name': 'sphere_600.mesh',    'fixed_id': [0]},
    {'dir_name': 'sphere_10k',    'obj_name': 'sphere_10k.mesh',    'fixed_id': [0]},
]

DEFAULT_PARAMS = {
    'mu': 1e6,
    'maxiter': 20,
    'atol': 1e-4,
    'rtol': 1e-2,
    'maxiter_cg': 100,
    'tol_cg': 1e-5,
    'n_smooth': 2,
    'omega_jacobi': 2.0 / 3.0,
    'damp': 0.993,
    'gravity': [0.0, 0.0, 0.0],
    'fps': 60,
    'substeps': 5,
}


def run_solver(solver_class, mesh, params, n_frames, solver_name, aggregation=None):
  """Run a solver and return summary dict."""
  g = ti.Vector(params['gravity'])
  dt = 1.0 / params['fps'] / params['substeps']

  bench = SolverBenchmark(solver_name, mesh.dir_name if hasattr(mesh, 'dir_name') else 'unknown',
                          {'n_tets': mesh.n_tets})

  kwargs = dict(
      v_p=mesh.v_p, v_p_ref=mesh.v_p_ref, v_invm=mesh.v_invm,
      t_i=mesh.t_i, t_m=mesh.t_m, gravity=g, dt=dt,
      mu=params['mu'], damp=params['damp'],
      maxiter=params['maxiter'], atol=params['atol'], rtol=params['rtol'],
      maxiter_cg=params['maxiter_cg'], tol_cg=params['tol_cg'],
      n_smooth=params['n_smooth'], omega_jacobi=params['omega_jacobi'],
      use_line_search=True, benchmark=bench)

  if aggregation is not None:
    kwargs['aggregation'] = aggregation

  solver = solver_class(**kwargs)
  solver.init()

  for frame in range(n_frames):
    for _ in range(params['substeps']):
      solver.step(frame)

  summary = bench.summary()
  summary['n_tets'] = mesh.n_tets
  summary['fps'] = 1.0 / summary['avg_frame_time'] if summary['avg_frame_time'] > 0 else 0
  return summary


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_frames', type=int, default=30)
  parser.add_argument('--out_dir', default='out/experiments/resolution')
  cli = parser.parse_args()

  ti.init(arch=ti.vulkan)
  os.makedirs(cli.out_dir, exist_ok=True)

  results = []

  for mcfg in MESH_CONFIGS:
    mesh_path = f"assets/{mcfg['dir_name']}/{mcfg['obj_name']}"
    if not os.path.exists(mesh_path):
      print(f"Skipping {mesh_path} (not found)")
      continue

    # MGPBD-CPU
    print(f"\n=== MGPBD-CPU: {mcfg['dir_name']} ===")
    mesh = tet_data.load_tets(mesh_path, 1.0, (0, 0, 0), remove_duplicate=False)
    invm = mesh.v_invm.to_numpy()
    invm[mcfg['fixed_id']] = 0
    mesh.v_invm.from_numpy(invm)
    s = run_solver(MGPBDSolver, mesh, DEFAULT_PARAMS, cli.n_frames, 'MGPBD-CPU')
    results.append(s)
    print(f"  n_tets={s['n_tets']}, frame_time={s['avg_frame_time']:.4f}s")

    # MGPBD-GPU
    print(f"\n=== MGPBD-GPU: {mcfg['dir_name']} ===")
    mesh = tet_data.load_tets(mesh_path, 1.0, (0, 0, 0), remove_duplicate=False)
    invm = mesh.v_invm.to_numpy()
    invm[mcfg['fixed_id']] = 0
    mesh.v_invm.from_numpy(invm)
    s = run_solver(MGPBDSolverGPU, mesh, DEFAULT_PARAMS, cli.n_frames,
                   'MGPBD-GPU', aggregation='tet')
    results.append(s)
    print(f"  n_tets={s['n_tets']}, frame_time={s['avg_frame_time']:.4f}s")

  # Plot
  if results:
    plot_resolution_scaling(results,
                            title='Exp 1.2: Resolution Scaling',
                            save_path=os.path.join(cli.out_dir, 'resolution.png'))
    print(f"\nResults saved to {cli.out_dir}/")


if __name__ == '__main__':
  main()
