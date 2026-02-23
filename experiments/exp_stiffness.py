"""Experiment 1.3: Stiffness sensitivity.

Fix resolution (sphere_10k), sweep mu: 1e3 to 1e7.
Shows that dual-space multigrid is insensitive to stiffness.
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
from experiments.plot_utils import plot_stiffness_scaling


MESH_CONFIG = {
    'dir_name': 'sphere_10k',
    'obj_name': 'sphere_10k.mesh',
    'fixed_id': [0],
}

MU_VALUES = [1e3, 1e4, 1e5, 1e6, 1e7]

BASE_PARAMS = {
    'maxiter': 50,
    'atol': 1e-6,
    'rtol': 1e-4,
    'maxiter_cg': 100,
    'tol_cg': 1e-5,
    'n_smooth': 2,
    'omega_jacobi': 2.0 / 3.0,
    'damp': 0.993,
    'gravity': [0.0, -9.8, 0.0],
    'fps': 60,
    'substeps': 5,
}


def run_solver_mu(solver_class, mesh_path, fixed_id, mu, params, n_frames,
                  solver_name, aggregation=None):
  mesh = tet_data.load_tets(mesh_path, 1.0, (0, 0, 0), remove_duplicate=False)
  invm = mesh.v_invm.to_numpy()
  invm[fixed_id] = 0
  mesh.v_invm.from_numpy(invm)

  g = ti.Vector(params['gravity'])
  dt = 1.0 / params['fps'] / params['substeps']

  bench = SolverBenchmark(solver_name, 'sphere_10k', {'mu': mu})

  kwargs = dict(
      v_p=mesh.v_p, v_p_ref=mesh.v_p_ref, v_invm=mesh.v_invm,
      t_i=mesh.t_i, t_m=mesh.t_m, gravity=g, dt=dt,
      mu=mu, damp=params['damp'],
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
  summary['mu'] = mu
  return summary


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_frames', type=int, default=30)
  parser.add_argument('--out_dir', default='out/experiments/stiffness')
  cli = parser.parse_args()

  ti.init(arch=ti.vulkan)
  os.makedirs(cli.out_dir, exist_ok=True)

  mesh_path = f"assets/{MESH_CONFIG['dir_name']}/{MESH_CONFIG['obj_name']}"
  if not os.path.exists(mesh_path):
    # Fall back to simple_sphere
    print(f"{mesh_path} not found, falling back to simple_sphere")
    mesh_path = 'assets/simple_sphere/simple_sphere.mesh'
    MESH_CONFIG['fixed_id'] = [0, 11]

  results = []

  for mu in MU_VALUES:
    # MGPBD-CPU
    print(f"\n=== MGPBD-CPU mu={mu:.0e} ===")
    s = run_solver_mu(MGPBDSolver, mesh_path, MESH_CONFIG['fixed_id'],
                      mu, BASE_PARAMS, cli.n_frames, 'MGPBD-CPU')
    results.append(s)
    print(f"  iters={s['avg_iterations']:.1f}, time={s['avg_frame_time']:.4f}s")

    # MGPBD-GPU
    print(f"\n=== MGPBD-GPU mu={mu:.0e} ===")
    s = run_solver_mu(MGPBDSolverGPU, mesh_path, MESH_CONFIG['fixed_id'],
                      mu, BASE_PARAMS, cli.n_frames, 'MGPBD-GPU',
                      aggregation='tet')
    results.append(s)
    print(f"  iters={s['avg_iterations']:.1f}, time={s['avg_frame_time']:.4f}s")

  # Plot
  if results:
    plot_stiffness_scaling(results,
                           title='Exp 1.3: Stiffness Sensitivity',
                           save_path=os.path.join(cli.out_dir, 'stiffness.png'))
    print(f"\nResults saved to {cli.out_dir}/")


if __name__ == '__main__':
  main()
