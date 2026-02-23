"""Experiment 2.2: Tool-tissue interaction.

Prostate mesh with displacement-controlled tool interaction.
Compares visual quality and convergence across solvers.
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
from data import tet_data
from cons.mgpbd import MGPBDSolver
from cons.mgpbd_gpu import MGPBDSolverGPU
from cons.vbd import VBDSolver
from utils.bench import SolverBenchmark
from experiments.plot_utils import plot_convergence, plot_timing_bars, load_benchmark


MESH_CONFIGS = [
    {'dir_name': 'prostate_tet', 'obj_name': 'prostate.mesh', 'fixed_id': [0]},
    {'dir_name': 'simple_sphere', 'obj_name': 'simple_sphere.mesh', 'fixed_id': [0, 11]},
]

BASE_PARAMS = {
    'mu': 1e5,
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


def apply_tool_displacement(mesh, operating_id, frame, speed=0.3, scale=1.0):
  """Apply sinusoidal displacement to operating point (simulates tool)."""
  t = frame / 60.0
  p = mesh.v_p.to_numpy()
  p_ref = mesh.v_p_ref.to_numpy()
  disp = np.array([0.0, 1.0, 0.0], dtype=np.float32) * \
      math.sin(speed * t * 2 * math.pi) * scale * 0.1
  p[operating_id] = p_ref[operating_id] + disp
  mesh.v_p.from_numpy(p)


def run_experiment(solver_class, mesh, params, n_frames, solver_name,
                   operating_id, out_dir, aggregation=None):
  """Run a solver on the tool-tissue scenario."""
  g = ti.Vector(params['gravity'])
  dt = 1.0 / params['fps'] / params['substeps']

  bench = SolverBenchmark(solver_name, 'tool_tissue', {'n_tets': mesh.n_tets})

  if solver_class == VBDSolver:
    solver = VBDSolver(
        v_p=mesh.v_p, v_p_ref=mesh.v_p_ref, v_invm=mesh.v_invm,
        t_i=mesh.t_i, t_m=mesh.t_m, gravity=g, dt=dt,
        damp=params['damp'], youngs_modulus=params['mu'] * 3,
        poissons_ratio=0.3, n_iterations=params['maxiter'],
        benchmark=bench)
  else:
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
    apply_tool_displacement(mesh, operating_id, frame)
    for _ in range(params['substeps']):
      solver.step(frame)

  fname = f'tool_tissue_{solver_name.lower().replace("-", "_")}.json'
  bench.save(os.path.join(out_dir, fname))
  return bench.summary()


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_frames', type=int, default=60)
  parser.add_argument('--out_dir', default='out/experiments/tool_tissue')
  cli = parser.parse_args()

  ti.init(arch=ti.vulkan)
  os.makedirs(cli.out_dir, exist_ok=True)

  # Find available mesh
  mesh_path = None
  mcfg = None
  for mc in MESH_CONFIGS:
    p = f"assets/{mc['dir_name']}/{mc['obj_name']}"
    if os.path.exists(p):
      mesh_path = p
      mcfg = mc
      break
  if mesh_path is None:
    print("No suitable mesh found. Exiting.")
    return

  operating_id = mcfg['fixed_id'][-1]  # Use last fixed point as operating
  results = []
  benchmarks = []

  solvers = [
      ('MGPBD-CPU', MGPBDSolver, None),
      ('MGPBD-GPU', MGPBDSolverGPU, 'tet'),
      ('VBD', VBDSolver, None),
  ]

  for name, cls, agg in solvers:
    print(f"\n=== {name} ===")
    mesh = tet_data.load_tets(mesh_path, 1.0, (0, 0, 0), remove_duplicate=False)
    invm = mesh.v_invm.to_numpy()
    invm[mcfg['fixed_id']] = 0
    mesh.v_invm.from_numpy(invm)

    s = run_experiment(cls, mesh, BASE_PARAMS, cli.n_frames, name,
                       operating_id, cli.out_dir, aggregation=agg)
    results.append(s)
    print(f"  time={s['avg_frame_time']:.4f}s, iters={s['avg_iterations']:.1f}")

  # Plot timing comparison
  plot_timing_bars(results,
                   title='Exp 2.2: Tool-Tissue Solver Comparison',
                   save_path=os.path.join(cli.out_dir, 'tool_tissue_timing.png'))

  # Plot convergence if benchmark files exist
  bench_files = []
  for name, _, _ in solvers:
    fname = f'tool_tissue_{name.lower().replace("-", "_")}.json'
    fpath = os.path.join(cli.out_dir, fname)
    if os.path.exists(fpath):
      bench_files.append(load_benchmark(fpath))
  if bench_files:
    plot_convergence(bench_files,
                     title='Exp 2.2: Tool-Tissue Convergence',
                     save_path=os.path.join(cli.out_dir, 'tool_tissue_convergence.png'))

  print(f"\nResults saved to {cli.out_dir}/")


if __name__ == '__main__':
  main()
