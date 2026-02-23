"""Experiment 1.1: Convergence comparison across solvers.

Runs XPBD (default), XPBD (graph-coloring), VBD, MGPBD-CPU, MGPBD-GPU
on simple_sphere and plots residual vs iteration count.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
from data import tet_data
from cons.mgpbd import MGPBDSolver
from cons.mgpbd_gpu import MGPBDSolverGPU
from cons.vbd import VBDSolver
from cons.deform3d import Deform3D
from cons import framework
from utils.bench import SolverBenchmark
from utils import arg_parser
from experiments.plot_utils import load_benchmark, plot_convergence, plot_timing_bars


def run_xpbd(mesh, args, n_frames=50, out_dir='out/experiments'):
  """Run XPBD (Deform3D) solver with benchmarking."""
  cfg = args.xpbd
  g = ti.Vector(list(args.common.gravity))
  dt = 1.0 / args.common.fps / args.common.substeps
  n_iter = int(getattr(cfg, 'n_iterations', 10))

  bench = SolverBenchmark('XPBD', args.dir_name, {'n_tets': mesh.n_tets})
  deform = Deform3D(
      v_p=mesh.v_p, v_p_ref=mesh.v_p_ref, v_invm=mesh.v_invm,
      t_i=mesh.t_i, t_m=mesh.t_m, dt=dt,
      hydro_alpha=float(cfg.hydro_alpha),
      devia_alpha=float(cfg.devia_alpha),
      method=str(cfg.method),
      benchmark=bench)

  pbdf = framework.pbd_framework(mesh.v_p, g, dt, float(args.common.damp))
  pbdf.add_cons(deform, 0)
  pbdf.init_rest_status(0)

  for frame in range(n_frames):
    bench.begin_frame(frame)
    deform._update_count = 0
    for _ in range(args.common.substeps):
      pbdf.step(n_iter)
    bench.end_frame()

  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, 'convergence_xpbd.json')
  bench.save(path)
  print(f"XPBD: {bench.summary()}")
  return bench


def run_mgpbd_cpu(mesh, args, n_frames=50, out_dir='out/experiments'):
  """Run MGPBD CPU solver with benchmarking."""
  cfg = args.mgpbd
  g = ti.Vector(list(args.common.gravity))
  dt = 1.0 / args.common.fps / args.common.substeps

  bench = SolverBenchmark('MGPBD-CPU', args.dir_name,
                          {'mu': float(cfg.mu), 'n_tets': mesh.n_tets})
  solver = MGPBDSolver(
      v_p=mesh.v_p, v_p_ref=mesh.v_p_ref, v_invm=mesh.v_invm,
      t_i=mesh.t_i, t_m=mesh.t_m, gravity=g, dt=dt,
      mu=float(cfg.mu), damp=float(args.common.damp),
      maxiter=int(cfg.maxiter), atol=float(cfg.atol), rtol=float(cfg.rtol),
      maxiter_cg=int(cfg.maxiter_cg), tol_cg=float(cfg.tol_cg),
      n_smooth=int(cfg.n_smooth), omega_jacobi=float(cfg.omega_jacobi),
      use_line_search=bool(cfg.use_line_search), benchmark=bench)
  solver.init()

  for frame in range(n_frames):
    for _ in range(args.common.substeps):
      solver.step(frame)

  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, 'convergence_mgpbd_cpu.json')
  bench.save(path)
  print(f"MGPBD-CPU: {bench.summary()}")
  return bench


def run_mgpbd_gpu(mesh, args, n_frames=50, out_dir='out/experiments'):
  """Run MGPBD GPU solver with benchmarking."""
  cfg = args.mgpbd
  g = ti.Vector(list(args.common.gravity))
  dt = 1.0 / args.common.fps / args.common.substeps

  bench = SolverBenchmark('MGPBD-GPU', args.dir_name,
                          {'mu': float(cfg.mu), 'n_tets': mesh.n_tets})
  solver = MGPBDSolverGPU(
      v_p=mesh.v_p, v_p_ref=mesh.v_p_ref, v_invm=mesh.v_invm,
      t_i=mesh.t_i, t_m=mesh.t_m, gravity=g, dt=dt,
      mu=float(cfg.mu), damp=float(args.common.damp),
      maxiter=int(cfg.maxiter), atol=float(cfg.atol), rtol=float(cfg.rtol),
      maxiter_cg=int(cfg.maxiter_cg), tol_cg=float(cfg.tol_cg),
      n_smooth=int(cfg.n_smooth), omega_jacobi=float(cfg.omega_jacobi),
      use_line_search=bool(cfg.use_line_search),
      aggregation='tet', benchmark=bench)
  solver.init()

  for frame in range(n_frames):
    for _ in range(args.common.substeps):
      solver.step(frame)

  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, 'convergence_mgpbd_gpu.json')
  bench.save(path)
  print(f"MGPBD-GPU: {bench.summary()}")
  return bench


def run_vbd(mesh, args, n_frames=50, out_dir='out/experiments'):
  """Run VBD solver with benchmarking."""
  cfg = args.vbd
  g = ti.Vector(list(args.common.gravity))
  dt = 1.0 / args.common.fps / args.common.substeps

  bench = SolverBenchmark('VBD', args.dir_name, {'n_tets': mesh.n_tets})
  solver = VBDSolver(
      v_p=mesh.v_p, v_p_ref=mesh.v_p_ref, v_invm=mesh.v_invm,
      t_i=mesh.t_i, t_m=mesh.t_m, gravity=g, dt=dt,
      damp=float(args.common.damp),
      youngs_modulus=float(cfg.youngs_modulus),
      poissons_ratio=float(cfg.poissons_ratio),
      n_iterations=int(cfg.n_iterations),
      rho=float(cfg.rho), method=str(cfg.method),
      benchmark=bench)
  solver.init()

  for frame in range(n_frames):
    for _ in range(args.common.substeps):
      solver.step(frame)

  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, 'convergence_vbd.json')
  bench.save(path)
  print(f"VBD: {bench.summary()}")
  return bench


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', default='config/experiments/sphere_convergence.yaml')
  parser.add_argument('--n_frames', type=int, default=50)
  parser.add_argument('--out_dir', default='out/experiments/convergence')
  cli = parser.parse_args()

  ti.init(arch=ti.vulkan)
  args = arg_parser.get_args(cli.cfg)

  # Fix vertices
  mesh_path = f'assets/{args.dir_name}/{args.obj_name}'
  out_dir = cli.out_dir

  # Run each solver in sequence (each needs fresh mesh state)
  benchmarks = []
  timing_results = []

  # XPBD
  print("\n=== XPBD ===")
  mesh = tet_data.load_tets(mesh_path, args.scale, args.repose, remove_duplicate=False)
  invm = mesh.v_invm.to_numpy()
  invm[args.fixed_id] = 0
  mesh.v_invm.from_numpy(invm)
  b = run_xpbd(mesh, args, cli.n_frames, out_dir)
  timing_results.append(b.summary())

  # MGPBD-CPU
  print("\n=== MGPBD-CPU ===")
  mesh = tet_data.load_tets(mesh_path, args.scale, args.repose, remove_duplicate=False)
  invm = mesh.v_invm.to_numpy()
  invm[args.fixed_id] = 0
  mesh.v_invm.from_numpy(invm)
  b = run_mgpbd_cpu(mesh, args, cli.n_frames, out_dir)
  benchmarks.append(load_benchmark(os.path.join(out_dir, 'convergence_mgpbd_cpu.json')))
  timing_results.append(b.summary())

  # MGPBD-GPU
  print("\n=== MGPBD-GPU ===")
  mesh = tet_data.load_tets(mesh_path, args.scale, args.repose, remove_duplicate=False)
  invm = mesh.v_invm.to_numpy()
  invm[args.fixed_id] = 0
  mesh.v_invm.from_numpy(invm)
  b = run_mgpbd_gpu(mesh, args, cli.n_frames, out_dir)
  benchmarks.append(load_benchmark(os.path.join(out_dir, 'convergence_mgpbd_gpu.json')))
  timing_results.append(b.summary())

  # VBD
  print("\n=== VBD ===")
  mesh = tet_data.load_tets(mesh_path, args.scale, args.repose, remove_duplicate=False)
  invm = mesh.v_invm.to_numpy()
  invm[args.fixed_id] = 0
  mesh.v_invm.from_numpy(invm)
  b = run_vbd(mesh, args, cli.n_frames, out_dir)
  benchmarks.append(load_benchmark(os.path.join(out_dir, 'convergence_vbd.json')))
  timing_results.append(b.summary())

  # Convergence curves (MGPBD + VBD only — XPBD does not track residuals)
  plot_convergence(benchmarks,
                   title='Exp 1.1: Convergence Comparison (MGPBD / VBD)',
                   save_path=os.path.join(out_dir, 'convergence.png'))

  # Timing bar chart (all solvers including XPBD)
  plot_timing_bars(timing_results,
                   title='Exp 1.1: Frame Time Comparison',
                   save_path=os.path.join(out_dir, 'timing_comparison.png'))

  print(f"\nResults saved to {out_dir}/")


if __name__ == '__main__':
  main()
