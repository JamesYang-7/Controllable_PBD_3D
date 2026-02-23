"""Experiment 2.3: Spatially adaptive solving.

Tool-tissue setup with adaptive zones at 10%, 25%, 50%, 100% active fractions.
Measures speedup vs accuracy tradeoff.
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
from cons.adaptive_zone import AdaptiveZone
from cons.mgpbd_adaptive import MGPBDAdaptiveSolver
from utils.bench import SolverBenchmark
from experiments.plot_utils import plot_adaptive_tradeoff


MESH_CONFIGS = [
    {'dir_name': 'sphere_10k', 'obj_name': 'sphere_10k.mesh', 'fixed_id': [0]},
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

# Radii chosen to give approximately these active fractions
ZONE_RADII = [0.05, 0.15, 0.3, 1e6]  # last one = everything active
ZONE_LABELS = ['~10%', '~25%', '~50%', '100%']


def apply_tool_displacement(mesh, operating_id, frame, speed=0.3, scale=1.0):
  t = frame / 60.0
  p = mesh.v_p.to_numpy()
  p_ref = mesh.v_p_ref.to_numpy()
  disp = np.array([0.0, 1.0, 0.0], dtype=np.float32) * \
      math.sin(speed * t * 2 * math.pi) * scale * 0.1
  p[operating_id] = p_ref[operating_id] + disp
  mesh.v_p.from_numpy(p)
  return p_ref[operating_id] + disp


def run_full_solve(mesh, params, n_frames, operating_id):
  """Run full (non-adaptive) solve as reference. Returns final positions."""
  g = ti.Vector(params['gravity'])
  dt = 1.0 / params['fps'] / params['substeps']

  bench = SolverBenchmark('Full', 'adaptive_ref', {})
  solver = MGPBDSolver(
      v_p=mesh.v_p, v_p_ref=mesh.v_p_ref, v_invm=mesh.v_invm,
      t_i=mesh.t_i, t_m=mesh.t_m, gravity=g, dt=dt,
      mu=params['mu'], damp=params['damp'],
      maxiter=params['maxiter'], atol=params['atol'], rtol=params['rtol'],
      maxiter_cg=params['maxiter_cg'], tol_cg=params['tol_cg'],
      n_smooth=params['n_smooth'], omega_jacobi=params['omega_jacobi'],
      use_line_search=True, benchmark=bench)
  solver.init()

  for frame in range(n_frames):
    apply_tool_displacement(mesh, operating_id, frame)
    for _ in range(params['substeps']):
      solver.step(frame)

  return mesh.v_p.to_numpy(), bench.summary()


def run_adaptive_solve(mesh, params, n_frames, operating_id, radius):
  """Run adaptive solve with given zone radius."""
  g = ti.Vector(params['gravity'])
  dt = 1.0 / params['fps'] / params['substeps']

  bench = SolverBenchmark(f'Adaptive-r{radius}', 'adaptive', {'radius': radius})
  solver = MGPBDSolver(
      v_p=mesh.v_p, v_p_ref=mesh.v_p_ref, v_invm=mesh.v_invm,
      t_i=mesh.t_i, t_m=mesh.t_m, gravity=g, dt=dt,
      mu=params['mu'], damp=params['damp'],
      maxiter=params['maxiter'], atol=params['atol'], rtol=params['rtol'],
      maxiter_cg=params['maxiter_cg'], tol_cg=params['tol_cg'],
      n_smooth=params['n_smooth'], omega_jacobi=params['omega_jacobi'],
      use_line_search=True, benchmark=bench)
  solver.init()

  zone = AdaptiveZone(solver.n_tets, solver.t_i, solver.v_p)
  adaptive = MGPBDAdaptiveSolver(solver, zone, passive_iters=2)

  avg_fraction = 0.0
  for frame in range(n_frames):
    tool_pos = apply_tool_displacement(mesh, operating_id, frame)
    for _ in range(params['substeps']):
      adaptive.step(frame, tool_pos=tool_pos, tool_radius=radius)
    avg_fraction += zone.get_active_fraction()

  avg_fraction /= n_frames
  return mesh.v_p.to_numpy(), bench.summary(), avg_fraction


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_frames', type=int, default=30)
  parser.add_argument('--out_dir', default='out/experiments/adaptive')
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

  operating_id = mcfg['fixed_id'][-1]

  # Reference: full solve
  print("\n=== Full solve (reference) ===")
  mesh = tet_data.load_tets(mesh_path, 1.0, (0, 0, 0), remove_duplicate=False)
  invm = mesh.v_invm.to_numpy()
  invm[mcfg['fixed_id']] = 0
  mesh.v_invm.from_numpy(invm)
  ref_pos, ref_summary = run_full_solve(mesh, BASE_PARAMS, cli.n_frames, operating_id)
  ref_time = ref_summary['avg_frame_time']
  print(f"  time={ref_time:.4f}s")

  results = []
  for radius, label in zip(ZONE_RADII, ZONE_LABELS):
    print(f"\n=== Adaptive zone: {label} (radius={radius}) ===")
    mesh = tet_data.load_tets(mesh_path, 1.0, (0, 0, 0), remove_duplicate=False)
    invm = mesh.v_invm.to_numpy()
    invm[mcfg['fixed_id']] = 0
    mesh.v_invm.from_numpy(invm)

    pos, summary, frac = run_adaptive_solve(
        mesh, BASE_PARAMS, cli.n_frames, operating_id, radius)

    error = float(np.linalg.norm(pos - ref_pos))
    speedup = ref_time / summary['avg_frame_time'] if summary['avg_frame_time'] > 0 else 1.0

    results.append({
        'zone_fraction': frac,
        'speedup': speedup,
        'error': error,
        'label': label,
        'radius': radius,
    })
    print(f"  fraction={frac:.2%}, speedup={speedup:.2f}x, error={error:.6f}")

  # Plot
  if results:
    plot_adaptive_tradeoff(results,
                           title='Exp 2.3: Adaptive Tradeoff',
                           save_path=os.path.join(cli.out_dir, 'adaptive.png'))
    print(f"\nResults saved to {cli.out_dir}/")


if __name__ == '__main__':
  main()
