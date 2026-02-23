"""Master runner for all MGPBD experiments.

Usage:
    python -m experiments.run_all                    # run all
    python -m experiments.run_all --exp 1.1 1.2      # run specific experiments
    python -m experiments.run_all --list              # list available experiments
"""

import os
import sys
import subprocess
import argparse

EXPERIMENTS = {
    '1.1': {
        'name': 'Convergence Comparison',
        'module': 'experiments.exp_convergence',
        'description': 'XPBD vs VBD vs MGPBD convergence curves',
    },
    '1.2': {
        'name': 'Resolution Scaling',
        'module': 'experiments.exp_resolution',
        'description': 'Wall-clock scaling with mesh resolution',
    },
    '1.3': {
        'name': 'Stiffness Sensitivity',
        'module': 'experiments.exp_stiffness',
        'description': 'Iteration count vs material stiffness',
    },
    '2.2': {
        'name': 'Tool-Tissue Interaction',
        'module': 'experiments.exp_tool_tissue',
        'description': 'Prostate mesh with tool interaction',
    },
    '2.3': {
        'name': 'Adaptive Zones',
        'module': 'experiments.exp_adaptive',
        'description': 'Speedup vs accuracy with spatial adaptivity',
    },
    '3.1': {
        'name': 'Ablation Study',
        'module': 'experiments.exp_ablation',
        'description': 'Per-component speedup contribution',
    },
}


def list_experiments():
  print("\nAvailable experiments:")
  print("-" * 60)
  for key in sorted(EXPERIMENTS.keys()):
    exp = EXPERIMENTS[key]
    print(f"  {key:5s}  {exp['name']:30s}  {exp['description']}")
  print()


def run_experiment(key, extra_args=None):
  exp = EXPERIMENTS[key]
  print(f"\n{'=' * 60}")
  print(f"  Experiment {key}: {exp['name']}")
  print(f"{'=' * 60}")

  cmd = [sys.executable, '-m', exp['module']]
  if extra_args:
    cmd.extend(extra_args)

  result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  if result.returncode != 0:
    print(f"  WARNING: Experiment {key} exited with code {result.returncode}")
  return result.returncode


def main():
  parser = argparse.ArgumentParser(description='Run MGPBD experiments')
  parser.add_argument('--exp', nargs='+', default=None,
                      help='Experiment IDs to run (e.g., 1.1 1.2)')
  parser.add_argument('--list', action='store_true', help='List experiments')
  parser.add_argument('--n_frames', type=int, default=None,
                      help='Override frame count for all experiments')
  args = parser.parse_args()

  if args.list:
    list_experiments()
    return

  exp_keys = args.exp if args.exp else sorted(EXPERIMENTS.keys())

  # Validate
  for key in exp_keys:
    if key not in EXPERIMENTS:
      print(f"Unknown experiment: {key}")
      list_experiments()
      return

  extra = []
  if args.n_frames is not None:
    extra.extend(['--n_frames', str(args.n_frames)])

  print(f"Running {len(exp_keys)} experiment(s): {', '.join(exp_keys)}")

  results = {}
  for key in exp_keys:
    rc = run_experiment(key, extra)
    results[key] = 'OK' if rc == 0 else f'FAILED (rc={rc})'

  print(f"\n{'=' * 60}")
  print("  Summary")
  print(f"{'=' * 60}")
  for key, status in results.items():
    print(f"  {key}: {status}")


if __name__ == '__main__':
  main()
