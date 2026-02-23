"""GPU-native V-cycle + MGPCG solver using GPUSparseMatrix.

All smoothing, restriction, prolongation, and CG iterations run on GPU.
The only CPU involvement is the outer loop control flow and scalar convergence
checks.
"""

import taichi as ti
import numpy as np
import scipy.sparse as sp

from cons.sparse_gpu import (GPUSparseMatrix, gpu_dot, gpu_axpy, gpu_xpay,
                             gpu_copy, gpu_fill_zero, gpu_norm_sq,
                             gpu_negate_and_add)


@ti.data_oriented
class GPUMultigridSolver:
  """Multigrid-preconditioned CG solver, fully GPU-resident."""

  def __init__(self, n_smooth: int = 2, omega_jacobi: float = 2.0 / 3.0,
               maxiter_cg: int = 100, tol_cg: float = 1e-5):
    self.n_smooth = n_smooth
    self.omega_jacobi = omega_jacobi
    self.maxiter_cg = maxiter_cg
    self.tol_cg = tol_cg
    self.n_levels = 0
    self._initialized = False

  def setup(self, A_level0_scipy: sp.csr_matrix, Ps_scipy: list):
    """Set up the multigrid hierarchy from fine-level matrix and prolongation ops.

    1. Pre-compute coarse-level sparsity patterns on CPU.
    2. Build a "contribution map" for each coarse level so that coarse matrix
       values can be recomputed on GPU without SpMM.
    3. Transfer all CSR structures to GPU as GPUSparseMatrix.
    4. Allocate work vectors at each level.
    """
    n_levels = len(Ps_scipy) + 1
    self.n_levels = n_levels
    self._level_sizes = []

    A0 = A_level0_scipy.tocsr().astype(np.float64)
    Ps = [P.tocsr().astype(np.float64) for P in Ps_scipy]

    # Build coarse matrices via Galerkin projection
    As_scipy = [A0]
    for P in Ps:
      Ac = (P.T @ As_scipy[-1] @ P).tocsr()
      As_scipy.append(Ac)

    self._level_sizes = [A.shape[0] for A in As_scipy]

    # Build contribution maps for GPU coarse matrix assembly
    self._contrib_maps = []
    self._contrib_offsets = []

    for lvl in range(len(Ps)):
      P = Ps[lvl]
      A_fine = As_scipy[lvl]
      Ac = As_scipy[lvl + 1]
      n_coarse = Ac.shape[0]

      PT = P.T.tocsr()

      offsets = [0]
      all_fine_idx = []
      all_p_ki = []
      all_p_lj = []

      for i in range(n_coarse):
        for ptr_c in range(Ac.indptr[i], Ac.indptr[i + 1]):
          j = Ac.indices[ptr_c]

          fine_rows_i = PT.indices[PT.indptr[i]:PT.indptr[i + 1]]
          p_vals_i = PT.data[PT.indptr[i]:PT.indptr[i + 1]]

          fine_rows_j = PT.indices[PT.indptr[j]:PT.indptr[j + 1]]
          p_vals_j = PT.data[PT.indptr[j]:PT.indptr[j + 1]]

          for ki_idx, k in enumerate(fine_rows_i):
            pk_i = p_vals_i[ki_idx]
            for a_ptr in range(A_fine.indptr[k], A_fine.indptr[k + 1]):
              l = A_fine.indices[a_ptr]
              lj_pos = np.searchsorted(fine_rows_j, l)
              if lj_pos < len(fine_rows_j) and fine_rows_j[lj_pos] == l:
                pl_j = p_vals_j[lj_pos]
                all_fine_idx.append(a_ptr)
                all_p_ki.append(float(pk_i))
                all_p_lj.append(float(pl_j))

          offsets.append(len(all_fine_idx))

      self._contrib_maps.append({
          'fine_idx': np.array(all_fine_idx, dtype=np.int32),
          'p_ki': np.array(all_p_ki, dtype=np.float32),
          'p_lj': np.array(all_p_lj, dtype=np.float32),
      })
      self._contrib_offsets.append(np.array(offsets, dtype=np.int32))

    # Transfer CSR to GPU
    self._gpu_As = []
    for A in As_scipy:
      A32 = A.astype(np.float32).tocsr()
      gA = GPUSparseMatrix(A32.shape[0], A32.nnz)
      gA.from_scipy(A32)
      self._gpu_As.append(gA)

    self._gpu_Ps = []
    for P in Ps:
      P32 = P.astype(np.float32).tocsr()
      gP = GPUSparseMatrix(P32.shape[0], P32.nnz)
      gP.from_scipy(P32)
      self._gpu_Ps.append(gP)

    # P^T stored separately for restriction
    self._gpu_PTs = []
    for P in Ps:
      PT32 = P.T.tocsr().astype(np.float32)
      gPT = GPUSparseMatrix(PT32.shape[0], PT32.nnz)
      gPT.from_scipy(PT32)
      self._gpu_PTs.append(gPT)

    # Transfer contribution maps to GPU
    self._gpu_contrib_fine_idx = []
    self._gpu_contrib_pki = []
    self._gpu_contrib_plj = []
    self._gpu_contrib_offsets = []
    for lvl in range(len(Ps)):
      cm = self._contrib_maps[lvl]
      n_contrib = len(cm['fine_idx'])

      fi = ti.field(dtype=ti.i32, shape=max(n_contrib, 1))
      pki = ti.field(dtype=ti.f32, shape=max(n_contrib, 1))
      plj = ti.field(dtype=ti.f32, shape=max(n_contrib, 1))
      if n_contrib > 0:
        fi.from_numpy(cm['fine_idx'])
        pki.from_numpy(cm['p_ki'])
        plj.from_numpy(cm['p_lj'])

      co = self._contrib_offsets[lvl]
      off = ti.field(dtype=ti.i32, shape=co.shape[0])
      off.from_numpy(co)

      self._gpu_contrib_fine_idx.append(fi)
      self._gpu_contrib_pki.append(pki)
      self._gpu_contrib_plj.append(plj)
      self._gpu_contrib_offsets.append(off)

    # Allocate work vectors at each level
    self._b = []
    self._x = []
    self._x_new = []
    self._r = []
    self._tmp = []

    for lvl in range(n_levels):
      n = self._level_sizes[lvl]
      self._b.append(ti.field(dtype=ti.f32, shape=n))
      self._x.append(ti.field(dtype=ti.f32, shape=n))
      self._x_new.append(ti.field(dtype=ti.f32, shape=n))
      self._r.append(ti.field(dtype=ti.f32, shape=n))
      self._tmp.append(ti.field(dtype=ti.f32, shape=n))

    # Level-0 CG vectors
    n0 = self._level_sizes[0]
    self._z0 = ti.field(dtype=ti.f32, shape=n0)
    self._p0 = ti.field(dtype=ti.f32, shape=n0)
    self._Ap0 = ti.field(dtype=ti.f32, shape=n0)

    self._initialized = True
    sizes_str = ' -> '.join(str(s) for s in self._level_sizes)
    print(f"GPUMultigridSolver: {n_levels} levels ({sizes_str})")

  def update_fine_matrix_data(self, data_field):
    """Copy new A values into level-0 GPUSparseMatrix.data."""
    if isinstance(data_field, np.ndarray):
      self._gpu_As[0].data.from_numpy(data_field.astype(np.float32))
    else:
      gpu_copy(data_field, self._gpu_As[0].data, self._gpu_As[0].nnz)
    self._gpu_As[0].cache_diag_inv()

  @ti.kernel
  def _fill_coarse_values_kernel(self, coarse_data: ti.template(),
                                 fine_data: ti.template(),
                                 fine_idx: ti.template(),
                                 pki: ti.template(),
                                 plj: ti.template(),
                                 offsets: ti.template(),
                                 n_coarse_nnz: ti.i32):
    """Fill coarse matrix values using pre-computed contribution map."""
    for c_nnz in range(n_coarse_nnz):
      start = offsets[c_nnz]
      end = offsets[c_nnz + 1]
      val = 0.0
      for k in range(start, end):
        val += pki[k] * fine_data[fine_idx[k]] * plj[k]
      coarse_data[c_nnz] = val

  def update_all_coarse_matrices(self):
    """Recompute all coarse-level matrix values on GPU."""
    for lvl in range(self.n_levels - 1):
      n_coarse_nnz = self._gpu_As[lvl + 1].nnz
      self._fill_coarse_values_kernel(
          self._gpu_As[lvl + 1].data,
          self._gpu_As[lvl].data,
          self._gpu_contrib_fine_idx[lvl],
          self._gpu_contrib_pki[lvl],
          self._gpu_contrib_plj[lvl],
          self._gpu_contrib_offsets[lvl],
          n_coarse_nnz)
      self._gpu_As[lvl + 1].cache_diag_inv()

  def vcycle(self, level: int = 0):
    """One V-cycle. Uses self._b[level] as RHS and self._x[level] as solution."""
    if level == self.n_levels - 1:
      # Coarsest level: many Jacobi iterations as approximate direct solve
      n = self._level_sizes[level]
      A = self._gpu_As[level]
      for _ in range(min(50, n)):
        A.jacobi_smooth(self._b[level], self._x[level],
                        self._x_new[level], self.omega_jacobi)
        gpu_copy(self._x_new[level], self._x[level], n)
      return

    n = self._level_sizes[level]
    n_coarse = self._level_sizes[level + 1]
    A = self._gpu_As[level]
    P = self._gpu_Ps[level]
    PT = self._gpu_PTs[level]

    # Pre-smooth
    for _ in range(self.n_smooth):
      A.jacobi_smooth(self._b[level], self._x[level],
                      self._x_new[level], self.omega_jacobi)
      gpu_copy(self._x_new[level], self._x[level], n)

    # Compute residual: r = b - A*x
    A.spmv(self._x[level], self._r[level])
    gpu_negate_and_add(self._b[level], self._r[level], n)

    # Restrict: b_coarse = P^T * r
    PT.spmv(self._r[level], self._b[level + 1])

    # Zero initial guess at coarse level
    gpu_fill_zero(self._x[level + 1], n_coarse)

    # Recurse
    self.vcycle(level + 1)

    # Prolongate and correct: x += P * x_coarse
    P.spmv(self._x[level + 1], self._tmp[level])
    gpu_axpy(1.0, self._tmp[level], self._x[level], n)

    # Post-smooth
    for _ in range(self.n_smooth):
      A.jacobi_smooth(self._b[level], self._x[level],
                      self._x_new[level], self.omega_jacobi)
      gpu_copy(self._x_new[level], self._x[level], n)

  def solve(self, b_field, x_field):
    """CG with V-cycle preconditioner, all on GPU.

    Args:
        b_field: ti.field, right-hand side vector.
        x_field: ti.field, solution vector (zeroed on entry).
    """
    n = self._level_sizes[0]

    # Copy b, zero x
    gpu_copy(b_field, self._b[0], n)
    gpu_fill_zero(x_field, n)

    # r = b (since x=0)
    gpu_copy(self._b[0], self._r[0], n)

    # z = V-cycle(r)
    gpu_copy(self._r[0], self._b[0], n)
    gpu_fill_zero(self._x[0], n)
    self.vcycle(0)
    gpu_copy(self._x[0], self._z0, n)

    # p = z
    gpu_copy(self._z0, self._p0, n)

    rz = gpu_dot(self._r[0], self._z0, n)

    cg_iters = 0
    for k in range(self.maxiter_cg):
      # Ap = A * p
      self._gpu_As[0].spmv(self._p0, self._Ap0)

      pAp = gpu_dot(self._p0, self._Ap0, n)
      if abs(pAp) < 1e-30:
        break

      alpha = rz / pAp

      # x += alpha * p
      gpu_axpy(alpha, self._p0, x_field, n)

      # r -= alpha * Ap
      gpu_axpy(-alpha, self._Ap0, self._r[0], n)

      r_norm_sq = gpu_norm_sq(self._r[0], n)
      if r_norm_sq < self.tol_cg * self.tol_cg:
        cg_iters = k + 1
        break

      # z = V-cycle(r)
      gpu_copy(self._r[0], self._b[0], n)
      gpu_fill_zero(self._x[0], n)
      self.vcycle(0)
      gpu_copy(self._x[0], self._z0, n)

      rz_new = gpu_dot(self._r[0], self._z0, n)
      if abs(rz) < 1e-30:
        cg_iters = k + 1
        break

      beta = rz_new / rz

      # p = z + beta * p
      gpu_xpay(beta, self._z0, self._p0, n)

      rz = rz_new
      cg_iters = k + 1

    return cg_iters
