"""GPU-resident CSR sparse matrix operations using Taichi fields.

All operations (SpMV, Jacobi smoothing, dot products, etc.) run entirely on
the GPU.  CSR index arrays are loaded once from a sparsity pattern; only the
data array is updated per iteration.
"""

import taichi as ti
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------- #
#  Module-level GPU vector operations (free functions, not class methods)       #
# ---------------------------------------------------------------------------- #

@ti.kernel
def gpu_dot(a: ti.template(), b: ti.template(), n: ti.i32) -> ti.f32:
  s = 0.0
  for i in range(n):
    s += a[i] * b[i]
  return s


@ti.kernel
def gpu_axpy(alpha: ti.f32, x: ti.template(), y: ti.template(), n: ti.i32):
  """y += alpha * x"""
  for i in range(n):
    y[i] += alpha * x[i]


@ti.kernel
def gpu_xpay(alpha: ti.f32, x: ti.template(), y: ti.template(), n: ti.i32):
  """y = x + alpha * y"""
  for i in range(n):
    y[i] = x[i] + alpha * y[i]


@ti.kernel
def gpu_copy(src: ti.template(), dst: ti.template(), n: ti.i32):
  for i in range(n):
    dst[i] = src[i]


@ti.kernel
def gpu_fill_zero(x: ti.template(), n: ti.i32):
  for i in range(n):
    x[i] = 0.0


@ti.kernel
def gpu_scale(x: ti.template(), alpha: ti.f32, n: ti.i32):
  for i in range(n):
    x[i] *= alpha


@ti.kernel
def gpu_norm_sq(x: ti.template(), n: ti.i32) -> ti.f32:
  s = 0.0
  for i in range(n):
    s += x[i] * x[i]
  return s


@ti.kernel
def gpu_negate_and_add(b: ti.template(), r: ti.template(), n: ti.i32):
  """r = b - r (in place)."""
  for i in range(n):
    r[i] = b[i] - r[i]


# ---------------------------------------------------------------------------- #
#  GPUSparseMatrix class                                                        #
# ---------------------------------------------------------------------------- #


@ti.data_oriented
class GPUSparseMatrix:
  """CSR sparse matrix stored entirely in Taichi GPU fields."""

  def __init__(self, n_rows: int, nnz: int):
    self.n_rows = n_rows
    self.nnz = nnz

    # CSR arrays
    self.data = ti.field(dtype=ti.f32, shape=nnz)
    self.indices = ti.field(dtype=ti.i32, shape=nnz)
    self.indptr = ti.field(dtype=ti.i32, shape=n_rows + 1)

    # Cached inverse diagonal for Jacobi
    self.diag_inv = ti.field(dtype=ti.f32, shape=n_rows)

  def from_scipy(self, A: sp.csr_matrix):
    """Load sparsity pattern and values from a SciPy CSR matrix."""
    A = A.tocsr()
    assert A.shape[0] == self.n_rows
    assert A.nnz == self.nnz
    self.data.from_numpy(A.data.astype(np.float32))
    self.indices.from_numpy(A.indices.astype(np.int32))
    self.indptr.from_numpy(A.indptr.astype(np.int32))
    self.cache_diag_inv()

  def update_data(self, data_np: np.ndarray):
    """Update only the data array (sparsity pattern unchanged)."""
    self.data.from_numpy(data_np.astype(np.float32))

  @ti.kernel
  def cache_diag_inv(self):
    """Cache 1/diag(A) for Jacobi smoothing."""
    for i in range(self.n_rows):
      start = self.indptr[i]
      end = self.indptr[i + 1]
      diag_val = 0.0
      for k in range(start, end):
        if self.indices[k] == i:
          diag_val = self.data[k]
          break
      if ti.abs(diag_val) > 1e-30:
        self.diag_inv[i] = 1.0 / diag_val
      else:
        self.diag_inv[i] = 0.0

  @ti.kernel
  def spmv(self, x: ti.template(), y: ti.template()):
    """y = A @ x, fully on GPU."""
    for i in range(self.n_rows):
      start = self.indptr[i]
      end = self.indptr[i + 1]
      s = 0.0
      for k in range(start, end):
        s += self.data[k] * x[self.indices[k]]
      y[i] = s

  @ti.kernel
  def jacobi_smooth(self, b: ti.template(), x: ti.template(),
                    x_new: ti.template(), omega: ti.f32):
    """One weighted Jacobi step: x_new = x + omega * D^{-1} * (b - A*x)."""
    for i in range(self.n_rows):
      start = self.indptr[i]
      end = self.indptr[i + 1]
      Ax_i = 0.0
      for k in range(start, end):
        Ax_i += self.data[k] * x[self.indices[k]]
      x_new[i] = x[i] + omega * self.diag_inv[i] * (b[i] - Ax_i)
