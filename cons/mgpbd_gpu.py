"""Full GPU MGPBD solver — Multigrid-accelerated Global XPBD on GPU.

The inner solve loop has zero CPU-GPU data transfers.  The only CPU involvement
is the outer loop control flow (iteration counter, convergence check scalar).

Reuses Taichi kernels from cons/mgpbd.py where applicable.
"""

import taichi as ti
import numpy as np
import scipy.sparse as sp

from cons.mgpbd import _arap_gradient, _intersect_tets
from cons.tet_aggregation import build_tet_hierarchy
from cons.mgpcg_gpu import GPUMultigridSolver
from cons.sparse_gpu import GPUSparseMatrix, gpu_fill_zero


@ti.data_oriented
class MGPBDSolverGPU:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               v_invm: ti.Field,
               t_i: ti.Field,
               t_m: ti.Field,
               gravity: ti.Vector,
               dt: float,
               mu: float = 1e6,
               damp: float = 1.0,
               maxiter: int = 20,
               atol: float = 1e-4,
               rtol: float = 1e-2,
               maxiter_cg: int = 100,
               tol_cg: float = 1e-5,
               n_smooth: int = 2,
               omega_jacobi: float = 2.0 / 3.0,
               setup_interval: int = 10000,
               use_line_search: bool = True,
               aggregation: str = 'tet',
               benchmark=None) -> None:
    self.n_verts = v_p.shape[0]
    self.n_tets = t_i.shape[0] // 4
    self.v_p = v_p
    self.v_p_ref = v_p_ref
    self.v_invm = v_invm
    self.t_i = t_i
    self.t_m = t_m
    self.gravity = gravity
    self.dt = dt
    self.mu = mu
    self.maxiter = maxiter
    self.atol = atol
    self.rtol = rtol
    self.maxiter_cg = maxiter_cg
    self.tol_cg = tol_cg
    self.n_smooth = n_smooth
    self.omega_jacobi = omega_jacobi
    self.setup_interval = setup_interval
    self.use_line_search = use_line_search
    self.aggregation = aggregation
    self.benchmark = benchmark

    self.damp = ti.field(dtype=ti.f32, shape=())
    self.damp[None] = damp

    # Velocity and position caches
    self.v_v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    self.v_p_old = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    self.v_predict = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    self.dpos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)

    # Per-tet precomputed data
    self.Dm_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_tets)
    self.rest_vol = ti.field(dtype=ti.f32, shape=self.n_tets)
    self.alpha_tilde = ti.field(dtype=ti.f32, shape=self.n_tets)

    # Per-tet constraint data
    self.lambdaf = ti.field(dtype=ti.f32, shape=self.n_tets)
    self.constraints = ti.field(dtype=ti.f32, shape=self.n_tets)
    self.gradC = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_tets, 4))

    # GPU solve vectors (allocated in init)
    self._b_gpu = ti.field(dtype=ti.f32, shape=self.n_tets)
    self._dlam_gpu = ti.field(dtype=ti.f32, shape=self.n_tets)

    # GPU multigrid solver
    self._gpu_solver = None
    self._amg_Ps = None

  def init(self):
    """Precompute rest shape, build hierarchy, set up GPU solver."""
    self._precompute_rest_shape()
    self._build_sparsity_pattern()
    self._alpha_tilde_np = self.alpha_tilde.to_numpy()

    # Build initial A to get sparsity pattern
    self._compute_C_and_gradC()
    A = self._fill_A()

    # Build hierarchy
    self._build_hierarchy(A)

    # Setup GPU multigrid solver
    self._gpu_solver = GPUMultigridSolver(
        n_smooth=self.n_smooth,
        omega_jacobi=self.omega_jacobi,
        maxiter_cg=self.maxiter_cg,
        tol_cg=self.tol_cg)
    self._gpu_solver.setup(A, self._amg_Ps)

    print(f"MGPBD-GPU: {self.n_verts} verts, {self.n_tets} tets, "
          f"{self._nnz} nnz, aggregation={self.aggregation}")

  def _build_hierarchy(self, A):
    """Build multigrid hierarchy using specified aggregation strategy."""
    if self.aggregation == 'tet':
      t_i_np = self.t_i.to_numpy()
      v_p_np = self.v_p_ref.to_numpy()
      Ps, sizes = build_tet_hierarchy(t_i_np, v_p_np, self.n_tets)
      self._amg_Ps = Ps
    else:
      # Fall back to PyAMG
      import pyamg
      A64 = A.astype(np.float64)
      ml = pyamg.smoothed_aggregation_solver(
          A64, max_coarse=400, smooth=None,
          improve_candidates=None, symmetry='symmetric')
      self._amg_Ps = [
          level.P.tocsr().astype(np.float64) for level in ml.levels[:-1]
      ]
      sizes = [A64.shape[0]] + [P.shape[1] for P in self._amg_Ps]
      print(f"MGPBD-GPU (PyAMG): {len(sizes)} levels, sizes={sizes}")

  @ti.kernel
  def _precompute_rest_shape(self):
    for k in range(self.n_tets):
      i0 = self.t_i[k * 4]
      i1 = self.t_i[k * 4 + 1]
      i2 = self.t_i[k * 4 + 2]
      i3 = self.t_i[k * 4 + 3]
      r0 = self.v_p_ref[i0]
      r1 = self.v_p_ref[i1]
      r2 = self.v_p_ref[i2]
      r3 = self.v_p_ref[i3]
      Dm = ti.Matrix.cols([r1 - r0, r2 - r0, r3 - r0])
      self.Dm_inv[k] = Dm.inverse()
      vol = ti.abs(Dm.determinant()) / 6.0
      self.rest_vol[k] = vol
      self.alpha_tilde[k] = 1.0 / (self.mu * self.dt * self.dt * vol)

  def _build_sparsity_pattern(self):
    """Build CSR sparsity pattern for A (NT x NT) from tet adjacency."""
    t_i_np = self.t_i.to_numpy()
    nt = self.n_tets

    vert_to_tets = {}
    for t in range(nt):
      for l in range(4):
        v = t_i_np[t * 4 + l]
        if v not in vert_to_tets:
          vert_to_tets[v] = set()
        vert_to_tets[v].add(t)

    adj = {}
    for t in range(nt):
      neighbors = set()
      for l in range(4):
        v = t_i_np[t * 4 + l]
        neighbors |= vert_to_tets[v]
      neighbors.discard(t)
      adj[t] = sorted(neighbors)

    num_adj = np.array([len(adj[t]) for t in range(nt)], dtype=np.int32)
    nnz = int(np.sum(num_adj) + nt)
    indptr = np.zeros(nt + 1, dtype=np.int32)
    indices = np.zeros(nnz, dtype=np.int32)

    for i in range(nt):
      indptr[i + 1] = indptr[i] + num_adj[i] + 1
      indices[indptr[i]:indptr[i] + num_adj[i]] = adj[i]
      indices[indptr[i] + num_adj[i]] = i

    ii = np.zeros(nnz, dtype=np.int32)
    for i in range(nt):
      ii[indptr[i]:indptr[i + 1]] = i

    self._A_data = np.zeros(nnz, dtype=np.float32)
    self._A_indices = indices
    self._A_indptr = indptr
    self._A_ii = ii
    self._A_jj = indices.copy()
    self._nnz = nnz

  # ------------------------------------------------------------------ #
  #                         Timestep methods                            #
  # ------------------------------------------------------------------ #

  def step(self, frame: int = 0):
    """Full timestep: predict -> solve -> velocity update."""
    self.make_prediction()
    self.solve(frame)
    self.update_vel()

  @ti.kernel
  def make_prediction(self):
    for k in range(self.n_verts):
      self.v_v[k] += self.dt * self.gravity
      self.v_v[k] *= self.damp[None]
      self.v_p_old[k] = self.v_p[k]
      self.v_p[k] += self.dt * self.v_v[k]
      self.v_predict[k] = self.v_p[k]

  @ti.kernel
  def update_vel(self):
    for k in range(self.n_verts):
      self.v_v[k] = self.damp[None] * (self.v_p[k] -
                                        self.v_p_old[k]) / self.dt

  # ------------------------------------------------------------------ #
  #                      Constraint evaluation                          #
  # ------------------------------------------------------------------ #

  @ti.kernel
  def _compute_C_and_gradC(self):
    for t in range(self.n_tets):
      p0 = self.t_i[t * 4]
      p1 = self.t_i[t * 4 + 1]
      p2 = self.t_i[t * 4 + 2]
      p3 = self.t_i[t * 4 + 3]
      x0, x1, x2, x3 = self.v_p[p0], self.v_p[p1], self.v_p[p2], self.v_p[p3]

      D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
      F = D_s @ self.Dm_inv[t]
      U, S, V = ti.svd(F)
      self.constraints[t] = ti.sqrt((S[0, 0] - 1.0)**2 +
                                    (S[1, 1] - 1.0)**2 +
                                    (S[2, 2] - 1.0)**2)
      if self.constraints[t] > 1e-6:
        self.gradC[t, 0], self.gradC[t, 1], self.gradC[t, 2], \
            self.gradC[t, 3] = _arap_gradient(U, S, V, self.Dm_inv[t])
      else:
        self.gradC[t, 0] = ti.Vector([0.0, 0.0, 0.0])
        self.gradC[t, 1] = ti.Vector([0.0, 0.0, 0.0])
        self.gradC[t, 2] = ti.Vector([0.0, 0.0, 0.0])
        self.gradC[t, 3] = ti.Vector([0.0, 0.0, 0.0])

  # ------------------------------------------------------------------ #
  #                      RHS and residual on GPU                        #
  # ------------------------------------------------------------------ #

  @ti.kernel
  def _compute_rhs_gpu(self):
    """b = -(C + alpha_tilde * lambda), entirely on GPU."""
    for t in range(self.n_tets):
      self._b_gpu[t] = -(self.constraints[t] +
                          self.alpha_tilde[t] * self.lambdaf[t])

  @ti.kernel
  def _residual_norm_gpu(self) -> ti.f32:
    """Compute ||b||_2 on GPU."""
    s = 0.0
    for t in range(self.n_tets):
      b = -(self.constraints[t] + self.alpha_tilde[t] * self.lambdaf[t])
      s += b * b
    return ti.sqrt(s)

  # ------------------------------------------------------------------ #
  #                         Fill A matrix                               #
  # ------------------------------------------------------------------ #

  @ti.kernel
  def _fill_A_kernel(self, data: ti.types.ndarray(dtype=ti.f32),
                     ii: ti.types.ndarray(dtype=ti.i32),
                     jj: ti.types.ndarray(dtype=ti.i32), nnz: ti.i32):
    for n in range(nnz):
      i = ii[n]
      j = jj[n]
      if i == j:
        m0 = self.v_invm[self.t_i[i * 4]]
        m1 = self.v_invm[self.t_i[i * 4 + 1]]
        m2 = self.v_invm[self.t_i[i * 4 + 2]]
        m3 = self.v_invm[self.t_i[i * 4 + 3]]
        g0 = self.gradC[i, 0]
        g1 = self.gradC[i, 1]
        g2 = self.gradC[i, 2]
        g3 = self.gradC[i, 3]
        data[n] = (m0 * g0.norm_sqr() + m1 * g1.norm_sqr() +
                   m2 * g2.norm_sqr() + m3 * g3.norm_sqr() +
                   self.alpha_tilde[i])
      else:
        a = ti.Vector([
            self.t_i[i * 4], self.t_i[i * 4 + 1], self.t_i[i * 4 + 2],
            self.t_i[i * 4 + 3]
        ])
        b = ti.Vector([
            self.t_i[j * 4], self.t_i[j * 4 + 1], self.t_i[j * 4 + 2],
            self.t_i[j * 4 + 3]
        ])
        n_shared, shared_v, order_a, order_b = _intersect_tets(a, b)
        offdiag = 0.0
        for kv in range(n_shared):
          sv = shared_v[kv]
          o1 = order_a[kv]
          o2 = order_b[kv]
          offdiag += self.v_invm[sv] * self.gradC[i, o1].dot(
              self.gradC[j, o2])
        data[n] = offdiag

  def _fill_A(self):
    """Fill A values into CSR, return scipy CSR (used during init only)."""
    self._fill_A_kernel(self._A_data, self._A_ii, self._A_jj, self._nnz)
    A = sp.csr_matrix(
        (self._A_data.copy(), self._A_indices.copy(),
         self._A_indptr.copy()),
        shape=(self.n_tets, self.n_tets))
    return A

  def _fill_A_and_update_gpu(self):
    """Fill A values and push to GPU solver (no scipy intermediate)."""
    self._fill_A_kernel(self._A_data, self._A_ii, self._A_jj, self._nnz)
    self._gpu_solver.update_fine_matrix_data(self._A_data)
    self._gpu_solver.update_all_coarse_matrices()

  # ------------------------------------------------------------------ #
  #                   Scatter dlambda -> dpos                           #
  # ------------------------------------------------------------------ #

  @ti.kernel
  def _scatter_dlam_gpu(self):
    """Scatter dlambda -> dpos using GPU fields."""
    for i in range(self.n_tets):
      p0 = self.t_i[i * 4]
      p1 = self.t_i[i * 4 + 1]
      p2 = self.t_i[i * 4 + 2]
      p3 = self.t_i[i * 4 + 3]
      dl = self._dlam_gpu[i]
      self.lambdaf[i] += dl
      ti.atomic_add(self.dpos[p0], self.v_invm[p0] * dl * self.gradC[i, 0])
      ti.atomic_add(self.dpos[p1], self.v_invm[p1] * dl * self.gradC[i, 1])
      ti.atomic_add(self.dpos[p2], self.v_invm[p2] * dl * self.gradC[i, 2])
      ti.atomic_add(self.dpos[p3], self.v_invm[p3] * dl * self.gradC[i, 3])

  @ti.kernel
  def _apply_dpos(self, omega: ti.f32):
    for k in range(self.n_verts):
      if self.v_invm[k] > 0.0:
        self.v_p[k] += omega * self.dpos[k]

  @ti.kernel
  def _zero_dpos(self):
    for k in range(self.n_verts):
      self.dpos[k] = ti.Vector([0.0, 0.0, 0.0])

  # ------------------------------------------------------------------ #
  #                          Line search                                #
  # ------------------------------------------------------------------ #

  @ti.kernel
  def _calc_dual_norm_at(
      self, pos_temp: ti.types.ndarray(dtype=ti.math.vec3)) -> ti.f32:
    dual = 0.0
    for t in range(self.n_tets):
      p0 = self.t_i[t * 4]
      p1 = self.t_i[t * 4 + 1]
      p2 = self.t_i[t * 4 + 2]
      p3 = self.t_i[t * 4 + 3]
      x0, x1, x2, x3 = pos_temp[p0], pos_temp[p1], pos_temp[p2], pos_temp[p3]
      D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
      F = D_s @ self.Dm_inv[t]
      U, S, V = ti.svd(F)
      c = ti.sqrt((S[0, 0] - 1.0)**2 + (S[1, 1] - 1.0)**2 +
                  (S[2, 2] - 1.0)**2)
      r = -(c + self.alpha_tilde[t] * self.lambdaf[t])
      dual += r * r
    return ti.sqrt(dual)

  def _line_search(self, beta=0.5, max_steps=10):
    """Backtracking line search on dual residual norm."""
    pos_np = self.v_p.to_numpy()
    dpos_np = self.dpos.to_numpy()
    current_norm = self._calc_dual_norm_at(pos_np)
    omega = 1.0
    for _ in range(max_steps):
      trial_pos = (pos_np + omega * dpos_np).astype(np.float32)
      trial_norm = self._calc_dual_norm_at(trial_pos)
      if trial_norm < current_norm:
        return omega
      omega *= beta
    return omega

  # ------------------------------------------------------------------ #
  #                        Main solve loop                              #
  # ------------------------------------------------------------------ #

  def solve(self, frame: int = 0):
    """MGPBD outer solve loop — GPU pipeline."""
    if self.benchmark is not None:
      self.benchmark.begin_frame(frame)

    self.lambdaf.fill(0.0)
    r0 = 0.0

    for ite in range(self.maxiter):
      # Evaluate constraints and gradients (GPU)
      self._compute_C_and_gradC()

      # Compute residual norm (GPU -> scalar)
      r = float(self._residual_norm_gpu())

      if ite == 0:
        r0 = r
      if r < self.atol or (r0 > 1e-12 and r < self.rtol * r0):
        if self.benchmark is not None:
          self.benchmark.record_iteration(ite, r)
        break

      # Fill A values and update GPU hierarchy (GPU)
      self._fill_A_and_update_gpu()

      # Compute RHS on GPU
      self._compute_rhs_gpu()

      # Solve A * dlam = b on GPU
      gpu_fill_zero(self._dlam_gpu, self.n_tets)
      cg_iters = self._gpu_solver.solve(self._b_gpu, self._dlam_gpu)

      # Scatter dlam -> dpos (GPU)
      self._zero_dpos()
      self._scatter_dlam_gpu()

      # Line search
      if self.use_line_search:
        omega = self._line_search()
      else:
        omega = 1.0

      # Apply position correction (GPU)
      self._apply_dpos(omega)

      if self.benchmark is not None:
        self.benchmark.record_iteration(ite, r, cg_iters=cg_iters, omega=omega)

    if self.benchmark is not None:
      self.benchmark.end_frame()
