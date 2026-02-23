# Multigrid-accelerated Global XPBD (MGPBD) solver for tetrahedral elasticity.
#
# Instead of the local Gauss-Seidel updates used in standard XPBD, MGPBD
# assembles a global sparse system A * dlambda = b at each Newton iteration,
# where A is the constraint Jacobian Gram matrix and b is the dual residual.
# The system is solved with a Multigrid-Preconditioned Conjugate Gradient
# (MGPCG) solver: PyAMG builds a smoothed-aggregation hierarchy (once every
# `setup_interval` frames) and Galerkin RAP projection updates the coarse-level
# matrices each iteration.  An optional backtracking line search is applied
# after each position correction.
# 
# Constraint: ARAP (as-rigid-as-possible) tetrahedral deformation.
# Energy: C(x) = ||S - I||_F  where F = Ds * Dm_inv, F = U S V^T.

import taichi as ti
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------- #
#                          Taichi helper functions                              #
# ---------------------------------------------------------------------------- #


@ti.func
def _make_dFdx_block(bx, by, bz):
  return ti.Matrix([
      [bx, 0.0, 0.0, by, 0.0, 0.0, bz, 0.0, 0.0],
      [0.0, bx, 0.0, 0.0, by, 0.0, 0.0, bz, 0.0],
      [0.0, 0.0, bx, 0.0, 0.0, by, 0.0, 0.0, bz],
  ])


@ti.func
def _arap_gradient(U, S, V, B):
  C = ti.sqrt((S[0, 0] - 1.0)**2 + (S[1, 1] - 1.0)**2 +
              (S[2, 2] - 1.0)**2)
  dcdS = (1.0 / C) * ti.Vector(
      [S[0, 0] - 1.0, S[1, 1] - 1.0, S[2, 2] - 1.0])

  dFdp1T = _make_dFdx_block(B[0, 0], B[0, 1], B[0, 2])
  dFdp2T = _make_dFdx_block(B[1, 0], B[1, 1], B[1, 2])
  dFdp3T = _make_dFdx_block(B[2, 0], B[2, 1], B[2, 2])

  u00, u01, u02 = U[0, 0], U[0, 1], U[0, 2]
  u10, u11, u12 = U[1, 0], U[1, 1], U[1, 2]
  u20, u21, u22 = U[2, 0], U[2, 1], U[2, 2]
  v00, v01, v02 = V[0, 0], V[0, 1], V[0, 2]
  v10, v11, v12 = V[1, 0], V[1, 1], V[1, 2]
  v20, v21, v22 = V[2, 0], V[2, 1], V[2, 2]

  dcdF = ti.Vector([
      ti.Vector([u00 * v00, u01 * v01, u02 * v02]).dot(dcdS),
      ti.Vector([u10 * v00, u11 * v01, u12 * v02]).dot(dcdS),
      ti.Vector([u20 * v00, u21 * v01, u22 * v02]).dot(dcdS),
      ti.Vector([u00 * v10, u01 * v11, u02 * v12]).dot(dcdS),
      ti.Vector([u10 * v10, u11 * v11, u12 * v12]).dot(dcdS),
      ti.Vector([u20 * v10, u21 * v11, u22 * v12]).dot(dcdS),
      ti.Vector([u00 * v20, u01 * v21, u02 * v22]).dot(dcdS),
      ti.Vector([u10 * v20, u11 * v21, u12 * v22]).dot(dcdS),
      ti.Vector([u20 * v20, u21 * v21, u22 * v22]).dot(dcdS),
  ])

  g1 = dFdp1T @ dcdF
  g2 = dFdp2T @ dcdF
  g3 = dFdp3T @ dcdF
  g0 = -g1 - g2 - g3
  return g0, g1, g2, g3


@ti.func
def _intersect_tets(a, b):
  k = 0
  c = ti.Vector([-1, -1, -1])
  order_a = ti.Vector([-1, -1, -1])
  order_b = ti.Vector([-1, -1, -1])
  for i in ti.static(range(4)):
    for j in ti.static(range(4)):
      if a[i] == b[j]:
        c[k] = a[i]
        order_a[k] = i
        order_b[k] = j
        k += 1
  return k, c, order_a, order_b


# ---------------------------------------------------------------------------- #
#                              MGPBDSolver class                               #
# ---------------------------------------------------------------------------- #


@ti.data_oriented
class MGPBDSolver:

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
               use_line_search: bool = True) -> None:
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

    # AMG state
    self._amg_Ps = None
    self._amg_As = None

  def init(self):
    """Precompute rest shape and sparsity pattern. Call once after construction."""
    self._precompute_rest_shape()
    self._build_sparsity_pattern()
    self._alpha_tilde_np = self.alpha_tilde.to_numpy()
    print(f"MGPBD: {self.n_verts} verts, {self.n_tets} tets, {self._nnz} nnz")

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
    print("MGPBD: Building sparsity pattern...")
    t_i_np = self.t_i.to_numpy()
    nt = self.n_tets

    # vertex-to-tet adjacency
    vert_to_tets = {}
    for t in range(nt):
      for l in range(4):
        v = t_i_np[t * 4 + l]
        if v not in vert_to_tets:
          vert_to_tets[v] = set()
        vert_to_tets[v].add(t)

    # tet-to-tet adjacency
    adj = {}
    for t in range(nt):
      neighbors = set()
      for l in range(4):
        v = t_i_np[t * 4 + l]
        neighbors |= vert_to_tets[v]
      neighbors.discard(t)
      adj[t] = sorted(neighbors)

    num_adj = np.array([len(adj[t]) for t in range(nt)], dtype=np.int32)

    # Build CSR: off-diagonal entries first, then diagonal last per row
    nnz = int(np.sum(num_adj) + nt)
    indptr = np.zeros(nt + 1, dtype=np.int32)
    indices = np.zeros(nnz, dtype=np.int32)
    data = np.zeros(nnz, dtype=np.float32)

    for i in range(nt):
      indptr[i + 1] = indptr[i] + num_adj[i] + 1
      indices[indptr[i]:indptr[i] + num_adj[i]] = adj[i]
      indices[indptr[i] + num_adj[i]] = i
    assert indptr[-1] == nnz

    # COO row indices
    ii = np.zeros(nnz, dtype=np.int32)
    for i in range(nt):
      ii[indptr[i]:indptr[i + 1]] = i

    self._A_data = data
    self._A_indices = indices
    self._A_indptr = indptr
    self._A_ii = ii
    self._A_jj = indices.copy()
    self._nnz = nnz
    print(
        f"MGPBD: Sparsity built. nnz={nnz}, avg_adj={np.mean(num_adj):.1f}")

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
      x0, x1, x2, x3 = self.v_p[p0], self.v_p[p1], self.v_p[p2], self.v_p[
          p3]

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
    """Fill A values into pre-computed CSR pattern, return scipy CSR."""
    self._fill_A_kernel(self._A_data, self._A_ii, self._A_jj, self._nnz)
    A = sp.csr_matrix(
        (self._A_data.copy(), self._A_indices.copy(),
         self._A_indptr.copy()),
        shape=(self.n_tets, self.n_tets))
    return A

  # ------------------------------------------------------------------ #
  #                   Scatter dlambda -> dpos                           #
  # ------------------------------------------------------------------ #

  @ti.kernel
  def _scatter_dlam(self, dlambda: ti.types.ndarray(dtype=ti.f32)):
    for i in range(self.n_tets):
      p0 = self.t_i[i * 4]
      p1 = self.t_i[i * 4 + 1]
      p2 = self.t_i[i * 4 + 2]
      p3 = self.t_i[i * 4 + 3]
      dl = dlambda[i]
      self.lambdaf[i] += dl
      self.dpos[p0] += self.v_invm[p0] * dl * self.gradC[i, 0]
      self.dpos[p1] += self.v_invm[p1] * dl * self.gradC[i, 1]
      self.dpos[p2] += self.v_invm[p2] * dl * self.gradC[i, 2]
      self.dpos[p3] += self.v_invm[p3] * dl * self.gradC[i, 3]

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
  #                        Dual residual norm                           #
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
      x0, x1, x2, x3 = pos_temp[p0], pos_temp[p1], pos_temp[p2], pos_temp[
          p3]
      D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
      F = D_s @ self.Dm_inv[t]
      U, S, V = ti.svd(F)
      c = ti.sqrt((S[0, 0] - 1.0)**2 + (S[1, 1] - 1.0)**2 +
                  (S[2, 2] - 1.0)**2)
      r = -(c + self.alpha_tilde[t] * self.lambdaf[t])
      dual += r * r
    return ti.sqrt(dual)

  # ------------------------------------------------------------------ #
  #                          Line search                                #
  # ------------------------------------------------------------------ #

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
  #                         AMG setup                                   #
  # ------------------------------------------------------------------ #

  def _amg_setup(self, A):
    """Build multigrid hierarchy using PyAMG Unsmoothed Aggregation."""
    import pyamg
    A64 = A.astype(np.float64)
    ml = pyamg.smoothed_aggregation_solver(A64,
                                           max_coarse=400,
                                           smooth=None,
                                           improve_candidates=None,
                                           symmetry='symmetric')
    self._amg_Ps = [
        level.P.tocsr().astype(np.float64) for level in ml.levels[:-1]
    ]
    sizes = [A64.shape[0]] + [P.shape[1] for P in self._amg_Ps]
    print(f"MGPBD: AMG setup {len(sizes)} levels, sizes={sizes}")

  def _update_coarse_matrices(self, A):
    """Recompute coarse matrices via Galerkin projection (RAP)."""
    self._amg_As = [A.astype(np.float64)]
    for P in self._amg_Ps:
      Ac = P.T @ self._amg_As[-1] @ P
      self._amg_As.append(Ac.tocsr())

  # ------------------------------------------------------------------ #
  #                      V-cycle and MGPCG                              #
  # ------------------------------------------------------------------ #

  def _jacobi_smooth(self, A, b, x, n_iter):
    """Weighted Jacobi smoother."""
    D_inv = 1.0 / A.diagonal()
    omega = self.omega_jacobi
    for _ in range(n_iter):
      x = x + omega * D_inv * (b - A @ x)
    return x

  def _vcycle(self, b, x, level=0):
    """One V-cycle for AMG preconditioning."""
    if level == len(self._amg_Ps):
      return sp.linalg.spsolve(self._amg_As[level], b)

    A = self._amg_As[level]
    P = self._amg_Ps[level]

    # Pre-smooth
    x = self._jacobi_smooth(A, b, x, self.n_smooth)

    # Restrict
    r = b - A @ x
    b_coarse = P.T @ r

    # Recurse
    x_coarse = self._vcycle(b_coarse,
                            np.zeros(b_coarse.shape[0], dtype=np.float64),
                            level + 1)

    # Prolongate and correct
    x = x + P @ x_coarse

    # Post-smooth
    x = self._jacobi_smooth(A, b, x, self.n_smooth)

    return x

  def _mgpcg(self, A, b):
    """CG with V-cycle preconditioner."""
    b64 = b.astype(np.float64)
    x = np.zeros_like(b64)
    r = b64.copy()
    z = self._vcycle(r, np.zeros_like(r))
    p = z.copy()
    rz = r @ z

    for k in range(self.maxiter_cg):
      Ap = self._amg_As[0] @ p
      pAp = p @ Ap
      if abs(pAp) < 1e-30:
        break
      alpha = rz / pAp
      x += alpha * p
      r -= alpha * Ap

      if np.linalg.norm(r) < self.tol_cg:
        break

      z = self._vcycle(r, np.zeros_like(r))
      rz_new = r @ z
      if abs(rz) < 1e-30:
        break
      beta_val = rz_new / rz
      p = z + beta_val * p
      rz = rz_new

    return x.astype(np.float32)

  # ------------------------------------------------------------------ #
  #                        Main solve loop                              #
  # ------------------------------------------------------------------ #

  def solve(self, frame: int = 0):
    """MGPBD outer solve loop."""
    self.lambdaf.fill(0.0)
    r0 = 0.0

    for ite in range(self.maxiter):
      # Evaluate constraints and gradients at current positions
      self._compute_C_and_gradC()

      # RHS = dual residual: b = -(C + alpha_tilde * lambda)
      C_np = self.constraints.to_numpy()
      lam_np = self.lambdaf.to_numpy()
      b = -(C_np + self._alpha_tilde_np * lam_np)

      # Convergence check
      r = np.linalg.norm(b)
      if ite == 0:
        r0 = r
      if r < self.atol or (r0 > 1e-12 and r < self.rtol * r0):
        break

      # Fill system matrix A
      A = self._fill_A()

      # AMG setup (lazy: only on first iteration when needed)
      if ite == 0 and (self._amg_Ps is None or
                       frame % self.setup_interval == 0):
        self._amg_setup(A)

      # Update coarse matrices (RAP) every iteration since A changes
      self._update_coarse_matrices(A)

      # Solve A * dlam = b
      dlam = self._mgpcg(A, b)

      # Scatter dlam -> dpos, update lambda
      self._zero_dpos()
      self._scatter_dlam(dlam)

      # Apply position correction with line search
      if self.use_line_search:
        omega = self._line_search()
      else:
        omega = 1.0
      self._apply_dpos(omega)
