import taichi as ti
import numpy as np
from utils.graph_coloring import color_vertices


@ti.data_oriented
class VBDSolver:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               v_invm: ti.Field,
               t_i: ti.Field,
               t_m: ti.Field,
               gravity: ti.Vector,
               dt: float,
               damp: float = 1.0,
               youngs_modulus: float = 1e4,
               poissons_ratio: float = 0.3,
               damping: float = 0.0,
               n_iterations: int = 10,
               rho: float = 0.0,
               method: str = 'serial',
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
    self.n_iterations = n_iterations
    self.rho = rho
    self.damping_coeff = damping
    self.method = method
    self.benchmark = benchmark

    self.damp = ti.field(dtype=ti.f32, shape=())
    self.damp[None] = damp

    # Lamé parameters from Young's modulus and Poisson's ratio
    E = youngs_modulus
    nu = poissons_ratio
    self.mu = E / (2.0 * (1.0 + nu))
    self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    print(f"VBD: mu={self.mu:.2f}, lambda={self.lam:.2f}")

    # Velocity field
    self.v_v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)

    # Position caches
    self.v_p_old = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    self.x_tilde = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    self.x_prev = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)

    # Precomputed per-tet data
    self.Dm_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_tets)
    self.rest_vol = ti.field(dtype=ti.f32, shape=self.n_tets)

    # Build adjacency
    self._build_vertex_tet_adjacency()
    if self.method == 'color':
      self._compute_vertex_coloring()
    print(f"VBD: method={self.method}")

  def _build_vertex_tet_adjacency(self):
    """Build CSR-format vertex-to-tet adjacency with local indices."""
    print("VBD: Building vertex-tet adjacency...")
    tet_indices = self.t_i.to_numpy()

    vert_tets = [[] for _ in range(self.n_verts)]
    vert_local = [[] for _ in range(self.n_verts)]
    for t in range(self.n_tets):
      for l in range(4):
        v = tet_indices[t * 4 + l]
        vert_tets[v].append(t)
        vert_local[v].append(l)

    offsets = [0]
    flat_tets = []
    flat_local = []
    for v in range(self.n_verts):
      flat_tets.extend(vert_tets[v])
      flat_local.extend(vert_local[v])
      offsets.append(len(flat_tets))

    total = len(flat_tets)
    self.vert_tet_adj = ti.field(dtype=ti.i32, shape=total)
    self.vert_tet_adj.from_numpy(np.array(flat_tets, dtype=np.int32))
    self.vert_local_id = ti.field(dtype=ti.i32, shape=total)
    self.vert_local_id.from_numpy(np.array(flat_local, dtype=np.int32))
    self.vert_adj_offsets = ti.field(dtype=ti.i32, shape=self.n_verts + 1)
    self.vert_adj_offsets.from_numpy(np.array(offsets, dtype=np.int32))
    print(f"VBD: Adjacency built. Total vert-tet pairs: {total}")

  def _compute_vertex_coloring(self):
    print("VBD: Computing vertex coloring...")
    result = color_vertices(self.t_i.to_numpy(), self.n_tets, 4, self.n_verts)
    print(f"VBD: Vertex coloring completed using {result.num_colors} colors.")
    self.color_offsets = result.color_offsets
    self.color_vertex_order = ti.field(dtype=ti.i32, shape=self.n_verts)
    self.color_vertex_order.from_numpy(result.ordered_indices)

  def init(self):
    """Precompute rest shape matrices and volumes. Call once after construction."""
    self._precompute_rest_shape()

  @ti.kernel
  def _precompute_rest_shape(self):
    for k in range(self.n_tets):
      a = self.t_i[k * 4]
      b = self.t_i[k * 4 + 1]
      c = self.t_i[k * 4 + 2]
      d = self.t_i[k * 4 + 3]
      r1 = self.v_p_ref[a]
      r2 = self.v_p_ref[b]
      r3 = self.v_p_ref[c]
      r4 = self.v_p_ref[d]
      Dm = ti.Matrix.cols([r1 - r4, r2 - r4, r3 - r4])
      self.Dm_inv[k] = Dm.inverse()
      self.rest_vol[k] = ti.abs(Dm.determinant()) / 6.0

  @ti.kernel
  def make_prediction(self):
    """Compute inertia position and set as initial guess."""
    for k in range(self.n_verts):
      self.v_p_old[k] = self.v_p[k]
      self.x_tilde[k] = self.v_p[k] + self.v_v[k] * self.dt + \
                         self.gravity * self.dt * self.dt
      self.v_p[k] = self.x_tilde[k]

  def solve(self, frame: int = 0):
    """Run VBD iterations."""
    if self.benchmark is not None:
      self.benchmark.begin_frame(frame)

    omega = 1.0
    for n in range(self.n_iterations):
      self._copy_to_prev()
      if self.method == 'serial':
        self._solve_all_serial()
      else:
        for c in range(len(self.color_offsets) - 1):
          self._solve_vertex_range(self.color_offsets[c], self.color_offsets[c + 1])
      # Chebyshev acceleration
      if self.rho > 0.0 and n >= 1:
        if n == 1:
          omega = 2.0 / (2.0 - self.rho * self.rho)
        else:
          omega = 4.0 / (4.0 - self.rho * self.rho * omega)
        self._apply_chebyshev(omega)

      if self.benchmark is not None:
        self.benchmark.record_iteration(n, 0.0, omega=omega)

    if self.benchmark is not None:
      self.benchmark.end_frame()

  @ti.kernel
  def update_vel(self):
    """Update velocities from position change."""
    for k in range(self.n_verts):
      self.v_v[k] = self.damp[None] * (self.v_p[k] -
                                        self.v_p_old[k]) / self.dt

  def step(self, frame: int = 0):
    """Full timestep: predict → solve → velocity update."""
    self.make_prediction()
    self.solve(frame)
    self.update_vel()

  @ti.kernel
  def _copy_to_prev(self):
    for k in range(self.n_verts):
      self.x_prev[k] = self.v_p[k]

  @ti.kernel
  def _apply_chebyshev(self, omega: ti.f32):
    for k in range(self.n_verts):
      if self.v_invm[k] > 0.0:
        self.v_p[k] = self.v_p[k] + omega * (self.v_p[k] - self.x_prev[k])

  @ti.kernel
  def _solve_all_serial(self):
    """Serial Gauss-Seidel: process vertices one by one."""
    mu = self.mu
    lam = self.lam
    dt = self.dt
    h2 = dt * dt
    kd = self.damping_coeff

    ti.loop_config(serialize=True)
    for vi in range(self.n_verts):
      if self.v_invm[vi] == 0.0:
        continue

      mi = 1.0 / self.v_invm[vi]

      # Inertia term
      f_i = mi / h2 * (self.x_tilde[vi] - self.v_p[vi])
      H_i = (mi / h2) * ti.Matrix.identity(ti.f32, 3)

      # Gather from incident tets
      adj_start = self.vert_adj_offsets[vi]
      adj_end = self.vert_adj_offsets[vi + 1]

      for adj_idx in range(adj_start, adj_end):
        tet_id = self.vert_tet_adj[adj_idx]
        local_id = self.vert_local_id[adj_idx]

        ia = self.t_i[tet_id * 4]
        ib = self.t_i[tet_id * 4 + 1]
        ic = self.t_i[tet_id * 4 + 2]
        i_d = self.t_i[tet_id * 4 + 3]
        xa = self.v_p[ia]
        xb = self.v_p[ib]
        xc = self.v_p[ic]
        xd = self.v_p[i_d]

        D = ti.Matrix.cols([xa - xd, xb - xd, xc - xd])
        B = self.Dm_inv[tet_id]
        F = D @ B

        f1 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
        f2 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
        f3 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])

        J = F.determinant()
        cofF = ti.Matrix.cols([f2.cross(f3), f3.cross(f1), f1.cross(f2)])

        P = mu * F + lam * (J - 1.0) * cofF

        B_row0 = ti.Vector([B[0, 0], B[0, 1], B[0, 2]])
        B_row1 = ti.Vector([B[1, 0], B[1, 1], B[1, 2]])
        B_row2 = ti.Vector([B[2, 0], B[2, 1], B[2, 2]])
        b_l = ti.Vector([0.0, 0.0, 0.0])
        if local_id == 0:
          b_l = B_row0
        elif local_id == 1:
          b_l = B_row1
        elif local_id == 2:
          b_l = B_row2
        else:
          b_l = -(B_row0 + B_row1 + B_row2)

        V0 = self.rest_vol[tet_id]
        f_e = -V0 * (P @ b_l)
        f_i += f_e

        q_l = cofF @ b_l
        b_l_sqr = b_l.norm_sqr()
        H_e = V0 * (mu * b_l_sqr * ti.Matrix.identity(ti.f32, 3) +
                     lam * q_l.outer_product(q_l))
        H_i += H_e

      if kd > 0.0:
        f_i -= (kd / dt) * (self.v_p[vi] - self.v_p_old[vi])
        H_i += (kd / dt) * ti.Matrix.identity(ti.f32, 3)

      det_H = H_i.determinant()
      if ti.abs(det_H) > 1e-10:
        dx = H_i.inverse() @ f_i
        self.v_p[vi] += dx

  @ti.kernel
  def _solve_vertex_range(self, start: int, end: int):
    """Process all vertices in a color group in parallel."""
    mu = self.mu
    lam = self.lam
    dt = self.dt
    h2 = dt * dt
    kd = self.damping_coeff

    for idx in range(start, end):
      vi = self.color_vertex_order[idx]

      if self.v_invm[vi] == 0.0:
        continue

      mi = 1.0 / self.v_invm[vi]

      # Inertia term
      f_i = mi / h2 * (self.x_tilde[vi] - self.v_p[vi])
      H_i = (mi / h2) * ti.Matrix.identity(ti.f32, 3)

      # Gather from incident tets
      adj_start = self.vert_adj_offsets[vi]
      adj_end = self.vert_adj_offsets[vi + 1]

      for adj_idx in range(adj_start, adj_end):
        tet_id = self.vert_tet_adj[adj_idx]
        local_id = self.vert_local_id[adj_idx]

        # Get tet vertex positions
        ia = self.t_i[tet_id * 4]
        ib = self.t_i[tet_id * 4 + 1]
        ic = self.t_i[tet_id * 4 + 2]
        i_d = self.t_i[tet_id * 4 + 3]
        xa = self.v_p[ia]
        xb = self.v_p[ib]
        xc = self.v_p[ic]
        xd = self.v_p[i_d]

        # Deformation gradient F = D @ B
        D = ti.Matrix.cols([xa - xd, xb - xd, xc - xd])
        B = self.Dm_inv[tet_id]
        F = D @ B

        # Extract columns of F
        f1 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
        f2 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
        f3 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])

        # Determinant
        J = F.determinant()

        # Cofactor matrix (columns are cross products)
        cofF = ti.Matrix.cols([f2.cross(f3), f3.cross(f1), f1.cross(f2)])

        # First PK stress: P = μF + λ(J-1)·cofF
        P = mu * F + lam * (J - 1.0) * cofF

        # Direction vector b_l: row of B for local 0-2, negative sum for 3
        B_row0 = ti.Vector([B[0, 0], B[0, 1], B[0, 2]])
        B_row1 = ti.Vector([B[1, 0], B[1, 1], B[1, 2]])
        B_row2 = ti.Vector([B[2, 0], B[2, 1], B[2, 2]])
        b_l = ti.Vector([0.0, 0.0, 0.0])
        if local_id == 0:
          b_l = B_row0
        elif local_id == 1:
          b_l = B_row1
        elif local_id == 2:
          b_l = B_row2
        else:
          b_l = -(B_row0 + B_row1 + B_row2)

        V0 = self.rest_vol[tet_id]

        # Force contribution: f_e = -V₀ · P · b_l
        f_e = -V0 * (P @ b_l)
        f_i += f_e

        # Gauss-Newton Hessian: H_e = V₀[μ||b_l||²I + λ(cofF·b_l)(cofF·b_l)ᵀ]
        q_l = cofF @ b_l
        b_l_sqr = b_l.norm_sqr()
        H_e = V0 * (mu * b_l_sqr * ti.Matrix.identity(ti.f32, 3) +
                     lam * q_l.outer_product(q_l))
        H_i += H_e

      # Rayleigh damping (optional)
      if kd > 0.0:
        f_i -= (kd / dt) * (self.v_p[vi] - self.v_p_old[vi])
        H_i += (kd / dt) * ti.Matrix.identity(ti.f32, 3)

      # Solve 3x3 system and update
      det_H = H_i.determinant()
      if ti.abs(det_H) > 1e-10:
        dx = H_i.inverse() @ f_i
        self.v_p[vi] += dx
