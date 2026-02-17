import taichi as ti
import numpy as np
from utils.graph_coloring import color_edges

LENGTH_METHOD = ['default', 'jacobi', 'gauss_seidel', 'graph_coloring']

@ti.data_oriented
class LengthCons:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               e_i: ti.Field,
               e_rl: ti.Field,
               v_invm: ti.Field,
               dt,
               alpha=0.0,
               method='default') -> None:
    self.n = e_i.shape[0] // 2
    assert self.n == e_rl.shape[0]

    self.pos = v_p
    self.pos_ref = v_p_ref
    self.indices = e_i
    self.invm = v_invm

    self.rest_length = e_rl
    self.length = ti.field(dtype=ti.f32, shape=self.n)
    self.lambdaf = ti.field(dtype=ti.f32, shape=self.n)
    self.alpha = alpha / (dt * dt)
    
    # For Jacobi method: vertex corrections accumulator
    self.vertex_corrections = ti.Vector.field(3, dtype=ti.f32, shape=v_p.shape[0])
    
    # Default to Gauss-Seidel method for deterministic parallel execution
    self.method = method
    if method == 'graph_coloring':
      self.compute_graph_coloring()

  def init_rest_status(self):
    pass

  def preupdate_cons(self):
    self.lambdaf.fill(0.0)
    if self.method == 'jacobi':
      self.vertex_corrections.fill(0.0)

  def update_cons(self):
    if self.method == 'jacobi':
      self.solve_cons_jacobi()
    elif self.method == 'default':
      self.solve_cons_original()
    elif self.method == 'gauss_seidel':
      self.solve_cons_gauss_seidel()
    elif self.method == 'graph_coloring':
      self.solve_cons_graph_coloring()

    else:
      raise ValueError(f"Unknown method: {self.method}")
    
  
  def set_jacobi_method(self, jacobi):
    self.use_jacobi = jacobi

  @ti.kernel
  def solve_cons_original(self):
    for k in range(self.n):
      i = self.indices[k * 2]
      j = self.indices[k * 2 + 1]
      xi = self.pos[i]
      xj = self.pos[j]
      xij = xi - xj
      self.length[k] = xij.norm()
      C = self.length[k] - self.rest_length[k]
      wi = self.invm[i]
      wj = self.invm[j]
      delta_lambda = -(C + self.alpha * self.lambdaf[k]) / (wi + wj +
                                                            self.alpha)
      self.lambdaf[k] += delta_lambda
      xij = xij / xij.norm()
      self.pos[i] += wi * delta_lambda * xij.normalized()
      self.pos[j] += -wj * delta_lambda * xij.normalized()
  
  @ti.kernel
  def solve_cons_gauss_seidel(self):
    ti.loop_config(serialize=True)
    for k in range(self.n):
      i = self.indices[k * 2]
      j = self.indices[k * 2 + 1]
      xi = self.pos[i]
      xj = self.pos[j]
      xij = xi - xj
      self.length[k] = xij.norm()
      C = self.length[k] - self.rest_length[k]
      wi = self.invm[i]
      wj = self.invm[j]
      delta_lambda = -(C + self.alpha * self.lambdaf[k]) / (wi + wj +
                                                            self.alpha)
      self.lambdaf[k] += delta_lambda
      xij = xij / xij.norm()
      self.pos[i] += wi * delta_lambda * xij.normalized()
      self.pos[j] += -wj * delta_lambda * xij.normalized()

  def solve_cons_jacobi(self):
    # Two-phase Jacobi method
    self.compute_corrections_jacobi()
    self.apply_corrections()
  
  @ti.kernel
  def compute_corrections_jacobi(self):
    # Phase 1: Compute all corrections without modifying vertex positions (fully parallel)
    for k in range(self.n):
      i = self.indices[k * 2]
      j = self.indices[k * 2 + 1]
      xi = self.pos[i]
      xj = self.pos[j]
      xij = xi - xj
      self.length[k] = xij.norm()
      C = self.length[k] - self.rest_length[k]
      wi = self.invm[i]
      wj = self.invm[j]
      delta_lambda = -(C + self.alpha * self.lambdaf[k]) / (wi + wj +
                                                            self.alpha)
      self.lambdaf[k] += delta_lambda
      xij_normalized = xij.normalized()
      
      # Accumulate corrections for both vertices
      correction_i = wi * delta_lambda * xij_normalized
      correction_j = -wj * delta_lambda * xij_normalized
      
      ti.atomic_add(self.vertex_corrections[i], correction_i)
      ti.atomic_add(self.vertex_corrections[j], correction_j)
      
  @ti.kernel  
  def apply_corrections(self):
    # Phase 2: Apply accumulated corrections to all vertices (fully parallel)
    for i in range(self.pos.shape[0]):
      self.pos[i] += self.vertex_corrections[i]

  def compute_graph_coloring(self):
    print("Computing graph coloring for edges...")
    result = color_edges(self.indices.to_numpy(), self.n, self.pos.shape[0])
    print(f"Graph coloring completed using {result.num_colors} colors.")
    self.colored_edge_order = ti.field(dtype=ti.i32, shape=self.n)
    self.colored_edge_order.from_numpy(result.ordered_indices)
    self.color_offsets = result.color_offsets

  def solve_cons_graph_coloring(self):
    for i in range(len(self.color_offsets) - 1):
      start = self.color_offsets[i]
      end = self.color_offsets[i+1]
      self.solve_range(start, end)

  @ti.kernel
  def solve_range(self, start_idx: int, end_idx: int):
    for k in range(start_idx, end_idx):
      edge_idx = self.colored_edge_order[k]
      i = self.indices[edge_idx * 2]
      j = self.indices[edge_idx * 2 + 1]
      xi = self.pos[i]
      xj = self.pos[j]
      xij = xi - xj
      self.length[edge_idx] = xij.norm()
      C = self.length[edge_idx] - self.rest_length[edge_idx]
      wi = self.invm[i]
      wj = self.invm[j]
      delta_lambda = -(C + self.alpha * self.lambdaf[edge_idx]) / (wi + wj +
                                                            self.alpha)
      self.lambdaf[edge_idx] += delta_lambda
      xij = xij / xij.norm()
      self.pos[i] += wi * delta_lambda * xij.normalized()
      self.pos[j] += -wj * delta_lambda * xij.normalized()
