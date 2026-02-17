import taichi as ti
import numpy as np
from interface import render_funcs, mesh_render_3d
from data import tet_data
from cons.vbd import VBDSolver
from utils import arg_parser, anim
import math

ti.init(arch=ti.vulkan)


def test_vbd(args):
  # Load args
  dir_name = args.dir_name
  obj_name = args.obj_name
  fixed_id = args.fixed_id
  operating_points_id = args.operating_points_id
  gravity = args.pbd.gravity
  fps = args.pbd.fps
  substeps = args.pbd.substeps
  damp = args.pbd.damp
  scale = args.scale
  repose = args.repose
  speed = args.speed
  movement_direction = args.movement_direction
  cam_eye = args.get('cam_eye', (2.0, 2.0, 2.0))
  cam_center = args.get('cam_center', (0.0, 0.0, 0.0))
  cam_up = args.get('cam_up', (0.0, 1.0, 0.0))
  window_height = args.get('window_height', 700)
  window_width = args.get('window_width', 700)

  # VBD params
  vbd_cfg = args.vbd
  youngs_modulus = float(vbd_cfg.get('youngs_modulus', 1e4))
  poissons_ratio = float(vbd_cfg.get('poissons_ratio', 0.3))
  n_iterations = int(vbd_cfg.get('n_iterations', 10))
  damping = float(vbd_cfg.get('damping', 0.0))
  rho = float(vbd_cfg.get('rho', 0.0))
  method = str(vbd_cfg.get('method', 'serial'))

  # Load mesh
  model_path = f'assets/{dir_name}/{obj_name}.mesh'
  mesh = tet_data.load_tets(model_path, scale, repose, remove_duplicate=False)
  invm = mesh.v_invm.to_numpy()
  invm[fixed_id] = 0
  mesh.v_invm.from_numpy(invm)

  # Display points
  wireframe = [True]
  static_points_id = []
  for x in fixed_id:
    if x not in operating_points_id:
      static_points_id.append(x)
  particles_static = ti.Vector.field(3,
                                     dtype=ti.f32,
                                     shape=len(static_points_id))
  particles_dynamic = ti.Vector.field(3,
                                      dtype=ti.f32,
                                      shape=len(operating_points_id))

  def update_display_points():
    for i in range(len(static_points_id)):
      particles_static[i] = mesh.v_p[static_points_id[i]]
    for i in range(len(operating_points_id)):
      particles_dynamic[i] = mesh.v_p[operating_points_id[i]]

  # Init VBD solver
  g = ti.Vector(gravity)
  dt = 1.0 / fps / substeps
  solver = VBDSolver(v_p=mesh.v_p,
                     v_p_ref=mesh.v_p_ref,
                     v_invm=mesh.v_invm,
                     t_i=mesh.t_i,
                     t_m=mesh.t_m,
                     gravity=g,
                     dt=dt,
                     damp=damp,
                     youngs_modulus=youngs_modulus,
                     poissons_ratio=poissons_ratio,
                     damping=damping,
                     n_iterations=n_iterations,
                     rho=rho,
                     method=method)
  solver.init()

  # Init interface
  window = mesh_render_3d.MeshRender3D(res=(window_width, window_height),
                                       title='VBD Tet Mesh',
                                       kernel='taichi')
  window.set_background_color((1, 1, 1, 1))
  window.set_camera(eye=cam_eye, center=cam_center, up=cam_up)
  window.set_lighting((4, 4, -4), (0.96, 0.96, 0.96), (0.2, 0.2, 0.2))
  window.add_render_func(render_func=render_funcs.get_mesh_render_func(
      mesh.v_p, mesh.f_i, wireframe, color=(0.0, 0.0, 1.0)))
  window.add_render_func(
      render_funcs.get_particles_render_func(particles_static,
                                             color=(0.5, 0.5, 0.5),
                                             radius=0.02 * scale))
  window.add_render_func(
      render_funcs.get_particles_render_func(particles_dynamic,
                                             color=(0.0, 1.0, 0.0),
                                             radius=0.02 * scale))

  while window.running():
    anim.set_movement(window, mesh, operating_points_id, movement_direction, scale,
                 speed)
    for i in range(substeps):
      solver.step()
    update_display_points()

    window.pre_update()
    window.render()
    window.show()

  window.terminate()


if __name__ == '__main__':
  args = arg_parser.get_args('config/sphere_10k_tet_vbd.yaml')
  test_vbd(args)
