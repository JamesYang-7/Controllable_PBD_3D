import taichi as ti
import numpy as np
from interface import render_funcs, mesh_render_3d
from data import tet_data
from cons import deform3d, framework
from utils import arg_parser, anim
import math

ti.init(arch=ti.vulkan)
# ti.init(arch=ti.x64, cpu_max_num_threads=1) # cpu for performance test

def test_tet_mesh(args):
  '''
  Test simulation of tet mesh with constraints
  '''
  # Load args
  # data args
  dir_name = args.dir_name
  obj_name = args.obj_name
  fixed_id = args.fixed_id
  operating_points_id = args.operating_points_id
  # simulation args
  pbd = args.pbd
  arg_cons = args.cons
  gravity = pbd.gravity
  fps = pbd.fps
  substeps = pbd.substeps
  subsub = pbd.subsub
  damp = pbd.damp
  scale = args.scale
  repose = args.repose
  speed = args.speed
  movement_direction = args.movement_direction
  cam_eye = args.get('cam_eye', (2.0, 2.0, 2.0))
  cam_center = args.get('cam_center', (0.0, 0.0, 0.0))
  cam_up = args.get('cam_up', (0.0, 1.0, 0.0))
  window_height = args.get('window_height', 700)
  window_width = args.get('window_width', 700)

  # Load mesh data
  model_path = f'assets/{dir_name}/{obj_name}.mesh'
  mesh = tet_data.load_tets(model_path, scale, repose, remove_duplicate=False)
  invm = mesh.v_invm.to_numpy()
  invm[fixed_id] = 0
  mesh.v_invm.from_numpy(invm)

  # Set display points
  wireframe = [True]
  static_points_id = []
  for x in fixed_id:
    if x not in operating_points_id:
      static_points_id.append(x)
  particles_static = ti.Vector.field(3, dtype=ti.f32, shape=len(static_points_id))
  particles_dynamic = ti.Vector.field(3, dtype=ti.f32, shape=len(operating_points_id))

  def update_display_points():
    for i in range(len(static_points_id)):
      particles_static[i] = mesh.v_p[static_points_id[i]]
    for i in range(len(operating_points_id)):
      particles_dynamic[i] = mesh.v_p[operating_points_id[i]]

  # Init simulation
  g = ti.Vector(gravity)
  dt = 1.0 / fps / substeps
  pbd = framework.pbd_framework(mesh.v_p, g, dt, damp=damp)
  if arg_cons.deform3d.enabled:
    deform = deform3d.Deform3D(v_p=mesh.v_p,
                               v_p_ref=mesh.v_p_ref,
                               v_invm=mesh.v_invm,
                               t_i=mesh.t_i,
                               t_m=mesh.t_m,
                               dt=dt,
                               hydro_alpha=float(arg_cons.deform3d.hydro_alpha),
                               devia_alpha=float(arg_cons.deform3d.devia_alpha),
                               method=arg_cons.deform3d.method)
    pbd.add_cons(deform, 0)
  pbd.init_rest_status(0)

  # Init interface
  window = mesh_render_3d.MeshRender3D(res=(window_width, window_height),
                                       title='Simple Sphere Tet Mesh',
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
    anim.set_movement(window, mesh, operating_points_id, movement_direction, scale, speed)
    for i in range(substeps):
      pbd.make_prediction()
      pbd.preupdate_cons(0)
      for j in range(subsub):
        pbd.update_cons(0)
      pbd.update_vel()
    update_display_points()

    window.pre_update()
    window.render()
    window.show()

  window.terminate()

if __name__ == '__main__':
  args = arg_parser.get_args('config/simple_sphere_tet.yaml')
  test_tet_mesh(args)