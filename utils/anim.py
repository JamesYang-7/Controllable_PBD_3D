import numpy as np
import math
import taichi as ti
from interface import render_funcs, mesh_render_3d

def setup_display_points(v_p, fixed_id, operating_points_id):
  '''
  Create Taichi fields for rendering fixed and operating points.
  Returns (static_points_id, particles_static, particles_dynamic, update_fn).
  '''
  static_points_id = [x for x in fixed_id if x not in operating_points_id]
  particles_static = ti.Vector.field(3, dtype=ti.f32, shape=len(static_points_id))
  particles_dynamic = ti.Vector.field(3, dtype=ti.f32, shape=len(operating_points_id))

  def update_display_points():
    for i in range(len(static_points_id)):
      particles_static[i] = v_p[static_points_id[i]]
    for i in range(len(operating_points_id)):
      particles_dynamic[i] = v_p[operating_points_id[i]]

  return static_points_id, particles_static, particles_dynamic, update_display_points


def setup_window(mesh, wireframe, particles_static, particles_dynamic, scale, args, title):
  '''
  Create and configure a MeshRender3D window with standard camera, lighting, and render funcs.
  '''
  window_height = args.get('window_height', 700)
  window_width = args.get('window_width', 700)
  cam_eye = args.get('cam_eye', (2.0 * scale, 2.0 * scale, 2.0 * scale))
  cam_center = args.get('cam_center', (0.0, 0.0, 0.0))
  cam_up = args.get('cam_up', (0.0, 1.0, 0.0))

  window = mesh_render_3d.MeshRender3D(res=(window_width, window_height),
                                       title=title,
                                       kernel='taichi')
  window.set_background_color((1, 1, 1, 1))
  window.set_camera(eye=cam_eye, center=cam_center, up=cam_up)
  window.set_lighting((4, 4, -4), (0.96, 0.96, 0.96), (0.2, 0.2, 0.2))
  window.add_render_func(
      render_funcs.get_mesh_render_func(mesh.v_p, mesh.f_i, wireframe, color=(0.0, 0.0, 1.0)))
  window.add_render_func(
      render_funcs.get_particles_render_func(particles_static,
                                             color=(0.5, 0.5, 0.5),
                                             radius=0.02 * scale))
  window.add_render_func(
      render_funcs.get_particles_render_func(particles_dynamic,
                                             color=(0.0, 1.0, 0.0),
                                             radius=0.02 * scale))
  return window


def set_movement(window, mesh, operating_points_id, direction, scale, speed):
  '''
  Set the movement of the operating points
  '''
  t = window.get_time() - 1.0
  if t < 0.0:
    return
  p_input = mesh.v_p.to_numpy()
  p_input_ref = mesh.v_p_ref.to_numpy()
  p_input[operating_points_id] = p_input_ref[operating_points_id] + np.array(direction, dtype=np.float32) * math.sin(
      speed * t * (2.0 * math.pi)) * scale * 0.1
  mesh.v_p.from_numpy(p_input)