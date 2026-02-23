import taichi as ti
from data import tet_data, cloth_data
from cons import deform3d, length, bend, volume, framework
from utils import arg_parser, anim

ti.init(arch=ti.vulkan)

def test_pbd_mesh(args):
  '''
  Test simulation of mesh with constraints.
  Uses tet_data if asset ends with .mesh, otherwise cloth_data.
  '''
  # Load args
  dir_name = args.dir_name
  obj_name = args.obj_name
  fixed_id = args.fixed_id
  operating_points_id = args.operating_points_id
  arg_cons = args.cons
  gravity = args.common.gravity
  fps = args.common.fps
  substeps = args.common.substeps
  subsub = args.common.subsub
  damp = args.common.damp
  scale = args.scale
  repose = args.repose
  speed = args.speed
  movement_direction = args.movement_direction

  # Load mesh data based on file suffix
  mesh_path = f'assets/{dir_name}/{obj_name}'
  if mesh_path.endswith('.mesh'):
    mesh = tet_data.load_tets(mesh_path, scale, repose, remove_duplicate=False)
    use_tet = True
  else:
    mesh = cloth_data.load_cloth_mesh(mesh_path, scale, repose,
                                      remove_duplicate=False, reverse_side=False)
    use_tet = False

  invm = mesh.v_invm.to_numpy()
  invm[fixed_id] = 0
  mesh.v_invm.from_numpy(invm)

  # Set display points
  wireframe = [True]
  _, particles_static, particles_dynamic, update_display_points = \
      anim.setup_display_points(mesh.v_p, fixed_id, operating_points_id)

  # Init simulation
  g = ti.Vector(gravity)
  dt = 1.0 / fps / substeps
  pbd = framework.pbd_framework(mesh.v_p, g, dt, damp=damp)

  if use_tet:
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
  else:
    if arg_cons.length.enabled:
      cons_length = length.LengthCons(v_p=mesh.v_p,
                                      v_p_ref=mesh.v_p_ref,
                                      e_i=mesh.e_i,
                                      e_rl=mesh.e_rl,
                                      v_invm=mesh.v_invm,
                                      dt=dt,
                                      alpha=float(arg_cons.length.alpha),
                                      method=arg_cons.length.method)
      pbd.add_cons(cons_length, 0)
    if arg_cons.bend.enabled:
      cons_bend = bend.Bend3D(v_p=mesh.v_p,
                              v_p_ref=mesh.v_p_ref,
                              e_i=mesh.e_i,
                              e_s=mesh.e_s,
                              v_invm=mesh.v_invm,
                              dt=dt,
                              alpha=float(arg_cons.bend.alpha))
      pbd.add_cons(cons_bend, 0)
    if arg_cons.volume.enabled:
      cons_volume = volume.Volume(v_p=mesh.v_p,
                                  v_p_ref=mesh.v_p_ref,
                                  f_i=mesh.f_i,
                                  v_invm=mesh.v_invm,
                                  dt=dt,
                                  alpha=float(arg_cons.volume.alpha))
      pbd.add_cons(cons_volume, 0)

  pbd.init_rest_status(0)

  # Init interface
  window = anim.setup_window(mesh, wireframe, particles_static, particles_dynamic,
                                  scale, args, title='PBD Mesh Simulation')

  while window.running():
    anim.set_movement(window, mesh, operating_points_id, movement_direction, scale, speed)
    for _ in range(substeps):
      pbd.step(subsub)
    update_display_points()

    window.pre_update()
    window.render()
    window.show()

  window.terminate()

if __name__ == '__main__':
  args = arg_parser.get_args()
  test_pbd_mesh(args)
