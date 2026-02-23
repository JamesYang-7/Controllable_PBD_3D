import taichi as ti
from data import tet_data
from cons.vbd import VBDSolver
from utils import arg_parser, anim

ti.init(arch=ti.vulkan)


def test_vbd(args):
  # Load args
  dir_name = args.dir_name
  obj_name = args.obj_name
  fixed_id = args.fixed_id
  operating_points_id = args.operating_points_id
  gravity = args.common.gravity
  fps = args.common.fps
  substeps = args.common.substeps
  damp = args.common.damp
  scale = args.scale
  repose = args.repose
  speed = args.speed
  movement_direction = args.movement_direction

  # VBD params
  vbd_cfg = args.vbd
  youngs_modulus = float(vbd_cfg.get('youngs_modulus', 1e4))
  poissons_ratio = float(vbd_cfg.get('poissons_ratio', 0.3))
  n_iterations = int(vbd_cfg.get('n_iterations', 10))
  damping = float(vbd_cfg.get('damping', 0.0))
  rho = float(vbd_cfg.get('rho', 0.0))
  method = str(vbd_cfg.get('method', 'serial'))

  # Load mesh
  mesh_path = f'assets/{dir_name}/{obj_name}'
  mesh = tet_data.load_tets(mesh_path, scale, repose, remove_duplicate=False)
  invm = mesh.v_invm.to_numpy()
  invm[fixed_id] = 0
  mesh.v_invm.from_numpy(invm)

  # Display points
  wireframe = [True]
  _, particles_static, particles_dynamic, update_display_points = \
      anim.setup_display_points(mesh.v_p, fixed_id, operating_points_id)

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
  window = anim.setup_window(mesh, wireframe, particles_static, particles_dynamic,
                                  scale, args, title='VBD Tet Mesh')

  while window.running():
    anim.set_movement(window, mesh, operating_points_id, movement_direction, scale, speed)
    for _ in range(substeps):
      solver.step()
    update_display_points()

    window.pre_update()
    window.render()
    window.show()

  window.terminate()


if __name__ == '__main__':
  args = arg_parser.get_args('config/sphere_10k_tet_vbd.yaml')
  test_vbd(args)
