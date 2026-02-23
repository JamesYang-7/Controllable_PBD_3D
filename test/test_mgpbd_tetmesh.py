import taichi as ti
from data import tet_data
from cons.mgpbd import MGPBDSolver
from utils import arg_parser, anim

ti.init(arch=ti.vulkan)


def test_mgpbd_tetmesh(args):
  '''
  Test simulation of tet mesh with MGPBD solver (Multigrid-accelerated Global XPBD).
  '''
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

  # MGPBD params
  mgpbd_cfg = args.mgpbd
  mu = float(mgpbd_cfg.get('mu', 1e6))
  maxiter = int(mgpbd_cfg.get('maxiter', 20))
  atol = float(mgpbd_cfg.get('atol', 1e-4))
  rtol = float(mgpbd_cfg.get('rtol', 1e-2))
  maxiter_cg = int(mgpbd_cfg.get('maxiter_cg', 100))
  tol_cg = float(mgpbd_cfg.get('tol_cg', 1e-5))
  n_smooth = int(mgpbd_cfg.get('n_smooth', 2))
  omega_jacobi = float(mgpbd_cfg.get('omega_jacobi', 2.0 / 3.0))
  setup_interval = int(mgpbd_cfg.get('setup_interval', 10000))
  use_line_search = bool(mgpbd_cfg.get('use_line_search', True))

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

  # Init MGPBD solver
  g = ti.Vector(gravity)
  dt = 1.0 / fps / substeps
  solver = MGPBDSolver(v_p=mesh.v_p,
                       v_p_ref=mesh.v_p_ref,
                       v_invm=mesh.v_invm,
                       t_i=mesh.t_i,
                       t_m=mesh.t_m,
                       gravity=g,
                       dt=dt,
                       mu=mu,
                       damp=damp,
                       maxiter=maxiter,
                       atol=atol,
                       rtol=rtol,
                       maxiter_cg=maxiter_cg,
                       tol_cg=tol_cg,
                       n_smooth=n_smooth,
                       omega_jacobi=omega_jacobi,
                       setup_interval=setup_interval,
                       use_line_search=use_line_search)
  solver.init()

  # Init interface
  window = anim.setup_window(mesh, wireframe, particles_static, particles_dynamic,
                                  scale, args, title='MGPBD Tet Mesh')

  frame = 0
  while window.running():
    anim.set_movement(window, mesh, operating_points_id, movement_direction, scale, speed)
    for _ in range(substeps):
      solver.step(frame)
    update_display_points()
    frame += 1

    window.pre_update()
    window.render()
    window.show()

  window.terminate()


if __name__ == '__main__':
  args = arg_parser.get_args('config/simple_sphere_tet_mgpbd.yaml')
  test_mgpbd_tetmesh(args)
