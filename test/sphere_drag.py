import taichi as ti
import numpy as np

from interface import render_funcs, mesh_render_3d
from data import cloth_data
from cons import length, bend, volume, framework

from interface import usd_objs, usd_render
from utils import objs, geom2d

import time

ti.init(arch=ti.cpu)

# ========================== load data ==========================
modelname = 'sphere'
model_path = f'assets/{modelname}/{modelname}.obj'
scale = 1.0
repose = (0.0, 0.0, 0.0)

mesh = cloth_data.load_cloth_mesh(model_path, scale, repose, remove_duplicate=False)

markers = [[4047, 4038, 4039, 4040, 4041], [4048, 4034, 4035, 4036, 4037], [4046, 4042, 4043, 4044, 4045]]
marker_positions = np.load(f"assets/{modelname}/marker_positions.npy")
n_frames = marker_positions.shape[0]
fixed = np.concatenate(markers)
display_points_indices = [markers[0][0], markers[1][0], markers[2][0]]
corres = display_points_indices
invm = mesh.v_invm.to_numpy()
invm[fixed] = 0
mesh.v_invm.from_numpy(invm)
display_points = ti.Vector.field(3, dtype=ti.f32, shape=len(display_points_indices))

def update_display_points():
  for i in range(len(display_points_indices)):
    for k in range(3):
      display_points[i][k] = mesh.v_p[display_points_indices[i]][k]


wireframe = [True]

# ========================== init simulation ==========================
g = ti.Vector([0.0, 0.0, 0.0])
fps = 60
substeps = 5
subsub = 1
dt = 1.0 / fps / substeps

pbd = framework.pbd_framework(mesh.v_p, g, dt, damp=0.993)
cons_length = length.LengthCons(v_p=mesh.v_p,
                                v_p_ref=mesh.v_p_ref,
                                e_i=mesh.e_i,
                                e_rl=mesh.e_rl,
                                v_invm=mesh.v_invm,
                                dt=dt,
                                alpha=1e-2)
cons_bend = bend.Bend3D(v_p=mesh.v_p,
                        v_p_ref=mesh.v_p_ref,
                        e_i=mesh.e_i,
                        e_s=mesh.e_s,
                        v_invm=mesh.v_invm,
                        dt=dt,
                        alpha=100)
cons_volume = volume.Volume(v_p=mesh.v_p,
                            v_p_ref=mesh.v_p_ref,
                            f_i=mesh.f_i,
                            v_invm=mesh.v_invm,
                            dt=dt,
                            alpha=1.0)
pbd.add_cons(cons_length, 0)
# pbd.add_cons(cons_bend, 0)
# pbd.add_cons(cons_volume, 0)

# ground = objs.Quad(axis1=(10.0, 0.0, 0.0), axis2=(0.0, 0.0, -10.0), pos=0.0)
# pbd.add_collision(ground.collision)

# ========================== init interface ==========================
window = mesh_render_3d.MeshRender3D(res=(1000, 1000),
                                     title='sphere',
                                     kernel='taichi')
# window.add_render_func(ground.get_render_draw())
window.add_render_func(
    render_funcs.get_mesh_render_func(mesh.v_p,
                                      mesh.f_i,
                                      wireframe,
                                      color=(14 / 255, 87 / 255, 204 / 255)))
window.add_render_func(
    render_funcs.get_particles_render_func(display_points,
                                           color=(1.0, 0.0, 0.0),
                                           radius=0.04))
window.set_camera(eye=(2.5, 0.8, 0), center=(-0.5, -0.5, 0), up=(0.0, 1.0, 0.0))

# ========================== init status ==========================
pbd.init_rest_status(0)

# ========================== use input ==========================
import math

written = [True]

last_t = 0
cnt = 0
v_p_seq = []
def set_movement():
  global last_t, cnt, v_p_seq
  t = window.get_time() - 1.0
  delta_t = t - last_t
  if t < 0.0:
    return
  v_p = mesh.v_p.to_numpy()
  if delta_t >= 0.057:
    last_t = t
    for i in range(len(markers)):
      trans = marker_positions[cnt % n_frames][i] - v_p[markers[i][0]]
      for v_marker in markers[i]:
        v_p[v_marker] += trans
    # if cnt < n_frames:
    #   v_p_seq.append(np.copy(v_p))
    # elif cnt == n_frames:
    #   v_p_seq = np.stack(v_p_seq, axis=0)
    #   np.save("v_p_seq.npy", v_p_seq)
    #   print("write to v_p_seq.npy", v_p_seq.shape)
    cnt += 1
  else:
    return

  if abs(0.5 * t - 1.25) < 1e-2 and not written[0]:
    import meshio
    meshio.Mesh(v_p, [("triangle", mesh.faces_np.reshape(-1, 3))
                     ]).write("out/sphere_drag.obj")
    print("write to outputs/sphere_drag.obj")
    print(v_p[corres])
    written[0] = True
  mesh.v_p.from_numpy(v_p)


# ========================== USD ==========================
save_usd = False
if save_usd:
  stage = usd_render.UsdRender('out/sphere_drag.usdc',
                               startTimeCode=1,
                               endTimeCode=600,
                               fps=60,
                               UpAxis='Y')
  cage_point_color = np.zeros((len(fixed), 3), dtype=np.float32)
  cage_point_color[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
  usd_cage_points = usd_objs.SavePoints(stage.stage,
                                        '/root/cage_points',
                                        verts_np=mesh.v_p.to_numpy()[fixed],
                                        radius=0.05,
                                        per_vert_color=cage_point_color)
  usd_mesh = usd_objs.SaveMesh(stage.stage, '/root/mesh', mesh.verts_np,
                               mesh.faces_np)

  def update_usd(frame: int):
    if frame < stage.startTimeCode or frame > stage.endTimeCode:
      return
    usd_cage_points.update(mesh.v_p.to_numpy()[fixed], frame)
    usd_mesh.update(mesh.v_p.to_numpy(), frame)
    print("update usd file at frame", frame)


t_total = 0.0

while window.running():
  t = time.time()
  set_movement()

  for i in range(substeps):
    pbd.make_prediction()
    pbd.preupdate_cons(0)
    for j in range(subsub):
      pbd.update_cons(0)
    pbd.update_vel()
  t_total += time.time() - t
  if window.get_total_frames() == 480:
    print(f'average time: {t_total / 480}')

  update_display_points()

  if save_usd:
    update_usd(window.get_total_frames())

  window.pre_update()
  window.render()
  window.show()

window.terminate()
if save_usd:
  stage.save()