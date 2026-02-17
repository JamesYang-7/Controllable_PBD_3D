import numpy as np
import math

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