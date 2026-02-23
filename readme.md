## Usage


#### Triangle mesh
- Convert face normals to vertex normals in `utils/obj_utils.py` before loading the mesh.
- Create a config file in `config/` and run `python -m test.test_pbd_mesh --cfg <cfg_file_path>`.

#### Tet mesh
- Convert obj to tet mesh using `scripts/obj_to_tet_mesh.py`.
- Create a config file in `config/` and run `python -m test.test_pbd_mesh --cfg <cfg_file_path>`.

## Points Data
tgf: control points
weights: n_vertices * n_control_points

