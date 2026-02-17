## Usage


1. Triangle mesh
Convert face normals to vertex normals in `utils/obj_utils.py` before loading the mesh.
Create a config file in `config/`, change the filename in `test/test_mesh.py` and run `python -m test.test_mesh`.

2. Tet mesh
Convert obj to tet mesh using `scripts/obj_to_tet_mesh.py`.
Create a config file in `config/`, change the filename in `test/test_tet_mesh.py` and run `python -m test.test_tet_mesh`.

## Points Data
tgf: control points
weights: n_vertices * n_control_points

