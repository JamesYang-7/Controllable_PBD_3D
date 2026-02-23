import os
import meshio
import glob
import argparse
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(project_root)
from utils.obj_utils import face_normal_to_vertex_normal

def copy_asset(src_dir, dst_dir, asset_name):
    os.system(f'cp -r {os.path.join(src_dir, asset_name)} {dst_dir}')

def obj_to_ply(obj_path, ply_path):
    mesh = meshio.read(obj_path)
    meshio.write(ply_path, mesh, file_format="ply", binary=False)

def tri_to_tet(asset_name):
    """
    Convert .obj mesh to .mesh format using TetGen."""
    obj_path = f"assets/{asset_name}/{asset_name}.obj"
    ply_path = f"assets/{asset_name}/{asset_name}.ply"

    # Compute average triangle area and derive target regular tetrahedron volume
    mesh = meshio.read(obj_path)
    verts = mesh.points
    faces = mesh.cells_dict['triangle']
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    avg_area = np.mean(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1))
    # Regular tet: face area A = (sqrt(3)/4)*a^2  =>  a = sqrt(4A/sqrt(3))
    # Volume = a^3 * sqrt(2) / 12
    a = np.sqrt(4 * avg_area / np.sqrt(3))
    tet_volume = a**3 * np.sqrt(2) / 12

    obj_to_ply(obj_path, ply_path)
    tetgen_path = "dependencies/TetGen/build/tetgen"
    os.system(f"{tetgen_path} -pqv{tet_volume:.6f} -g {ply_path}") # call TetGen to generate .mesh file
    os.rename(f"assets/{asset_name}/{asset_name}.1.mesh", f"assets/{asset_name}/{asset_name}.mesh")
    for f in glob.glob(f"assets/{asset_name}/{asset_name}.1.*"):
        os.remove(f)
        print(f"Removed {f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('asset_name', type=str, help='Name of the asset to copy')
    args = parser.parse_args()

    src_dir = f'/mnt/f/Data/MRSurgery/MRSurgery_Sim/assets/'
    dst_dir = f'./assets/'
    asset_name = args.asset_name
    copy_asset(src_dir, dst_dir, asset_name) # copy to local directory
    conversion_applied = face_normal_to_vertex_normal(f'./assets/{asset_name}/{asset_name}.obj', f'./assets/{asset_name}/{asset_name}.obj')
    tri_to_tet(asset_name) # convert to tet mesh
    if conversion_applied:
        copy_asset(dst_dir, src_dir, asset_name)
        print(f'Normal conversion applied to {asset_name}, copy back to source directory')
    

