import meshio
import os
import glob

def obj_to_ply(obj_path, ply_path):
    mesh = meshio.read(obj_path)
    meshio.write(ply_path, mesh, file_format="ply", binary=False)

if __name__ == "__main__":
    asset_name = "simple_sphere_example"   
    obj_path = f"assets/{asset_name}/{asset_name}.obj"
    ply_path = f"assets/{asset_name}/{asset_name}.ply"
    obj_to_ply(obj_path, ply_path)
    tetgen_path = "dependencies/TetGen/build/tetgen"
    os.system(f"{tetgen_path} -pq -g {ply_path}")
    os.rename(f"assets/{asset_name}/{asset_name}.1.mesh", f"assets/{asset_name}/{asset_name}.mesh")
    for f in glob.glob(f"assets/{asset_name}/{asset_name}.1.*"):
        os.remove(f)
        print(f"Removed {f}")
    
    