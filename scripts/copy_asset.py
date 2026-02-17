import os
import shutil
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(project_root)
from utils.obj_utils import face_normal_to_vertex_normal

def copy_asset(src_dir, dst_dir, asset_name):
    os.system(f'cp -r {os.path.join(src_dir, asset_name)} {dst_dir}')

if __name__ == '__main__':
    src_dir = f'/mnt/f/Data/MRSurgery/MRSurgery_Sim/assets/'
    dst_dir = f'./assets/'
    asset_name = 'sphere_10k'
    copy_asset(src_dir, dst_dir, asset_name)
    conversion_applied = face_normal_to_vertex_normal(f'./assets/{asset_name}/{asset_name}.obj', f'./assets/{asset_name}/{asset_name}.obj')
    if conversion_applied:
        copy_asset(dst_dir, src_dir, asset_name)
        print(f'Normal conversion applied to {asset_name}, copy back to source directory')
