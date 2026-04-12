import open3d as o3d
import numpy as np
import colorsys, random, os, sys
from glob import glob
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere

out_root = 'GrowSP++_vis/primitives_S3DIS_obj2'
ply_files = []
for root, dirs, files in os.walk('/home/zihui/SSD2/GrowSP++_vis/primitives_S3DIS2/'):
    for file in files:
        if file.endswith('.ply'):
            # Construct the full file path
            file_path = os.path.join(root, file)
            ply_files.append(file_path)

for scene in ply_files:
    data = read_ply(scene)
    print(len(data))
    data = data[np.random.choice(len(data), len(data)//2, replace=False)]
    points = data[:, 0:3]
    points = points - points.mean(0, keepdims=True)
    colors = data[:, 3:6]

    # mesh = pc_segm_to_sphere(points, segm=labels, radius=0.01, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
    mesh = pc_segm_to_sphere(points, segm=np.arange(len(data)), radius=0.02, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
    # o3d.visualization.draw_geometries([mesh])

    out_folder = os.path.join(out_root, scene.split('/')[-2])
    os.makedirs(out_folder, exist_ok=True)
    save_file = os.path.join(out_folder, scene.split('/')[-1][0:-4] + '0.02.obj')
    o3d.io.write_triangle_mesh(save_file, mesh)
