import open3d as o3d
import numpy as np
import colorsys, random, os, sys
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere
import os.path as osp
import trimesh
from glob import glob

# scene_id_list = ['04_000800', '05_000828', '06_000838', '07_000975', '08_000996', '08_000998', '08_000999']
scene_id_list = ['07_000927']
# all_scene_names = sorted(glob(os.path.join('/home/zihui/SSD/ndf/baseline_safe2', 'EFEM','*.ply')))
all_scene_names = sorted(glob(os.path.join('/home/zihui/SSD/ndf/ckpt_safe_scene2/HDBSCAN_500_1/vis', '*.ply')))

out_foler = 'safe2_0.4_add/HDBSCAN'
scene_names = []
for scene in all_scene_names:
    for scene_id in scene_id_list:
        if scene_id in scene:
            scene_names.append(scene)

for scene in scene_names:

    data = read_ply(scene)
    print(len(data))
    # data = data[np.random.choice(len(data), len(data)//5, replace=False)]
    points = data[:, 0:3]
    points = points - points.mean(0, keepdims=True)
    colors = data[:, 3:6]

    mesh = pc_segm_to_sphere(points, segm=np.arange(len(data)), radius=0.04, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
    # mesh = pc_segm_to_sphere(points, radius=0.04, resolution=3, with_background=False)### 0.05 radius for ScanNet
    # o3d.visualization.draw_geometries([mesh])

    os.makedirs(out_foler, exist_ok=True)
    save_file = os.path.join(out_foler, scene.split('/')[-1][0:-4] + '0.04_3.obj')
    o3d.io.write_triangle_mesh(save_file, mesh)
