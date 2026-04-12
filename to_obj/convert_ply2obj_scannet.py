import open3d as o3d
import numpy as np
import os
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere
from glob import glob


# scene_name = ['/home/zihui/SSD/GrowSP_newExt/openning_fig/0000_00/growsp.ply', '/home/zihui/SSD/GrowSP_newExt/openning_fig/0000_00/gt.ply',
#               '/home/zihui/SSD/GrowSP_newExt/openning_fig/0000_00/initsp.ply', '/home/zihui/SSD/GrowSP_newExt/openning_fig/0000_00/input.ply',
#               '/home/zihui/SSD/GrowSP_newExt/openning_fig/0000_00/preds.ply']
# scene_name = ['/home/zihui/SSD/GrowSP_newExt/initsp/0465_00/input.ply', '/home/zihui/SSD/GrowSP_newExt/initsp/0465_00/sp.ply']
path = '/home/zihui/SSD/GrowSP_newExt/draw_sp'
save_path = os.path.join(path, 'obj')
os.makedirs(save_path, exist_ok=True)
scene_name = glob(os.path.join(path, '*.ply'))

for scene in scene_name:
    data = read_ply(scene)
    print(len(data))
    points = data[:, 0:3]
    points = points - points.mean(0, keepdims=True)
    colors = data[:, 3:6]

    # mesh = pc_segm_to_sphere(points, segm=labels, radius=0.01, resolution=3, with_background=False, default_color=colors)### 0.02/0.03 radius for ScanNet
    mesh = pc_segm_to_sphere(points, segm=np.arange(len(data)), radius=0.05, resolution=5, with_background=False, default_color=colors)### 0.04/0.05 radius for ScanNet, 0.02/0.01 for s3dis, maybe 0.1 for semkitti?
    # o3d.visualization.draw_geometries([mesh])
    save_file = os.path.join(save_path, scene.split('/')[-1][0:-4] + '0.05.obj')
    o3d.io.write_triangle_mesh(save_file, mesh)