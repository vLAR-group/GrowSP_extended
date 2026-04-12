import open3d as o3d
import numpy as np
import os
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere
from glob import glob


scene_name = ['birmingham_block_1_1', 'birmingham_block_1_2', 'birmingham_block_3_0', 'cambridge_block_10_0', 'cambridge_block_26_0', 'cambridge_block_26_1']
for meth in ['growsp', 'GT', 'input', 'pointdc_dinov2', 'pointdc', 'ours', 'IIC', 'PICIE']:
    collect_scene = []
    folder = os.path.join('Urban3D', meth)
    scene_names = sorted(glob(os.path.join('/home/zihui/SSD2/GrowSP++_vis/SensatUrban/', meth, '*.ply')))
    for s in scene_names:
        for s2 in scene_name:
            if s2 in s:
                collect_scene.append(s)

    out_foler = os.path.join('objs', folder)

    for scene in collect_scene:
        data = read_ply(scene)
        print(len(data))
        # data = data[np.random.choice(len(data), len(data)//5, replace=False)]
        points = data[:, 0:3]
        points = points - points.mean(0, keepdims=True)
        colors = data[:, 3:6]

        # mesh = pc_segm_to_sphere(points, segm=labels, radius=0.01, resolution=3, with_background=False, default_color=colors)### 0.02/0.03 radius for ScanNet
        mesh = pc_segm_to_sphere(points, segm=np.arange(len(data)), radius=0.2, resolution=3, with_background=False, default_color=colors)### 0.02/0.03 radius for ScanNet, 0.02/0.01 for s3dis, maybe 0.1 for semkitti?
        # o3d.visualization.draw_geometries([mesh])

        os.makedirs(out_foler, exist_ok=True)
        save_file = os.path.join(out_foler, scene.split('/')[-1][0:-4] + '0.05.obj')
        o3d.io.write_triangle_mesh(save_file, mesh)