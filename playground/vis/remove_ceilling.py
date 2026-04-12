import open3d as o3d
import numpy as np
import colorsys, random, os, sys
from lib.helper_ply import read_ply
import os.path as osp
from visual_util import pc_segm_to_sphere
from glob import glob


########### Indoor Scene #####################
input_path = '/home/user/SSD2/GrowSP_extension/data/S3DIS/input/'
sp_path = '/home/user/SSD2/GrowSP_extension/data/S3DIS/initial_superpoints_RegionGrowing/'

''' Reading Data'''
folders = sorted(glob(input_path + '/*.ply'))
for _, file in enumerate(folders):
    if 'Area_1_conferenceRoom_2' in file or 'Area_3_conferenceRoom_1' in file or 'Area_4_conferenceRoom_3' in file or 'Area_6_conferenceRoom_1' in file:
        name = file.replace(input_path, '')[0:-4]
        spname = os.path.join(sp_path, 'vis', name+'.ply')

        label = read_ply(file)['class'].astype(np.int32)
        data = read_ply(spname)
        xyz, color = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T
        mask = (label != 0)&(label!=12)

        mesh = pc_segm_to_sphere(xyz[mask], segm=color[mask], radius=0.01, resolution=3, with_background=False, default_color=color[mask])### 0.05 radius for ScanNet
        # o3d.visualization.draw_geometries([mesh])

        save_path = 'spvis_conf/RegionGrowing/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_file = osp.join(save_path, name+'.obj')
        o3d.io.write_triangle_mesh(save_file, mesh, print_progress=True)