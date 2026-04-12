import open3d as o3d
import numpy as np
import colorsys, random, os, sys
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere
import os.path as osp
from glob import glob


disabled_name = ['bildstein_station5_xyz_intensity_rgb', 'domfountain_station2_xyz_intensity_rgb', 'domfountain_station3_xyz_intensity_rgb', 'neugasse_station1_xyz_intensity_rgb',
                 'sg27_station1_intensity_rgb', 'sg27_station5_intensity_rgb', 'sg27_station9_intensity_rgb']

dataset_path = '/home/user/SSD2/GrowSP_extension/visualizations/visualizations/Semantic3d/'
save_path = '/home/user/SSD2/semantic3d_obj/'

folders = np.sort(os.listdir(dataset_path))
for folder in folders:
    path = os.path.join(dataset_path, folder+'/')
    plyfiles = sorted(glob(path + '/*.ply'))
    for plyfile in plyfiles:
        name = plyfile.replace(os.path.join(dataset_path, folder+'/'), '')[:-4]
        process = False
        if 'GT' in name:
            if name[:-2] not in disabled_name:
                process = True
        else:
            if name not in disabled_name:
                process = True
        if process:
            print(name)
            data = read_ply(plyfile)
            coords, colors = data[:, 0:3], data[:, 3:6]
            labels = -np.ones(len(coords))

            mesh = pc_segm_to_sphere(coords, segm=labels, radius=0.1, resolution=3, with_background=False,
                                     default_color=colors)  ### 0.05 radius for ScanNet
            # o3d.visualization.draw_geometries([mesh])


            # save_path = os.path.join(save_path, plyfile.replace(dataset_path, '')[:-4])
            save_folder = os.path.join(save_path, plyfile.split('/')[-2])
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_file = osp.join(save_folder, name + '.obj')
            o3d.io.write_triangle_mesh(save_file, mesh)


# disabled_name = []
#
# dataset_path = '/home/user/SSD2/GrowSP_extension/visualizations/visualizations/SensatUrban'
# save_path = '/home/user/SSD2/SensatUrban_obj_0.5/'
# present_name = ['cambridge_block_7_0', 'cambridge_block_7_1', 'cambridge_block_7_2', 'cambridge_block_7_3', 'cambridge_block_10_3',
#                 'cambridge_block_7_0GT', 'cambridge_block_7_1GT', 'cambridge_block_7_2GT', 'cambridge_block_7_3GT', 'cambridge_block_10_3GT',
#                 'cambridge_block_7_0input', 'cambridge_block_7_1input', 'cambridge_block_7_2input', 'cambridge_block_7_3input', 'cambridge_block_10_3input']
#
# folders = np.sort(os.listdir(dataset_path))
# for folder in folders:
#     # if folder == 'GrowSP_0.5':
#     path = os.path.join(dataset_path, folder+'/')
#     plyfiles = sorted(glob(path + '/*.ply'))
#     for plyfile in plyfiles:
#         name = plyfile.replace(os.path.join(dataset_path, folder+'/'), '')[:-4]
#         if name in present_name:
#             process = False
#             if 'GT' in name:
#                 if name[:-2] not in disabled_name:
#                     process = True
#             else:
#                 if name not in disabled_name:
#                     process = True
#             if process:
#                 print(name)
#                 data = read_ply(plyfile)
#                 coords, colors = data[:, 0:3], data[:, 3:6]
#                 labels = -np.ones(len(coords))
#
#                 mesh = pc_segm_to_sphere(coords, segm=labels, radius=0.5, resolution=3, with_background=False,
#                                          default_color=colors)  ### 0.05 radius for ScanNet
#                 # o3d.visualization.draw_geometries([mesh])
#
#
#                 # save_path = os.path.join(save_path, plyfile.replace(dataset_path, '')[:-4])
#                 save_folder = os.path.join(save_path, plyfile.split('/')[-2])
#                 if not os.path.exists(save_folder):
#                     os.makedirs(save_folder)
#
#                 save_file = osp.join(save_folder, name + '.obj')
#                 o3d.io.write_triangle_mesh(save_file, mesh, print_progress=True)
