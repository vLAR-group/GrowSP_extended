import open3d as o3d
import numpy as np
import colorsys, random, os, sys
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere
import os.path as osp


########### Indoor Scene #####################
file = '/home/user/SSD/ndf/0000_00.ply'
data = read_ply(file)
points = data[:, 0:3]
colors = data[:, 3:]

# colors = np.vstack((data['red'], data['green'], data['blue'])).T
# labels = data[:, 6].astype(np.int32)
# mask = (labels != 0)&(labels!=12)
# mask = labels != -1
# points, colors, labels = points[mask], colors[mask], labels[mask]
# points, labels = points[mask], labels[mask]


# file2 = 'vis/growing_sp_full/Area_3_conferenceRoom_1.ply'
# data2 = read_ply(file2)
# colors = data[:, 3:6]
# colors = colors[mask]
#
# mask2 = colors[:,1]>30
# points, labels, colors = points[mask2], labels[mask2], colors[mask2]

# coords = points
# bound_min = np.min(coords, 0).astype(float)
# bound_max = np.max(coords, 0).astype(float)
# bound_size = bound_max - bound_min
# center = bound_min + bound_size * 0.5
# lim = 4
#
# clip_inds = ((coords[:, 0] >= (-lim + center[0])) & \
#              (coords[:, 0] < (lim + center[0])) & \
#              (coords[:, 1] >= (-lim + center[1])) & \
#              (coords[:, 1] < (lim + center[1])) & \
#              (coords[:, 2] >= (-lim + center[2])) & \
#              (coords[:, 2] < (lim + center[2])))
#
# points, colors = points[clip_inds], colors[clip_inds]

# mesh = pc_segm_to_sphere(points, segm=labels, radius=0.01, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
mesh = pc_segm_to_sphere(points, segm=np.arange(len(data)), radius=0.05, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
o3d.visualization.draw_geometries([mesh])

# save_path = 'video/s3dis/Area_3_conferenceRoom_1/grow_sp/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
save_path = 'tmp2'
save_file = osp.join(save_path+'.obj')
o3d.io.write_triangle_mesh(save_file, mesh)


# ############### SemanticKITTI ######################################3
# raw_data = 'data/SemanticKITTI/dataset/sequences/08/raw_ply/'
# sp_path = '/home/user/SSD2/unsup_seg/aligned_kitti_unsupervised_supervoxel&regiongrowing2/08/raw_ply/'
#
# for f in np.sort(os.listdir(raw_data)):
#     if int(f[:-4])>=3800 and int(f[:-4])<= 4000:
#         file = os.path.join(raw_data, f)
#         data = read_ply(file)
#         points = data[:, 0:3]
#         points -= points.mean(0)
#     # colors = np.vstack((data['red'], data['green'], data['blue'])).T
#     # colors = np.ones_like(colors)*128
#         labels = data[:, 6].astype(np.int32)
#         colors = np.ones_like(points)*128
#
#     # mask = labels != 0
#     # points, colors, labels = points[mask], colors[mask], labels[mask]
#     # points, labels = points[mask], labels[mask]
#
#     # file2 = '/home/user/SSD2/unsup_seg/vis/preds_kitti/ours/08/003000GT.ply'
#     #     file2 = sp_path + f[:-4]+'.ply'
#     #     data2 = read_ply(file2)
#     #     colors = np.vstack((data2['red'], data2['green'], data2['blue'])).T
#         mask2 = np.sqrt(((points) ** 2).sum(-1)) < 20
#         colors = colors[mask2]
#         points, colors, labels = points[mask2], colors[mask2], labels[mask2]
#         mask = labels != 0
#         points, colors, labels = points[mask], colors[mask], labels[mask]
#
#         mesh = pc_segm_to_sphere(points, segm=labels, radius=0.1, resolution=3, with_background=False, default_color=colors)
#
#         save_path = 'video/kitti_20/initial_3700-4100/input/'
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#
#         save_file = osp.join(save_path+f[:-4]+'.obj')
#         o3d.io.write_triangle_mesh(save_file, mesh)
