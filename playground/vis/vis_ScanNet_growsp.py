import torch
import torch.nn.functional as F
from datasets.ScanNet import Scannetvis, cfl_collate_fn_vis
import numpy as np
import random
import os
import spconv.pytorch as spconv
from models.fpn import Res16FPN18
from lib.vis_utils import construct_growing_superpoints, get_fixclassifier, construct_growing_primitive
from utils_degrowsp import get_fixclassifier
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from lib.helper_ply import read_ply, write_ply
import warnings
import argparse
import colorsys
from typing import List, Tuple
import functools
from distinctipy import distinctipy
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
colormap = []
for _ in range(30):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)[:, 0:3]
# @functools.lru_cache(20)
# def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
#     HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
#     return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))
# colormap = get_evenly_distributed_colors(400)
# colormap = distinctipy.get_colors(400)

seed = 2022
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
###
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='../data/ScanNet/processed/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= '../data/ScanNet/initial_superpoints/',
                        help='initial sp path')
    parser.add_argument('--save_path', type=str, default='/home/zihui/SSD/GrowSP_newExt/vis/ScanNet_growspNA/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=20, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()


def vis_preds(epoch, args):
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/ScanNet_growsp/primitive_grow_abl/distillv2_80-40sp_kmeans_elsaug_t3_fixmodel_multi0.9_nodecaylr_end30/model_300_checkpoint.pth'))
    model.eval()

    current_growsp = 400

    trainval_dataset = Scannetvis(args)
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    for i, data in enumerate(trainval_loader):
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region, full_coords, full_colors, full_labels = data

            ori_region = region.clone()

            cloud_name = trainval_loader.dataset.name[i]

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            valid_mask = region.squeeze()!=-1
            features = features.cuda()
            features = features[valid_mask]
            # normals = normals.cuda()
            # normals = normals[valid_mask]
            feats = feats[valid_mask]
            region = region[valid_mask].long()
            ##
            # pc_rgb = features[:, 0:3]
            # pc_xyz = features[:, 3:]*args.voxel_size
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            # region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
            # region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
            # region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

            region_feats= F.normalize(region_feats, dim=-1)
            # rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
            # region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
            # region_feats = torch.cat((region_feats, 3*region_rgb, 3*region_xyz), dim=-1)
            #
            # if region_feats.size(0)<current_growsp:
            #     current_growsp = region_feats.size(0)
            # grwosp_labels = torch.from_numpy(KMeans(n_clusters=current_growsp, n_init=10, random_state=0, n_jobs=10).fit_predict(region_feats.cpu().numpy())).long()
            grwosp_labels = region

            '''Visualization for Growing Superpoints'''
            grwosp_labels = grwosp_labels[region].squeeze()

            final_growsp_labels = ori_region.squeeze().numpy()
            final_growsp_labels[valid_mask] = grwosp_labels
            #

            colors = 255 * (np.array(colormap)[final_growsp_labels.astype(np.int32)])  # [:, 0:3]
            colors[~valid_mask] = np.zeros(3)
            colors = colors.astype(np.uint8)

            savepath = args.save_path
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            test_name = savepath + cloud_name + '.ply'
            write_ply(test_name, [full_coords.numpy(), colors[inverse_map]], ['x', 'y', 'z', 'red', 'green', 'blue'])


if __name__ == '__main__':
    args = parse_args()
    vis_preds(-1, args)
