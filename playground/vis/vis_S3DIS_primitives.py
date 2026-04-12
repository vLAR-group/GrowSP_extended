import torch
import torch.nn.functional as F
from datasets.S3DIS import S3DISvis, cfl_collate_fn_vis
import numpy as np
import random
import spconv.pytorch as spconv
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from utils_degrowsp import get_fixclassifier
import os
from lib.helper_ply import write_ply
import warnings
import argparse
from distinctipy import distinctipy

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
    parser.add_argument('--data_path', type=str, default='../data/S3DIS/input_0.010/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='../data/S3DIS/initial_superpoints_growsp/',
                        help='initial sp path')
    parser.add_argument('--save_path', type=str, default='/home/zihui/SSD2/GrowSP++_vis/S3DIS_primitive_sp2/',
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
    parser.add_argument('--primitive_num', type=int, default=12, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=12, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=12, help='invalid label')
    return parser.parse_args()

colormap = distinctipy.get_colors(8)

def vis_preds(epoch, args):
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/ScanNet_growsp/primitive_grow_abl/distillv2_80-40sp_kmeans_elsaug_t3_fixmodel_multi0.9_nodecaylr_end30/model_300_checkpoint.pth'))
    model.eval()

    # cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls = torch.nn.Linear(args.feats_dim, 30, bias=False).cuda()
    cls.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/ScanNet_growsp/primitive_grow_abl/distillv2_80-40sp_kmeans_elsaug_t3_fixmodel_multi0.9_nodecaylr_end30/cls_300_checkpoint.pth'))
    cls.eval()

    trainval_dataset = S3DISvis(args)
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    use_sp = True
    for data in trainval_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region, full_coords, full_colors, full_labels = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            region = region.squeeze()
            # feats = feats[labels!=-1]
            # region = region[labels!=-1]
            # labels = labels[labels!=-1]
            #
            if use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats[valid_mask].mean(0, keepdim=True))
                region_feats = torch.cat(region_feats, dim=0)
                #
                scores = F.linear(F.normalize(feats), F.normalize(cls.weight))
                preds = torch.argmax(scores, dim=1).cpu()

                region_scores = F.linear(F.normalize(region_feats), F.normalize(cls.weight))
                region_no = 0
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        preds[valid_mask] = torch.argmax(region_scores, dim=1).cpu()[region_no]
                        region_no +=1
            else:
                scores = F.linear(F.normalize(feats), F.normalize(cls.weight))
                preds = torch.argmax(scores, dim=1).cpu()

            preds_full = preds[inverse_map.long()].long()
            preds_full = preds_full[(full_labels!=-1)&(full_labels!=0)]
            full_coords = full_coords[(full_labels!=-1)&(full_labels!=0)]
            ###
            cloud_name = trainval_loader.dataset.name[index[0]]
            # uni_idx = torch.unique(preds_full)
            uni_idx = [0, 3, 6, 18, 26, 28]
            colormap_tmp = np.zeros((30, 3))
            colormap_tmp[0] = colormap[1]
            colormap_tmp[3] = colormap[2]
            colormap_tmp[6] = colormap[3]
            colormap_tmp[18] = colormap[4]
            colormap_tmp[26] = colormap[5]
            colormap_tmp[28] = colormap[6]

            for p in uni_idx:
                if p !=-1 and (preds_full==p).sum()>10000:
                    # primitive_path = os.path.join(args.save_path, str(p.item()))
                    primitive_path = os.path.join(args.save_path, str(p))
                    os.makedirs(primitive_path, exist_ok=True)

                    colors = np.array(colormap_tmp)[preds_full]
                    colors = colors[:, 0:3]*255
                    colors[preds_full!=p] = np.array([64, 64, 64])

                    write_ply(os.path.join(primitive_path, cloud_name+'.ply'), [full_coords.numpy(), colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])



if __name__ == '__main__':
    args = parse_args()
    vis_preds(-1, args)