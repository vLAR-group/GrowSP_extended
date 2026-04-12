import torch
import torch.nn.functional as F
from datasets.SensatUrban import SensatUrbanvis, SensatUrbanval, cfl_collate_fn_val, cfl_collate_fn_vis
import numpy as np
import random
import os
import spconv.pytorch as spconv
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, estimate_bandwidth, DBSCAN, SpectralClustering
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from lib.helper_ply import read_ply, write_ply
import warnings
import argparse
import colorsys, random, os, sys
# import KPConv_modules.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
warnings.filterwarnings('ignore')

colormap = np.array([[85, 107, 47],  # ground -> OliveDrab
                          [0, 255, 0],  # tree -> Green
                          [255, 165, 0],  # building -> orange
                          [41, 49, 101],  # Walls ->  darkblue
                          [0, 0, 0],  # Bridge -> black
                          [0, 0, 255],  # parking -> blue
                          [255, 0, 255],  # rail -> Magenta
                          [200, 200, 200],  # traffic Roads ->  grey
                          [89, 47, 95],  # Street Furniture  ->  DimGray
                          [255, 0, 0],  # cars -> red
                          [255, 255, 0],  # Footpath  ->  deeppink
                          [0, 255, 255],  # bikes -> cyan
                          [0, 191, 255]])

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
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='/home/zihui/SSD2/GrowSP_extension/data/SensatUrban/grid_0.200split/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= '/home/zihui/SSD2/GrowSP_extension/data/SensatUrban/initial_superpoints_graphcut_split_reg0.5/',
                        help='initial sp path')
    ###
    parser.add_argument('--save_path', type=str, default='vis/SensatUrban/',
                        help='model savepath')
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    parser.add_argument('--workers', type=int, default=4, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.4, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--semantic_class', type=int, default=13, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    return parser.parse_args()


def vis_preds(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size, config=args)
    model.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/ckpt/SensatUrban/model_260_checkpoint.pth'))
    model.eval().cuda()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False)
    # cls = torch.nn.Linear(args.feats_dim, 13, bias=False)
    cls.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/ckpt/SensatUrban/cls_260_checkpoint.pth'))
    cls.eval().cuda()
    # ############################# PICIE ##########################
    # model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    # model.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/PICIE/PICIE-S_PFH/SensatUrban/model_30_checkpoint.pth'))
    # model.eval()
    #
    # cls = torch.nn.Linear(args.feats_dim, 300, bias=False).cuda()
    # cls.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/PICIE/PICIE-S_PFH/SensatUrban/cls0_30_checkpoint.pth'))
    # cls.eval()
    # # ############################## IIC ###########################
    # model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    # model.classifier = torch.nn.Linear(128+13, 300, bias=False).cuda()
    # model.classifier2 = torch.nn.Linear(128+13, 3*13, bias=False).cuda()
    # model.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/IIC/IIC-S_PFH/SensatUrban/model_150_checkpoint.pth'))
    # model.eval()

    # primitive_centers = model.classifier.weight.data[:, 0:128]###[300, 128]
    ##########################################################
    primitive_centers = cls.weight.data###[300, 128]
    print('merging primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0, n_jobs=10).fit_predict(primitive_centers.cpu().numpy())

    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    trainval_dataset = SensatUrbanvis(args)
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    use_sp = False#True
    all_preds, all_labels, all_coords, all_inverse = [], [], [], []
    all_input_rgb = []
    for data in trainval_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region, proj, xyz, rgb = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)
            # feats = features
            # feats = F.normalize(feats, dim=1)

            region = region.squeeze()

            if use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats[valid_mask].mean(0, keepdim=True))
                region_feats = torch.cat(region_feats, dim=0)
                #
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

                region_scores = F.linear(F.normalize(region_feats), F.normalize(classifier.weight))
                region_no = 0
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        preds[valid_mask] = torch.argmax(region_scores, dim=1).cpu()[region_no]
                        region_no +=1
            else:
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()
            preds = preds[inverse_map.long()]
            # preds = preds[proj.long()]

            ### Kmeans
            # centroids, error, _, preds = faiss_kmeans(np.ascontiguousarray(feats.cpu().numpy()), centroids_num=13, centroids_dim=6)
            # preds = preds[inverse_map.long()]
            # preds = preds[proj.long()]
            ### Kmeans
            preds_full = preds
            all_preds.append(preds_full), all_labels.append(labels.numpy()), all_coords.append(xyz.numpy()), all_input_rgb.append(rgb)#, all_inverse.append(inverse_map)

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    ##
    sem_num = args.semantic_class
    mask = (labels >= 0) & (labels < sem_num)
    histogram = np.bincount(sem_num * labels[mask] + preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum()*100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1))*100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]
    # get final metrics
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)

    print('Visualize')
    m_resort = m[np.argsort(m[:,1])]
    for i, coords in enumerate(all_coords):

        label = all_labels[i]
        mask = (label!=-1)
        preds = all_preds[i]
        preds = m_resort[preds, 0]

        colors = colormap[preds]
        colors[~mask] = np.zeros(3)
        colors = colors.astype(np.uint8)

        colors_GT = colormap[label]
        colors_GT[~mask] = np.zeros(3)
        colors_GT = colors_GT.astype(np.uint8)

        # Save plys
        cloud_name = trainval_loader.dataset.name[i]

        pred_path, GT_path, input_path = os.path.join(args.save_path, 'growsp'), os.path.join(args.save_path, 'GT'), os.path.join(args.save_path, 'input')
        os.makedirs(pred_path, exist_ok=True)
        os.makedirs(GT_path, exist_ok=True)
        os.makedirs(input_path, exist_ok=True)

        pred_filename = os.path.join(pred_path, cloud_name+'growsp.ply')
        GT_filename = os.path.join(GT_path, cloud_name+'GT.ply')
        input_filename = os.path.join(input_path, cloud_name+'input.ply')

        final_coords, final_colors, final_colors_GT, input_colors = coords[mask], colors[mask], colors_GT[mask], all_input_rgb[i][mask].numpy()

        write_ply(pred_filename, [final_coords, final_colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # write_ply(GT_filename, [final_coords, final_colors_GT.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # write_ply(input_filename, [final_coords, input_colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])


if __name__ == '__main__':
    args = parse_args()
    vis_preds(-1, args)



