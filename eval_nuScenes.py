import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
from datasets.nuScenes import nuScenesval, cfl_collate_fn_val
import numpy as np
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from sparse_fpn import Res16FPN18
from utils_degrowsp import get_fixclassifier
from lib.helper_ply import read_ply, write_ply
import argparse
import os
colormap = np.array(
    [[220,220,  0], [119, 11, 32], [0, 60, 100], [0, 0, 250], [230,230,250],
     [0, 0, 230], [220, 20, 60], [250, 170, 30], [200, 150, 0], [0, 0, 110],
     [128, 64, 128], [0,250, 250], [244, 35, 232], [152, 251, 152], [70, 70, 70],
     [107,142, 35]])

###
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/zihui/SSD/LogoSP/data/nuScenes/nuscenes_3d/train/',
                        help='pont cloud data path')
    parser.add_argument('--val_input_path', type=str, default='/home/zihui/SSD/LogoSP/data/nuScenes/nuscenes_3d/val',
                        help='pont cloud data path')
    parser.add_argument('--save_path', type=str, default='/home/user/SSD2/GrowSP/ckpt/SemanticKITTI/',
                        help='model savepath')
    ###
    parser.add_argument('--workers', type=int, default=16, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4,help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--batch_size', type=int, default=8, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.1, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')  ### 6 for XYZGB
    parser.add_argument('--semantic_class', type=int, default=16, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()


def eval_once(args, model, test_loader, classifier):

    all_preds, all_label = [], []
    for data in test_loader:
        with torch.no_grad():
            coords, features, labels, inverse_map, index = data

            if not coords.is_contiguous():
                coords = coords.contiguous()
            in_field = spconv.SparseConvTensor(features=features.cuda(), indices=coords.int().cuda(),
                    spatial_shape=list(coords.max(0)[0] + 16)[1:], batch_size=coords.max(0)[0][0].item()+1)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map[0]]
            preds = preds[labels!=args.ignore_label]
            labels = labels[labels!=args.ignore_label]
            all_preds.append(preds), all_label.append(labels)

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return all_preds, all_label


def eval(epoch, args):
    model = Res16FPN18(in_channels=3).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()
    primitive_centers = cls.weight.data
    print('Merging Primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0).fit_predict(primitive_centers.cpu().numpy())#.astype(np.float64))

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

    val_dataset = nuScenesval(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=args.cluster_workers, pin_memory=True)

    preds, labels = eval_once(args, model, val_loader, classifier)
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[0], m[1]].sum() / histogram.sum()*100.
    m_Acc = np.mean(histogram[m[0], m[1]] / histogram.sum(1))*100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[1][idx]]

    '''Final Metrics'''
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)

    return o_Acc, m_Acc, s

if __name__ == '__main__':

    args = parse_args()
    for epoch in range(0, 2):
        if epoch==1:
            o_Acc, m_Acc, s = eval(epoch, args)
            print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)

