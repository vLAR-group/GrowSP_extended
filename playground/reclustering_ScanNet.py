import torch
import torch.nn.functional as F
from datasets.ScanNet import Scannetval, cfl_collate_fn_val
import numpy as np
import me_compat as ME
from torch.utils.data import DataLoader
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from models.MinkNet import MinkUNet18A
from utils_degrowsp import get_fixclassifier
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/ScanNet/processed/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= 'data/ScanNet/initial_superpoints/',
                        help='initial sp path')
    parser.add_argument('--save_path', type=str, default='ckpt_seg/ScanNet/primitive_grow/distillv2_80-30sp_kmeans_elsaug_t3_fixmodel/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--semantic_class', type=int, default=20, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=768, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    ###
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    return parser.parse_args()


def eval_once(args, model, test_loader, classifier, use_sp=False):

    all_preds, all_label = [], []
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            region = region.squeeze()
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
            all_preds.append(preds[labels!=args.ignore_label]), all_label.append(labels[[labels!=args.ignore_label]])

    return all_preds, all_label


def collect_all_spfeats(args, model, test_loader, use_sp=False):

    all_feats = []
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            region = region.squeeze()
            #
            if use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats[valid_mask].mean(0, keepdim=True))
                feats = torch.cat(region_feats, dim=0)

            all_feats.append(feats[np.random.choice(len(feats), int(len(feats)//5), replace=False)])

    return all_feats



def eval(epoch, args):

    test_dataset = Scannetval(args)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=4, pin_memory=True)

    # model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=5, config=args).cuda()
    # model.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/ScanNet_growsp/primitive_grow_abl/distillv2_80-40sp_kmeans_elsaug_t3_fixmodel_multi0.9_nodecaylr_end30/model_300_checkpoint.pth'))
    # model.eval()
    model = MinkUNet18A(in_channels=3, out_channels=768).cuda()
    dict = torch.load('/home/zihui/SSD/GrowSP_newExt/scannet_openseg.pth.tar')['state_dict']
    dict = {key.replace('net3d.', ''): value for key, value in dict.items()}
    model.load_state_dict(dict)
    model.eval()

    all_feats = collect_all_spfeats(args, model, test_loader, use_sp=False)
    all_feats = torch.cat(all_feats)
    # all_feats = all_feats[np.random.choice(len(all_feats), 100000, replace=False)]

    print('Merging Primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0, n_jobs=10).fit_predict(all_feats.cpu().numpy())#.astype(np.float64))

    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = all_feats[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    preds, labels = eval_once(args, model, test_loader, classifier, use_sp=False)
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum()*100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1))*100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]

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
    o_Acc, m_Acc, s = eval(-1, args)
    print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(-1, o_Acc, m_Acc), s)