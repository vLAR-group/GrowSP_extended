import torch
import torch.nn.functional as F
from datasets.SemanticKITTI import KITTIvis, cfl_collate_fn_vis
import numpy as np
import random
import os
import spconv.pytorch as spconv
from models.fpn import Res16FPN18
from utils_degrowsp import get_fixclassifier
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from lib.helper_ply import read_ply, write_ply
import warnings
import argparse

warnings.filterwarnings('ignore')
colormap = np.array(
    [[245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80], [255, 0, 0], [30, 30, 255],
     [200, 40, 255], [90, 30, 150], [255, 0, 255], [255, 150, 255], [75, 0, 75], [75, 0, 175], [0, 200, 255],
     [50, 120, 255], [0, 175, 0], [0, 60, 135], [80, 240, 150], [150, 240, 255], [0, 0, 255], [0, 0, 0] ])

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
    parser.add_argument('--data_path', type=str, default='/home/zihui/SSD2/SemanticKITTI/dataset/sequences/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='/home/zihui/SSD2/SemanticKITTI/initial_superpoints/sequences/',
                        help='initial sp path')
    parser.add_argument('--save_path', type=str, default='/home/zihui/SSD2/GrowSP++_vis/KITTI/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=19, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()


def vis_preds(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/SemanticKITTI/primitive_grow/80-20_0.7/model_25_checkpoint.pth'))
    model.eval()

    # cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls = torch.nn.Linear(args.feats_dim, 71, bias=False).cuda()
    cls.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/SemanticKITTI/primitive_grow/80-20_0.7/cls_25_checkpoint.pth'))
    cls.eval()
    ############################ PICIE ##########################
    # model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    # model.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/PICIE/PICIE-S_PFH/KITTI/model_50_checkpoint.pth'))
    # model.eval()
    #
    # cls = torch.nn.Linear(args.feats_dim, 500, bias=False).cuda()
    # cls.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/PICIE/PICIE-S_PFH/KITTI/cls0_50_checkpoint.pth'))
    # cls.eval()
    ############################## IIC ###########################
    # model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    # model.classifier = torch.nn.Linear(128+13, 300, bias=False).cuda()
    # model.classifier2 = torch.nn.Linear(128+13, 3*19, bias=False).cuda()
    # model.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/IIC/IIC-S_PFH/KITTI/model_20_checkpoint.pth'))
    # model.eval()
    #
    # primitive_centers = model.classifier.weight.data[:, 0:128]###[300, 128]
    # ##########################################################

    primitive_centers = cls.weight.data###[500, 128]
    print('merging primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=5, random_state=0, n_jobs=5).fit_predict(primitive_centers.cpu().numpy())

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

    val_dataset = KITTIvis(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    all_preds, all_labels, all_coords, all_inverse = [], [], [], []
    for data in val_loader:
        with torch.no_grad():
            coords, inverse_map, labels, index, raw_coords = data

            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map.long()]
            preds_full = preds[labels!=args.ignore_label]
            raw_coords = raw_coords[labels!=args.ignore_label]
            labels = labels[labels!=args.ignore_label]

            full_mask = np.sqrt(((raw_coords) ** 2).sum(-1)) < 40

            all_preds.append(preds_full.numpy()[full_mask]), all_labels.append(labels.numpy()[full_mask]), all_coords.append(raw_coords.numpy()[full_mask])#, all_inverse.append(inverse_map)


    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    ##
    sem_num = args.semantic_class
    mask = (labels >= 0) & (labels < sem_num)
    histogram = np.bincount(sem_num * labels[mask] + preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum()
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
    print('Epoch: {:02d}, Test acc: {:.5f}  Test IoU'.format(epoch, o_Acc), s)

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
        cloud_name = val_loader.dataset.name[i]

        pred_path, GT_path, input_path = os.path.join(args.save_path, 'PICIE'), os.path.join(args.save_path, 'GT'), os.path.join(args.save_path, 'input')
        os.makedirs(pred_path, exist_ok=True)
        os.makedirs(GT_path, exist_ok=True)
        os.makedirs(input_path, exist_ok=True)

        pred_filename = os.path.join(pred_path, cloud_name+'PICIE.ply')
        GT_filename = os.path.join(GT_path, cloud_name+'GT.ply')
        input_filename = os.path.join(input_path, cloud_name+'input.ply')

        write_ply(pred_filename, [coords[mask], colors[mask]], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # write_ply(GT_filename, [coords[mask], colors_GT[mask]], ['x', 'y', 'z', 'red', 'green', 'blue'])


# def vis_growsp(epoch, args):
#     model = Res16FPN18(in_channels=args.input_dim, out_channels=args.centroids_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
#     model.load_state_dict(torch.load(args.save_path + 'model_' + str(epoch) + '_checkpoint.pth'))
#     model.eval()
#
#     val_dataset = KITTItrain(args, split='val')
#     val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn(), num_workers=4, pin_memory=True)
#
#     from distinctipy import distinctipy
#     current_growsp = 30
#     colormap = distinctipy.get_colors(current_growsp)
#     construct_kitti_growing_superpoints(args, val_loader, model, colormap, current_growsp=current_growsp, epoch=epoch)



if __name__ == '__main__':
    args = parse_args()
    vis_preds(-1, args)
    # for epoch in range(3000):
    #     if epoch >=10 and epoch %760 ==0:
    #         vis_preds(epoch, args)
            # vis_growsp(epoch, args)



