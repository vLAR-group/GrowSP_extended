import torch
import torch.nn.functional as F
from datasets.SemanticKITTI import KITTIval, cfl_collate_fn_val, KITTItest, cfl_collate_fn_test
import numpy as np
import me_compat as ME
from torch.utils.data import DataLoader
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier
from lib.helper_ply import read_ply, write_ply
import warnings
import argparse
import random
import os
import matplotlib.pyplot as plt

label_inverse_map = {
  0: 0,      # "unlabeled", and others ignored
  1: 10,    # "car"
  2: 11,     # "bicycle"
  3: 15,     # "motorcycle"
  4: 18,     # "truck"
  5: 20,     # "other-vehicle"
  6: 30,     # "person"
  7: 31,     # "bicyclist"
  8: 32,     # "motorcyclist"
  9: 40,     # "road"
  10: 44,    # "parking"
  11: 48,    # "sidewalk"
  12: 49,    # "other-ground"
  13: 50,    # "building"
  14: 51,    # "fence"
  15: 70,    # "vegetation"
  16: 71,    # "trunk"
  17: 72,    # "terrain"
  18: 80,    # "pole"
  19: 81,    # "traffic-sign"
}

colormap = np.array(
    [[0, 0, 0], [245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80], [255, 0, 0], [30, 30, 255],
     [200, 40, 255], [90, 30, 150], [255, 0, 255], [255, 150, 255], [75, 0, 75], [75, 0, 175], [0, 200, 255],
     [50, 120, 255], [0, 175, 0], [0, 60, 135], [80, 240, 150], [150, 240, 255], [0, 0, 255]])

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='/home/zihui/SSD2/SemanticKITTI/dataset/sequences/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='/home/zihui/SSD2/SemanticKITTI/initial_superpoints/sequences/',
                        help='initial sp path')
    parser.add_argument('--save_path', type=str, default='ckpt_seg/SemanticKITTI/',
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


def eval_once(args, model, test_loader, classifier):

    all_preds, all_label = [], []
    for data in test_loader:
        with torch.no_grad():
            coords, inverse_map, labels, index = data

            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map.long()]
            preds = preds[labels!=args.ignore_label]
            labels = labels[labels!=args.ignore_label]
            all_preds.append(preds), all_label.append(labels)

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return all_preds, all_label


def eval(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=5, config=args).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()

    primitive_centers = cls.weight.data###[300, 128]
    print('Merging Primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0, n_jobs=10).fit_predict(primitive_centers.cpu().numpy())#.astype(np.float64))

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

    val_dataset = KITTIval(args)
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


def test_once(matching, model, test_loader, classifier):

    for data in test_loader:
        with torch.no_grad():
            coords, inverse_map, index = data

            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()

            preds_tmp = preds.clone()
            for i in range(len(matching)):
                preds[preds_tmp == matching[i, 1]] = i

            preds = preds[inverse_map.long()]+1
            scene_name = test_loader.dataset.name[index[0]]

            pred_ori_labels = np.array([label_inverse_map[x.item()] for x in preds])
            os.makedirs('online_testing_kitti/sequences/'+scene_name[0:3]+'/predictions/', exist_ok=True)
            savepath = 'online_testing_kitti/sequences/'+scene_name[0:3]+'/predictions/'+scene_name[3:]+'.label'
            pred_ori_labels = pred_ori_labels.astype(np.uint32)
            pred_ori_labels.tofile(savepath)

            plypath = 'online_testing_kitti/'
            colors = np.array([colormap[x] for x in preds])
            colors = np.array(colors).astype(np.uint8)

            if not os.path.exists(plypath+scene_name[0:3]):
                os.makedirs(plypath+scene_name[0:3])

            # write_ply(plypath+scene_name+'.ply', [coords[:, 1:].numpy()[inverse_map]*args.voxel_size, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])


def online_test(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/SemanticKITTI/primitive_grow/80-20_0.7/model_25_checkpoint.pth'))
    model.eval()

    # cls = torch.nn.Linear(args.centroids_dim, args.centroids_num, bias=False)
    cls = torch.nn.Linear(384, 71, bias=False)
    cls.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/SemanticKITTI/primitive_grow/80-20_0.7/cls_25_checkpoint.pth'))
    cls.eval()

    primitive_centers = cls.weight.data.cpu()###[300, 128]
    print('Merging Primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0, n_jobs=10).fit_predict(primitive_centers.numpy())#.astype(np.float64))

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

    test_dataset = KITTItest(args)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=args.cluster_workers, pin_memory=True)

    val_dataset = KITTIval(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=args.cluster_workers, pin_memory=True)

    preds, labels = eval_once(args, model, val_loader, classifier)
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    matching = linear_assignment(histogram.max() - histogram)
    test_once(matching, model, test_loader, classifier)


import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.3f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':

    args = parse_args()
    # epoch=-1
    # o_Acc, m_Acc, s = eval(epoch, args)
    # print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)
    online_test(-100,args)
    # for epoch in range(1, 100):
    #     if epoch%70==0:
    #         o_Acc, m_Acc, s = eval(epoch, args)
    #         print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)

