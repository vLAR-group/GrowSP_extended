import torch
import torch.nn.functional as F
from datasets.SensatUrban import SensatUrbanval, cfl_collate_fn_val, SensatUrbantest, cfl_collate_fn_test
import numpy as np
import spconv.pytorch as spconv
from torch.utils.data import DataLoader
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier
from lib.helper_ply import read_ply, write_ply
import warnings
import argparse
import colorsys, random, os, sys

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
###
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='/home/zihui/SSD2/GrowSP_extension/data/SensatUrban/grid_0.200split/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= '/home/zihui/SSD2/GrowSP_extension/data/SensatUrban/initial_superpoints_graphcut_split_reg0.5/',
                        help='initial sp path')
    ###
    parser.add_argument('--save_path', type=str, default='ckpt_seg/SensatUrban_bs1_0.4grid/primitive_grow/70-20_0.9/',
                        help='model savepath')
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    parser.add_argument('--workers', type=int, default=4, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.2, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--semantic_class', type=int, default=13, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384*1, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    return parser.parse_args()


def eval_once(args, model, test_loader, classifier, use_sp=True):

    all_preds, all_label = [], []
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region, proj = data

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
            preds = preds[proj.long()]

            all_preds.append(preds[labels!=args.ignore_label]), all_label.append(labels[[labels!=args.ignore_label]])

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

    val_dataset = SensatUrbanval(args)
    # val_dataset.args.voxel_size = 0.2
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=4, pin_memory=True)

    preds, labels = eval_once(args, model, val_loader, classifier)
    del val_dataset
    del val_loader
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


def test_once(matching, model, test_loader, classifier, test_name, use_sp=False):
    full_preds_list = []
    full_point_num = 0
    coords_list = []
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, index, region, proj, splitmask = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

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

            preds_tmp = preds.clone()
            for i in range(len(matching)):
                preds[preds_tmp == matching[i, 1]] = i

            preds = preds[inverse_map.long()]
            preds = preds[proj.long()]
            full_preds_list.append((preds, splitmask))
            full_point_num = len(preds) + full_point_num

            coords_cur = coords[:, 1:].numpy() * args.voxel_size
            coords_cur = coords_cur[inverse_map]
            coords_list.append(coords_cur[proj.long()])

    # full_preds = -np.ones(full_point_num)
    # for (preds, splitmask) in full_preds_list:
    #     full_preds[splitmask==True] = preds
    full_preds = preds

    if not os.path.exists('online_testing_sensaturban/'):
        os.mkdir('online_testing_sensaturban/')
    save_name = test_name + '.label'
    savepath = 'online_testing_sensaturban/' + save_name
    full_preds = np.array(full_preds).astype(np.uint8)
    full_preds.tofile(savepath)

    if not os.path.exists('online_testing_sensaturban/vis/'):
        os.mkdir('online_testing_sensaturban/vis/')
    colors = np.array([colormap[x] for x in full_preds])
    colors = np.array(colors).astype(np.uint8)
    # coords = np.concatenate(coords_list, axis=0)
    coords = coords_list[0]
    write_ply('online_testing_sensaturban/vis/' + test_name + '.ply', [coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])


def online_test(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/SensatUrban_bs1_0.4grid/primitive_grow/70-20_0.5/model_40_checkpoint.pth'))
    model.eval()

    # cls = torch.nn.Linear(args.centroids_dim, args.centroids_num, bias=False)
    cls = torch.nn.Linear(384, 37, bias=False)
    cls.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/SensatUrban_bs1_0.4grid/primitive_grow/70-20_0.5/cls_40_checkpoint.pth'))
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

    trainval_dataset = SensatUrbanval(args)
    trainval_dataset.args.voxel_size = 0.4
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=4, pin_memory=True)

    preds, labels = eval_once(args, model, trainval_loader, classifier)
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    matching = linear_assignment(histogram.max() - histogram)
    test_name =  ['birmingham_block_2', 'birmingham_block_8', 'cambridge_block_15', 'cambridge_block_22', 'cambridge_block_27', 'cambridge_block_16']
    for i in range(len(test_name)):
        test_dataset = SensatUrbantest(args, test_name = test_name[i])
        print(test_name[i])
        test_dataset.args.voxel_size = 0.4
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=4)
        test_once(matching, model, test_loader, classifier, test_name[i])


if __name__ == '__main__':

    args = parse_args()
    online_test(-100,args)
    # epoch =
    # o_Acc, m_Acc, s = eval(epoch, args)
    # print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)
    #
    # for epoch in range(1, 200):
    #     if epoch%10==0:
    #         o_Acc, m_Acc, s = eval(epoch, args)
    #         print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)
    #     if epoch%10==0:
    #         args.primitive_num = int(args.primitive_num * 0.9)
    #         if args.primitive_num < args.semantic_class:
    #             args.primitive_num = args.semantic_class