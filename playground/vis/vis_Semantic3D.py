import torch
import torch.nn.functional as F
from datasets.Semantic3D import Semantic3Dvis, cfl_collate_fn_vis, Semantic3Dval, cfl_collate_fn_val
import numpy as np
import random
import os
import spconv.pytorch as spconv
from models.fpn import Res16FPN18
from lib.vis_utils import construct_growing_superpoints, get_fixclassifier, construct_growing_primitive
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, estimate_bandwidth, DBSCAN, SpectralClustering
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from lib.helper_ply import read_ply, write_ply
import warnings
import argparse
from lib.utils import faiss_kmeans
import KPConv_modules.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

warnings.filterwarnings('ignore')
colormap = np.array(
    [[128, 128, 128], [ 79, 121, 66], [50, 205, 50], [253, 213, 13], [199, 0, 57], [218, 112, 214], [125, 249, 255], [255, 95, 21]])

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
    parser.add_argument('--data_path', type=str, default='/home/user/SSD2/GrowSP_extension/data/semantic3d/input_0.060/',
                        help='pont cloud data path')
    parser.add_argument('--label_path', type=str, default='/home/user/SSD2/GrowSP_extension/data/semantic3d/original_ply/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= '/home/user/SSD2/GrowSP_extension/data/semantic3d/initial_superpoints_graphcut_reg1/',
                        help='initial sp path')
    ###
    parser.add_argument('--save_path', type=str, default='/home/user/SSD2/GrowSP_extension/visualizations/visualizations/Semantic3d/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=4, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--centroids_num', type=int, default=300, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=8, help='ground truth semantic class')
    parser.add_argument('--centroids_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--growsp_start', type=int, default=1000, help='the start number of growing superpoint')
    parser.add_argument('--growsp_end', type=int, default=30, help='the end number of grwoing superpoint')
    parser.add_argument('--drop_threshold', type=int, default=10, help='ignore superpoints with few points')
    parser.add_argument('--w_rgb', type=float, default=5/5, help='weight for RGB in merging superpoint')
    parser.add_argument('--w_xyz', type=float, default=1/5, help='weight for XYZ in merging superpoint')
    parser.add_argument('--w_norm', type=float, default=4/5, help='weight for Normal in merging superpoint')
    parser.add_argument('--c_rgb', type=float, default=3, help='weight for RGB in clustering primitives')
    parser.add_argument('--c_shape', type=float, default=3, help='weight for PFH in clustering primitives')
    return parser.parse_args()


def vis_preds(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.centroids_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    # model.load_state_dict(torch.load('/home/user/SSD2/GrowSP_extension/PICIE/PICIE/Semantic3D/model_20_checkpoint.pth'))
    # model.load_state_dict(torch.load('/home/user/SSD2/GrowSP_extension/ckpt/Semantic3d/model_120_checkpoint.pth'))
    # model.eval()
    model.classifier = torch.nn.Linear(args.centroids_dim+0, args.centroids_num, bias=False).cuda()
    model.classifier2 = torch.nn.Linear(args.centroids_dim+0, 3*args.semantic_class, bias=False).cuda()
    model.load_state_dict(torch.load('/home/user/SSD2/GrowSP_extension/IIC/IIC/Semantic3D/model_540_checkpoint.pth'))
    model.eval()

    primitive_centers = model.classifier.weight.data[:, 0:args.centroids_dim]###[300, 128]
    # cls = torch.nn.Linear(args.centroids_dim, args.centroids_num, bias=False)
    # cls.load_state_dict(torch.load('/home/user/SSD2/GrowSP_extension/PICIE/PICIE/Semantic3D/cls0_20_checkpoint.pth'))
    # cls.load_state_dict(torch.load('/home/user/SSD2/GrowSP_extension/ckpt/SensatUrban/cls_240_checkpoint.pth'))
    # cls.eval()
    #
    # primitive_centers = cls.weight.data###[300, 128]
    print('merging primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0, n_jobs=10).fit_predict(primitive_centers.cpu().numpy())

    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.centroids_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.centroids_dim, centroids_num=args.semantic_class, centroids=centroids)
    classifier.eval()

    trainval_dataset = Semantic3Dvis(args)
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    use_sp = False
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
            preds = preds[proj.long()]

            # ### Kmeans
            # centroids, error, _, preds = faiss_kmeans(np.ascontiguousarray(feats.cpu().numpy()), centroids_num=8, centroids_dim=6)
            # preds = preds[inverse_map.long()]
            # preds = preds[proj.long()]
            ### Kmeans
            preds_full = preds
            all_preds.append(preds_full), all_labels.append(labels.numpy()), all_coords.append(xyz.numpy()), all_input_rgb.append(rgb)#, all_inverse.append(inverse_map)


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
    save_path = args.save_path + str(epoch) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

        test_name = save_path+cloud_name+'.ply'
        test_name_GT = save_path + cloud_name + 'GT.ply'
        test_name_input = save_path + cloud_name + 'input.ply'

        final_coords, final_colors = cpp_subsampling.subsample(coords[mask], features=colors[mask], sampleDl=0.1)
        _, final_colors_GT = cpp_subsampling.subsample(coords[mask], features=colors_GT[mask], sampleDl=0.1)
        _, input_colors = cpp_subsampling.subsample(coords[mask], features=all_input_rgb[i][mask], sampleDl=0.1)

        # write_ply(test_name, [final_coords, final_colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # write_ply(test_name_GT, [final_coords, final_colors_GT.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply(test_name_input, [final_coords, input_colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])


def vis_growsp(epoch, args):
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.centroids_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(args.save_path + 'model_' + str(epoch) + '_checkpoint.pth'))
    model.eval()

    trainval_dataset = Scannettrain(args)
    trainval_dataset.path_file = 'data_prepare/ScanNet_splits/scannetv2_trainval.txt'
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn(), num_workers=4, pin_memory=True)

    from distinctipy import distinctipy
    current_growsp = 30
    colormap = distinctipy.get_colors(current_growsp)
    construct_growing_superpoints(args, trainval_loader, model, colormap, current_growsp=current_growsp, epoch=epoch)

def vis_primitive(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.centroids_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(args.save_path + 'model_' + str(epoch) + '_checkpoint.pth'))
    model.eval()

    cls = torch.nn.Linear(args.centroids_dim, args.centroids_num, bias=False).cuda()
    cls.load_state_dict(torch.load(args.save_path + 'cls_' + str(epoch) + '_checkpoint.pth'))
    cls.eval()

    from distinctipy import distinctipy
    colormap = distinctipy.get_colors(args.centroids_num)

    # test_dataset = S3DIStest(args, areas=test_areas)
    test_dataset = S3DIStest(args, areas=['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6'])
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=0, pin_memory=True)

    use_sp = True
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            region = region.squeeze()
            # valid_mask = region!=-1
            # feats = feats[valid_mask]
            # region = region[valid_mask]
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

            # labels = labels[inverse_map.long()]
            # preds_full = -np.ones_like(labels)
            preds = preds.numpy()[inverse_map.long()]
            points = coords[:, 1:].numpy()[inverse_map.long()]
            mask = (labels != 0) & (labels != -1)
            # mask = labels!=-1

            colors = 255 * (np.array(colormap)[preds])[:, 0:3]
            # colors[~mask] = np.zeros(3)
            colors = colors.astype(np.uint8)

            # Save plys
            save_path = '/home/user/SSD/GrowSP_extension/visualizations/visualizations/S3DIS/primitive/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cloud_name = test_loader.dataset.name[index[0]]
            test_name = save_path + cloud_name + '.ply'
            write_ply(test_name, [points[mask], colors[mask]], ['x', 'y', 'z', 'red', 'green', 'blue'])

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))


def vis_growing_primitive(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.centroids_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(args.save_path + 'model_' + str(epoch) + '_checkpoint.pth'))
    model.eval()

    cls = torch.nn.Linear(args.centroids_dim, args.centroids_num, bias=False).cuda()
    cls.load_state_dict(torch.load(args.save_path + 'cls_' + str(epoch) + '_checkpoint.pth'))
    cls.eval()

    clusterset = Scannettrain(args)
    cluster_loader = DataLoader(clusterset, batch_size=1, collate_fn=cfl_collate_fn(), num_workers=3, pin_memory=True)
    cluster_loader.dataset.mode = 'cluster'

    from distinctipy import distinctipy
    current_growsp = 30
    colormap = distinctipy.get_colors(args.centroids_num)
    construct_growing_primitive(args, cluster_loader, model, colormap, current_growsp=current_growsp, epoch=epoch, classifier=cls)


if __name__ == '__main__':
    args = parse_args()
    # for epoch in range(3000):
    #     if epoch >=10 and epoch %500 ==0:
    #         # vis_preds(epoch, args)
    #         # vis_growsp(epoch, args)
    #         vis_growing_primitive(epoch, args)
    vis_preds(-1, args)



