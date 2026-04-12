import torch
import torch.nn.functional as F
from datasets.ScanNet import Scannetvis, cfl_collate_fn_vis
import numpy as np
import random
import os
import spconv.pytorch as spconv
from models.fpn import Res16FPN18
# from lib.vis_utils import construct_growing_superpoints, get_fixclassifier, construct_growing_primitive
from utils_degrowsp import get_fixclassifier
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, estimate_bandwidth, DBSCAN, SpectralClustering
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from lib.helper_ply import read_ply, write_ply
import warnings
import argparse

warnings.filterwarnings('ignore')
colormap = np.array(
    [[245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
     [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240],
     [  0, 128, 128], [230,  25,  75], [170, 110,  40], [255, 250, 200], [128,   0,   0],
     [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128]])

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
    parser.add_argument('--save_path', type=str, default='/home/zihui/SSD/GrowSP_newExt/vis/ScanNet/',
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
    parser.add_argument('--primitive_num', type=int, default=30, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=20, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()


def vis_preds(epoch, args):
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/ScanNet_growsp/primitive_grow_abl/distillv2_80-40sp_kmeans_elsaug_t3_fixmodel_multi0.9_nodecaylr_end30/model_300_checkpoint.pth'))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    # cls = torch.nn.Linear(args.feats_dim, 30, bias=False).cuda()
    cls.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/ScanNet_growsp/primitive_grow_abl/distillv2_80-40sp_kmeans_elsaug_t3_fixmodel_multi0.9_nodecaylr_end30/cls_300_checkpoint.pth'))
    cls.eval()

    ############################# PICIE ##########################
    # model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    # model.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/PICIE/PICIE-S_PFH/ScanNet/model_80_checkpoint.pth'))
    # model.eval()
    #
    # cls = torch.nn.Linear(args.feats_dim, 300, bias=False).cuda()
    # cls.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/PICIE/PICIE-S_PFH/ScanNet/cls1_80_checkpoint.pth'))
    # cls.eval()
    # ############################## IIC ###########################
    # model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    # model.classifier = torch.nn.Linear(128+13, 300, bias=False).cuda()
    # model.classifier2 = torch.nn.Linear(128+13, 3*20, bias=False).cuda()
    # model.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP_extension/IIC/IIC-S_PFH/ScanNet/model_50_checkpoint.pth'))
    # model.eval()
    #
    # primitive_centers = model.classifier.weight.data[:, 0:128]###[300, 128]
    # ##########################################################

    primitive_centers = cls.weight.data  ###[500, 128]
    print('merging primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0, n_jobs=10).fit_predict(primitive_centers.cpu().numpy())

    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred == cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    trainval_dataset = Scannetvis(args)
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    use_sp = False
    all_full_preds, all_full_labels, all_full_coords, all_full_colors = [], [], [], []
    voxel_preds, voxel_labels = [], []
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

            # preds_full = -np.ones_like(labels)
            # preds_full[labels!=-1] = preds.numpy()
            preds_full = preds[inverse_map.long()]
            voxel_preds.append(preds), voxel_labels.append(labels)
            all_full_preds.append(preds_full), all_full_labels.append(full_labels), all_full_coords.append(full_coords), all_full_colors.append(full_colors)


    preds = np.concatenate(voxel_preds)
    labels = np.concatenate(voxel_labels)
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

    all_miou = []
    for i, coords in enumerate(all_full_coords):

        label = all_full_labels[i]
        mask = (label!=-1)
        preds = all_full_preds[i]
        preds = m_resort[preds, 0]

        ## compute mIoU for this scenes
        hist = np.bincount(sem_num * label[mask] + preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
        # get final metrics
        tp = np.diag(hist)
        fp = np.sum(hist, 0) - tp
        fn = np.sum(hist, 1) - tp
        IoUs = tp / (tp + fp + fn + 1e-8)
        m_IoU = np.nanmean(IoUs)
        all_miou.append(m_IoU)
        ##

        # coords = coords[mask].numpy()
        coords = coords.numpy()

        colors = colormap[preds]
        # colors = colors[mask].astype(np.uint8)
        colors = colors.astype(np.uint8)

        colors_GT = colormap[label]
        # colors_GT = colors_GT[mask].astype(np.uint8)
        colors_GT = colors_GT.astype(np.uint8)

        colors_input = all_full_colors[i].numpy()
        # colors_input = colors_input[mask].astype(np.uint8)
        colors_input = colors_input.astype(np.uint8)

        # Save plys
        cloud_name = trainval_loader.dataset.name[i]

        pred_path, GT_path, input_path = os.path.join(args.save_path, 'preds'), os.path.join(args.save_path, 'GT'), os.path.join(args.save_path, 'input')
        os.makedirs(pred_path, exist_ok=True)
        os.makedirs(GT_path, exist_ok=True)
        os.makedirs(input_path, exist_ok=True)

        pred_filename = os.path.join(pred_path, cloud_name+'preds.ply')
        GT_filename = os.path.join(GT_path, cloud_name+'GT.ply')
        input_filename = os.path.join(input_path, cloud_name+'input.ply')

        write_ply(pred_filename, [coords, colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply(GT_filename, [coords, colors_GT.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply(input_filename, [coords, colors_input.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Pair the numbers with the strings
    paired_list = list(zip(all_miou, trainval_loader.dataset.name))
    # Sort the paired list by the numbers in descending order
    sorted_paired_list = sorted(paired_list, key=lambda x: x[0], reverse=True)
    # Unzip the sorted pairs back into two lists
    sorted_numbers, sorted_strings = zip(*sorted_paired_list)
    # Convert the tuples back to lists (if needed)
    sorted_numbers = list(sorted_numbers)
    sorted_strings = list(sorted_strings)
    for i in range(100):
        print(sorted_strings[i], sorted_numbers[i])

if __name__ == '__main__':
    args = parse_args()
    vis_preds(-1, args)


# scene0101_01 0.36460010756615974
# scene0177_01 0.34295718644735573
# scene0092_04 0.32144464051397836
# scene0177_00 0.31809673692823714
# scene0276_01 0.3103076136383866
# scene0591_02 0.3072165824783227
# scene0591_00 0.3033182428315436
# scene0276_00 0.30278154374445665
# scene0474_05 0.3027631178392348
# scene0698_01 0.30243989677375277
# scene0207_00 0.30144983379542795
# scene0101_02 0.29780751406662687
# scene0698_00 0.29669145807726355
# scene0207_01 0.2939711577236633
# scene0092_02 0.2916163668223583
# scene0092_01 0.2893074872968875
# scene0073_01 0.2872227782749216
# scene0101_00 0.28708622451963006
# scene0101_03 0.2859904905899041
# scene0645_01 0.2834332793380546
# scene0395_02 0.2817120052027479
# scene0515_02 0.28126566249187557
# scene0000_00 0.2807218561133981
# scene0392_01 0.2806696202750623
# scene0515_01 0.2803917325688946
# scene0092_00 0.27931239836932453
# scene0207_02 0.27747873332207823
# scene0211_00 0.2772220893057746
# scene0054_00 0.2769750397190717
# scene0092_03 0.2764576196557893
# scene0262_01 0.27578536967563294
# scene0591_01 0.2753546676424687
# scene0640_00 0.2751047929843654
# scene0040_01 0.2745045644962655
# scene0101_05 0.2736911740772091
# scene0515_00 0.2734372950846428
# scene0025_00 0.27288300402234456
# scene0472_01 0.271920003276537
# scene0392_00 0.27188665783895155
# scene0051_01 0.27045644364263344
# scene0040_00 0.2698811210873088
# scene0420_01 0.2691535182827117
# scene0395_00 0.2680474042321297
# scene0673_04 0.26743416009503396
# scene0329_02 0.26640751212438946
# scene0590_00 0.2663032437065661
# scene0640_02 0.26599931702572943
# scene0181_01 0.264639633614501
# scene0474_04 0.26418617504768893
# scene0177_02 0.26415680862581054
# scene0474_03 0.26397763303715
# scene0673_01 0.26297776249293103
# scene0640_01 0.26297661024893243
# scene0262_00 0.262950814606702
# scene0061_01 0.2621569201106872
# scene0288_00 0.2621084710370532
# scene0279_02 0.26101354273307453
# scene0211_01 0.2609029287443266
# scene0000_01 0.2608306683742314
# scene0101_04 0.26066492536423597
# scene0362_01 0.26049534315670436
# scene0073_03 0.2597309560986395
# scene0420_02 0.2594841256419645
# scene0000_02 0.259186292372967
# scene0395_01 0.2590941527651034
# scene0051_02 0.2590094759372096
# scene0166_00 0.2588669501093365
# scene0051_03 0.2576665062297421
# scene0017_00 0.25707043009596514
# scene0282_00 0.256920166604735
# scene0623_01 0.2568522514129267
# scene0211_02 0.25670270161502806
# scene0699_00 0.2566953455227369
# scene0166_02 0.25626570806571475
# scene0472_00 0.2551464746048574
# scene0645_00 0.25416568103644394
# scene0181_02 0.25380587082375067
# scene0288_02 0.25367064928888267
# scene0297_02 0.2536666133271305
# scene0505_00 0.25295936826999743
# scene0673_00 0.2518867963558472
# scene0181_00 0.2511379203108343
# scene0474_00 0.2500891458998532
# scene0651_02 0.2498036357157099
# scene0645_02 0.24932970732623208
# scene0297_01 0.2491767632714888
# scene0369_00 0.24913125329616864
# scene0695_00 0.24902887941944898
# scene0653_00 0.24892367914040983
# scene0186_00 0.24840625496821228
# scene0117_00 0.24782564372464538
# scene0403_00 0.246852032432876
# scene0472_02 0.24672113418589303
# scene0403_01 0.24377375280974906
# scene0288_01 0.2437737493066031
# scene0705_02 0.24375777826756373
# scene0373_01 0.24319420124596508
# scene0051_00 0.24235726924420348
# scene0630_00 0.2422806344676181
# scene0419_02 0.24202160711745288
