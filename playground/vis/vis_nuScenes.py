import torch
import torch.nn.functional as F
from datasets.nuScenes import nuScenesvis, cfl_collate_fn_vis
import numpy as np
import spconv.pytorch as spconv
from torch.utils.data import DataLoader
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from utils_degrowsp import get_fixclassifier
import argparse
import os
from lib.helper_ply import write_ply

colormap = np.array(
    [[220,220,  0], [119, 11, 32], [0, 60, 100], [0, 0, 250], [230,230,250],
     [0, 0, 230], [220, 20, 60], [250, 170, 30], [200, 150, 0], [0, 0, 110],
     [128, 64, 128], [0,250, 250], [244, 35, 232], [152, 251, 152], [70, 70, 70],
     [107,142, 35]])

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='../../LogoSP/data/nuScenes/nuscenes_3d/train/',
                        help='pont cloud data path')
    parser.add_argument('--val_input_path', type=str, default='../../LogoSP/data/nuScenes/nuscenes_3d/val',
                        help='pont cloud data path')
    parser.add_argument('--vis_path', type=str, default='vis/nuScenes/',
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
    parser.add_argument('--primitive_num', type=int, default=16, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=16, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()


def vis_preds(epoch, args):
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/nuScenes/primitive_grow/distillv2_80-30sp_kmeans_elsaug_t3_300primitives_fixmodel_0.9/model_30_checkpoint.pth'))
    model.eval()

    # cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls = torch.nn.Linear(args.feats_dim, 176, bias=False).cuda()
    cls.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/nuScenes/primitive_grow/distillv2_80-30sp_kmeans_elsaug_t3_300primitives_fixmodel_0.9/cls_30_checkpoint.pth'))
    cls.eval()

    primitive_centers = cls.weight.data###[300, 128]
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
    classifier = get_fixclassifier(in_channel=args.semantic_class, centroids_num=args.feats_dim, centroids=centroids).cuda()
    classifier.eval()

    val_dataset = nuScenesvis(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    all_full_preds, all_full_labels, all_full_coords = [], [], []
    voxel_preds, voxel_labels = [], []
    for data in val_loader:
        with torch.no_grad():
            coords, inverse_map, labels, index, full_coords, full_labels = data

            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)
            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()

            preds_full = preds[inverse_map.long()]
            preds_full, full_labels, full_coords = preds_full[full_labels!=-1], full_labels[full_labels!=-1], full_coords[full_labels!=-1]
            full_mask = np.sqrt(((full_coords) ** 2).sum(-1)) < 40

            voxel_preds.append(preds[labels!=--1]), voxel_labels.append(labels[labels!=--1])
            all_full_preds.append(preds_full[full_mask]), all_full_labels.append(full_labels[full_mask]), all_full_coords.append(full_coords[full_mask])


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
    vis_path_preds = os.path.join(args.vis_path, 'preds')
    vis_path_gt = os.path.join(args.vis_path, 'gt')
    vis_path_input = os.path.join(args.vis_path, 'input')

    os.makedirs(vis_path_preds, exist_ok=True)
    os.makedirs(vis_path_gt, exist_ok=True)
    os.makedirs(vis_path_input, exist_ok=True)

    m_resort = m[np.argsort(m[:,1])]

    all_miou = []
    for i, coords in enumerate(all_full_coords):

        label = all_full_labels[i]
        mask = torch.logical_and(label!=-1, label!=0)
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

        coords = coords[mask].numpy()

        colors = colormap[preds]
        colors = colors[mask].astype(np.uint8)

        colors_GT = colormap[label]
        colors_GT = colors_GT[mask].astype(np.uint8)

        colors_input = np.ones_like(colors_GT)*128
        colors_input = colors_input.astype(np.uint8)


        # Save plys
        cloud_name = val_loader.dataset.name[i]

        test_name = os.path.join(vis_path_preds, cloud_name+'preds.ply')
        test_name_gt = os.path.join(vis_path_gt, cloud_name+'gt.ply')
        test_name_input = os.path.join(vis_path_input, cloud_name+'input.ply')

        write_ply(test_name, [coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply(test_name_gt, [coords, colors_GT], ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply(test_name_input, [coords, colors_input], ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Pair the numbers with the strings
    paired_list = list(zip(all_miou, val_loader.dataset.name))
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
    print(args)
    vis_preds(5, args)


# 281b92269fd648d4b52d06ac06ca6d65 0.3046737687274622
# ce60ca0e7f1749f380461559de3e8113 0.2967491334224719
# 284234f5451d495fa01bd24bd04993dc 0.29603420248726114
# 26ce0c76786c43158b3bac999b27e76b 0.29525832290739873
# 355be35606a1455e83b171ce89822e54 0.29492116753811803
# 6c7261cf1e9746b5a8bfeb89e9ce22e8 0.2943457906636411
# b1cc984e17a341e093c53c1d271acce6 0.29190783431226547
# 8e5cf22369314278a08862383f647c40 0.2911916255427515
# 2002b712735749e0bb54ef30a62a6b58 0.2908510263733105
# cbfdfa1a34184ff895e94ae3ec7928bd 0.28957014537888676
# f457a12f1a354958a3238e4fbbcc7059 0.289495267076027
# 7d47a8f93f564e4e9546107d21de0551 0.2877770603142513
# e78da0feb7e4407db87b40d012052b1e 0.28750647917932376
# bd08b6c8ca814798bc6e62220b14ab0c 0.2874124645977788
# f7b3cd928a034c48b606db29a5f2b3c5 0.2870064083071453
# 345ffff158a245bc80572dbde4344f28 0.28567479582158783
# 088886ce0f6c461eaae9de7bf3d49c5b 0.2846809073336807
# cc72799740754b58bc17183e88010df3 0.2842489452195094
# 6f88910ce826420fb9739c7da9d2e54e 0.28405685388752744
# 358fbfb1ebac44878a94f49577c9c658 0.2827042998690241
# 45df585aa5fb4972986093b353bae03f 0.28268466840604145
# 72ab6eb24ebe4b6188b86363d6e99feb 0.28263793007589955
# 1b7102e9393a456e9710a63c35c6af40 0.28127021696314747
# 264f7a3db349473c9032d9ee63b9b828 0.28048109254718273
# 4e171b7de025407384e7d9414b8a8c7b 0.2801824466534951
# 72505908976e4fbe8addb92ed4dcd250 0.27984718598509606
# 0645e15ddf57496faec8fb83a3068faa 0.27955218206366844
# bb0abb64c37b49058b885392c22d3b29 0.27955160964532144
# d79002ca6d834f1d9970941c2c441591 0.2793136052036642
# bc06995dc3674a41b2c90ba219954932 0.27872764534019023
# 102fadd49f0a42bebb20a558c0f95031 0.2781539606879555
# 5850f2ddde324e80bb28d792bfefb632 0.276169362504061
# 4208e9982fbb49329271ee6dc0dbcf64 0.2761653893166265
# dcc9ee6964da49848340b18753a0dc96 0.27610085370357035
# 9ed18ab8b4c2468ea0dfa3b7c6ced37e 0.2757329112293707
# 4c8b6c42e0294113b78fb64a4fa14fa7 0.27515987786235196
# 007c057cac8b490fa8b67e72e630b18a 0.27512661088191354
# 618d34a8ce0a4f898952fce1d0ea8fbb 0.2750326075800006
# 7654157b2905440eb11f710e21486ab6 0.274709813708129
# 6bacbd0e67734ed5a76a0ffb798a2f40 0.274608821182744
# 551d1fdf87cc41fd92f33dc0048cf997 0.2745581951006012
# 6cd73bca1f384f968b06466eaf509383 0.27447844758322826
# 8f6fc5405819408986cc0822197bfbd9 0.2739297694845844
# f177de608c51420c904470e3f7610f0b 0.27383941938774425
# cc0bd4bd1055472a98db7f0507546b44 0.27352790140289795
# 6402fd1ffaf041d0b9162bd92a7ba0a2 0.2734317981710993
# 9c79076d8e9646c5b77fca8fbe189d64 0.2731818246799048
# 4176983e28014670b8d4599c5e70166a 0.27315815456868464
# 789b705cdbcd4e5bb43812186d612d7d 0.273056102091662
# 704f3a805e16454aa3acade23171f2c3 0.2728849287384354
# 90493070491d4adeb6098ced8980d14b 0.27285664096034534
# a4747e221be5477a8b0cba64d9032bcd 0.27255627016563444
# 1f7c7394281b4a5ab8be31646be276e7 0.27229666759149274
# 8cd9b9f28b6b44e3933216e65bbfbbd4 0.2722469321665175
# 5bbb8e5b260a4a13870b0239bbd764b4 0.2718868215361089
# 07036c3a872444c8a7f369d796aa401f 0.2715637876504602
# 8c394cbb9706486aae5c0edc1d704178 0.27118749140990916
# 1144cb46dfd6402cb2bd2b0d0a485594 0.2709812243734164
# cc6d0652e2eb4240b6f07ad1be99109c 0.27067358227569716
# a11fc6258e564dc4ab612bdf98e90fd1 0.27063692110923904
# 1af97f63a062426e846250a28346a735 0.27060595971517826
# 5b790dd0fb6e4545bc7f3273aa178d4c 0.2705274653343027
# 8820971be64f48249658358bee1f50a5 0.27033324045234164
# f40544fd4f5d42abbcfa948eeaf86850 0.2702654642541005
# 134627f07b8b47f196af46f88013d112 0.27011071412710985
# 5b29d1a2b0e646368622dee5e8fa5108 0.2697855258871109
# ecb9d6d3ce7944bc9d6a599aa9d25abc 0.26977708447333326
# 95f0230140d24bf49f707545141c5044 0.2696798014553414
# 6a85bfb241cc42a3a7f1e8d516f74ff5 0.2696140610373774
# 778c13ddbcd640fbb27d6b5cb18a76a9 0.26949297842538766
# c686a80773ef45c3a850112e98f21df6 0.26932312260148644
# 749272e58d20497282225997c6c99602 0.2690794331807065
# 57f43a802933455389ec31a2a16c551f 0.26906701300789954
# 81884f7b4b1f45a3bd952bfbcf5651f5 0.26862477656063144
# f666b893286a4511afd9c799841d427b 0.2679213754408101
# b9ea04a6121d4a8bb00199b885aa5ef0 0.26787957372213883
# 0b3bc272ceff4fc8aa5aff143b36a275 0.26780241967904905
# ba8eeec4aec74e2e8cfbdd78af5484c1 0.2670280609150176
# 9645597ea6ce4637ba0c155041d888a9 0.2665104102299308
# 069b2210186c49c383b08dfded66ff35 0.26634088539007283
# 45266b136cf846b0a7d398d2360f9a17 0.26609193022792743
# 7f6f982ee5894e2d87c654abb159b117 0.26590416379810444
# 1263172dcdf2494c9d4b252875233079 0.26583944260215187
# d09b065aa1c74e31b0742ec36052f9c9 0.26573700266295563
# 78b6b787491441fa864d6bd9382d5701 0.26557005118807286
# 2493d6ff221e4dfda32f3f46dfd02fa3 0.2655641016237119
# 14c2fb7926b5423a842f8a7f1240e977 0.2655299331877483
# 87e772078a494d42bd34cd16172808bc 0.2654652128849958
# 6513a70d52f44052a7e7014e11f30a3a 0.2653777857571121
# dafe7793990d41a0ae1e50260899f8b4 0.2649743561888973
# f2d691553d5040a6bd82b7ea63f78042 0.2648386030022887
# 26b713b54ec04cbaba823597730cbbea 0.26480672331431343
# ac7e7a21d29b41e6acc78ce8ef5ade66 0.2646902153974329
# 970f41b6fc4544b388fe7ccd578988a9 0.2645881043594898
# c2a4c914bd064ede98dc6b3c193050a8 0.2643848620999141
# 5ddf6aeb821146de8fd98fd73dfe02df 0.26417360881966945
# c223c6a23bdb4aed96e70a0ebee9dfed 0.2637239752142785
# eba35b7afd6d4b9f8f4eebc6477933de 0.263699878774001
# cd14e6505cdb4627844cc6065d6e3f02 0.2635668623965611
# 2483f870bba9443aa33cda63424b0e37 0.26348262992977006