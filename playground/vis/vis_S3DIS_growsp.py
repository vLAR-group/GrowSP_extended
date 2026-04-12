import torch
import torch.nn.functional as F
from datasets.S3DIS import S3DISvis, cfl_collate_fn_vis
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

import matplotlib.pyplot as plt
colormap = []
for k in range(12):
    colormap.append(plt.cm.Set3(k))
colormap.append([0, 0, 0, 0])
colormap = np.array(colormap)[:, 0:3]*255

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='../data/S3DIS/input_0.010/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='../data/S3DIS/initial_superpoints_growsp/',
                        help='initial sp path')
    parser.add_argument('--save_path', type=str, default='/home/zihui/SSD2/GrowSP++_vis/S3DIS/',
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


def vis_preds(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load('/home/zihui/SSD/GrowSP_newExt/ckpt_seg/S3DIS_Area6/primitive_grow/distillv2_80-20sp_kmeans_elsaug_t3_fixmodel_0.9_recls/model_300_checkpoint.pth'))
    model.eval()

    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.semantic_class, centroids_num=args.feats_dim, centroids=centroids).cuda()
    classifier.eval()

    trainval_dataset = S3DISvis(args, areas=['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6'])
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    use_sp = False
    all_full_preds, all_full_labels, all_full_coords, all_full_colors = [], [], [], []
    for data in trainval_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region, full_coords, full_colors, full_labels = data

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

            preds_full = preds[inverse_map.long()]
            all_full_preds.append(preds_full[full_labels!=args.ignore_label]), all_full_labels.append(full_labels[full_labels!=args.ignore_label]), \
            all_full_coords.append(full_coords[full_labels!=args.ignore_label]), all_full_colors.append(full_colors[full_labels!=args.ignore_label])


    preds = np.concatenate(all_full_preds)
    labels = np.concatenate(all_full_labels)
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
        mask = (label!=-1)&(label!=0)
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

        colors_input = all_full_colors[i].numpy()
        colors_input = colors_input[mask].astype(np.uint8)

        # Save plys
        cloud_name = trainval_loader.dataset.name[i]

        pred_path, GT_path, input_path = os.path.join(args.save_path, 'IIC'), os.path.join(args.save_path, 'GT'), os.path.join(args.save_path, 'input')
        os.makedirs(pred_path, exist_ok=True)
        os.makedirs(GT_path, exist_ok=True)
        os.makedirs(input_path, exist_ok=True)

        pred_filename = os.path.join(pred_path, cloud_name+'IIC.ply')
        GT_filename = os.path.join(GT_path, cloud_name+'GT.ply')
        input_filename = os.path.join(input_path, cloud_name+'input.ply')

        write_ply(pred_filename, [coords, colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # write_ply(GT_filename, [coords, colors_GT.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # write_ply(input_filename, [coords, colors_input.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])


    # Pair the numbers with the strings
    paired_list = list(zip(all_miou, trainval_loader.dataset.name))
    # Sort the paired list by the numbers in descending order
    sorted_paired_list = sorted(paired_list, key=lambda x: x[0], reverse=True)
    # Unzip the sorted pairs back into two lists
    sorted_numbers, sorted_strings = zip(*sorted_paired_list)
    # Convert the tuples back to lists (if needed)
    sorted_numbers = list(sorted_numbers)
    sorted_strings = list(sorted_strings)
    for i in range(30):
        print(sorted_strings[i], sorted_numbers[i])


if __name__ == '__main__':

    args = parse_args()
    print(args)
    vis_preds(-1, args)



# Area_1_office_12 0.5133396111618604
# Area_1_office_6 0.4916121665951052
# Area_1_office_28 0.48322874265059007
# Area_1_office_20 0.4825551090860322
# Area_1_office_18 0.47692808751901605
# Area_1_office_21 0.4703886028803555
# Area_1_office_13 0.4696265693892647
# Area_1_office_17 0.4691076176658022
# Area_1_office_22 0.45958500174539885
# Area_1_office_26 0.4506369316945706
# Area_1_office_11 0.4499542905415839
# Area_1_office_14 0.4423278414845262
# Area_1_office_19 0.4290455553945491
# Area_1_office_16 0.4278655063506311
# Area_1_office_23 0.42280886484584673
# Area_1_office_7 0.4158590511080516
# Area_1_office_4 0.41528336108062963
# Area_1_office_25 0.41502474169657894
# Area_1_office_24 0.4039782075596851
# Area_1_conferenceRoom_2 0.40022673659549374
# Area_1_conferenceRoom_1 0.396254738403202
# Area_1_office_5 0.3951263324722522
# Area_1_office_1 0.38777517029443037
# Area_1_office_9 0.38303023959654015
# Area_1_office_3 0.37064357989063795
# Area_1_office_30 0.35721973937076185
# Area_1_office_27 0.3563245363471977
# Area_1_office_2 0.35041432797399136
# Area_1_office_8 0.34464528210733664
# Area_1_office_15 0.34013817988232414


# Area_2_office_5 0.444796691842247
# Area_2_office_13 0.405486705037565
# Area_2_office_3 0.39394194063664584
# Area_2_office_10 0.3751003148904488
# Area_2_office_12 0.3682534904223928
# Area_2_storage_3 0.36423329790360587
# Area_2_office_1 0.35778467240895045
# Area_2_office_9 0.35571992856667073
# Area_2_storage_4 0.35204747938837055
# Area_2_office_4 0.34602763377925744
# Area_2_office_7 0.32572931410704103
# Area_2_office_2 0.31600319282928074
# Area_2_office_11 0.2970449794667475
# Area_2_WC_2 0.2886061906315421
# Area_2_storage_2 0.28392299359062895
# Area_2_conferenceRoom_1 0.2805003991552781
# Area_2_storage_7 0.2688879852079755
# Area_2_office_8 0.2616786667379985
# Area_2_hallway_11 0.24916505495982597
# Area_2_hallway_3 0.24729330725738552
# Area_2_auditorium_1 0.2459547688038282
# Area_2_hallway_10 0.24201522009555212
# Area_2_auditorium_2 0.2341374080867801
# Area_2_hallway_5 0.2312644828082192
# Area_2_storage_1 0.23056827446277398
# Area_2_office_14 0.2190650313634569
# Area_2_hallway_7 0.21804889445094058
# Area_2_WC_1 0.21675473200387377
# Area_2_hallway_12 0.2163886666142898
# Area_2_hallway_2 0.21054568771556192


# Area_3_office_6 0.5257793138528067
# Area_3_office_2 0.5208295082930204
# Area_3_office_5 0.486092566247446
# Area_3_office_1 0.47915127844697686
# Area_3_office_4 0.46181873885249575
# Area_3_office_8 0.46046778730723464
# Area_3_office_10 0.44730225008644736
# Area_3_conferenceRoom_1 0.43487268611173197
# Area_3_office_7 0.42711191663992215
# Area_3_office_9 0.4185740026217542
# Area_3_office_3 0.3957590826016307
# Area_3_lounge_2 0.2973730051366697
# Area_3_WC_2 0.23921690854843203
# Area_3_storage_1 0.2364706210757109
# Area_3_WC_1 0.22697045756794268
# Area_3_hallway_3 0.2042226185600413
# Area_3_lounge_1 0.20131583073474782
# Area_3_hallway_5 0.1613920921635783
# Area_3_hallway_6 0.15771402211253457
# Area_3_hallway_4 0.13766322084252133
# Area_3_storage_2 0.13323829283052135
# Area_3_hallway_2 0.11876494350434207
# Area_3_hallway_1 0.10342104996418951

# Area_4_office_19 0.42874886608238233
# Area_4_office_20 0.3970048457034951
# Area_4_office_17 0.39673860490794693
# Area_4_office_2 0.3918097380167125
# Area_4_office_6 0.38427131566450234
# Area_4_office_16 0.3824094514625959
# Area_4_office_3 0.37084497727899435
# Area_4_office_9 0.36987945369232844
# Area_4_office_15 0.36633794978374407
# Area_4_office_22 0.36613955081544763
# Area_4_office_13 0.3575070840651808
# Area_4_office_4 0.3516319572480611
# Area_4_office_5 0.3429665204225801
# Area_4_office_12 0.33687049476048797
# Area_4_office_8 0.33439799154238997
# Area_4_office_7 0.3334738962543795
# Area_4_office_18 0.3327242303098063
# Area_4_office_21 0.3283458098489373
# Area_4_office_10 0.3124842025281716
# Area_4_office_11 0.3115051430544707
# Area_4_lobby_1 0.3077139411140751
# Area_4_hallway_14 0.2949878264404766
# Area_4_lobby_2 0.2943496317750913
# Area_4_conferenceRoom_2 0.291862238922492
# Area_4_conferenceRoom_3 0.28233153617222395
# Area_4_WC_4 0.2805068454702752
# Area_4_office_14 0.2798349335493145
# Area_4_conferenceRoom_1 0.2728157357672854
# Area_4_storage_3 0.26678239927248265
# Area_4_office_1 0.2511011358492665

# Area_5_office_9 0.4479514406765111
# Area_5_office_28 0.4424544232683582
# Area_5_office_4 0.4407278474418351
# Area_5_office_5 0.43478904346725794
# Area_5_office_10 0.4328024413478506
# Area_5_office_22 0.4236724051834857
# Area_5_office_11 0.4229913233242201
# Area_5_office_6 0.42242781651529304
# Area_5_office_33 0.4192434220882757
# Area_5_office_7 0.4160457361859235
# Area_5_office_2 0.4133750663241953
# Area_5_office_8 0.4098346132552688
# Area_5_conferenceRoom_2 0.40567694903811985
# Area_5_office_14 0.4019271582437858
# Area_5_office_13 0.4001534313767514
# Area_5_office_26 0.39922034075484336
# Area_5_office_18 0.39640125795181286
# Area_5_office_27 0.39436324162382824
# Area_5_office_31 0.3920492125099626
# Area_5_office_23 0.3886513147334369
# Area_5_office_24 0.3821144878828111
# Area_5_office_32 0.3821080869446305
# Area_5_office_3 0.380646321430317
# Area_5_office_12 0.37932544551995767
# Area_5_office_42 0.37676566415197116
# Area_5_office_29 0.37289476223627066
# Area_5_office_25 0.3720365399677518
# Area_5_office_41 0.36726261888077477
# Area_5_conferenceRoom_1 0.365034977656755
# Area_5_office_30 0.34681030373161853

# Area_6_office_29 0.5219356293400774
# Area_6_office_34 0.5071922617882628
# Area_6_office_35 0.48778799983997684
# Area_6_office_23 0.48742192058177763
# Area_6_office_25 0.4847275147505385
# Area_6_office_12 0.480151503024088
# Area_6_office_26 0.4742587676464988
# Area_6_office_31 0.47116816247510723
# Area_6_office_10 0.4677998692286652
# Area_6_office_36 0.46127969435831156
# Area_6_office_20 0.46003886178314596
# Area_6_office_24 0.4549220498675526
# Area_6_office_3 0.4529239816743133
# Area_6_office_28 0.45089518316355964
# Area_6_office_11 0.44663394943441354
# Area_6_office_27 0.4444519203380206
# Area_6_office_32 0.44242252196174015
# Area_6_office_2 0.4388670927486682
# Area_6_office_21 0.43216665916439595
# Area_6_office_17 0.432116759937797
# Area_6_office_19 0.4283816671400306
# Area_6_office_33 0.4268712892912714
# Area_6_office_22 0.4177037533555004
# Area_6_office_6 0.4163640286605113
# Area_6_office_5 0.4063707402177023
# Area_6_office_1 0.4052473223798357
# Area_6_office_15 0.3995576855051441
# Area_6_office_30 0.3910535113617633
# Area_6_office_8 0.39095352246870685
# Area_6_office_37 0.39037674513202064