import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from sklearn.cluster import KMeans, SpectralClustering
import warnings
from lib.helper_ply import read_ply, write_ply
from sparse_fpn import Res16FPN18
from sklearn.preprocessing import LabelEncoder
import colorsys
from typing import List, Tuple
import functools
warnings.filterwarnings('ignore')
import spconv.pytorch as spconv
import matplotlib.pyplot as plt
voxel_size= 0.05
colormap_sp = []
for _ in range(30):
    for k in range(12):
        colormap_sp.append(plt.cm.Set3(k))
    for k in range(9):
        colormap_sp.append(plt.cm.Set1(k))
    for k in range(8):
        colormap_sp.append(plt.cm.Set2(k))
colormap_sp.append((0, 0, 0, 0))
colormap_sp = np.array(colormap_sp)[:, 0:3]*255

colormap_gt = np.array(
    [[245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
     [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240],
     [  0, 128, 128], [230,  25,  75], [170, 110,  40], [255, 250, 200], [128,   0,   0],
     [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128]])

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))

def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines

def contin_label(label):
    Encoder = LabelEncoder()
    return Encoder.fit_transform(label)


def voxelize(coords):
    scale = 1 / voxel_size
    coords = coords - coords.min(0)
    grids = np.floor(coords * scale)
    grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
    return grids, unique_map, inverse_map

def augment_coords_to_feats(coords, colors):
    coords_center = coords.float().mean(0, keepdims=True)
    coords_center[0, 2] = 0
    norm_coords = (coords - coords_center)

    feats = norm_coords
    feats = np.concatenate((colors, feats), axis=-1)
    return norm_coords, feats

model = Res16FPN18()
model.load_state_dict(torch.load('ckpt_seg/ScanNet_growsp/primitive_grow_abl/distillv2_80-30sp_kmeans_elsaug_t3_fixmodel_multi0.9_nodecaylr_end30/model_300_checkpoint.pth'))
model.eval().cuda()

plypath = 'data/ScanNet/processed'
sp_path = 'data/ScanNet/initial_superpoints'
semantic_num = 20
primitive_num = 20
segment_num = 30
feats_dim = 384
colormap = colormap_sp#get_evenly_distributed_colors(primitive_num)
save_path = os.path.join('vis_'+str(primitive_num)+'_'+str(segment_num))
os.makedirs(save_path, exist_ok=True)
train_scene_id = read_txt('data_prepare/ScanNet_splits/scannetv2_train.txt')#[0:100]
time_start = time.time()

context, all_sp_feats = {}, []
acc_no = 0
for scene_id in train_scene_id:
    print(scene_id)
    data = read_ply(os.path.join(plypath, scene_id))
    coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
    colors = colors.astype(np.float32)/255-0.5
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)

    # os.makedirs(os.path.join(save_path, 'gt'), exist_ok=True)
    # color_gt = np.ones_like(coords) * 128
    # for cate in range(semantic_num):
    #     color_gt[cate == labels] = colormap_gt[cate]
    # write_ply(os.path.join(os.path.join(save_path, 'gt'), scene_id + '_gt.ply'), [coords,
    #                 color_gt.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

    context[scene_id] = {'full_gt': labels, 'full_color': colors, 'full_coords': coords}

    grids, unique_map, inv_map = voxelize(coords)
    coords, colors, labels = coords[unique_map], colors[unique_map], labels[unique_map]
    context[scene_id]['vox_coords'] = coords
    context[scene_id]['inv_map'] = inv_map

    region_file = os.path.join(sp_path, scene_id[:-4]+'_superpoint.npy')
    region = np.load(region_file)
    region = region[unique_map]

    ######### modify spuerpoints idx ###############
    region[labels == -1] = -1

    for q in np.unique(region):
        mask = q == region
        if mask.sum() < 10 and q != -1:
            region[mask] = -1

    valid_region = region[region != -1]
    valid_region = contin_label(valid_region)

    region[region != -1] = valid_region
    region = torch.from_numpy(region).long()
    ####################################################
    """valid mask"""
    valid_mask = region!=-1
    coords, grids, colors, labels, sp = coords[valid_mask], grids[valid_mask], colors[valid_mask], labels[valid_mask], region[valid_mask]
    feats = augment_coords_to_feats(grids, torch.from_numpy(colors))[1]
    colors, coords, feats, labels = torch.from_numpy(colors).cuda(), torch.from_numpy(coords).cuda(), torch.from_numpy(feats).cuda(), torch.from_numpy(labels)

    with torch.no_grad():
        grids = torch.cat((torch.zeros_like(grids)[:, 0][:, None], grids), -1).int()
        in_field = spconv.SparseConvTensor(features=feats.float().cuda(), indices=grids.int().cuda(),
                spatial_shape=list(grids.max(0)[0] + 16)[1:], batch_size=1)
        feats = model(in_field)

    # context[scene_id] = {'gt': labels, 'initial_sp': sp, 'coords': coords, 'label':labels}
    context[scene_id]['coords'] = coords
    context[scene_id]['initial_sp'] = sp
    context[scene_id]['label'] = labels

    ''' cluster initial spuerpoints to 20, means one step growing, may not needed'''
    sp_num = len(torch.unique(sp))
    sp_corr = torch.zeros(sp.size(0), sp_num)  # ?
    sp_corr.scatter_(1, sp.view(-1, 1), 1)
    sp_corr = sp_corr.cuda()
    per_sp_num = sp_corr.sum(0, keepdims=True).t()

    sp_feats = F.linear(sp_corr.t(), feats.t()) / per_sp_num
    sp_feats = F.normalize(sp_feats, dim=-1)

    if sp_feats.size(0) < segment_num:
        n_segments = sp_feats.size(0)
    else:
        n_segments = segment_num
    sp_idx = KMeans(n_clusters=n_segments, n_init=10, random_state=0, n_jobs=-1).fit_predict(sp_feats.cpu().numpy())
    sp_idx = contin_label(sp_idx)
    sp_idx = torch.from_numpy(sp_idx).long()
    sp = sp_idx[sp]
    ## can use to vis curr spuerpoints
    color = np.ones_like(coords.cpu().numpy()) * 128
    for cate in torch.unique(sp):
        color[cate == sp] = colormap_sp[cate]

    dist = torch.cdist(coords.half(), torch.from_numpy(context[scene_id]['vox_coords']).cuda().half())
    context[scene_id]['corr'] = dist.min(0)[1].cpu().long()
    # os.makedirs(os.path.join(save_path, 'sp'), exist_ok=True)
    # write_ply(os.path.join(os.path.join(save_path, 'sp'), scene_id + '_sp.ply'), [context[scene_id]['full_coords'],
    #                 color[context[scene_id]['corr']][context[scene_id]['inv_map']].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

    ''' get new sp idx for the 20 segments'''
    sp_num = len(torch.unique(sp))
    sp_corr = torch.zeros(sp.size(0), sp_num)
    sp_corr.scatter_(1, sp.view(-1, 1), 1)
    sp_corr = sp_corr.cuda()
    per_sp_num = sp_corr.sum(0, keepdims=True).t()
    sp_feats = F.linear(sp_corr.t(), feats.t()) / per_sp_num
    sp_feats = F.normalize(sp_feats, dim=-1)
    ''' get sp features (initial or growed)'''
    context[scene_id]['final_sp'] = sp
    context[scene_id]['acc_sp'] = sp+acc_no
    all_sp_feats.append(sp_feats)
    acc_no += len(sp[sp!=-1].unique())

    torch.cuda.empty_cache()
    torch.cuda.synchronize(torch.device("cuda"))
print('sp features extracted')

all_sp_feats = torch.cat(all_sp_feats)
primitive_labels = KMeans(n_clusters=primitive_num, n_jobs=-1).fit_predict(all_sp_feats.cpu().numpy().astype(np.float32))
primitive_labels = contin_label(primitive_labels)
##
for idx, scene_id in enumerate(train_scene_id):
    coords, acc_sp = context[scene_id]['coords'].cpu().numpy(), context[scene_id]['acc_sp']
    labels = context[scene_id]['label'].cpu().numpy().astype(np.int32)
    color = np.ones_like(coords) * 128
    color_gt = np.ones_like(coords) * 128
    cur_primitive_labels = primitive_labels[acc_sp.long()]
    for cate in range(primitive_num):
        color[cate == cur_primitive_labels] = colormap[cate]
    for cate in range(semantic_num):
        color_gt[cate == labels] = colormap_gt[cate]
    os.makedirs(os.path.join(save_path, 'pri2'), exist_ok=True)
    write_ply(os.path.join(os.path.join(save_path, 'pri2'), scene_id + '_votecate.ply'), [context[scene_id]['full_coords'],
                color[context[scene_id]['corr']][context[scene_id]['inv_map']].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

