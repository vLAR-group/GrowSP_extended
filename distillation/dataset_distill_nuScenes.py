import torch, sys, os, pickle
current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import numpy as np
from torch.utils.data import Dataset
from lib.aug_tools import rota_coords, scale_coords, trans_coords
from os.path import join
from tqdm import tqdm

class nuScenesdistill(Dataset):
    def __init__(self, args, scene_idx, split='train'):
        self.args = args
        self.label_to_names = {0: 'barrier',
                               1: 'bicycle',
                               2: 'bus',
                               3: 'car',
                               4: 'construction vehicle',
                               5: 'motorcycle',
                               6: 'person',
                               7: 'traffic cone',
                               8: 'trailer',
                               9: 'truck',
                               10: 'drivable surface',
                               11: 'other flat',
                               12: 'sidewalk',
                               13: 'terrain',
                               14: 'manmade',
                               15: 'vegetation',
                               -1: 'unlabeled'}
        self.split = split
        self.train_path_list = []
        self.feats_list = []
        self.feats_datas = []
        self.points_datas = []

        scene_list = np.sort(os.listdir(args.data_path))
        for scene_id in scene_list:
            scene_path = join(args.data_path, scene_id)
            self.train_path_list.append(scene_path)
            ##
            feat_path = join(args.feats_path, scene_id[:-4] + '.pt')
            self.feats_list.append(feat_path)

        self.random_select_sample(scene_idx)
        self.preload_data()

        self.rota_coords = rota_coords(rotation_bound=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%

    def voxelize(self, coords, feature):
        scale = 1 / self.args.voxel_size
        coords = coords - coords.min(0)
        grids = np.floor(coords * scale)
        grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
        return grids, feature[unique_map], unique_map, inverse_map

    def random_select_sample(self, scene_idx):
        self.name = []
        self.file_selected = []
        self.feats_selected = []
        for i in scene_idx:
            self.file_selected.append(self.train_path_list[i])
            self.feats_selected.append(self.feats_list[i])
            self.name.append(self.train_path_list[i][0:-4].replace(self.args.data_path, ''))

    def preload_data(self):
        for featpath, filepath in tqdm(zip(self.feats_selected, self.file_selected), desc='Pre Load Datas(1500)'):
            point_feat = torch.load(featpath)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.feats_datas.append(point_feat)
            self.points_datas.append(data)

    def augs(self, coords):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords

    def __len__(self):
        return len(self.file_selected)

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=self.collate_fn, shuffle=shuffle, drop_last=True)

    def __getitem__(self, index):
        data = self.points_datas[index]
        pc = data['coords']
        labels = data['labels']
        label_mask = labels == 255
        labels[label_mask] = -1
        pc = pc.astype(np.float32)
        pc -= pc.mean(0)
        pc = self.augs(pc)

        grids, feature, unique_map, inverse_map = self.voxelize(pc, pc)

        feats = self.feats_datas[index]
        feat_3d = feats['feat']
        mask_chunk = feats['mask_full']
        mask_chunk = torch.from_numpy(mask_chunk)

        ## DINO feats
        mask = mask_chunk[unique_map]
        mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
        index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
        index1[mask_ind] = mask_ind
        index1 = index1[unique_map]
        chunk_ind = index1[index1 != -1]
        index2 = torch.zeros(mask_chunk.shape[0])
        index2[mask_ind] = 1
        index3 = torch.cumsum(index2, dim=0, dtype=int)
        # get the indices of corresponding masked point features after voxelization
        indices = index3[chunk_ind] - 1
        feat_3d = feat_3d[indices]  ##点特征的体素化
        return grids, feature, labels, inverse_map, feat_3d, mask


    def collate_fn(self, batch):
        coords, feats, labels, inverse_map, dino_feats, mask = list(zip(*batch))
        coords_batch, feats_batch, labels_batch, inverse_batch, dino_feats_batch, mask_batch = [], [], [], [], [], []

        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(inverse_map[batch_id])
            dino_feats_batch.append(dino_feats[batch_id])
            mask_batch.append(mask[batch_id])

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        dino_feats = torch.cat(dino_feats_batch, 0)
        return coords_batch, feats_batch, labels_batch, inverse_batch, dino_feats, torch.cat(mask_batch)