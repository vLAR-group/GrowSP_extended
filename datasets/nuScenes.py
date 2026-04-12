import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
from os.path import join
from lib.aug_tools import rota_coords, scale_coords, trans_coords
from tqdm import tqdm

class nuScenesvis(Dataset):
    def __init__(self, args):
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
        self.name = []
        self.mode = 'val'
        self.file = []

        # val_path_list = []
        scene_list = np.sort(os.listdir(args.val_input_path))#[0:1000]
        for scene_id in scene_list:
            scene_path = join(args.val_input_path, scene_id)
            self.file.append(scene_path)
            self.name.append(scene_id[0:-4])

    def voxelize(self, coords, feature):
        scale = 1 / self.args.voxel_size
        coords = coords - coords.min(0)
        grids = np.floor(coords * scale)
        grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
        return grids, feature[unique_map], unique_map, inverse_map

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        with open(file, 'rb') as f:
            data = pickle.load(f)
        pc = data['coords']
        labels = data['labels']
        ##
        label_mask = labels == 255
        labels[label_mask] = -1
        pc = pc.astype(np.float32)

        grids, feature, unique_map, inverse_map = self.voxelize(pc, pc)
        voxel_labels = labels[unique_map]

        return grids, feature, inverse_map, np.ascontiguousarray(voxel_labels), index, pc, labels


class cfl_collate_fn_vis:
    def __call__(self, list_data):
        grids, feature, inverse_map, voxel_labels, index, pc, labels = list(zip(*list_data))
        grids_batch, voxel_labels_batch, feature_batch = [], [], []
        pc_batch, labels_batch = [], []
        for batch_id, _ in enumerate(grids):
            num_points = grids[batch_id].shape[0]
            grids_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(grids[batch_id]).int()), 1))
            voxel_labels_batch.append(torch.from_numpy(voxel_labels[batch_id]).int())
            feature_batch.append(torch.from_numpy(feature[batch_id]).float())
            pc_batch.append(torch.from_numpy(pc[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]))
        #
        # Concatenate all lists
        grids_batch = torch.cat(grids_batch, 0).int()
        voxel_labels_batch = torch.cat(voxel_labels_batch, 0).int()
        feature_batch = torch.cat(feature_batch, 0).float()
        pc_batch = torch.cat(pc_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).int()
        return grids_batch, feature_batch, inverse_map, voxel_labels_batch, index, pc_batch, labels_batch


class cfl_collate_fn:
    def __call__(self, list_data):
        grids, feats, labels, inverse_map, pseudo, region, index = list(zip(*list_data))
        grids_batch, feats_batch, labels_batch, pseudo_batch = [], [], [], []
        region_batch = []
        accm_num = 0
        for batch_id, _ in enumerate(grids):
            num_points = grids[batch_id].shape[0]
            grids_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(grids[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            region_batch.append(torch.from_numpy(region[batch_id])[:,None])
            accm_num += grids[batch_id].shape[0]

        # Concatenate all lists
        grids_batch = torch.cat(grids_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        region_batch = torch.cat(region_batch, 0).int()
        return grids_batch, feats_batch, labels_batch, inverse_map, pseudo_batch, region_batch, index


class nuScenestrain(Dataset):
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
        self.mode = 'train'
        self.split = split
        self.train_path_list = []

        scene_list = np.sort(os.listdir(args.data_path))
        for scene_id in scene_list:
            scene_path = join(args.data_path, scene_id)
            self.train_path_list.append(scene_path)

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
        self.random_select_sample(scene_idx)

    def random_select_sample(self, scene_idx):
        self.name = []
        self.file_selected = []
        for i in scene_idx:
            self.file_selected.append(self.train_path_list[i])
            self.name.append(self.train_path_list[i][0:-4].replace(self.args.data_path, '').split('/')[1])

    def augs(self, coords):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords

    def voxelize(self, coords, feature):
        scale = 1 / self.args.voxel_size
        coords = coords - coords.min(0)
        grids = np.floor(coords * scale)
        grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
        return grids, feature[unique_map], unique_map, inverse_map

    def __len__(self):
        return len(self.file_selected)

    def __getitem__(self, index):
        file = self.file_selected[index]
        with open(file, 'rb') as f:
            data = pickle.load(f)
        pc = data['coords']
        labels = data['labels']
        label_mask = labels == 255
        labels[label_mask] = -1
        pc = pc.astype(np.float32)

        mask = np.sqrt((pc**2).sum(-1))< 50
        region_file = os.path.join(self.args.sp_path, self.name[index] + '_superpoint.npy')
        region = np.load(region_file)

        pc, feature, region, labels = pc[mask], pc[mask], region[mask], labels[mask]
        if self.mode == 'train':
            pc = self.augs(pc)

        grids, feature, unique_map, inverse_map = self.voxelize(pc, pc)
        region = region[unique_map]
        labels = labels[unique_map]

        '''mode must be cluster or train'''
        if self.mode == 'cluster':
            region[labels == -1] = -1
            original_region = region.copy()
            for q in np.unique(region):
                mask = q == region
                if mask.sum() < self.args.drop_threshold and q != -1:
                    region[mask] = -1
                    if np.all(region == -1):
                        region = original_region
                        break

            valid_region = region[region != -1]
            unique_vals = np.unique(valid_region)
            unique_vals.sort()
            valid_region = np.searchsorted(unique_vals, valid_region)
            region[region != -1] = valid_region
            pseudo = -np.ones_like(labels).astype(np.int32)

        else:
            scene_name = self.name[index]
            file_path = os.path.join(self.args.save_path, self.args.pseudo_label_path, scene_name+'.npy')
            pseudo = np.array(np.load(file_path), dtype=np.int32)[unique_map]
        return grids, feature, labels, inverse_map, pseudo, region, index


class nuScenesval(Dataset):
    def __init__(self, args):
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
        self.name = []
        self.mode = 'val'
        self.file = []

        scene_list = np.sort(os.listdir(args.val_input_path))
        for scene_id in scene_list:
            scene_path = join(args.val_input_path, scene_id)
            self.file.append(scene_path)

    def voxelize(self, coords, feature):
        scale = 1 / self.args.voxel_size
        coords = coords - coords.min(0)
        grids = np.floor(coords * scale)
        grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
        return grids, feature[unique_map], unique_map, inverse_map

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        with open(file, 'rb') as f:
            data = pickle.load(f)
        pc = data['coords']
        labels = data['labels']
        ##
        label_mask = labels == 255
        labels[label_mask] = -1
        pc = pc.astype(np.float32)

        grids, feature, unique_map, inverse_map = self.voxelize(pc, pc)
        return grids, feature, np.ascontiguousarray(labels), inverse_map, index


class cfl_collate_fn_val:
    def __call__(self, list_data):
        grids, feats, labels, inverse_map, index = list(zip(*list_data))
        grids_batch, feats_batch, labels_batch = [], [], []
        accm_num = 0
        for batch_id, _ in enumerate(grids):
            num_points = grids[batch_id].shape[0]
            grids_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(grids[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            accm_num += grids[batch_id].shape[0]

        # Concatenate all lists
        grids_batch = torch.cat(grids_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).int()
        return grids_batch, feats_batch, labels_batch, inverse_map, index


class nuScenestest(Dataset):
    def __init__(self, args):
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

        self.mode = 'test'
        self.file = []

        self.name = []
        scene_list = np.sort(os.listdir(args.test_input_path))
        for scene_id in scene_list:
            scene_path = join(args.test_input_path, scene_id)
            name = scene_path.replace(args.test_input_path, '').split('/')[1]
            self.name.append(name[0:-4])
            self.file.append(scene_path)


    def voxelize(self, coords, feature):
        scale = 1 / self.args.voxel_size
        coords = coords - coords.min(0)
        grids = np.floor(coords * scale)
        grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
        return grids, feature[unique_map], unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        pc = np.load(file)
        ##
        scene_name = self.name[index]
        grids, feature, unique_map, inverse_map = self.voxelize(pc, pc)
        return grids, feature, inverse_map.numpy(), index, pc, scene_name


class cfl_collate_fn_test:
    def __call__(self, list_data):
        grids, feature, inverse_map, index, pc, scene_name= list(zip(*list_data))
        grids_batch, feature_batch, pc_batch, scene_name_batch= [], [], [],[]
        for batch_id, _ in enumerate(grids):
            num_points = grids[batch_id].shape[0]
            grids_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(grids[batch_id]).int()), 1))
            pc_batch.append(torch.from_numpy(pc[batch_id]))
            scene_name_batch.append(scene_name[batch_id])
            feature_batch.append(feature[batch_id])
        #
        # Concatenate all lists
        grids_batch = torch.cat(grids_batch, 0).int()
        pc_batch = torch.cat(pc_batch, 0).float()
        feature_batch = torch.cat(feature_batch, 0).float()
        return grids_batch, feature_batch, inverse_map, index, pc_batch, scene_name_batch