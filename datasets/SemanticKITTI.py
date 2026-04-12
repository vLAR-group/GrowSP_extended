import torch
import numpy as np
from lib.helper_ply import read_ply, write_ply
from torch.utils.data import Dataset
import me_compat as ME
import os
import open3d as o3d
from lib.aug_tools import rota_coords, scale_coords, trans_coords

class cfl_collate_fn:

    def __call__(self, list_data):
        coords, labels, inverse_map, pseudo, region, index, scene_name = list(zip(*list_data))
        coords_batch, labels_batch, inverse_batch, pseudo_batch = [], [], [], []
        region_batch = []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()  # .int()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        region_batch = torch.cat(region_batch, 0)

        return coords_batch, labels_batch, inverse_batch, pseudo_batch, region_batch, index, scene_name


class KITTItrain(Dataset):
    def __init__(self, args, scene_idx, split='train'):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.mode = 'train'
        self.split = split
        self.val_split = '08'
        self.file = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if self.split == 'train':
                if seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))

            elif self.split == 'val':
                if seq_id == '08':
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))
                    scene_idx = range(len(self.file))

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
        self.random_select_sample(scene_idx)

    def random_select_sample(self, scene_idx):
        self.name = []
        self.file_selected = []
        for i in scene_idx:
            self.file_selected.append(self.file[i])
            self.name.append(self.file[i][0:-4].replace(self.args.data_path, ''))


    def augs(self, coords):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords


    def augment_coords_to_feats(self, coords):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords

    def voxelize(self, coords, labels):
        # nuscenes feats =None
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), features=None,
                                                                           labels=labels, ignore_label=-1,
                                                                           return_index=True, return_inverse=True)
        return coords, labels, unique_map, inverse_map

    def __len__(self):
        return len(self.file_selected)

    def __getitem__(self, index):
        file = self.file_selected[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)
        labels -= 1

        mask = np.sqrt((coords**2).sum(-1))< self.args.r_crop
        coords, feats, labels = coords[mask], feats[mask], labels[mask]

        # coords, feats, voxel_labels, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords, labels, unique_map, inverse_map = self.voxelize(coords, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)
        region = region[mask]
        region = region[unique_map]

        if self.mode == 'train':
            coords = self.augs(coords)
        coords = self.augment_coords_to_feats(coords)

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
            pseudo = -np.ones_like(labels).astype(np.long)

        else:
            scene_name = self.name[index]
            file_path = self.args.save_path + '/'+self.args.pseudo_label_path + '/' + scene_name + '.npy'
            pseudo = np.array(np.load(file_path), dtype=np.long)
        return coords, labels, inverse_map, pseudo, region, index, self.name[index]



class KITTIval(Dataset):
    def __init__(self, args, split='val'):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.name = []
        self.mode = 'val'
        self.split = split
        self.val_split = '08'
        self.file = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if self.split == 'val':
                if seq_id == '08':
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))
                        self.name.append(os.path.join(seq_path, f)[0:-4].replace(self.args.data_path, ''))


    def augment_coords_to_feats(self, coords):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords

    def voxelize(self, coords, labels):
        # nuscenes feats =None
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), features=None,
                                                                           labels=labels, ignore_label=-1,
                                                                           return_index=True, return_inverse=True)
        return coords, labels, unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)
        labels = labels -1

        coords, _, unique_map, inverse_map = self.voxelize(coords, labels)
        coords = coords.astype(np.float32)

        coords = self.augment_coords_to_feats(coords)
        return coords, np.ascontiguousarray(labels), inverse_map, index



class KITTItest(Dataset):
    def __init__(self, args, split='test'):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.name = []
        self.mode = 'test'
        self.split = split
        self.val_split = '08'
        self.file = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if self.split == 'test':
                if seq_id in ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']:
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))
                        self.name.append(os.path.join(seq_path, f)[0:-4].replace(self.args.data_path, ''))


    def augment_coords_to_feats(self, coords):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        feats = norm_coords
        return norm_coords

    def voxelize(self, coords):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords),
                                                                                  ignore_label=-1, return_index=True,
                                                                                  return_inverse=True)
        return coords.numpy(), unique_map, inverse_map.numpy()


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, unique_map, inverse_map = self.voxelize(coords)
        coords = coords.astype(np.float32)

        coords = self.augment_coords_to_feats(coords)
        return coords, inverse_map, index



class KITTIKmeans(Dataset):
    def __init__(self, args, scene_idx):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.file = []
        self.name = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                for f in np.sort(os.listdir(seq_path)):
                    self.file.append(os.path.join(seq_path, f))
                    self.name.append(os.path.join(seq_path, f)[0:-4].replace(self.args.data_path, ''))

        self.random_select_sample(scene_idx)

    def random_select_sample(self, scene_idx):
        self.name = []
        self.file_selected = []
        for i in scene_idx:
            self.file_selected.append(self.file[i])
            self.name.append(self.file[i][0:-4].replace(self.args.data_path, ''))

    def augment_coords_to_feats(self, coords, feats, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        feats = norm_coords
        return norm_coords, feats, labels

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats, labels, unique_map, inverse_map.numpy()


    def __len__(self):
        return len(self.file_selected)

    def __getitem__(self, index):
        file = self.file_selected[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, feats, labels, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)
        region = region[unique_map]

        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels)
        labels = labels -1

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
        normals = np.array(pcd.normals)

        feats = np.concatenate((feats, normals),axis=1)
        return coords, feats, (labels), inverse_map, region, index


class cfl_collate_fn_val:
    def __call__(self, list_data):
        coords, labels, inverse_map, index = list(zip(*list_data))
        coords_batch, inverse_batch, labels_batch = [], [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()

        return coords_batch, inverse_batch, labels_batch, index


class cfl_collate_fn_test:
    def __call__(self, list_data):
        coords, inverse_map, index = list(zip(*list_data))
        coords_batch, inverse_batch = [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()

        return coords_batch, inverse_batch, index



class KITTIvis(Dataset):
    def __init__(self, args, split='val'):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.name = []
        self.mode = 'val'
        self.split = split
        self.val_split = '08'
        self.file = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if self.split == 'val':
                if seq_id == '08':
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))
                        self.name.append(f[0:-4])


    def augment_coords_to_feats(self, coords):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords

    def voxelize(self, coords, labels):
        # nuscenes feats =None
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), features=None,
                                                                           labels=labels, ignore_label=-1,
                                                                           return_index=True, return_inverse=True)
        return coords, labels, unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)
        labels = labels -1

        coords, _, unique_map, inverse_map = self.voxelize(coords, labels)
        coords = coords.astype(np.float32)

        coords = self.augment_coords_to_feats(coords)
        return coords, np.ascontiguousarray(labels), inverse_map, index, np.array([data['x'], data['y'], data['z']], dtype=np.float32).T


class cfl_collate_fn_vis:
    def __call__(self, list_data):
        coords, labels, inverse_map, index, raw_coords = list(zip(*list_data))
        coords_batch, inverse_batch, labels_batch, raw_coords_batch = [], [], [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            raw_coords_batch.append(torch.from_numpy(raw_coords[batch_id]))
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        raw_coords_batch = torch.cat(raw_coords_batch, 0).float()

        return coords_batch, inverse_batch, labels_batch, index, raw_coords_batch