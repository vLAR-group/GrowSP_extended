import torch
from glob import glob
import numpy as np
from torch.utils.data import Dataset
import scipy
from lib.aug_tools import rota_coords, scale_coords, trans_coords, elastic_coords
from lib.helper_ply import read_ply as read_ply
import os, pickle

class S3DISvis(Dataset):
    def __init__(self, args, areas=['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']):
        self.args = args
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.name = []
        self.file = []

        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.split('/')[-1]
            if plyname[0:6] in areas and 'auditorium' not in plyname:
                self.name.append(plyname[:-4])
                self.file.append(file)

    def voxelize(self, coords, feature):
        scale = 1 / self.args.voxel_size
        coords = coords - coords.min(0)
        grids = np.floor(coords * scale)
        grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
        return grids, feature[unique_map], unique_map, inverse_map

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        pc, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        pc= pc.astype(np.float32)
        full_pc, full_labels, full_colors = pc.copy(), labels.copy(), colors.copy()
        pc = pc - pc.mean(0)

        grids, feats, unique_map, inverse_map = self.voxelize(pc, np.concatenate([colors/ 255.0, pc], 1))
        region_file = os.path.join(self.args.sp_path, self.name[index] + '_superpoint.npy')
        region = np.load(region_file)

        labels[labels == self.args.ignore_label] = -1
        full_labels[full_labels == self.args.ignore_label] = -1
        region = region[unique_map]
        voxel_labels = labels[unique_map]

        valid_region = region[region != -1]
        unique_vals = np.unique(valid_region)
        unique_vals.sort()
        valid_region = np.searchsorted(unique_vals, valid_region)

        region[region != -1] = valid_region
        return grids, feats, inverse_map, voxel_labels, index, region, full_pc, full_labels, full_colors


class cfl_collate_fn_vis:
    def __call__(self, list_data):
        voxel_coords, feats, inverse_map, voxel_labels, index, region, full_coords, full_labels, full_colors = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch = [], [], []
        region_batch = []
        full_coors_batch, full_colors_batch, full_labels_batch = [], [], []
        for batch_id, _ in enumerate(voxel_coords):
            num_points = voxel_coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(voxel_coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(voxel_labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])

            full_coors_batch.append(torch.from_numpy(full_coords[batch_id]))
            full_colors_batch.append(torch.from_numpy(full_colors[batch_id]))
            full_labels_batch.append(torch.from_numpy(full_labels[batch_id]))

        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)

        full_coors_batch = torch.cat(full_coors_batch, 0).float()
        full_colors_batch = torch.cat(full_colors_batch, 0).float()
        full_labels_batch = torch.cat(full_labels_batch, 0).int()

        return coords_batch, feats_batch, inverse_map, labels_batch, index, region_batch, full_coors_batch, full_colors_batch, full_labels_batch




class cfl_collate_fn:
    def __call__(self, list_data):
        coords, feats, labels, inverse_map, pseudo, region, index = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch, pseudo_batch = [], [], [], []
        region_batch = []
        accm_num = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            region_batch.append(torch.from_numpy(region[batch_id])[:,None])
            accm_num += coords[batch_id].shape[0]

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        region_batch = torch.cat(region_batch, 0)
        return coords_batch, feats_batch, labels_batch, inverse_map, pseudo_batch, region_batch, index


class S3DIStrain(Dataset):
    def __init__(self, args, areas=['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']):
        self.args = args
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.name = []
        self.mode = 'train'
        self.clip_bound = 4 # 4m
        self.file = []

        ''' Reading Data'''
        files = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(files):
            plyname = file.split('/')[-1]
            if plyname[0:6] in areas:
                self.name.append(plyname[0:-4])
                self.file.append(file)

        '''Initial Augmentations'''
        self.rota_coords = rota_coords(rotation_bound=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
        self.trans_coords = trans_coords(shift_ratio=50) ### 50%

    def voxelize(self, coords, feature):
        scale = 1 / self.args.voxel_size
        coords = coords - coords.min(0)
        grids = np.floor(coords * scale)
        grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
        return grids, feature[unique_map], unique_map, inverse_map

    def elastic_distortion(self, pointcloud, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords = pointcloud[:, :3]
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [np.linspace(d_min, d_max, d)
              for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim)]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        pointcloud[:, :3] = coords + interp(coords) * magnitude
        return pointcloud

    def clip(self, coords, center=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound

        if isinstance(self.clip_bound, (int, float)):
            if bound_size.max() < self.clip_bound:
                return None
            else:
                clip_inds = ((coords[:, 0] >= (-lim + center[0])) & (coords[:, 0] < (lim + center[0])) & \
                             (coords[:, 1] >= (-lim + center[1])) & (coords[:, 1] < (lim + center[1])) & \
                             (coords[:, 2] >= (-lim + center[2])) & (coords[:, 2] < (lim + center[2])))
                return clip_inds

    def augs(self, pc):
        pc = self.rota_coords(pc)
        pc = self.trans_coords(pc)
        pc = self.scale_coords(pc)
        if np.random.random() < 0.9:
            for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                pc = self.elastic_distortion(pc, granularity, magnitude)
        return pc

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        pc, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        pc= pc.astype(np.float32)
        pc = pc.astype(np.float32)
        pc = pc - pc.mean(0)

        if self.mode == 'train':
            pc = self.augs(pc)

        grids, feats, unique_map, inverse_map = self.voxelize(pc, np.concatenate([colors/ 255.0, pc], 1))
        region_file = os.path.join(self.args.sp_path, self.name[index] + '_superpoint.npy')
        region = np.load(region_file)
        region = region[unique_map]
        labels[labels == self.args.ignore_label] = -1
        labels = labels[unique_map]

        '''mode must be cluster or train'''
        if self.mode == 'cluster':
            region[labels==-1] = -1
            for q in np.unique(region):
                mask = q == region
                if mask.sum() < self.args.drop_threshold and q != -1:
                    region[mask] = -1
            valid_region = region[region != -1]
            unique_vals = np.unique(valid_region)
            unique_vals.sort()
            valid_region = np.searchsorted(unique_vals, valid_region)
            region[region != -1] = valid_region
            pseudo = -np.ones_like(labels).astype(np.int32)
        else:
            scene_name = self.name[index]
            file_path = os.path.join(self.args.save_path , self.args.pseudo_label_path, scene_name + '.npy')
            pseudo = np.array(np.load(file_path), dtype=np.int32)[unique_map]
        return grids, feats, labels.copy(), inverse_map, pseudo, region, index


class S3DIStest(Dataset):
    def __init__(self, args, areas=['Area_5']):
        self.args = args
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.name = []
        self.file = []

        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.split('/')[-1]
            if plyname[0:6] in areas and 'auditorium' not in plyname:
                self.name.append(plyname[:-4])
                self.file.append(file)

    def voxelize(self, coords, feature):
        scale = 1 / self.args.voxel_size
        coords = coords - coords.min(0)
        grids = np.floor(coords * scale)
        grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
        return grids, feature[unique_map], unique_map, inverse_map

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        pc, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        pc = pc.astype(np.float32)
        pc = pc - pc.mean(0)

        grids, feats, unique_map, inverse_map = self.voxelize(pc, np.concatenate([colors/ 255.0, pc], 1))
        region_file = os.path.join(self.args.sp_path, self.name[index] + '_superpoint.npy')
        region = np.load(region_file)#[select_idx]
        # region = labels

        labels[labels == self.args.ignore_label] = -1
        region[labels == -1] = -1
        region = region[unique_map]

        valid_region = region[region != -1]
        unique_vals = np.unique(valid_region)
        unique_vals.sort()
        valid_region = np.searchsorted(unique_vals, valid_region)
        region[region != -1] = valid_region
        return grids, feats, inverse_map, np.ascontiguousarray(labels), index, region



class cfl_collate_fn_test:
    def __call__(self, list_data):
        coords, feats, inverse_map, labels, index, region = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch = [], [], []
        region_batch = []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)

        return coords_batch, feats_batch, inverse_map, labels_batch, index, region_batch