import torch
import numpy as np
from lib.helper_ply import read_ply, write_ply
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import open3d as o3d
from lib.aug_tools import rota_coords, scale_coords, trans_coords
from glob import glob
import pickle

class cfl_collate_fn:
    def __call__(self, list_data):
        coords, feats, labels, inverse_map, pseudo, inds, region, index = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch, inverse_batch, pseudo_batch, inds_batch = [], [], [], [], [], []
        region_batch = []
        accm_num = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            inds_batch.append(torch.from_numpy(inds[batch_id] + accm_num).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:,None])
            accm_num += coords[batch_id].shape[0]

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()#.int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        inds_batch = torch.cat(inds_batch, 0)
        region_batch = torch.cat(region_batch, 0)

        return coords_batch, feats_batch, labels_batch, inverse_batch, pseudo_batch, inds_batch, region_batch, index


class SensatUrbantrain(Dataset):
    def __init__(self, args):
        self.args = args
        self.label_to_names = {0: 'Ground',
                               1: 'High Vegetation',
                               2: 'Buildings',
                               3: 'Walls',
                               4: 'Bridge',
                               5: 'Parking',
                               6: 'Rail',
                               7: 'traffic Roads',
                               8: 'Street Furniture',
                               9: 'Cars',
                               10: 'Footpath',
                               11: 'Bikes',
                               12: 'Water'}
        self.val_name = ['birmingham_block_1', 'birmingham_block_5', 'cambridge_block_10', 'cambridge_block_7']
        self.test_name = ['birmingham_block_2', 'birmingham_block_8', 'cambridge_block_15', 'cambridge_block_22', 'cambridge_block_16', 'cambridge_block_27']
        self.name = []
        self.mode = 'train'
        # self.clip_bound = 100
        self.file = []
        self.input_xyz = []
        self.input_colors = []
        self.input_labels = []
        self.splitmask = []

        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.replace(self.args.data_path, '')
            print(plyname[:-6])
            if plyname[:-6] not in self.val_name and plyname[:-6] not in self.test_name and 'GT' not in plyname:
                name = file.replace(self.args.data_path, '')
                self.name.append(name[:-4])
                self.file.append(file)
                self.splitmask.append(file[0:-4] + 'splitmask.npy')

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))


    def augs(self, coords):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords

    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)

        feats = norm_coords#[:, -1][:, None]
        feats = np.concatenate((colors, feats), axis=-1)
        return norm_coords, feats

    # def clip(self, coords, center=None):
    #     bound_min = np.min(coords, 0).astype(float)
    #     bound_max = np.max(coords, 0).astype(float)
    #     bound_size = bound_max - bound_min
    #     if center is None:
    #         center = bound_min + bound_size * 0.5
    #     lim = self.clip_bound
    #
    #     if isinstance(self.clip_bound, (int, float)):
    #         if bound_size.max() < self.clip_bound:
    #             return None
    #         else:
    #             clip_inds = ((coords[:, 0] >= (-lim + center[0])) & (coords[:, 0] < (lim + center[0])) & \
    #                          (coords[:, 1] >= (-lim + center[1])) & (coords[:, 1] < (lim + center[1])) & \
    #                          (coords[:, 2] >= (-lim + center[2])) & (coords[:, 2] < (lim + center[2])))
    #             return clip_inds

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords, feats, labels, unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class'].astype(np.float32)
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T.copy()

        coords, colors, labels, unique_map, inverse_map = self.voxelize(coords, colors, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)
        region = region[unique_map]

        if self.mode == 'train':
            coords = self.augs(coords)

        # '''Clip if Scene includes much Points'''
        # if clip_inds is not None:
        #     region = region[clip_inds]
        # region = region[unique_map]

        inds = np.arange(coords.shape[0])

        coords, feats = self.augment_coords_to_feats(coords, colors/255-0.5)
        labels[labels == self.args.ignore_label] = -1

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

            pseudo = -np.ones_like(labels).astype(np.long)

        else:
            scene_name = self.name[index]
            file_path = self.args.save_path + '/'+self.args.pseudo_label_path + '/' + scene_name + '.npy'
            pseudo = np.array(np.load(file_path), dtype=np.long)
        return coords, feats, labels, inverse_map, pseudo, inds, region, index



class SensatUrbanval(Dataset):
    def __init__(self, args):
        self.args = args
        self.label_to_names = {0: 'Ground',
                               1: 'High Vegetation',
                               2: 'Buildings',
                               3: 'Walls',
                               4: 'Bridge',
                               5: 'Parking',
                               6: 'Rail',
                               7: 'traffic Roads',
                               8: 'Street Furniture',
                               9: 'Cars',
                               10: 'Footpath',
                               11: 'Bikes',
                               12: 'Water'}
        self.val_name = ['birmingham_block_1', 'birmingham_block_5', 'cambridge_block_10', 'cambridge_block_7']
        self.test_name = ['birmingham_block_2', 'birmingham_block_8', 'cambridge_block_15', 'cambridge_block_22', 'cambridge_block_16', 'cambridge_block_27']
        self.name = []
        self.file = []
        self.proj = []
        self.label = []

        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.replace(self.args.data_path, '')
            if plyname[:-6] in self.val_name and 'GT' not in plyname:
                name = file.replace(self.args.data_path, '')
                self.name.append(name[:-4])
                self.file.append(file)
                self.proj.append(file[:-4] + 'proj.pkl')
                self.label.append(file[:-4] + 'ori_label.npy')


    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)

        feats = norm_coords#[:, -1][:, None]
        feats = np.concatenate((colors, feats), axis=-1)
        return norm_coords, feats

    def voxelize(self, coords, feats):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats, unique_map, inverse_map.numpy()


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)
        labels = np.load(self.label[index])#data['class'].astype(np.float32)#read_ply(self.label[index])['class']
        # labels = data['class'].astype(np.float32)

        coords, colors, unique_map, inverse_map = self.voxelize(coords, colors)
        coords = coords.astype(np.float32)
        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)#[unique_map]

        labels[labels == self.args.ignore_label] = -1
        # region[labels == -1] = -1
        region = region[unique_map]

        valid_region = region[region != -1]
        unique_vals = np.unique(valid_region)
        unique_vals.sort()
        valid_region = np.searchsorted(unique_vals, valid_region)

        region[region != -1] = valid_region

        coords, feats = self.augment_coords_to_feats(coords, colors/255-0.5)

        proj_file = self.proj[index]
        with open(proj_file, 'rb') as f:
            proj_inds = pickle.load(f)[0]

        return coords, feats, inverse_map, np.ascontiguousarray(labels), index, region, proj_inds


class SensatUrbantest(Dataset):
    def __init__(self, args, test_name = ['birmingham_block_2', 'birmingham_block_8', 'cambridge_block_15', 'cambridge_block_22', 'cambridge_block_16', 'cambridge_block_27']):
        self.args = args
        self.label_to_names = {0: 'Ground',
                               1: 'High Vegetation',
                               2: 'Buildings',
                               3: 'Walls',
                               4: 'Bridge',
                               5: 'Parking',
                               6: 'Rail',
                               7: 'traffic Roads',
                               8: 'Street Furniture',
                               9: 'Cars',
                               10: 'Footpath',
                               11: 'Bikes',
                               12: 'Water'}
        self.name = []
        self.file = []
        self.proj = []
        self.splitmask = []

        test_path = '/home/zihui/SSD2/GrowSP_extension/data/SensatUrban/grid_0.200test/'
        # folders = sorted(glob(self.args.data_path + '/*.ply'))
        folders = sorted(glob(test_path + '/*.ply'))
        for _, file in enumerate(folders):
            # plyname = file.replace(self.args.data_path, '')
            plyname = file.replace(test_path, '')
            # if plyname[:-6] in test_name and 'GT' not in plyname:
            if plyname[:-4] in test_name and 'GT' not in plyname:
                # name = file.replace(self.args.data_path, '')
                name = file.replace(test_path, '')
                self.name.append(name[:-4])
                self.file.append(file)
                self.proj.append(file[:-4] + 'proj.pkl')
                # self.splitmask.append(file[0:-4] + 'splitmask.npy')

    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)

        feats = norm_coords
        feats = np.concatenate((colors, feats), axis=-1)
        return norm_coords, feats

    def voxelize(self, coords, feats):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats, unique_map, inverse_map.numpy()

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)
        # splitmask = np.load(self.splitmask[index])
        splitmask = -np.ones((len(coords), 1))

        coords, colors, unique_map, inverse_map = self.voxelize(coords, colors)
        coords = coords.astype(np.float32)
        # region_file = self.args.sp_path + '/' + self.name[index] + '_superpoint.npy'
        # region = np.load(region_file)[unique_map]

        # valid_region = region[region != -1]
        # unique_vals = np.unique(valid_region)
        # unique_vals.sort()
        # valid_region = np.searchsorted(unique_vals, valid_region)
        #
        # region[region != -1] = valid_region
        region = -np.ones((len(coords), 1))

        coords, feats = self.augment_coords_to_feats(coords, colors / 255 - 0.5)

        proj_file = self.proj[index]
        with open(proj_file, 'rb') as f:
            proj_inds = pickle.load(f)[0]

        return coords, feats, inverse_map, index, region, proj_inds, splitmask



class SensatUrbanKmeans(Dataset):
    def __init__(self, args):
        self.args = args
        self.label_to_names = {0: 'Ground',
                               1: 'High Vegetation',
                               2: 'Buildings',
                               3: 'Walls',
                               4: 'Bridge',
                               5: 'Parking',
                               6: 'Rail',
                               7: 'traffic Roads',
                               8: 'Street Furniture',
                               9: 'Cars',
                               10: 'Footpath',
                               11: 'Bikes',
                               12: 'Water'}
        self.val_name = ['birmingham_block_1', 'birmingham_block_5', 'cambridge_block_10', 'cambridge_block_7']
        self.test_name = ['birmingham_block_2', 'birmingham_block_8', 'cambridge_block_15', 'cambridge_block_22', 'cambridge_block_16', 'cambridge_block_27']

        self.name = []
        self.file = []
        self.proj = []
        self.label = []

        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.replace(self.args.data_path, '')
            if plyname[:-6] not in self.val_name and plyname[:-6] not in self.test_name and 'GT' not in plyname:
                print(plyname[:-4])
                name = file.replace(self.args.data_path, '')
                self.name.append(name[:-4])
                self.file.append(file)
                self.proj.append(file[:-4] + 'proj.pkl')
                self.label.append(file[:-4] + 'ori_label.npy')

    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)

        feats = norm_coords
        feats = np.concatenate((colors, feats), axis=-1)
        return norm_coords, feats

    def voxelize(self, coords, feats):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats, unique_map, inverse_map.numpy()


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)
        labels = np.load(self.label[index])

        coords, colors, unique_map, inverse_map = self.voxelize(coords, colors)
        coords = coords.astype(np.float32)
        # coords -= coords.mean(0)
        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)

        region[data['class'].astype(np.float32) == -1] = -1
        region = region[unique_map]

        valid_region = region[region != -1]
        unique_vals = np.unique(valid_region)
        unique_vals.sort()
        valid_region = np.searchsorted(unique_vals, valid_region)

        region[region != -1] = valid_region

        coords, feats = self.augment_coords_to_feats(coords, colors/255-0.5)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
        normals = np.array(pcd.normals)
        feats = np.concatenate((feats, normals),axis=1)

        proj_file = self.proj[index]
        with open(proj_file, 'rb') as f:
            proj_inds = pickle.load(f)[0]

        return coords, feats, inverse_map, np.ascontiguousarray(labels), index, region, proj_inds



class SensatUrbanvis(Dataset):
    def __init__(self, args):
        self.args = args
        self.label_to_names = {0: 'Ground',
                               1: 'High Vegetation',
                               2: 'Buildings',
                               3: 'Walls',
                               4: 'Bridge',
                               5: 'Parking',
                               6: 'Rail',
                               7: 'traffic Roads',
                               8: 'Street Furniture',
                               9: 'Cars',
                               10: 'Footpath',
                               11: 'Bikes',
                               12: 'Water'}
        self.val_name = ['birmingham_block_1', 'birmingham_block_5', 'cambridge_block_10', 'cambridge_block_7']
        self.test_name = ['birmingham_block_2', 'birmingham_block_8', 'cambridge_block_15', 'cambridge_block_22', 'cambridge_block_16', 'cambridge_block_27']
        self.name = []
        self.file = []
        self.proj = []
        self.label = []

        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.replace(self.args.data_path, '')
            # if plyname[:-6] not in self.test_name and 'GT' not in plyname:
            if plyname[:-6] not in self.test_name and 'GT' not in plyname:
                name = file.replace(self.args.data_path, '')
                self.name.append(name[:-4])
                self.file.append(file)
                self.proj.append(file[:-4] + 'proj.pkl')
                self.label.append(file[:-4] + 'ori_label.npy')


    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)

        feats = norm_coords
        feats = np.concatenate((colors, feats), axis=-1)
        return norm_coords, colors#feats

    def voxelize(self, coords, feats):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats, unique_map, inverse_map.numpy()


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)
        # labels = np.load(self.label[index])#data['class'].astype(np.float32)#read_ply(self.label[index])['class']
        labels = data['class'].astype(np.float32)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T

        coords, colors, unique_map, inverse_map = self.voxelize(coords, colors)
        coords = coords.astype(np.float32)
        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)#[unique_map]

        labels[labels == self.args.ignore_label] = -1
        # region[labels == -1] = -1
        region = region[unique_map]

        valid_region = region[region != -1]
        unique_vals = np.unique(valid_region)
        unique_vals.sort()
        valid_region = np.searchsorted(unique_vals, valid_region)

        region[region != -1] = valid_region

        coords, feats = self.augment_coords_to_feats(coords, colors/255-0.5)

        proj_file = self.proj[index]
        with open(proj_file, 'rb') as f:
            proj_inds = pickle.load(f)[0]

        return coords, feats, inverse_map, np.ascontiguousarray(labels), index, region, proj_inds, xyz, np.vstack((data['red'], data['green'], data['blue'])).T

class cfl_collate_fn_val:
    # def __init__(self, limit_numpoints=0):
    #   self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, inverse_map, labels, index, region, proj = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, labels_batch, proj_batch = [], [], [], [], []
        region_batch = []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
            proj_batch.append(torch.from_numpy(proj[batch_id]))

        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)
        proj_batch = torch.cat(proj_batch, 0)

        return coords_batch, feats_batch, inverse_batch, labels_batch, index, region_batch, proj_batch

class cfl_collate_fn_test:
    # def __init__(self, limit_numpoints=0):
    #   self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, inverse_map, index, region, proj, splitmask = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, region_batch, proj_batch, splitmask_batch = [], [], [], [], [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
            proj_batch.append(torch.from_numpy(proj[batch_id]))
            splitmask_batch.append(torch.from_numpy(splitmask[batch_id]))
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)
        proj_batch = torch.cat(proj_batch, 0)
        splitmask_batch = torch.cat(splitmask_batch, 0)

        return coords_batch, feats_batch, inverse_batch, index, region_batch, proj_batch, splitmask_batch

class cfl_collate_fn_vis:
    # def __init__(self, limit_numpoints=0):
    #   self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, inverse_map, labels, index, region, proj, xyz, rgb = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, labels_batch, proj_batch, xyz_batch = [], [], [], [], [], []
        region_batch, rgb_batch = [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
            proj_batch.append(torch.from_numpy(proj[batch_id]))
            xyz_batch.append(torch.from_numpy(xyz[batch_id]))
            rgb_batch.append(torch.from_numpy(rgb[batch_id]))

        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)
        proj_batch = torch.cat(proj_batch, 0)
        xyz_batch = torch.cat(xyz_batch, 0)
        rgb_batch = torch.cat(rgb_batch, 0)

        return coords_batch, feats_batch, inverse_batch, labels_batch, index, region_batch, proj_batch, xyz_batch, rgb_batch