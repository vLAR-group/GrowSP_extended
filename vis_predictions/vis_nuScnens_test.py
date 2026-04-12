import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import random
import os
from os.path import join


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
            name = scene_path.replace(args.test_input_path, '')
            self.name.append(name[0:-4])
            self.file.append(scene_path)


    def augment_coords_to_feats(self, coords):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords

    def voxelize(self, coords):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords,unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords),return_index=True, return_inverse=True)
        return coords, unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        coords = np.load(file)
        ##
        scene_name = self.name[index]
        original_coords =coords.copy()
        coords -= coords.mean(0)
        coords,unique_map, inverse_map = self.voxelize(coords)
        coords = coords.numpy().astype(np.float32)
        coords= self.augment_coords_to_feats(coords)

        return coords,inverse_map.numpy(), index,original_coords,scene_name


class cfl_collate_fn_test:

    def __call__(self, list_data):
        coords, inverse_map,index,original_coords,scene_name= list(zip(*list_data))
        coords_batch, inverse_batch,original_coords_batch,scene_name_batch= [], [], [],[]
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            original_coords_batch.append(torch.from_numpy(original_coords[batch_id]))
            scene_name_batch.append(scene_name[batch_id])
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        original_coords_batch =torch.cat(original_coords_batch, 0).float()
        return coords_batch,inverse_batch, index,original_coords_batch,scene_name_batch