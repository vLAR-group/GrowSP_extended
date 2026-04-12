import torch, sys, os
current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import numpy as np
from lib.helper_ply import read_ply, write_ply
from torch.utils.data import Dataset
from lib.aug_tools import rota_coords, scale_coords, trans_coords
import scipy
from glob import glob


class S3DISdistill(Dataset):
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
        self.mode = 'distill'
        self.clip_bound = 4 # 4m
        self.file = []
        self.dino_file = []

        ''' Reading Data'''
        files = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(files):
            plyname = file.split('/')[-1]
            dino_file = glob(os.path.join(self.args.feats_path, plyname[0:-4] + '_*.pt'))
            if plyname[0:6] in areas:
                if plyname[0:-4] not in ['Area_2_storage_8', 'Area_3_hallway_5', 'Area_3_storage_2', 'Area_4_hallway_5', 'Area_4_hallway_6'] and 'auditorium' not in plyname:
                    self.name.append(plyname[0:-4])
                    self.file.append(file)
                    self.dino_file.append(dino_file)

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50) ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound = (0.9, 1.1))

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

    def __len__(self):
        return len(self.file)

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=self.collate_fn, shuffle=shuffle, drop_last=True)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        pc, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        pc = pc.astype(np.float32)
        pc = pc - pc.mean(0)
        pc = pc.astype(np.float32)
        clip_ind = self.clip(pc)
        if clip_ind is not None:
            pc, colors, labels = pc[clip_ind], colors[clip_ind], labels[clip_ind]
        # pc = self.augs(pc)

        # loading DINO feats
        dino_files = self.dino_file[index]
        dino_file = dino_files[np.random.randint(len(dino_files))]
        processed_data = torch.load(dino_file)
        feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']

        grids, feature, unique_map, inverse_map = self.voxelize(pc, np.concatenate([colors/ 255.0, pc], 1))

        mask = mask_chunk[unique_map]  # voxelized visible mask for entire point cloud
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

        # get the corresponding features after voxelization
        feat_3d = feat_3d[indices]
        return grids, feature, labels, inverse_map, feat_3d, mask


    def collate_fn(self, batch):
        coords, feats, labels, inverse_map, dino_feats, mask = list(zip(*batch))
        coords_batch, feats_batch, labels_batch, dino_feats_batch, mask_batch = [], [], [], [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            dino_feats_batch.append(dino_feats[batch_id])
            mask_batch.append(mask[batch_id])

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        dino_feats = torch.cat(dino_feats_batch, 0)
        return coords_batch, feats_batch, labels_batch, inverse_map, dino_feats, torch.cat(mask_batch)