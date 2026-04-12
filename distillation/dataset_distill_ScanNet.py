import torch, sys, os
current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import numpy as np
from lib.helper_ply import read_ply, write_ply
from torch.utils.data import Dataset
from lib.aug_tools import rota_coords, scale_coords, trans_coords
import yaml
import scipy
from glob import glob

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


class Scannetdistill(Dataset):
    def __init__(self, args):
        self.args = args
        self.plypath = read_txt(self.args.splits)
        self.file = []
        self.dino_files = []
        self.limit_numpoints = 1500000

        for plyname in self.plypath:
            file = os.path.join(self.args.data_path, plyname[0:12]+'.ply')
            self.file.append(file)

            scene_name = plyname[0:12]
            dino_files = glob(os.path.join(self.args.feats_path, scene_name + '_*.pt'))
            self.dino_files.append(dino_files)

        # N = 10
        # self.file.extend(self.file*N)
        # self.dino_files.extend(self.dino_files*N)

        self.rota_coords = rota_coords(rotation_bound = (None, None, (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound = (0.9, 1.1))

    def voxelize(self, coords, feature):
        scale = 1 / self.args.voxel_size
        coords = coords - coords.min(0)
        grids = np.floor(coords * scale)
        grids, unique_map, inverse_map = np.unique(grids, return_index=True, return_inverse=True, axis=0)
        return grids, feature[unique_map], unique_map, inverse_map

    def load_yaml(self, filepath):
        with open(filepath) as f:
            file = yaml.load(f, Loader = yaml.FullLoader)
        return file

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

        # loading DINO feats
        dino_files = self.dino_files[index]
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

        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                print(f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points ' f'limit. '
                      f'Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.')
                break
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            dino_feats_batch.append(dino_feats[batch_id])
            mask_batch.append(mask[batch_id])

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        dino_feats = torch.cat(dino_feats_batch, 0)
        return coords_batch, feats_batch, labels_batch, inverse_map, dino_feats, torch.cat(mask_batch)