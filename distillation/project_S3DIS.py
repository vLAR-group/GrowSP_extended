import glob, os, sys
from os.path import join, exists, dirname, abspath
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
import torch
import numpy as np
import torch.nn.functional as F
from lib.helper_ply import read_ply, write_ply
from Point_feature_dataset_S3DIS import PointFeature, cfl_collate_fn
from torch.utils.data import DataLoader
import time
import json
import imageio

clip_bound = 4

class PointCloudToImageMapper(object):
    def __init__(self, image_dim=(1080, 1080), visibility_threshold=0.25, cut_bound=10):
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound

    def compute_mapping(self, extrinsic, coords, depth, intrinsic):
        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"
        # world_to_camera = np.linalg.inv(camera_to_world) ### should invert camera pose, not extr
        # p = np.matmul(world_to_camera, coords_new)
        p = np.matmul(extrinsic, coords_new)###
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]

        pi = np.round(p).astype(int)  # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                      * (pi[0] < self.image_dim[0] - self.cut_bound) * (pi[1] < self.image_dim[1] - self.cut_bound)

        depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
        occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= self.vis_thres * depth_cur  ### N*1

        inside_mask[inside_mask == True] = occlusion_mask
        mapping[0][inside_mask] = pi[1][inside_mask]  #### u coordinate in pixel-coordinate
        mapping[1][inside_mask] = pi[0][inside_mask]  #### v coordinate in pixel-coordinate
        mapping[2][inside_mask] = 1
        return mapping.T

def clip(coords, center=None):
    bound_min = np.min(coords, 0).astype(float)
    bound_max = np.max(coords, 0).astype(float)
    bound_size = bound_max - bound_min
    if center is None:
        center = bound_min + bound_size * 0.5
    lim = clip_bound

    if isinstance(clip_bound, (int, float)):
        if bound_size.max() < clip_bound:
            return None
        else:
            clip_inds = ((coords[:, 0] >= (-lim + center[0])) & (coords[:, 0] < (lim + center[0])) & \
                         (coords[:, 1] >= (-lim + center[1])) & (coords[:, 1] < (lim + center[1])) & \
                         (coords[:, 2] >= (-lim + center[2])) & (coords[:, 2] < (lim + center[2])))
            return clip_inds


if __name__ == "__main__":

    depth_scale = 512
    feat_dim = 384
    feat_save_path = '../data/S3DIS/dinov2s14'
    data_2d = '/media/HDD/2D_3D_S/'
    data_3d = '../data/S3DIS/unalign/'

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    model = model.cuda().eval()

    print("projecting multiview features to point cloud...")
    mapper = PointCloudToImageMapper()

    Areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6', 'Area_5']
    for Area in Areas:
        ### list all 3D and 2D data
        pc_dict = {}
        depth_dict, pose_dict, rgb_dict = {}, {}, {}
        files = sorted(glob.glob(data_3d + '/*.ply'))
        for file in files:
            if Area in file:
                scene_name = file.replace(data_3d, '')
                # if scene_name == 'Area_5_office_21.ply':
                scene_name = scene_name.replace(Area, '')[:-4]
                pc_dict[scene_name] = file
                depth_dict[scene_name], pose_dict[scene_name], rgb_dict[scene_name] = [], [], []

        if Area == 'Area_5':
            depth_path_a, pose_path_a, rgb_path_a = os.path.join(data_2d, 'Area_5a', 'data/depth'), os.path.join(data_2d, 'Area_5a', 'data/pose'), os.path.join(data_2d, 'Area_5a', 'data/rgb')
            depth_imgs_a, pose_json_a, rgb_imgs_a = sorted(glob.glob(depth_path_a+ '/*.png')), sorted(glob.glob(pose_path_a+ '/*.json')), sorted(glob.glob(rgb_path_a+ '/*.png'))
            depth_path_b, pose_path_b, rgb_path_b = os.path.join(data_2d, 'Area_5b', 'data/depth'), os.path.join(data_2d, 'Area_5b', 'data/pose'), os.path.join(data_2d, 'Area_5b', 'data/rgb')
            depth_imgs_b, pose_json_b, rgb_imgs_b = sorted(glob.glob(depth_path_b+ '/*.png')), sorted(glob.glob(pose_path_b+ '/*.json')), sorted(glob.glob(rgb_path_b+ '/*.png'))
            depth_imgs, pose_json, rgb_imgs = depth_imgs_a+depth_imgs_b, pose_json_a+pose_json_b, rgb_imgs_a+rgb_imgs_b
        else:
            depth_path, pose_path, rgb_path = os.path.join(data_2d, Area, 'data/depth'), os.path.join(data_2d, Area, 'data/pose'), os.path.join(data_2d, Area, 'data/rgb')
            depth_imgs, pose_json, rgb_imgs = sorted(glob.glob(depth_path+ '/*.png')), sorted(glob.glob(pose_path+ '/*.json')), sorted(glob.glob(rgb_path+ '/*.png'))
        for scene in depth_dict.keys():
            depth_dict[scene] = [path for path in depth_imgs if scene in path]
            pose_dict[scene] = [path for path in pose_json if scene in path]
            rgb_dict[scene] = [path for path in rgb_imgs if scene in path]

        for scene in depth_dict.keys():
            if 'auditorium' in scene:
                continue # can reserve if CPU or GPU memory is enough
                # print(Area+scene)
            cur_depth_list = depth_dict[scene]
            cur_pose_list = pose_dict[scene]
            cur_rgb_list = rgb_dict[scene]
            pc = pc_dict[scene]

            n = 10
            if os.path.exists(os.path.join(feat_save_path, Area+scene + '_%d.pt' % (n-1))):
                print('Exist', os.path.join(feat_save_path, Area+scene + '_%d.pt' % (n-1)))
                continue
            #
            print(Area+scene, 'image num: ', len(cur_rgb_list))
            if len(cur_rgb_list) == 0:
                print(Area+scene)
                continue
            if len(cur_rgb_list)>1500:
                sample_idx = np.random.choice(np.arange(len(cur_rgb_list)), 1500, replace=False)
                cur_depth_list, cur_pose_list, cur_rgb_list = [cur_depth_list[i] for i in sample_idx], [cur_pose_list[i] for i in sample_idx], [cur_rgb_list[i] for i in sample_idx]

            # load 3D data (point cloud)
            pc = read_ply(pc)
            pc = np.vstack((pc['x'], pc['y'], pc['z'])).T.astype(np.float32)

            ###### for each scene #################3
            point_feature_dataset = PointFeature(cur_rgb_list)
            data_loader = DataLoader(point_feature_dataset, batch_size=4, shuffle=False, collate_fn=cfl_collate_fn(), num_workers=4, pin_memory=True)
            frame_start_time = time.time()
            image_embeddings_list = []
            with torch.no_grad():
                for batch_idx, data in enumerate(data_loader):
                    one_batch_time = time.time()
                    color, index = data
                    ##
                    color = color.cuda()
                    features = model(color, is_training=True)
                    image_embedding = features["x_norm_patchtokens"].cpu()
                    batch_size = image_embedding.size(0)
                    image_embedding = image_embedding.reshape(batch_size, 77, 77, feat_dim).permute(0, 3, 1, 2)
                    image_embeddings_list.append(image_embedding)
                    print(f" one batch processing time: {time.time() - one_batch_time:.2f} seconds")
            ##
            image_embeddings = torch.cat(image_embeddings_list, dim=0)#.half() ### image features from all views
            frame_end_time = time.time()
            print(f" image processing time: {frame_end_time - frame_start_time:.2f} seconds")

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

            project_time = time.time()
            # project
            counter = torch.zeros((pc.shape[0], 1))#.cuda()
            sum_features = torch.zeros((pc.shape[0], feat_dim))#.cuda()
            vis_id = torch.zeros((pc.shape[0], len(image_embeddings)), dtype=int)
            for i in range(len(image_embeddings)):
                print(f" project image:", i)

                feat = image_embeddings[i].cuda()
                feat_2d = F.interpolate(feat[None, ...], (1080, 1080), mode='bicubic', align_corners=False).squeeze(0).cpu()
                pose_file, loc_in, depth_file = cur_pose_list[i], pc, cur_depth_list[i]
                # load pose
                # Open the JSON file
                with open(pose_file) as file:
                    # Load the JSON data
                    data = json.load(file)
                intrinsic, extrinsic = np.zeros((4, 4)), np.zeros((4, 4))
                intrinsic[0:3, 0:3], extrinsic[0:3] = np.array(data["camera_k_matrix"]), np.array(data["camera_rt_matrix"])
                intrinsic[-1, -1], extrinsic[-1, -1] = 1, 1
                # load depth and convert to meter
                depth = imageio.v2.imread(depth_file) / depth_scale
                # calculate the 3d-2d mapping based on the depth
                mapping = np.ones([pc.shape[0], 4], dtype=int)
                mapping[:, 1:4] = mapper.compute_mapping(extrinsic, loc_in, depth, intrinsic)
                if mapping[:, 3].sum() == 0:  # no points corresponds to this image, skip
                    continue

                mapping = torch.from_numpy(mapping)#.cuda()
                mask = mapping[:, 3]
                vis_id[:, i] = mask

                feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

                counter[mask != 0] += 1
                sum_features[mask != 0] += feat_2d_3d[mask != 0]#.cpu()

            # indicator = counter.clone().cpu().numpy()
            counter[counter == 0] = 1e-5
            feat_bank = sum_features / counter

            # save
            os.makedirs(feat_save_path, exist_ok=True)
            coords = pc.astype(np.float32)
            coords = coords - coords.mean(0)
            ## clip
            clip_inds = clip(coords)
            if clip_inds is not None:
                coords, feat_bank, vis_id = coords[clip_inds], feat_bank[clip_inds], vis_id[clip_inds]

            point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])
            n_points = coords.shape[0]
            ##
            for n in range(10):
                if n_points < 20000:
                    n_points_cur = n_points  # to handle point cloud numbers less than n_split_points
                else:
                    n_points_cur = 20000

                rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)
                mask_entire = torch.zeros(n_points, dtype=torch.bool)
                mask_entire[rand_ind] = True
                mask = torch.zeros(n_points, dtype=torch.bool)
                mask[point_ids] = True
                mask_entire = mask_entire & mask

                torch.save({"feat": feat_bank[mask_entire].half(), "mask_full": mask_entire}, os.path.join(feat_save_path, Area+scene + '_%d.pt' % (n)))
                print(os.path.join(feat_save_path, Area+scene + '_%d.pt' % (n)) + ' is saved!')

            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(feat_bank.cpu().numpy())
            min_vals = pca_features.min(axis=0)
            max_vals = pca_features.max(axis=0)
            feat_color = 255 * (pca_features - min_vals) / (max_vals - min_vals)
            feat_color = feat_color.astype(np.uint8)
            #
            # write_ply(os.path.join(feat_save_path, Area+scene + '.ply'), [coords, feat_color],
            #           ['x', 'y', 'z', 'red', 'green', 'blue'])
            print(f"project  processing time: {time.time() - project_time:.2f} seconds")
    print("done!")


### some scenes have no rgb images
# Area_2_storage_8
# Area_3_hallway_5
# Area_3_storage_2
# Area_4_hallway_5
# Area_4_hallway_6