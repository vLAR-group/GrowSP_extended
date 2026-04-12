from os.path import join, exists, dirname
from sklearn.neighbors import KDTree
from lib.helper_ply import write_ply, read_ply
import numpy as np
import os, pickle, argparse
import matplotlib.pyplot as plt
colormap = []
for k in range(12):
    colormap.append(plt.cm.Set3(k))
for k in range(9):
    colormap.append(plt.cm.Set1(k))
colormap = np.array(colormap)
test_name = ['birmingham_block_2', 'birmingham_block_8', 'cambridge_block_15', 'cambridge_block_22', 'cambridge_block_16', 'cambridge_block_27']


def prepare(pc_path):
    for sample_type in preparation_types:
        # for pc_path in files:
        cloud_name = pc_path.split('/')[-1][:-4]
        print('start to process:', cloud_name)
        if cloud_name not in test_name:

            # create output directory
            out_folder = join(dirname(dataset_path), sample_type + '_{:.3f}'.format(grid_size) +'test')
            os.makedirs(out_folder) if not exists(out_folder) else None

            data = read_ply(pc_path)
            if pc_path in train_files:
                xyz, rgb, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
            else:
                xyz, rgb = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T
                labels = np.zeros(len(xyz), dtype=np.uint8)
            labels = labels.astype(np.float32)
            # split
            xyz = xyz.astype(np.float32)
            xyz -= xyz.mean(0)


            spilt_mask1 = (xyz[:, 0] >= 0) & (xyz[:, 1] >= 0)
            spilt_mask2 = (xyz[:, 0] < 0) & (xyz[:, 1] >= 0)
            spilt_mask3 = (xyz[:, 0] >= 0) & (xyz[:, 1] < 0)
            spilt_mask4 = (xyz[:, 0] < 0) & (xyz[:, 1] < 0)

            spilt_mask = []
            pc_thresh = 0
            if spilt_mask1.sum() > pc_thresh:
                spilt_mask.append(spilt_mask1)
            if spilt_mask2.sum() > pc_thresh:
                spilt_mask.append(spilt_mask2)
            if spilt_mask3.sum() > pc_thresh:
                spilt_mask.append(spilt_mask3)
            if spilt_mask4.sum() > pc_thresh:
                spilt_mask.append(spilt_mask4)

            if len(spilt_mask) < 4:
                print('!!!', pc_path)

            # if cloud_name in test_name:
            #     print('!!!!!!!!! val set, labels', np.unique(labels))

            for i in range(len(spilt_mask)):
                curr_xyz = xyz[spilt_mask[i]]
                curr_rgb = rgb[spilt_mask[i]]
                curr_labels = labels[spilt_mask[i]]

                # if cloud_name in test_name:
                #     print('!!!!!!!!! val set, curr labels', i, np.unique(curr_labels))
                #
                ###
                sub_ply_file = join(out_folder, cloud_name + '_' + str(i) + '.ply')
                if sample_type == 'grid':
                    sub_xyz, sub_rgb, sub_labels = DP.grid_sub_sampling(curr_xyz, curr_rgb, curr_labels.astype(np.int32), grid_size)
                else:
                    sub_xyz, sub_rgb, sub_labels = DP.random_sub_sampling(curr_xyz, curr_rgb, curr_labels.astype(np.int32), random_sample_ratio)

                # sub_rgb = sub_rgb / 255.0
                sub_labels = sub_labels.astype(np.float32)
                sub_labels = np.squeeze(sub_labels)
                write_ply(sub_ply_file, [sub_xyz, sub_rgb.astype(np.uint8), sub_labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
                ### vis GT
                GT_rgb = -np.ones_like(sub_rgb)
                for p in range(GT_rgb.shape[0]):
                    GT_rgb[p] = 255 * (colormap[sub_labels[p].astype(np.int32)])[:3]
                write_ply(join(out_folder, cloud_name + '_' + str(i) + 'GT.ply'), [sub_xyz, GT_rgb.astype(np.uint8)],
                          ['x', 'y', 'z', 'red', 'green', 'blue'])
                ### KDTree
                search_tree = KDTree(sub_xyz, leaf_size=50)
                kd_tree_file = join(out_folder, cloud_name + '_' + str(i) + 'KDTree.pkl')
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)
                ## Proj
                proj_idx = np.squeeze(search_tree.query(curr_xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                proj_save = join(out_folder, cloud_name + '_' + str(i) + 'proj.pkl')
                with open(proj_save, 'wb') as f:
                    pickle.dump([proj_idx, curr_labels], f)
                ### splitmask and ori labels
                np.save(join(out_folder, cloud_name + '_' + str(i) + 'splitmask.npy'), spilt_mask[i])
                np.save(join(out_folder, cloud_name + '_' + str(i) + 'ori_label.npy'), curr_labels)
                print(sub_ply_file)
            print('finished', pc_path)


        #  no split, for testing data
        #     sub_ply_file = join(out_folder, cloud_name + '.ply')
        #     if sample_type == 'grid':
        #         sub_xyz, sub_rgb, sub_labels = DP.grid_sub_sampling(xyz, rgb, labels.astype(np.int32), grid_size)
        #     else:
        #         sub_xyz, sub_rgb, sub_labels = DP.random_sub_sampling(xyz, rgb, labels.astype(np.int32), random_sample_ratio)
        #
        #     # sub_rgb = sub_rgb / 255.0
        #     sub_labels = sub_labels.astype(np.float32)
        #     sub_labels = np.squeeze(sub_labels)
        #     write_ply(sub_ply_file, [sub_xyz, sub_rgb.astype(np.uint8), sub_labels],
        #               ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
        #     ### vis GT
        #     GT_rgb = -np.ones_like(sub_rgb)
        #     for p in range(GT_rgb.shape[0]):
        #         GT_rgb[p] = 255 * (colormap[sub_labels[p].astype(np.int32)])[:3]
        #     write_ply(join(out_folder, cloud_name+ 'GT.ply'), [sub_xyz, GT_rgb.astype(np.uint8)],
        #               ['x', 'y', 'z', 'red', 'green', 'blue'])
        #     ### KDTree
        #     search_tree = KDTree(sub_xyz, leaf_size=50)
        #     kd_tree_file = join(out_folder, cloud_name + 'KDTree.pkl')
        #     with open(kd_tree_file, 'wb') as f:
        #         pickle.dump(search_tree, f)
        #     ## Proj
        #     proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
        #     proj_idx = proj_idx.astype(np.int32)
        #     proj_save = join(out_folder, cloud_name + 'proj.pkl')
        #     with open(proj_save, 'wb') as f:
        #         pickle.dump([proj_idx, labels], f)
        # print('finished', pc_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/home/user/SSD2/GrowSP_extension/data/SensatUrban', help='the number of GPUs to use [default: 0]')
    FLAGS = parser.parse_args()
    dataset_name = 'SensatUrban'
    dataset_path = FLAGS.dataset_path
    preparation_types = ['grid']  # Grid sampling & Random sampling
    grid_size = 0.2
    random_sample_ratio = 10
    train_files = np.sort([join(dataset_path, 'train', i) for i in os.listdir(join(dataset_path, 'train'))])
    test_files = np.sort([join(dataset_path, 'test', i) for i in os.listdir(join(dataset_path, 'test'))])
    files = np.sort(np.hstack((train_files, test_files)))
    for file in files:
        prepare(file)




