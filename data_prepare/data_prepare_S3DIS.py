from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob
import argparse

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import write_ply, read_ply

sub_grid_size = 0.010
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/S3DIS/raw', help='raw data path')
parser.add_argument('--save_path', type=str, default='./data/S3DIS/unalign'.format(sub_grid_size))
# parser.add_argument('--processed_data_path', type=str, default='../data/S3DIS/input_{:.3f}'.format(sub_grid_size))
args = parser.parse_args()

anno_paths = [line.rstrip() for line in open(join(BASE_DIR, 'S3DIS_anno_paths.txt'))]
anno_paths = [join(args.data_path, p) for p in anno_paths]

gt_class = [x.rstrip() for x in open(join(BASE_DIR, 'S3DIS_class_names.txt'))]
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}

if not exists(args.output_path):
    os.makedirs(args.output_path)
out_format = '.ply'

def convert_pc2ply(anno_path, file_name):
    data_list = []

    instance_counter = 0
    for f in sorted(glob.glob(os.path.join(anno_path, '*.txt'))):
        class_name = os.path.basename(f).split('_')[0]
        if class_name not in gt_class:  # note: in some room there is 'staris' class..
            class_name = 'clutter'
        pc = pd.read_csv(f, header=None, sep='\s+').values
        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
        instance_labels = instance_counter*np.ones_like(labels)
        instance_counter += 1
        data_list.append(np.concatenate([pc, labels, instance_labels], 1))  # Nx7

    pc_info = np.concatenate(data_list, 0)

    coords = pc_info[:, :3]
    colors = pc_info[:, 3:6].astype(np.float32).astype(np.uint8)
    labels = pc_info[:, 6]
    instance_labels = pc_info[:, 7]
    # write_ply(pc_info, save_path, with_label=True, verbose=False)

    save_path = join(args.output_path, file_name)
    write_ply(save_path, [coords, colors, labels, instance_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'instance'])

    # _, _, collabels, inds = ME.utils.sparse_quantize(np.ascontiguousarray(coords), colors, labels, return_index=True, ignore_label=-1, quantization_size=sub_grid_size)
    # sub_coords, sub_colors, sub_labels, sub_instance_labels = coords[inds], colors[inds], collabels, instance_labels[inds]
    #
    # sub_ply_file = join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
    # # write_ply(np.concatenate((sub_coords, sub_colors, sub_labels[:,None], sub_instance_labels[:,None]), axis=1), sub_ply_file, with_label=True, verbose=False)
    # write_ply(sub_ply_file, [sub_coords, sub_colors, sub_labels[:,None], sub_instance_labels[:,None]], ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'instance'])


print('start preprocess')
# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for annotation_path in anno_paths:
    print(annotation_path)
    elements = str(annotation_path).split('/')
    out_file_name = elements[-3] + '_' + elements[-2] + out_format
    convert_pc2ply(annotation_path, out_file_name)