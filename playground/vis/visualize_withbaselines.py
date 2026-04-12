import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import argparse
import random
import numpy as np
import tqdm

import open3d as o3d

from utils.visual_util import pc_segm_to_sphere
from common_util import filter_label, compress_label, segm_to_mask, align_insts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kittisf', help='Dataset name')
    parser.add_argument('--select', dest='select', default=False, action='store_true',
                        help='Selection mode or Visualization mode')
    parser.add_argument('--save', dest='save', default=False, action='store_true',
                        help='Save visualizations or not')
    parser.add_argument('--preview', dest='preview', default=False, action='store_true',
                        help='Preview prediction results or not')
    args = parser.parse_args()

    # Setup the dataset
    if args.dataset == 'sapien':
        from datasets.dataset_sapien import SapienDataset as TestDataset
        DATA_ROOT = '/home/ziyang/Desktop/Datasets/MBS_SAPIEN/mbs-sapien'
        radius = 0.02
        resolution = 10
    elif args.dataset == 'ogcdr':
        from datasets.dataset_ogcdr import OGCDynamicRoomDataset as TestDataset
        DATA_ROOT = '/home/ziyang/Desktop/Datasets/OGC_DynamicRoom'
        radius = 0.01
        resolution = 5
    elif args.dataset == 'ogcdrsv':
        from datasets.dataset_ogcdr import OGCDynamicRoomDataset as TestDataset
        DATA_ROOT = '/home/ziyang/Desktop/Datasets/OGC_DynamicRoom_SingleView'
        radius = 0.01
        resolution = 5
    elif args.dataset == 'kittisf':
        from datasets.dataset_kittisf import KITTISceneFlowDataset as TestDataset
        DATA_ROOT = '/home/ziyang/Desktop/Datasets/KITTI_SceneFlow_downsampled'
        radius = 0.1
        resolution = 3
    else:
        raise KeyError('Unrecognized dataset!')

    if args.dataset == 'kittisf':
        mapping_path = 'data_prepare/kittisf/splits/val.txt'
        view_sels = [[0, 1], [1, 0]]
        dataset = TestDataset(data_root=DATA_ROOT,
                              mapping_path=mapping_path,
                              downsampled=True,
                              view_sels=view_sels)
        ignore_npoint_thresh = 50
    else:
        split = 'test'
        view_sels = [[0, 1], [1, 2], [2, 3], [3, 2]]
        dataset = TestDataset(data_root=DATA_ROOT,
                              split=split,
                              view_sels=view_sels)
        ignore_npoint_thresh = 0

    # Predictions from xxx methods
    if args.dataset == 'kittisf':       # In KITTI-SF, specially handle the ground for baseline methods
        # methods = ['TrajAffn_HG', 'SSC_HG', 'WardLink_HG', 'DBSCAN_HG', 'OGCsup', 'OGC_R2']
        methods = ['TrajAffn', 'TrajAffn_HG', 'SSC', 'SSC_HG', 'WardLink', 'WardLink_HG', 'DBSCAN', 'DBSCAN_HG', 'OGCsup', 'OGC_R2']
    else:
        # methods = ['TrajAffn', 'SSC', 'WardLink', 'DBSCAN', 'OGCsup', 'OGC_R2']
        methods = ['OGC_R2']
    PRED_ROOT = osp.join(DATA_ROOT, 'segm_preds')
    save_path = 'visualize/%s_qual'%(args.dataset)
    os.makedirs(save_path, exist_ok=True)


    if args.select:
        """
        Select good samples according to quantitative results
        """
        import torch
        from metrics.seg_metric import accumulate_eval_results, calculate_PQ_F1, ClusteringMetrics
        mbs_eval = ClusteringMetrics(spec=[ClusteringMetrics.IOU, ClusteringMetrics.RI])
        sample_ids = []
        sample_f1, sample_pq, sample_miou, sample_n_object = [], [], [], []

        save_stat_path = 'visualize/%s_qual_stats'%(args.dataset)
        os.makedirs(save_stat_path, exist_ok=True)

        pbar = tqdm.tqdm(total=len(dataset))
        for sid in list(range(len(dataset))):
            pbar.update(1)
            idx, view_sel_idx = sid // len(view_sels), sid % len(view_sels)

            # Load GT segmentation
            pcs, segms, _, _ = dataset[sid]
            pc, gt_segm = pcs[0], segms[0]
            gt_segm = filter_label(gt_segm, ignore_npoint_thresh)
            gt_segm = compress_label(gt_segm)
            n_object = np.unique(gt_segm).shape[0]

            # Load predicted segmentation
            sample_path = dataset.data_ids[idx]
            if isinstance(sample_path, int):
                sample_path = '%06d'%(sample_path)
            load_path = osp.join(PRED_ROOT, 'OGC_R2', sample_path)
            if args.dataset == 'kittisf':
                load_file = osp.join(load_path, 'segm%d.npy' % (view_sel_idx + 1))
            else:
                load_file = osp.join(load_path, 'segm_%02d.npy' % (view_sel_idx))
            segm = np.load(load_file)
            mask = segm_to_mask(segm)
            # if mask.shape[1] != n_object:
            #     continue

            # Quantitative evaluation
            gt_segm = torch.from_numpy(gt_segm).unsqueeze(0).cuda()
            mask = torch.from_numpy(mask).unsqueeze(0).cuda()
            Pred_IoU, Pred_Matched, Confidence, N_GT_Inst = accumulate_eval_results(gt_segm, mask)
            pq, f1, _, _ = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
            per_scan_mbs = mbs_eval(mask, gt_segm)
            miou = per_scan_mbs['iou'][0]
            sample_f1.append(f1)
            sample_pq.append(pq)
            sample_miou.append(miou)
            sample_n_object.append(n_object)
            sample_ids.append(sid)

        sample_ids, sample_n_object = np.array(sample_ids, dtype=np.int32), np.array(sample_n_object, dtype=np.int32)
        sample_f1, sample_pq, sample_miou = np.array(sample_f1, dtype=np.float32), np.array(sample_pq, dtype=np.float32), np.array(sample_miou, dtype=np.float32)
        np.save(osp.join(save_stat_path, 'sample_ids.npy'), sample_ids)
        np.save(osp.join(save_stat_path, 'sample_n_object.npy'), sample_n_object)
        np.save(osp.join(save_stat_path, 'sample_f1.npy'), sample_f1)
        np.save(osp.join(save_stat_path, 'sample_pq.npy'), sample_pq)
        np.save(osp.join(save_stat_path, 'sample_miou.npy'), sample_miou)

    else:
        """
        Visualize selected samples
        """
        # Visualization config
        if args.dataset == 'kittisf':
            interval = 50
            with_background = True
        else:
            interval = 1.2
            with_background = False

        if args.dataset == 'sapien':
            sample_ids = [144, 165, 316, 880, 2168, 2177, 2402]
            # sample_ids = [916, 1383, 1975, 2575, 2728]        # Newly-added for 1-min demo
        elif args.dataset == 'ogcdr':
            sample_ids = [570, 854, 963, 1951, 2432, 3105, 3808]
        elif args.dataset == 'ogcdrsv':
            sample_ids = [284, 817, 1093, 1660, 1835, 1907, 2962]
        else:       # KITTI-SF
            sample_ids = [46, 49, 79, 122]


        pbar = tqdm.tqdm(total=len(sample_ids))
        for sid in sample_ids:
            idx, view_sel_idx = sid // len(view_sels), sid % len(view_sels)
            print ('%d: scene %s, view %d'%(sid, dataset.data_ids[idx], view_sel_idx))

            # Load GT segmentation
            pcs, segms, _, _ = dataset[sid]
            pc, gt_segm = pcs[0], segms[0]
            gt_segm = filter_label(gt_segm, ignore_npoint_thresh)
            gt_segm = compress_label(gt_segm)

            # Flip the point cloud for better visualization
            if (args.dataset == 'sapien') and (sid in [144, 880, 916, 1383, 2168, 2177, 2402]):
                pc[:, 1] *= -1

            # Load predicted segmentation
            segm_preds = []
            for pid, method in enumerate(methods):
                sample_path = dataset.data_ids[idx]
                if isinstance(sample_path, int):
                    sample_path = '%06d'%(sample_path)
                load_path = osp.join(PRED_ROOT, method, sample_path)
                if args.dataset == 'kittisf':
                    load_file = osp.join(load_path, 'segm%d.npy' % (view_sel_idx + 1))
                else:
                    load_file = osp.join(load_path, 'segm_%02d.npy' % (view_sel_idx))
                segm = np.load(load_file)
                segm = compress_label(segm)
                segm = align_insts(gt_segm, segm)
                segm_preds.append(segm)

            # Visualize
            if args.preview:
                meshes = []
                for pid, segm_pred in enumerate(segm_preds):
                    mesh = pc_segm_to_sphere(pc, segm_pred, radius=radius, resolution=resolution, with_background=with_background)
                    meshes.append(mesh.translate([pid * interval, 0.0, 0.0]))
                mesh = pc_segm_to_sphere(pc, gt_segm, radius=radius, resolution=resolution, with_background=with_background)
                meshes.append(mesh.translate([len(segm_preds) * interval, 0.0, 0.0]))
                o3d.visualization.draw_geometries(meshes)

            # Save
            if args.save:
                mesh = pc_segm_to_sphere(pc, gt_segm, radius=radius, resolution=resolution, with_background=with_background)
                save_file = osp.join(save_path, '%06d_gt.obj' % (sid))
                o3d.io.write_triangle_mesh(save_file, mesh)
                # Save predictions
                for method, segm_pred in zip(methods, segm_preds):
                    mesh = pc_segm_to_sphere(pc, segm_pred, radius=radius, resolution=resolution, with_background=with_background)
                    save_file = osp.join(save_path, '%06d_%s.obj' % (sid, method))
                    o3d.io.write_triangle_mesh(save_file, mesh)

            pbar.update(1)