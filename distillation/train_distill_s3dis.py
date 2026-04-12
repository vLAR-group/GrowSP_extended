import argparse
import time
import os
import numpy as np
from dataset_distill_S3DIS import S3DISdistill
import torch
import warnings
warnings.filterwarnings('ignore')
import spconv.pytorch as spconv
from sparse_fpn import Res16FPN18
from glob import glob
import logging
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/S3DIS/unalign',
                        help='pont cloud data path')
    parser.add_argument('--feats_path', type=str, default='../data/S3DIS/dinov2s14',
                        help='project dino point feature')
    parser.add_argument('--save_path', type=str, default='../ckpt_distill/S3DIS',
                        help='model savepath')
    parser.add_argument('--max_epoch', type=int, default=500, help='max epoch')
    parser.add_argument('--max_iter', type=int, default=500*100, help='max iter')
    ###
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='how many workers for loading data')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--log-interval', type=int, default=20, help='log interval')
    parser.add_argument('--batch_size', type=int, default=10, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--semantic_class', type=int, default=12, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384*1, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--drop_threshold', type=int, default=30, help='ignore superpoints with few points')
    return parser.parse_args()


def main(args, logger):
    trainset = S3DISdistill(args)
    train_loader = trainset.get_loader(True)

    '''Prepare Model/Optimizer'''
    model = Res16FPN18(in_channels=6)
    logger.info(model)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = PolyLR(optimizer, max_iter=args.max_iter)

    checkpoints = glob(args.save_path+'/*tar')
    if len(checkpoints) == 0:
        print('No checkpoints found at {}'.format(args.save_path))
        epoch = 0
    else:
        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        print(args.save_path)
        path = os.path.join(args.save_path,  'checkpoint_{}.tar').format(checkpoints[-1])
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

    '''Train'''
    for epoch in range(epoch+1, args.max_epoch + 1):
        train(train_loader, logger, model, optimizer, epoch, scheduler)
        if epoch % 50 == 0:
            ckpt_path = os.path.join(args.save_path, 'checkpoint_{}.tar').format(epoch)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}, ckpt_path)

def train(train_loader, logger, model, optimizer, epoch, scheduler):
    model.train()
    loss_display = 0
    time_curr = time.time()
    for batch_idx, data in enumerate(train_loader):
        iteration = (epoch - 1) * len(train_loader) + batch_idx+1
        coords, features, labels, inverse_map, feats_2d, mask = data

        if not coords.is_contiguous():
            coords = coords.contiguous()
        in_field = spconv.SparseConvTensor(features=features.cuda(), indices=coords.int().cuda(),
                spatial_shape=list(coords.max(0)[0] + 16)[1:], batch_size=coords.max(0)[0][0].item()+1)
        feats_3d = model(in_field)
        feats_3d = feats_3d[torch.where(mask)[0]]
        loss = (1 - torch.nn.functional.cosine_similarity(feats_3d, feats_2d.cuda())).mean()

        loss_display += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        if (batch_idx+1) % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.10f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, (batch_idx+1), len(train_loader), 100. * (batch_idx+1) / len(train_loader),
                    iteration, loss_display, optimizer.param_groups[0]['lr'], time_used, args.log_interval))
            time_curr = time.time()
            loss_display = 0

from torch.optim.lr_scheduler import LambdaLR
class LambdaStepLR(LambdaLR):
  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v

class PolyLR(LambdaStepLR):
  """DeepLab learning rate policy"""
  def __init__(self, optimizer, max_iter=300*151, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    return logger

if __name__ == '__main__':
    args = parse_args()

    '''Setup logger'''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = set_logger(os.path.join(args.save_path, 'train.log'))
    os.system(f"cp {__file__} {args.save_path}")
    os.system(f"cp -r {'../distillation/'} {args.save_path}")
    os.system(f"cp {'../sparse_unet.py'} {args.save_path}")
    main(args, logger)