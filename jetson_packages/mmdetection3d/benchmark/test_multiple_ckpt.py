import os
from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.logging import print_log
import torch
import os.path as osp
import argparse
from torch.utils.tensorboard import SummaryWriter
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument(
        '--work_dir',
        help='the root directory with the chekpoints')
    args = parser.parse_args()
    return args

def extract_epoch_number(filename):
    match = re.search(r'epoch_(\d+)\.pth', filename)
    return int(match.group(1)) if match else -1

def main():
    args = parse_args()  
    checkpoint_dir =  'work_dirs/bevfusion_kitti3_lidar-cam' + '/ckpt'
    cfg = Config.fromfile('projects/BEVFusion/configs/bevfusion_kitti3_lidar-cam.py')
    cfg.work_dir = 'work_dirs/bevfusion_kitti3_lidar-cam'
        
    cfg.visualizer.vis_backends = [dict(type='TensorboardVisBackend')]

    runner = Runner.from_cfg(cfg)

    writer = SummaryWriter(log_dir=os.path.join('work_dirs/bevfusion_kitti3_lidar-cam', 'evall_all_logs'))

    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pth')],
        key=extract_epoch_number
    )

    for file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, file)
        num_ckpt = file.split('.')[0].split('_')[1]
        print_log(f"Evaluating: {file}", 'current')
        runner.load_checkpoint(checkpoint_path, map_location='cpu')
        metrics = runner.test()
        for metrica, valor in metrics.items():
            writer.add_scalar(metrica, valor, num_ckpt )

    writer.close()


if __name__ == '__main__':
    main()
