
import argparse
import numpy as np
import os
import pickle
from pathlib import Path
from PIL import Image
from LiDAR_corruptions import rain_sim, fog_sim, scene_glare_noise, snow_sim
import torch
from tqdm import tqdm

LD_CORRUPTIONS = ['snow', 'fog', 'rain', 'sunlight', 'darkness']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate corrupted kitti dataset for image data')
    parser.add_argument('-c', '--corruption', help='corruption type', type=str,
                        choices=LD_CORRUPTIONS, default='fog')
    parser.add_argument('-r', '--dataset_root', help='root folder of dataset', type=str,
                        default='/media/ainara/ainara/dataset/kitti')
    parser.add_argument('-s', '--save_root', help='savefolder of dataset', type=str,
                        default='/media/ainara/ainara/dataset/kittiCorrupt')
    parser.add_argument('-f', '--severity', help='severity level {1, 2, 3, 4, 5}', type=int,
                        default=1)
    parser.add_argument('--seed', help='random seed', type=int,
                        default=2025)
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()

    return arguments

def save_points(save_path, points_np):  
    points_np = points_np.flatten().astype(np.float32)
    points_np.tofile(save_path)
    
def load_points(points_path):
    points = np.fromfile(points_path, dtype=np.float32)
    points = points.reshape((-1, 4))
    return points

def main():

    args = parse_arguments()    
    print(f'Preparing lidar dataset for {args.corruption} with severity {args.severity}')
    print(f'Using {args.seed} as numpy random seed')
    np.random.seed(args.seed)

    imageset = os.path.join(args.dataset_root, "kitti_infos_val.pkl")

    with open(imageset, 'rb') as f:
        infos = pickle.load(f)  

    save_path = os.path.join(args.save_root, args.corruption, f'severity_{args.severity}', 'training/velodyne')
    Path(save_path).mkdir(parents=True, exist_ok=True) 
    
    progress_bar = tqdm(total=len(infos)) 
    
    start = 0

    if args.resume:
        files = os.listdir(save_path)
        start = len(files)
        progress_bar.update(start)

    for info in infos[start:]:
        points_aug = None
        frame_idx = info['point_cloud']['lidar_idx']
        filename = f'{frame_idx}.bin'
        points = load_points(os.path.join(args.dataset_root,'training/velodyne',filename))
        if args.corruption == 'fog':
            points_aug = fog_sim(points, args.severity)
        elif args.corruption == 'rain':
            points_aug = rain_sim(points, args.severity)
        elif args.corruption == 'snow':
            points_aug = snow_sim(points, args.severity)
        elif args.corruption == 'sunlight':
            points_aug = scene_glare_noise(points, args.severity)
        elif args.corruption == 'darkness':
            points_aug = points
        elif args.corruption == 'no_lidar':
            save_path = os.path.join(args.save_root, args.corruption,'training/velodyne')
            points_aug = np.empty((0, 4), dtype=np.float32)
        else:
            raise NotImplementedError('Corruption not implemented')
        
        save_points(os.path.join(save_path, filename), points_aug[:,:4])
        progress_bar.update(1)


if __name__ == '__main__':
    main()
