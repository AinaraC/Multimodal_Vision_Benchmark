
import argparse
import numpy as np
import os
import pickle
from pathlib import Path
from PIL import Image
from Camera_corruptions import ImageAddSnow,ImageAddFog,ImageAddRain,ImageAddDarkness, ImageAddSunlight
from tqdm import tqdm

IMG_CORRUPTIONS = ['snow', 'fog', 'rain', 'sunlight', 'darkness', 'no_camera']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate corrupted kitti dataset for pointcloud data')
    parser.add_argument('-c', '--corruption', help='corruption type', type=str,
                        choices=IMG_CORRUPTIONS, default='fog')
    parser.add_argument('-r', '--dataset_root', help='root folder of dataset', type=str,
                        default='/media/ainara/ainara/kitti_dataset_organizado/kitti')
    parser.add_argument('-s', '--save_root', help='savefolder of dataset', type=str,
                        default='/media/ainara/ainara/kitti_dataset_organizado/kittiCorrupt')
    parser.add_argument('-f', '--severity', help='severity level {1, 2, 3, 4, 5}', type=int,
                        default=1)
    parser.add_argument('--seed', help='random seed', type=int,
                        default=2025)
    arguments = parser.parse_args()

    return arguments

def load_image(image_path):
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_np = np.array(image)
    image_rgb_uint8_np = image_np.astype(np.uint8)

    return image_rgb_uint8_np

def save_image(result_path, image_aug):
    image_aug = Image.fromarray(image_aug)
    path = os.path.join(result_path)
    image_aug.save(path)

def main():

    args = parse_arguments()    
    print(f'Preparing image dataset for {args.corruption} with severity {args.severity}')

    print(f'Using {args.seed} as numpy random seed')

    imageset = os.path.join(args.dataset_root, "kitti_infos_val.pkl")

    with open(imageset, 'rb') as f:
        infos = pickle.load(f)  

    save_path = os.path.join(args.save_root, args.corruption, f'severity_{args.severity}', 'training/image_2')
    Path(save_path).mkdir(parents=True, exist_ok=True)  

    sim = None
    if args.corruption == 'fog':
        sim = ImageAddFog(args.severity, args.seed)
    elif args.corruption == 'rain':
        sim = ImageAddRain(args.severity,args.seed)
    elif args.corruption == 'snow':
        sim = ImageAddSnow(args.severity, args.seed)
    elif args.corruption == 'sunlight':
        sim = ImageAddSunlight(args.severity, args.seed)
    elif args.corruption == 'darkness':
        sim = ImageAddDarkness(args.severity, args.seed)
    elif args.corruption == 'no_camera':
    	sim = None
    else:
        raise NotImplementedError('Corruption not implemented')
    
    progress_bar = tqdm(total=len(infos))
 
    for info in infos:
        frame_idx = info['image']['image_idx']
        filename = f'{frame_idx}.png'
        img = load_image(os.path.join(args.dataset_root,'training/image_2',filename))
        if args.corruption == 'no_camera':
        	save_path = os.path.join(args.save_root, args.corruption, 'training/image_2')
        	img_aug = np.zeros_like(original)
        else:
        	img_aug = sim(img)
        save_image(os.path.join(save_path, filename), img_aug)
        progress_bar.update(1)


if __name__ == '__main__':
    main()
