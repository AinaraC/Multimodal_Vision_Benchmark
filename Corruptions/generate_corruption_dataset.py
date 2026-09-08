from pathlib import Path
import os
import shutil
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Complete corrupted dataset info')
    parser.add_argument('-dr',
                    '--dataset_root',
                    type = str,
                    default = '/media/ainara/ainara/dataset/kitti',
                    help='root path to original dataset')
    parser.add_argument('-cr',
                    '--corrupted_root',
                    type = str,
                    default = '/media/ainara/ainara/dataset/kittiCorrupt',
                    help='root path to corrupted dataset')
    arguments = parser.parse_args()
    return arguments

args = parse_arguments()

path_imageset = os.path.join(args.dataset_root, 'ImageSets/val.txt')
with open(path_imageset, 'r') as f:
    ids = [int(line.strip()) for line in f]

path_label = Path(os.path.join(args.dataset_root, 'training/label_2'))
file_label = sorted([f.name for f in path_label.iterdir() if f.is_file()])

path_calib = Path(os.path.join(args.dataset_root, 'training/calib'))
file_calib = sorted([f.name for f in path_calib.iterdir() if f.is_file()])

corruption_names = [d for d in os.listdir(args.corrupted_root) if os.path.isdir(os.path.join(args.corrupted_root, d))]

for corruption in corruption_names:

    corruption_path = Path(os.path.join(args.corrupted_root, corruption))
    nums_severity = [d[-1] for d in os.listdir(corruption_path) if os.path.isdir(os.path.join(corruption_path, d))]

    for severity in nums_severity:
        
        path_imageset_out = Path(os.path.join(args.corrupted_root, corruption, f'severity_{severity}/ImageSets'))
        path_imageset_out.mkdir(parents=True, exist_ok=True)
        shutil.copy(path_imageset, os.path.join(path_imageset_out, 'val.txt'))

        out_path_label = Path(os.path.join(corruption_path, f'severity_{severity}/training/label_2'))
        out_path_label.mkdir(parents=True, exist_ok=True)

        out_path_calib = Path(os.path.join(corruption_path, f'severity_{severity}/training/calib'))
        Path(out_path_calib).mkdir(parents=True, exist_ok=True)

        for i in ids:
            shutil.copy(os.path.join(path_label, file_label[i]), out_path_label)
            shutil.copy(os.path.join(path_calib, file_calib[i]), out_path_calib)