from pcdet.datasets.kitti.kitti_dataset_mm import KittiDatasetMM
from pathlib import Path
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description= '')
    parser.add_argument('--corruption', type=str, default="rain", help='specify the corruption')
    parser.add_argument('--severity', type=str, default="1", help='specify the severity level')
    
    args = parser.parse_args()
    return args
    
def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDatasetMM(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    val_split =  'val'

    val_filename = Path(f'{save_path}/kitti_infos_{val_split}.pkl')

    print('---------------Start to generate val data infos---------------')

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    args = parse_args()
    dataset_cfg = EasyDict(yaml.safe_load(open(Path('../../tools/cfgs/dataset_configs/kitti_dataset.yaml'))))

    if (args.corruption != 'no_camera') and (args.corruption != 'no_lidar'):
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path= Path(f'{args.corruption}/severity_{args.severity}'),
            save_path= Path(f'{args.corruption}/severity_{args.severity}')
        )
    else:
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path= Path(f'{args.corruption}'),
            save_path= Path(f'{args.corruption}')
        )