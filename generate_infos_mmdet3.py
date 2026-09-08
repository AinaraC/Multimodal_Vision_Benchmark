import argparse
from os import path as osp
from pathlib import Path

from mmengine import print_log, dump
from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

def parse_args():
    parser = argparse.ArgumentParser(description= '')
    parser.add_argument('--corruption', type=str, default="rain", help='specify the corruption')
    parser.add_argument('--severity', type=str, default="1", help='specify the severity level')
    
    args = parser.parse_args()
    return args


def create_kitti_info_file(data_path,
                           pkl_prefix='kitti',
                           with_plane=False,
                           save_path=None,
                           relative_path=True):
    """Create info file of KITTI dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'kitti'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    val_img_ids = kitti._read_imageset_file(str(imageset_folder / 'val.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
        
    kitti_infos_val = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        with_plane=with_plane,
        image_ids=val_img_ids,
        relative_path=relative_path)
    kitti._calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Kitti info val file is saved to {filename}')
    dump(kitti_infos_val, filename)

def create_reduced_point_cloud(data_path,
                               pkl_prefix,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        val_info_path (str, optional): Path of validation set info.
            Default: None.
        save_path (str, optional): Path to save reduced point cloud data.
            Default: None.
    """
    if val_info_path is None:
        val_info_path = Path(data_path) / f'{pkl_prefix}_infos_val.pkl'

    print('create reduced point cloud for validation set')
    kitti._create_reduced_point_cloud(data_path, val_info_path, save_path)


def kitti_data_prep(root_path,
                    out_dir
                   ):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        out_dir (str): Output directory of the groundtruth database info.
    """
    create_kitti_info_file(root_path, 'kitti' )
    create_reduced_point_cloud(root_path, 'kitti')

    info_val_path = osp.join(out_dir, 'kitti_infos_val.pkl')

    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    args = parse_args()
    init_default_scope('mmdet3d')

    if args.corruption != 'no_lidar' and args.corruption != 'no_camera':
        kitti_data_prep(root_path=f'{args.corruption}/severity_{args.severity}',
                        out_dir=f'{args.corruption}/severity_{args.severity}')
    else:
        kitti_data_prep(root_path=f'{args.corruption}',
                out_dir=f'{args.corruption}')
