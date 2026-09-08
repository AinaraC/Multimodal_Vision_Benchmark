import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet3d.evaluation.functional.kitti_utils.eval import d3_box_overlap, image_box_overlap
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root
import copy

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('--config', default='configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py', help='test config file path')
    parser.add_argument(
        '--prediction_path',default='results/kitti-3class/mvxnet_results/pred_instances_3d.pkl', help='prediction path where test .pkl result')
    parser.add_argument(
        '--save_dir',
        default='tools/analysis_tools/matrix_mvxnet',
        help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--normalize', action='store_true', help='normalize confusion matrix')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--kitti_eval_classes', action='store_true', help='show only kitti evaluation classes in the confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='Blues',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--dc-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to filter Dontcare areas')
    parser.add_argument(
        '--title',
        default='Confusion Matrix MVXnet',
        help='title for the confusion matrix image')
    parser.add_argument(
        '--difficulty',
        type=int,
        default=1,
        help='kitti difficulty level: Easy(0), Medium(1), Hard(2)')
    parser.add_argument(
        '--iou_type',
        type=int,
        default=3,
        help='iou 3D or 2D')
    args = parser.parse_args()
    return args

def clean_data(gts_dicts, dt, difficulty, dc_iou_thr, current_classes):
    gt = {'name': [],'bbox': [], 'bbox_3d': [], 'difficulty': [] }
    for gt_dict in gts_dicts:
        gt['name'].append(np.array(gt_dict['bbox_label']))
        gt['bbox'].append(np.array(gt_dict['bbox']))
        gt['bbox_3d'].append(np.array(gt_dict['bbox_3d']))
        gt['difficulty'].append(np.array(gt_dict['difficulty']))

    gt_valid_classes = [ind for ind, name in enumerate(gt['name']) if name != -1 ]
    valid_gt = {key:np.array([value[i] for i in gt_valid_classes]) for key, value in gt.items()}

    dc_boxes = [box for name, box in zip(gt['name'], gt['bbox']) if name == -1 ]
    
    # Filtrar detecciones si hay DontCare
    MIN_HEIGHT = [40, 25, 25]
    if len(dc_boxes) > 0:
        dt_box = dt['bbox']
        dc_boxes = np.array(dc_boxes)
        ious = image_box_overlap(dt_box, dc_boxes, 0)
        valid_dt_ids = []
        for i, box in enumerate(dt_box):
            height = abs(box[3] - box[1])
            if np.all(ious[i] < dc_iou_thr) and height > MIN_HEIGHT[difficulty]:
                valid_dt_ids.append(i)
        valid_dt = {key: np.array([value[i] for i in valid_dt_ids]) for key, value in dt.items()}
    else:
        valid_dt = dt

    return valid_gt, valid_dt
                
def calculate_metrics(confusion_matrix, class_names, save_dir):
    result = []
    for id in range(len(class_names)):
        tp = confusion_matrix[id, id]
        fp = confusion_matrix[-1, id]
        fn = confusion_matrix[id, -1]
        f1_scr = (2 * tp) / (2 * tp + fp + fn)
        recall = tp / (tp + fn)
        sens = tp / (tp + fn)

        result.append(f'**************{class_names[id]}**************')
        result.append(f'f1 score: {f1_scr}')
        result.append(f'recall: {recall}')
        result.append(f'sensitivity: {sens}')

    with open(f"{save_dir}/metricas.txt", "w") as f:
        for lin in result:
            f.write(lin + "\n")

def calculate_confusion_matrix(dataset,
                               class_names,
                               results,
                               difficulty,
                               score_thr,
                               tp_iou_thr,
                               dc_iou_thr,
                               iou_type):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (dict): A dictionary of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    num_classes = len(class_names)
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)

    for idx, per_img_res in enumerate(results):
        gt = dataset.get_data_info(idx)['instances']
        gt, dt= clean_data(gt, per_img_res, difficulty, dc_iou_thr, class_names)
        analyze_per_img_dets(confusion_matrix, gt, dt, difficulty, score_thr,
                             tp_iou_thr, iou_type)
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gts,
                         dt_unsorted,
                         difficulty,
                         score_thr,
                         tp_iou_thr,
                         iou_type):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_classes, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_classes).
        result (dict): Detection results.
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    class_to_index ={'Pedestrian':0, 'Cyclist':1, 'Car':2, 'Van':3, 'Truck':4,
                    'Person_sitting':5, 'Tram':6, 'Misc':7}

    # Obtener los índices ordenados por score descendente
    sorted_inds = np.argsort(-dt_unsorted['score'])

    # Reordenar todos los arrays del dict usando esos índices
    dt = {
        k: v[sorted_inds] for k, v in dt_unsorted.items()
    }
    det_labels = np.array([class_to_index[label] for label in dt['name']])
    gt_labels = gts['name']
    gt_difficulty = np.array(gts['difficulty'])

    # Caso sin detecciones: solo FNs para los GTs válidos
    if len(dt.get('name', [])) == 0:
        for j in range(len(gt_labels)):
            if gt_difficulty[j] == difficulty:
                gt_label = gt_labels[j]
                confusion_matrix[gt_label, -1] += 1
        return
    
        # Máscara de GTs válidos según dificultad
    valid_gt_mask = (gt_difficulty == difficulty)
    assigned_gt = np.zeros(len(gt_labels), dtype=bool)
    
    if iou_type == 2:
        gt_bboxes = gts['bbox']
        dt_bboxes = dt['bbox']
        ious = image_box_overlap(dt_bboxes, gt_bboxes)
    elif iou_type == 3:
        gt_bboxes = gts['bbox_3d']
        loc = dt['location']
        dims = dt['dimensions']
        rots = dt['rotation_y']
        dt_bboxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        ious = d3_box_overlap(dt_bboxes, gt_bboxes).astype(np.float64)
    else:
        raise ValueError("IoU type not supported")
    
    for i, score in enumerate(dt['score']):
        if score < score_thr:
            continue

        det_label = det_labels[i]
        best_iou = 0
        best_gt_idx = -1

        for j in range(len(gt_labels)):
            if assigned_gt[j]:
                continue

            iou = ious[i, j]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Si hay match
        if best_iou >= tp_iou_thr and best_gt_idx >= 0:
            if valid_gt_mask[best_gt_idx]:
                # GT es de la dificultad correcta → asignar
                assigned_gt[best_gt_idx] = True
                gt_label = gt_labels[best_gt_idx]
                confusion_matrix[gt_label, det_label] += 1
            else:
                # El único match es con GT inválido → ignorar la detección (ni FP, ni TP)
                continue
        else:
            # No hay ningún match suficiente con ningún GT → FP
            confusion_matrix[-1, det_label] += 1

        # Añadir FNs (GTs válidos no asignados)
    for j in range(len(gt_labels)):
        if valid_gt_mask[j] and not assigned_gt[j]:
            gt_label = gt_labels[j]
            confusion_matrix[gt_label, -1] += 1


def plot_confusion_matrix(confusion_matrix,
                          kitti_eval_classes,
                          difficulty,
                          score_thr,
                          tp_iou_thr,
                          iou_type,
                          save_dir=None,
                          show=False,
                          title='Confusion Matrix',
                          color_theme='Blues',
                          normalize=False):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `Blues`.
    """
    class_to_index ={'Pedestrian':0, 'Cyclist':1, 'Car':2, 'Van':3, 'Truck':4,
                    'Person_sitting':5, 'Tram':6, 'Misc':7}

    if not kitti_eval_classes:
        used_classes =  ['Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc']
    else:
        used_classes =  ['Pedestrian', 'Cyclist', 'Car']
        confusion_matrix[-1, :] += confusion_matrix[[3,4,5,6,7], :].sum(axis=0)  
        confusion_matrix[-1,2] -= confusion_matrix[3,2] #van confundido por coche
        confusion_matrix[-1,0] -= confusion_matrix[5,0] #person sitting confundido por pedestrian
        confusion_matrix[-1,-1] = 0
        confusion_matrix = np.delete(confusion_matrix, [3,4,5,6,7], axis=0) 
        confusion_matrix = np.delete(confusion_matrix, [3,4,5,6,7], axis=1) 
    if normalize:
        per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix = confusion_matrix.astype(np.float32) / per_label_sums * 100

   
    difficulty_levels  = ['Easy','Moderate','Hard','Unknown']

    used_classes.append('Background')
    num_classes = len(used_classes)

    fig, ax = plt.subplots(figsize=(num_classes * 1.5, num_classes * 1.5) ) 
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    plt.suptitle(title + f' {iou_type}D', fontsize=12, ha='center', weight = 'extra bold')
    ax.set_title(f'Difficulty level {difficulty_levels[difficulty]}', size=12, style = "italic")
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(used_classes)
    ax.set_yticklabels(used_classes)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    threshold = np.max(confusion_matrix) / 2
    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            if normalize:
                text = '{:.2f}%'.format(
                    float(confusion_matrix[
                        i,
                        j]) if not np.isnan(confusion_matrix[i, j]) else -1)
            else: 
                text = '{}'.format(
                    int(confusion_matrix[
                        i,
                        j]) if not np.isnan(confusion_matrix[i, j]) else -1)
                
            color = 'white' if confusion_matrix[i,j] > threshold else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, f'confusion_matrix_level{difficulty}_tp-th{tp_iou_thr}_scr-th{score_thr}_{iou_type}D.png'), format='png')
    if show:
        plt.show()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    results = load(args.prediction_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    class_names =  ['Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc']

    confusion_matrix = calculate_confusion_matrix(dataset,
                                                  class_names,
                                                  results,
                                                  args.difficulty,
                                                  args.score_thr,
                                                  args.tp_iou_thr,
                                                  args.dc_iou_thr,
                                                  args.iou_type)

    plot_confusion_matrix(confusion_matrix, 
                           args.kitti_eval_classes,
                           args.difficulty,
                           args.score_thr,
                           args.tp_iou_thr,
                           args.iou_type,
                           args.save_dir,
                           args.show,
                           args.title, 
                           args.color_theme,
                           args.normalize)

    if args.kitti_eval_classes:
        calculate_metrics(confusion_matrix, ['Pedestrian', 'Cyclist', 'Car'], args.save_dir)

if __name__ == '__main__':
    main()