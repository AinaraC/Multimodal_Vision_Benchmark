import argparse
import os

import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from pcdet.datasets.kitti.kitti_object_eval_python.eval import image_box_overlap, d3_box_overlap
import pcdet.datasets.kitti.kitti_object_eval_python.kitti_common as kitti
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
import copy



cfg_from_yaml_file('cfgs/models/kitti/VirConv-L_3class.yaml', cfg)

dataset, val_loader, sampler = build_dataloader(
dataset_cfg=cfg.DATA_CONFIG,
class_names=cfg.CLASS_NAMES,
batch_size=1, training=True, dist=False, workers=0)

difficulty_list = [0, 1, 2]
dist_list = [(0, 80)]
class_names = ['Car', 'Cyclist', 'Pedestrian']
gts = dataset.kitti_infos
for cl in class_names:
    for dif in difficulty_list:
        for pair in dist_list:
            min_dist = pair[0]
            max_dist = pair[1]
            sum = 0
            for gt in gts:
                anno = gt['annos']
                valid_class = [ind for ind, name in enumerate(anno['name']) if name == cl ]
                valid_dif = [ind for ind, diff in enumerate(anno['difficulty']) if diff == dif ]
                valid_dist = [ind for ind, dist in enumerate(anno['location']) if ((dist[2]  >= min_dist) and (dist[2] <= max_dist)) ]
                valid_ids = list(set(valid_class) & set(valid_dif) & set(valid_dist)) 
                sum += len(valid_ids)
            with open("clases_por_distancia.txt", "a") as f:
                f.write(f"{cl}, {dif}, {pair}, num: {sum}" + "\n")

