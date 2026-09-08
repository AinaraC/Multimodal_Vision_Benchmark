import pickle
import time
from pathlib import Path
import numpy as np
import torch
import os
from pcdet.datasets import build_dataloader
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import time
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import build_network
import datetime
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.path.abspath('/VirConv/tools/PENet')) # Agregar la ruta de PENet para que encuentre el archivo model.py
from PENet.build_model_inference import build_completion_model
from PENet.vis_utils import depth_to_points
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--samples', type=int, default=200, help='samples to benchmark')
    parser.add_argument( '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument('--network', default='vir', help='model to evaluate')
    parser.add_argument('--path', default='latency_eval', help='logger dirname')
    parser.add_argument('--detpath', default='../data/kitti/training', help='logger path')
    parser.add_argument('--in_seconds', action='store_true', help='compute time in seconds instead of fps')
    parser.add_argument('--num_warmup', type=int, default=5, help='Dont count the first <num_warmup> iterations. Normally the first iterations are slower' )
    args = parser.parse_args()
    return args

class latency_evaluation():
    def __init__(self, args):
        self.args = args

        np.random.seed(1024)
            
    def __define_logger(self):
        os.makedirs(self.args.path, exist_ok=True)
        log_file = self.args.path + '/' + ('log_latency_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=0)

        return logger

    def __create_val_loader(self):
        cfg_from_yaml_file(self.args.config, cfg)
        val_set, val_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, training=False, dist=False, workers=0)

        return val_loader


    def __virconv_model(self):
        logger = self.__define_logger()
        val_loader = self.__create_val_loader()
        cfg_from_yaml_file(self.args.config, cfg)
        cfg.TAG = Path(self.args.config).stem
        cfg.EXP_GROUP_PATH = '/'.join(self.args.config.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_loader.dataset)
        model.load_params_from_file(filename=self.args.checkpoint, logger=logger)
        model.cuda()
        model.eval()

        return model
    
    def __depth_model(self):
        model_creator = build_completion_model(network_model=self.args.network,evaluate=self.args.checkpoint, detpath=self.args.detpath, test=True, cpu=False)
        depth_model, completion_loader = model_creator.create_model()
        depth_model.eval()
        torch.cuda.empty_cache()

        return depth_model, completion_loader
    
    def virconv_inference_time(self):
        if self.args.samples <= self.args.num_warmup:
            raise ValueError(f'Samples param must be greater than {self.args.num_warmup}. If you want to change this limitation, change num_warmup value')
        
        model = self.__virconv_model()
        val_loader = self.__create_val_loader()
        start_time = 0
        elapsed = 0
        pure_inf_time = 0

        print('*************** STARTING LATENCY EVALUATION OF VIRCONV NETWORK *****************')

        # https://github.com/hailanyi/VirConv/issues/51
        for i, batch_dict in enumerate(val_loader):
            torch.cuda.synchronize() 
            start_time = time.perf_counter()
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict, batch_dict = model(batch_dict)
            torch.cuda.synchronize()   # https://github.com/JUGGHM/PENet_ICRA2021
            elapsed = time.perf_counter() - start_time

            if i >= self.args.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % self.args.log_interval == 0:
                    if self.args.in_seconds:
                        seconds = pure_inf_time / (i + 1 - self.args.num_warmup)
                        print(f'Done sample [{i + 1:<3}/ {self.args.samples}], '
                            f'latency: {seconds:.2f} seconds')
                    else:
                        fps = (i + 1 - self.args.num_warmup) / pure_inf_time
                        print(f'Done sample [{i + 1:<3}/ {self.args.samples}], '
                            f'fps: {fps:.2f} sample / s')

            if (i + 1) == self.args.samples:
                if self.args.in_seconds:
                    seconds = pure_inf_time / (i + 1 - self.args.num_warmup)
                    print(f'Overall [{i + 1:<3}/ {self.args.samples}], '
                       f'latency: {seconds:.2f} seconds')
                    
                    return seconds
                
                else:
                    fps = (i + 1 - self.args.num_warmup) / pure_inf_time
                    print(f'Overall [{i + 1:<3}/ {self.args.samples}], '
                        f'fps: {fps:.2f} sample / s')
                    
                    return fps

    
    def penet_inference_time(self):

        if self.args.samples <= self.args.num_warmup:
            raise ValueError(f'Samples param must be greater than {self.args.num_warmup}. If you want to change this limitation, change num_warmup value')


        cuda = torch.cuda.is_available()
        if cuda:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("=> using '{}' for computation.".format(device))

        depth_model, completion_loader = self.__depth_model()
        start_time = 0
        elapsed = 0
        pure_inf_time = 0



        if self.args.network == 'pe':
            depth_network = 'PENET'
        else:
            depth_network = 'ENET'

        print(f'*************** STARTING LATENCY EVALUATION OF {depth_network} NETWORK *****************')

        for i, batch_data in enumerate(completion_loader):
            batch_data = {
                key: val.to(device)
                for key, val in batch_data.items() if val is not None
            }
            if(depth_network == 'ENET'):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    st1_pred, st2_pred, pred = depth_model(batch_data)
                    print(pred)
                depth_to_points(pred, i, self.args.detpath)
                torch.cuda.synchronize()   # https://github.com/JUGGHM/PENet_ICRA2021   
                elapsed = time.perf_counter() - start_time
            else:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    pred = depth_model(batch_data)
                depth_to_points(pred, i, self.args.detpath)
                torch.cuda.synchronize()   # https://github.com/JUGGHM/PENet_ICRA2021
                elapsed = time.perf_counter() - start_time

            if i >= self.args.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % self.args.log_interval == 0:
                    if self.args.in_seconds:
                        seconds = pure_inf_time / (i + 1 - self.args.num_warmup)
                        print(f'Done sample [{i + 1:<3}/ {self.args.samples}], '
                            f'latency: {seconds:.2f} seconds')
                    else:
                        fps = (i + 1 - self.args.num_warmup) / pure_inf_time
                        print(f'Done sample [{i + 1:<3}/ {self.args.samples}], '
                            f'fps: {fps:.2f} sample / s')
            if (i + 1) == self.args.samples:
                if self.args.in_seconds:
                    seconds = pure_inf_time / (i + 1 - self.args.num_warmup)
                    print(f'Overall [{i + 1:<3}/ {self.args.samples}], '
                    f'latency: {seconds:.2f} seconds')

                    return seconds
                
                else:
                    fps = (i + 1 - self.args.num_warmup) / pure_inf_time
                    print(f'Overall [{i + 1:<3}/ {self.args.samples}], '
                        f'fps: {fps:.2f} sample / s')
                    
                    return fps
    

if __name__ == '__main__':
    args = parse_args()
    latency_calculator = latency_evaluation(args)
    if args.network == 'vir':
        total_time = latency_calculator.virconv_inference_time()
    elif args.network == 'pe' or args.network == 'e':
        total_time = latency_calculator.penet_inference_time()
    else:
        print('Model not found')

    with open(f"InferenceTime.txt", "a") as f:
        if args.in_seconds:
            postfix = 's'
        else: 
            postfix = 'fps'
        f.write(f'{args.network}: {total_time} {postfix} \n')
        
    
    