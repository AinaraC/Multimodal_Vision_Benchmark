# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmengine import Config
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, autocast, load_checkpoint

from mmdet3d.registry import MODELS
from tools.misc.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples',  type=int, default=200, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Whether to use automatic mixed precision inference')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--in_seconds', action='store_true', help='compute time in seconds instead of fps')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_default_scope('mmdet3d')

    # build config and set cudnn_benchmark
    cfg = Config.fromfile(args.config)

    if cfg.env_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build dataloader
    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # build model and load checkpoint
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    model.to(get_device())
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(dataloader):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with autocast(enabled=args.amp):
            model.test_step(data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                if args.in_seconds:
                    t_time = pure_inf_time / (i + 1 - num_warmup)
                    print(f'Done sample [{i + 1:<3}/ {args.samples}], '
                            f'latency: {t_time:.2f} seconds')
                else:
                    t_time = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done sample [{i + 1:<3}/ {args.samples}], '
                            f'fps: {t_time:.1f} sample / s')
  
        if (i + 1) == args.samples:
            if args.in_seconds:
                t_time = pure_inf_time / (i + 1 - num_warmup)
                print(f'Overall [{i + 1:<3}/ {args.samples}], '
                f'latency: {t_time:.2f} seconds')

            
            else:
                t_time = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall [{i + 1:<3}/ {args.samples}], '
                    f'fps: {t_time:.2f} sample / s')
                
            with open(f"InferenceTime.txt", "a") as f:
                if args.in_seconds:
                    postfix = 's'
                else: 
                    postfix = 'fps'

                name = args.config.split('/')[-2]
                f.write(f'{name}: {t_time} {postfix} \n')

            return
                



if __name__ == '__main__':
    main()
