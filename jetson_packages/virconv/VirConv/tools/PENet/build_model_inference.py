import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch

from model import ENet
#from model import PENet_C4_train (Not Implemented)
from model import PENet_C1
from model import PENet_C2
from model import PENet_C4
from dataloaders.kitti_loader import KittiDepth
import time

# Default self for test completion VirConv
class build_completion_model():
    def __init__(self,network_model='pe',workers=4,epochs=100,start_epoch=0,start_epoch_bias=0,criterion='l2',batch_size=1,lr=1e-3,weight_decay=1e-6,print_freq=10,resume='',data_folder='/data/dataset/kitti_depth/depth',data_folder_rgb='/data/dataset/kitti_raw',data_folder_save='',detpath='/VirConv/data/kitti/training',input='rgbd',val='select',jitter=0.1,rank_metric='rmse',evaluate='/VirConv/tools/PENet/pe.pth.tar',freeze_backbone=False,test=True,cpu=False,not_random_crop=False,random_crop_height=320,random_crop_width=1216,convolutional_layer_encoding='xyz',dilation_rate=2,result=os.path.join('..', 'results'), val_h=352,val_w=1216):
        self.network_model = network_model
        self.workers = workers
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.start_epoch_bias = start_epoch_bias
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.print_freq = print_freq
        self.resume = resume
        self.data_folder = data_folder
        self.data_folder_rgb = data_folder_rgb
        self.data_folder_save = ''
        self.detpath = detpath
        self.input = input
        self.val = val
        self.jitter = jitter
        self.rank_metric = rank_metric
        self.evaluate = evaluate
        self.freeze_backbone = freeze_backbone
        self.test = test
        self.cpu = cpu
        self.not_random_crop = False
        self.random_crop_height = random_crop_height
        self.random_crop_width = random_crop_width
        self.convolutional_layer_encoding = convolutional_layer_encoding
        self.dilation_rate = dilation_rate
        self.result = result
        self.use_rgb = ('rgb' in self.input)
        self.use_d = 'd' in self.input
        self.use_g = 'g' in self.input
        self.val_h = val_h
        self.val_w = val_w

# Create a model to measure de test completion latency
    def create_model(self):
        print("Building test completion model...")
        cuda = torch.cuda.is_available() and not self.cpu
        if cuda:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("=> using '{}' for computation.".format(device))
        checkpoint = None
        is_eval = False
        self_new = self
        if os.path.isfile(self.evaluate):
            print("=> loading checkpoint '{}' ... ".format(self.evaluate),
                end='')
            checkpoint = torch.load(self.evaluate, map_location=device)
            #self = checkpoint['self']
            self.start_epoch = checkpoint['epoch'] + 1
            self.data_folder = self_new.data_folder
            self.val = self_new.val
            is_eval = True
            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(self.evaluate))
            return

        print("=> creating model and optimizer ... ", end='')
        model = None
        penet_accelerated = False
        if (self.network_model == 'e'):
            model = ENet(self).to(device)
        else:
            if (self.dilation_rate == 1):
                model = PENet_C1(self).to(device)
                penet_accelerated = True
            elif (self.dilation_rate == 2):
                model = PENet_C2(self).to(device)
                penet_accelerated = True
            elif (self.dilation_rate == 4):
                model = PENet_C4(self).to(device)
                penet_accelerated = True

        if (penet_accelerated == True):
            model.encoder3.requires_grad = False
            model.encoder5.requires_grad = False
            model.encoder7.requires_grad = False

        model.load_state_dict(checkpoint['model'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")


        test_dataset = None
        test_loader = None
        test_completion_dataset = KittiDepth('test_completion', self )
        test_loader = torch.utils.data.DataLoader(
            test_completion_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

        return model, test_loader

if __name__ == '__main__':
    model_creator = build_completion_model()
    model, test_loader = model_creator.create_model()