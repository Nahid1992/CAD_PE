import argparse
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from pretrainedmodels.senet import se_resnet50, se_resnext50_32x4d, se_resnet101, se_resnext101_32x4d
# from train0_SeResNet50 import seresnext50 || model = seresnext50()
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
import albumentations
import pydicom
import copy
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from random import randrange

import time
from model_pytorch import Classifier_model, get_weight_name, ProgressMeter, save_checkpoint
numSeed = randrange(25000)

DATA_DIR = '/ocean/projects/bcs190005p/nahid92/Data/RSNA_PE/train/'  # Bridge
# DATA_DIR = '/mnt/dataset/shared/zguo32/rsna-pe-detection/train/'

def get_weight_name_from(number):
    if number == 0:
        return "insdis"
    elif number == 1:
        return "moco-v1"
    elif number == 2:
        return "moco-v2"
    elif number == 3:
        return "pcl-v1"
    elif number == 4:
        return "pcl-v2"
    elif number == 5:
        return "pirl"
    elif number == 6:
        return "sela-v2"
    elif number == 7:
        return "infomin"
    elif number == 8:
        return "byol"
    elif number == 9:
        return "deepcluster-v2"
    elif number == 10:
        return "swav"
    elif number == 11:
        return "simclr-v1"
    elif number == 12:
        return "simclr-v2"
    else:
        return "None"

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

def convert_to_SE_ResNet50(model):
    reduction_ratio = 16
    for index in range(0,len(model.layer1)): # layer1[0-2]
        # if index == 0:
        #     kkk = 1
        #     model.layer1[index] = nn.Sequential(model.layer1[index].conv1, 
        #                                                   model.layer1[index].bn1, 
        #                                                   model.layer1[index].conv2,
        #                                                   model.layer1[index].bn2,
        #                                                   model.layer1[index].conv3,
        #                                                   model.layer1[index].bn3,
        #                                                   model.layer1[index].relu,
        #                                                   SEModule(model.layer1[index].bn3.num_features, reduction_ratio),
        #                                                   model.layer1[index].downsample)
        # else:
        #     model.layer1[index] = nn.Sequential(model.layer1[index], SEModule(model.layer1[index].bn3.num_features, reduction_ratio))
        model.layer1[index] = nn.Sequential(model.layer1[index], SEModule(model.layer1[index].bn3.num_features, reduction_ratio))
        model.layer1[index][1].fc1.weight.data.normal_(mean=0.0, std=0.01)
        model.layer1[index][1].fc1.bias.data.zero_()
        model.layer1[index][1].fc2.weight.data.normal_(mean=0.0, std=0.01)
        model.layer1[index][1].fc2.bias.data.zero_()

    for index in range(0, len(model.layer2)): # layer2[0-3]
        # if index == 0:
        #     kkk = 1
        #     model.layer2[index] = nn.Sequential(model.layer2[index].conv1, 
        #                                                   model.layer2[index].bn1, 
        #                                                   model.layer2[index].conv2,
        #                                                   model.layer2[index].bn2,
        #                                                   model.layer2[index].conv3,
        #                                                   model.layer2[index].bn3,
        #                                                   model.layer2[index].relu,
        #                                                   SEModule(model.layer2[index].bn3.num_features, reduction_ratio),
        #                                                   model.layer2[index].downsample)
        # else:
        #     model.layer2[index] = nn.Sequential(model.layer2[index], SEModule(model.layer2[index].bn3.num_features, reduction_ratio))
        model.layer2[index] = nn.Sequential(model.layer2[index], SEModule(model.layer2[index].bn3.num_features, reduction_ratio))
        model.layer2[index][1].fc1.weight.data.normal_(mean=0.0, std=0.01)
        model.layer2[index][1].fc1.bias.data.zero_()
        model.layer2[index][1].fc2.weight.data.normal_(mean=0.0, std=0.01)
        model.layer2[index][1].fc2.bias.data.zero_()
            
    for index in range(0, len(model.layer3)): # layer3[0-5]
        # if index == 0:
        #     kkk = 1
        #     model.layer3[index] = nn.Sequential(model.layer3[index].conv1, 
        #                                                   model.layer3[index].bn1, 
        #                                                   model.layer3[index].conv2,
        #                                                   model.layer3[index].bn2,
        #                                                   model.layer3[index].conv3,
        #                                                   model.layer3[index].bn3,
        #                                                   model.layer3[index].relu,
        #                                                   SEModule(model.layer3[index].bn3.num_features, reduction_ratio),
        #                                                   model.layer3[index].downsample)
        # else:
        #     model.layer3[index] = nn.Sequential(model.layer3[index], SEModule(model.layer3[index].bn3.num_features, reduction_ratio))
        model.layer3[index] = nn.Sequential(model.layer3[index], SEModule(model.layer3[index].bn3.num_features, reduction_ratio))
        model.layer3[index][1].fc1.weight.data.normal_(mean=0.0, std=0.01)
        model.layer3[index][1].fc1.bias.data.zero_()
        model.layer3[index][1].fc2.weight.data.normal_(mean=0.0, std=0.01)
        model.layer3[index][1].fc2.bias.data.zero_()

    for index in range(0, len(model.layer4)): # layer4[0-2]
        # if index == 0:
        #     kkk = 1
        #     model.layer4[index] = nn.Sequential(model.layer4[index].conv1, 
        #                                                   model.layer4[index].bn1, 
        #                                                   model.layer4[index].conv2,
        #                                                   model.layer4[index].bn2,
        #                                                   model.layer4[index].conv3,
        #                                                   model.layer4[index].bn3,
        #                                                   model.layer4[index].relu,
        #                                                   SEModule(model.layer4[index].bn3.num_features, reduction_ratio),
        #                                                   model.layer4[index].downsample)
        # else:
        #     model.layer4[index] = nn.Sequential(model.layer4[index], SEModule(model.layer4[index].bn3.num_features, reduction_ratio))
        model.layer4[index] = nn.Sequential(model.layer4[index], SEModule(model.layer4[index].bn3.num_features, reduction_ratio))
        model.layer4[index][1].fc1.weight.data.normal_(mean=0.0, std=0.01)
        model.layer4[index][1].fc1.bias.data.zero_()
        model.layer4[index][1].fc2.weight.data.normal_(mean=0.0, std=0.01)
        model.layer4[index][1].fc2.bias.data.zero_()
    return model

def convert_to_SE_Xception(model, reduction_ratio=16):
    model.block1.rep = nn.Sequential(model.block1.rep, SEModule(model.block1.rep[4].num_features, reduction_ratio))
    model.block1.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block1.rep[1].fc1.bias.data.zero_()
    model.block1.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block1.rep[1].fc2.bias.data.zero_()
        
    model.block2.rep = nn.Sequential(model.block2.rep, SEModule(model.block2.rep[5].num_features, reduction_ratio))
    model.block2.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block2.rep[1].fc1.bias.data.zero_()
    model.block2.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block2.rep[1].fc2.bias.data.zero_()
    
    model.block3.rep = nn.Sequential(model.block3.rep, SEModule(model.block3.rep[5].num_features, reduction_ratio))
    model.block3.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block3.rep[1].fc1.bias.data.zero_()
    model.block3.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block3.rep[1].fc2.bias.data.zero_()
    
    model.block4.rep = nn.Sequential(model.block4.rep, SEModule(model.block4.rep[8].num_features, reduction_ratio))
    model.block4.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block4.rep[1].fc1.bias.data.zero_()
    model.block4.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block4.rep[1].fc2.bias.data.zero_()
    
    model.block5.rep = nn.Sequential(model.block5.rep, SEModule(model.block5.rep[8].num_features, reduction_ratio))
    model.block5.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block5.rep[1].fc1.bias.data.zero_()
    model.block5.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block5.rep[1].fc2.bias.data.zero_()
    
    model.block6.rep = nn.Sequential(model.block6.rep, SEModule(model.block6.rep[8].num_features, reduction_ratio))
    model.block6.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block6.rep[1].fc1.bias.data.zero_()
    model.block6.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block6.rep[1].fc2.bias.data.zero_()
    
    model.block7.rep = nn.Sequential(model.block7.rep, SEModule(model.block7.rep[8].num_features, reduction_ratio))
    model.block7.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block7.rep[1].fc1.bias.data.zero_()
    model.block7.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block7.rep[1].fc2.bias.data.zero_()
    
    model.block8.rep = nn.Sequential(model.block8.rep, SEModule(model.block8.rep[8].num_features, reduction_ratio))
    model.block8.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block8.rep[1].fc1.bias.data.zero_()
    model.block8.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block8.rep[1].fc2.bias.data.zero_()
    
    model.block9.rep = nn.Sequential(model.block9.rep, SEModule(model.block9.rep[8].num_features, reduction_ratio))
    model.block9.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block9.rep[1].fc1.bias.data.zero_()
    model.block9.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block9.rep[1].fc2.bias.data.zero_()
    
    model.block10.rep = nn.Sequential(model.block10.rep, SEModule(model.block10.rep[8].num_features, reduction_ratio))
    model.block10.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block10.rep[1].fc1.bias.data.zero_()
    model.block10.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block10.rep[1].fc2.bias.data.zero_()
    
    model.block11.rep = nn.Sequential(model.block11.rep, SEModule(model.block11.rep[8].num_features, reduction_ratio))
    model.block11.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block11.rep[1].fc1.bias.data.zero_()
    model.block11.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block11.rep[1].fc2.bias.data.zero_()
    
    model.block12.rep = nn.Sequential(model.block12.rep, SEModule(model.block12.rep[5].num_features, reduction_ratio))
    model.block12.rep[1].fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.block12.rep[1].fc1.bias.data.zero_()
    model.block12.rep[1].fc2.weight.data.normal_(mean=0.0, std=0.01)
    model.block12.rep[1].fc2.bias.data.zero_()
    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2 # 400 to -300
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    # X = (X*255.0).astype('uint8')
    return X

class PEDataset(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size, transform):
        self.image_dict=image_dict  # 1790594
        self.bbox_dict=bbox_dict  # should be 6,279
        self.image_list=image_list  # should be 6,279
        self.target_size=target_size
        self.transform=transform
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,index):
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]
        data1 = pydicom.dcmread(DATA_DIR+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_minus1']+'.dcm')
        data2 = pydicom.dcmread(DATA_DIR+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        data3 = pydicom.dcmread(DATA_DIR+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_plus1']+'.dcm')
        x1 = data1.pixel_array
        x2 = data2.pixel_array
        x3 = data3.pixel_array
        x1 = x1*data1.RescaleSlope+data1.RescaleIntercept
        x2 = x2*data2.RescaleSlope+data2.RescaleIntercept
        x3 = x3*data3.RescaleSlope+data3.RescaleIntercept
        x1 = np.expand_dims(window(x1, WL=100, WW=700), axis=2)
        x2 = np.expand_dims(window(x2, WL=100, WW=700), axis=2)
        x3 = np.expand_dims(window(x3, WL=100, WW=700), axis=2)
        x = np.concatenate([x1, x2, x3], axis=2)
        # print("CHECK: x shape = " + str(x.shape))
        bbox = self.bbox_dict[self.image_dict[self.image_list[index]]['series_id']]
        # print("CHECK: bbox = " + str(bbox))
        x = x[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        # print("CHECK: x shape = " + str(x.shape))
        x = cv2.resize(x, (self.target_size,self.target_size))
        x = self.transform(image=x)['image']
        x = x.transpose(2, 0, 1)
        y = self.image_dict[self.image_list[index]]['pe_present_on_image']
        return x, y

class PEDataset_val(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size):
        self.image_dict=image_dict  # # 1790594
        self.bbox_dict=bbox_dict  # should be 1,000
        self.image_list=image_list  # should be 1,000
        self.target_size=target_size
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,index):
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]
        data1 = pydicom.dcmread(DATA_DIR+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_minus1']+'.dcm')
        data2 = pydicom.dcmread(DATA_DIR+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        data3 = pydicom.dcmread(DATA_DIR+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_plus1']+'.dcm')
        x1 = data1.pixel_array
        x2 = data2.pixel_array
        x3 = data3.pixel_array
        x1 = x1*data1.RescaleSlope+data1.RescaleIntercept
        x2 = x2*data2.RescaleSlope+data2.RescaleIntercept
        x3 = x3*data3.RescaleSlope+data3.RescaleIntercept
        x1 = np.expand_dims(window(x1, WL=100, WW=700), axis=2)
        x2 = np.expand_dims(window(x2, WL=100, WW=700), axis=2)
        x3 = np.expand_dims(window(x3, WL=100, WW=700), axis=2)
        x = np.concatenate([x1, x2, x3], axis=2)
        bbox = self.bbox_dict[self.image_dict[self.image_list[index]]['series_id']]
        x = x[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        x = cv2.resize(x, (self.target_size,self.target_size))
        x = transforms.ToTensor()(x)
        # x = transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])(x) # old
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x) # New
        y = self.image_dict[self.image_list[index]]['pe_present_on_image']
        return x, y

class seresnext50(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
        # self.net = se_resnext50_32x4d(num_classes=1000, pretrained=None) # Random Init
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

class seresnet50(nn.Module):
    def __init__(self, pretrainedModel='ImageNet'):
        super().__init__()
        if pretrainedModel == 'ImageNet':
            self.net = se_resnet50(num_classes=1000, pretrained='imagenet')
        else:
            self.net = se_resnet50(num_classes=1000, pretrained=None) # Random Init
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

class my_resnet50(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.net = models.resnet50(num_classes=1000, pretrained=pretrained)
        # self.net = se_resnext50_32x4d(num_classes=1000, pretrained=None) # Random Init
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.fc.in_features
        self.fc = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

class seresnet101(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.net = se_resnet101(num_classes=1000, pretrained=None)
        if pretrained == "ImageNet":
            state_dict = torch.load("pretrain_model_weights/se_resnet101-7e38fcc6.pth", map_location="cpu") # extra
            self.net.load_state_dict(state_dict) # was url
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

class seresnext101(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.net = se_resnext101_32x4d(num_classes=1000, pretrained=pretrained)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


def main():
    # python -m torch.distributed.launch --nproc_per_node=4 train0_SeResNet50.py --train_task=1 --val_task=0 --manual_load=0 --backboneName seresnext50 --runV _v001_
    # python train0_SeResNet50.py --train_task=0 --val_task=1 --manual_load=1 --backboneName seresnext50 --runV _v001_
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    parser.add_argument("--train_task", type=int, default=1, help="train or not")
    parser.add_argument("--val_task", type=int, default=1, help="val or not")
    parser.add_argument("--manual_load", type=int, default=0, help="model load during val or not")

    parser.add_argument("--backboneName", type=str, default="seresnext50", help="resnet18 | resnet50 | densenet121 | xception")
    parser.add_argument("--runV", type=str, default="_v0_", help="model load during val or not")

    parser.add_argument("--redu", type=int, default=100, help="Reduced Data")

    parser.add_argument("--BS", type=int, default=20, help="Batch Size")

    parser.add_argument("--loadW", type=str, default="ImageNet", help="Weights")

    parser.add_argument("--nGPU", type=int, default=4, help="number of GPUs")

    parser.add_argument("--nEpoch", type=int, default=1, help="number of Epochs")

    parser.add_argument("--worker", type=int, default=12, help="number of Epochs")

    args = parser.parse_args()

    train_task = args.train_task
    val_task = args.val_task
    manually_load = args.manual_load
    runV = args.runV
    backboneName = args.backboneName
    redu = args.redu
    loadW = args.loadW

    # hyperparameters
    learning_rate = 0.0004 # was 0.0004
    batch_size = args.BS # was 32
    image_size = 576 # was 576
    num_epoch = args.nEpoch
    number_of_gpu = args.nGPU # was 4

    gwn = backboneName + "_576_" + loadW + runV

    title_name = 'FineTune_Reduced_' + str(redu) + "_"
    out_dir = title_name + gwn + '/'

    # prepare input
    import pickle5 as pickle
    with open('../process_input/split2/image_dict.pickle', 'rb') as f:
        image_dict = pickle.load(f) 
    with open('../lung_localization/split2/bbox_dict_train.pickle', 'rb') as f:
        bbox_dict_train = pickle.load(f) 

    if redu == 100:
        with open('../process_input/split2/image_list_train.pickle', 'rb') as f:
            image_list_train = pickle.load(f) 
    elif redu == 200:
        with open('../process_input/split2/RSNA_PE_shiv/image_dict.pickle', 'rb') as f:
            image_dict = pickle.load(f) 
        with open('../process_input/split2/RSNA_PE_shiv/image_list_train.pickle', 'rb') as f:
            image_list_train = pickle.load(f) 
        with open('../process_input/split2/RSNA_PE_shiv/bbox_dict_train.pickle', 'rb') as f:
            bbox_dict_train = pickle.load(f) 
        print("[INFO] Training data loaded - Shiv's Data Split")
    else: # for reduced training data
        with open('../process_input/split2/image_list_train_'+ str(redu) + '.pickle', 'rb') as f:
            image_list_train = pickle.load(f) 
        print("[INFO] Reduced Training Data loaded: ", redu, "%")



    if train_task == 1:
        print("Model is training...")
        ## ---------------------------- Parameter ---------------------------- ##
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.device = device

        seed = numSeed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        print("Data is ready...")
        print(len(image_list_train), len(image_dict), len(bbox_dict_train))

        print("Learning Rate: " + str(learning_rate))
        print("Batch Size: " + str(batch_size))
        print("Image Size: " + str(image_size))
        print("Number of Epoch: " + str(num_epoch))



        ## ---------------------------- Data Loading ---------------------------- ##

        # training
        train_transform = albumentations.Compose([
            albumentations.RandomContrast(limit=0.2, p=1.0),
            albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
            # albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0) # Old
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0, p=1.0) # New
        ])

        # iterator for training
        datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train, target_size=image_size, transform=train_transform)
        sampler = DistributedSampler(datagen)
        generator = DataLoader(dataset=datagen, sampler=sampler, batch_size=batch_size, num_workers=args.worker, pin_memory=True)




        

        ## ---------------------------- Model: Training ---------------------------- ##
        # build model
        if args.local_rank != 0:
            torch.distributed.barrier()

        if backboneName == "seresnext50":
            model = seresnext50()
        elif backboneName == "seresnet50":
            model = seresnet50(pretrained=loadW)
        elif backboneName == "seresnet101":
            model = seresnet101(pretrained=loadW)
        elif backboneName == "seresnext101":
            if loadW == "ImageNet":
                model = seresnext101(pretrained="imagenet")
            else:
                model = seresnext101(pretrained=None)
        elif backboneName == "seresnet50_manual": # manually created SE + ResNet50
            if loadW == "ImageNet":
                model = models.resnet50(pretrained=True) # true or false
            else:
                model = models.resnet50(pretrained=False) # true or false
            kernelCount = model.fc.in_features
            model.fc = nn.Linear(kernelCount, 1)
            model = convert_to_SE_ResNet50(model)
            print("ResNet50 Loaded with conversion to SE (manually).")
        elif backboneName == "xception": # manually created SE + Xception
            from xception_copiedModel import xception
            if loadW == "ImageNet":
                model = xception(num_classes=1000, pretrained="imagenet")
            else:
                model = xception(num_classes=1000, pretrained=None) # needs to be imagenet
            kernelCount = model.last_linear.in_features
            model.last_linear = nn.Sequential(nn.Linear(kernelCount, 1))
            # model = convert_to_SE_Xception(model, reduction_ratio=16)
            print("Xception Loaded...")
        elif backboneName == "sexception": # newly created SE + Xception
            from xception_copiedModel import xception
            model = xception(num_classes=1000, pretrained=None)
            model = convert_to_SE_Xception(model, reduction_ratio=16)
            if loadW == "ImageNet": # loading pre-trained model
                checkpoint = torch.load('pretrain_model_weights/sexception.pth.tar', map_location="cpu") 
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('module.'):
                        # remove prefix
                        state_dict[k[len("module."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                msg = model.load_state_dict(state_dict, strict=False)
            kernelCount = model.last_linear.in_features
            model.last_linear = nn.Sequential(nn.Linear(kernelCount, 1))
            print("SE-Xception Loaded with conversion to SE (newly_created).")
        
        print("Model Backbone: " + backboneName)
        print("Model Weights: " + gwn)

        print("Model is ready...") 

        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(args.device)

        num_train_steps = int(len(image_list_train)/(batch_size*number_of_gpu)*1)   # 4 GPUs # (batch_size*number_of_gpu)*num_epoch) to (batch_size*number_of_gpu)*1)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        criterion = nn.BCEWithLogitsLoss().to(args.device) # old with Se_resNet50


        ## Write configurations
        output_text_file_name = title_name + '_outputLines' + '_' + gwn + '.txt'
        output_text_file = open(output_text_file_name, 'w')
        output_text_file.write("Learning Rate: " + str(learning_rate) + "\n")
        output_text_file.write("Batch Size: " + str(batch_size) + "\n")
        output_text_file.write("Image Size: " + str(image_size) + "\n")
        output_text_file.write("Number of Epoch: " + str(num_epoch) + "\n")
        output_text_file.write("Model Backbone: " + "ResNet50" + "\n")
        output_text_file.write("Model Weights: " + gwn + "\n")
        output_text_file.write("DataSize: " + str(redu) + "\n")
        output_text_file.write("RunVersion: " + runV + "\n")
        output_text_file.write("--------------------- \n")
        output_text_file.close()

        
        output_text_file = open(output_text_file_name, 'a')
        list_ep_avgLoss = []
        for ep in range(num_epoch):
            losses = AverageMeter()
            model.train()
            for j,(images,labels) in enumerate(generator):
                images = images.to(args.device)
                labels = labels.float().to(args.device)
                # print("[CHECK]", images.shape)
                logits = model(images)
                loss = criterion(logits.view(-1),labels) # was with BCEwithLogitsLoss
                losses.update(loss.item(), images.size(0))

                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss = losses.avg
                if args.local_rank == 0:
                    if j < 50:
                        print('epoch: {}| step {}/{} train_loss: {}'.format(ep,j,len(generator),losses.avg), flush=True)
                    if j % 100 == 0:
                        print('epoch: {}| step {}/{} train_loss: {}'.format(ep,j,len(generator),losses.avg), flush=True)
                    string_msg = 'epoch: {}| step {}/{} train_loss: {} \n'.format(ep,j,len(generator),losses.avg)
                    output_text_file.write(string_msg)


            if args.local_rank == 0:
                print('epoch: {} train_loss: {}'.format(ep, losses.avg), flush=True)
                string_msg = 'Training => loss:{}\n'.format(losses.avg)
                output_text_file.write(string_msg)


            if args.local_rank == 0:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                torch.save(model.module.state_dict(), out_dir+'epoch{}'.format(ep))
                save_checkpoint({
                              'epoch': ep,
                              'lossMIN': train_loss,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict(),
                            }, False, filename=out_dir+"_SavedModel_"+str(ep)+"_"+gwn)
            list_ep_avgLoss.append([ep, losses.avg]) ## Storing average loss for each epoch

        print()
        print('Epoch and their Average Losses')
        print(' --------------------- ')
        for index in range(0, num_epoch):
            print(str(list_ep_avgLoss[index][1]) + " --- " + str(list_ep_avgLoss[index][1]))
        print(' --------------------- ')
        print()

        print("Model training done...")
        output_text_file.close()

        del datagen, sampler, generator


    ## ---------------------------- Model: Validation ---------------------------- ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch_index in range(0, num_epoch):
        if val_task == 1:
            if manually_load == 1:
                # Loading model manually
                if backboneName == "seresnext50":
                    model = seresnext50()
                elif backboneName == "seresnet50":
                    model = seresnet50(pretrained=loadW)
                elif backboneName == "seresnet101":
                    model = seresnet101(pretrained=loadW)
                elif backboneName == "seresnext101":
                    if loadW == "ImageNet":
                        model = seresnext101(pretrained="imagenet")
                    else:
                        model = seresnext101(pretrained=None)
                elif backboneName == "seresnet50_manual": # manually created SE + ResNet50
                    if loadW == "ImageNet":
                        model = models.resnet50(pretrained=True) # true or false
                    else:
                        model = models.resnet50(pretrained=False) # true or false
                    kernelCount = model.fc.in_features
                    model.fc = nn.Linear(kernelCount, 1)
                    model = convert_to_SE_ResNet50(model)
                    print("ResNet50 Loaded with conversion to SE (manually).")
                elif backboneName == "xception": # manually created SE + Xception
                    from xception_copiedModel import xception
                    if loadW == "ImageNet":
                        model = xception(num_classes=1000, pretrained="imagenet")
                    else:
                        model = xception(num_classes=1000, pretrained=None) # needs to be imagenet
                    kernelCount = model.last_linear.in_features
                    model.last_linear = nn.Sequential(nn.Linear(kernelCount, 1))
                    # model = convert_to_SE_Xception(model, reduction_ratio=16)
                    print("Xception Loaded...")
                elif backboneName == "sexception": # newly created SE + Xception
                    from xception_copiedModel import xception
                    model = xception(num_classes=1000, pretrained=None)
                    model = convert_to_SE_Xception(model, reduction_ratio=16)
                    if loadW == "ImageNet": # loading pre-trained model
                        checkpoint = torch.load('pretrain_model_weights/sexception.pth.tar', map_location="cpu") 
                        state_dict = checkpoint['state_dict']
                        for k in list(state_dict.keys()):
                            if k.startswith('module.'):
                                # remove prefix
                                state_dict[k[len("module."):]] = state_dict[k]
                            # delete renamed or unused k
                            del state_dict[k]
                        msg = model.load_state_dict(state_dict, strict=False)
                    kernelCount = model.last_linear.in_features
                    model.last_linear = nn.Sequential(nn.Linear(kernelCount, 1))
                    print("SE-Xception Loaded with conversion to SE (newly_created).")
                    
                model.load_state_dict(torch.load(out_dir + "epoch"+str(epoch_index))) # was epoch0
                model = model.cuda()
                optimizer = optim.Adam(model.parameters(), lr=0.0004)
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)  
                criterion = nn.BCEWithLogitsLoss().to(device)
                print(backboneName + ' 576 ' + loadW + ' weight Model loaded manually')

            import pickle5 as pickle
            print("Validation Testing Started")
            with open('../process_input/split2/image_list_valid.pickle', 'rb') as f:
                image_list_valid = pickle.load(f) 
            with open('../lung_localization/split2/bbox_dict_valid.pickle', 'rb') as f:
                bbox_dict_valid = pickle.load(f)
            if redu == 200:
                with open('../process_input/split2/RSNA_PE_shiv/image_dict.pickle', 'rb') as f:
                    image_dict = pickle.load(f) 
                with open('../process_input/split2/RSNA_PE_shiv/image_list_valid.pickle', 'rb') as f:
                    image_list_valid = pickle.load(f) 
                with open('../process_input/split2/RSNA_PE_shiv/bbox_dict_valid.pickle', 'rb') as f:
                    bbox_dict_valid = pickle.load(f)

            pred_prob = np.zeros((len(image_list_valid),),dtype=np.float32)
            datagen = PEDataset_val(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=image_list_valid, target_size=image_size)
            generator = DataLoader(dataset=datagen, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

            model.eval()
            losses = AverageMeter()
            for i, (images, labels) in tqdm(enumerate(generator), total=len(generator)):        
                images, labels = images.float().to(device), labels.float().to(device)
                with torch.no_grad():
                    start = i*batch_size
                    end = start+batch_size
                    if i == len(generator)-1:
                        end = len(generator.dataset)
                    logits = model(images)
                    loss = criterion(logits.view(-1), labels) # was with BCEwithLogitsLoss
                    losses.update(loss.item(), images.size(0))
                    pred_prob[start:end] = np.squeeze(logits.sigmoid().cpu().data.numpy()) # no need of sigmoid for BCEloss


            label = np.zeros((len(image_list_valid),),dtype=int)        
            for i in range(len(image_list_valid)):
                label[i] = image_dict[image_list_valid[i]]['pe_present_on_image']
            auc = roc_auc_score(label, pred_prob)

            print(backboneName + "_" + gwn)
            print("Epoch: " + str(epoch_index))
            print('loss:{}, auc:{}'.format(losses.avg, auc), flush=True)
            print()

            np.save(out_dir + 'groundTruth.npy', label)
            np.save(out_dir + 'predicted_label.npy', pred_prob)


            string_msg = 'Validation => loss:{}, auc:{}\n'.format(losses.avg, auc)
            output_text_file_name = title_name + '_outputLines' + '_' + gwn + '.txt'
            output_text_file = open(output_text_file_name, 'a')
            string_msg0 = "[INFO] Reduced Training Data loaded: " + str(redu) + "%\n"
            output_text_file.write(string_msg0)
            output_text_file.write(string_msg)

    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


    string_msg = "Time: {:0>2}:{:0>2}:{:05.2f} \n".format(int(hours),int(minutes),seconds) 
    output_text_file = open(output_text_file_name, 'a')
    output_text_file.write(string_msg)
    output_text_file.close()

    exit()

if __name__ == "__main__":
    main()
