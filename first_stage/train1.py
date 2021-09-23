# python -m torch.distributed.launch --nproc_per_node=1 train1_ResNet50.py --train_task=1 --val_task=0 --manual_load=0 --backboneName resnet50 --runV _v101_ --loadW ImageNet --redu 100 --BS 32 --numGPU 1
# python train1_ResNet50.py --train_task=0 --val_task=1 --manual_load=1 --backboneName resnet50 --runV _v101_ --loadW ImageNet --redu 100 --BS 32 --numGPU 1

# BridgeAI
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
from pretrainedmodels.senet import se_resnext50_32x4d, senet154
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

# DATA_DIR = '/data/jliang12/zzhou82/holy_grail/rsnastr20/train/'  # Agave
DATA_DIR = '/ocean/projects/bcs190005p/nahid92/Data/RSNA_PE/train/'  # Bridge

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

class resnext50_(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        import torchvision.models as models
        self.net = models.resnext50_32x4d(pretrained=pretrained)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.fc.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

class senet154_(nn.Module):
    def __init__(self ):
        super().__init__()
        # self.net = senet154(num_classes=1000, pretrained='imagenet')
        self.net = se_resnext50_32x4d(num_classes=1000, pretrained=None) # Random Init
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

class DRN_A_50(nn.Module):
    def __init__(self ):
        from drn_copiedModel import drn_a_50 as drn_model
        super().__init__()
        self.net = drn_model(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.fc.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

class DRN_D_54(nn.Module):
    def __init__(self ):
        from drn_copiedModel import drn_d_54 as drn_model
        super().__init__()
        self.net = drn_model(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.fc.in_channels
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


def transferLearningModels(backboneName, _weight):
    if _weight == "ImageNet":
        PRETRAINED = True
    else:
        PRETRAINED = False
        
    if backboneName == "densenet121":
        net = models.densenet121(pretrained=PRETRAINED)
        net.classifier = nn.Linear(1024, 1)
        print("=> DenseNet121 loaded model: " + _weight)
        return net
    elif backboneName == "vgg16":
        net = models.vgg16(pretrained=PRETRAINED)
        net.classifier = nn.Linear(25088, 1)
        print("=> VGG16 loaded model: " + _weight)
        return net
    elif backboneName == "xception":
        from xception_copiedModel import xception
        if _weight == "ImageNet":
            PRETRAINED = 'imagenet'
        else:
            PRETRAINED = None
        model = xception(num_classes=1000, pretrained=PRETRAINED)
        print("Xception: => loaded model: " + _weight)
        kernelCount = model.last_linear.in_features
        model.last_linear = nn.Sequential(nn.Linear(kernelCount, 1))
    elif backboneName == "resnet50":
        model = models.resnet50(pretrained=PRETRAINED)
        kernelCount = model.fc.in_features
        model.fc = nn.Linear(kernelCount, 1)
        print("=> ResNet50 loaded model: " + _weight)
    elif backboneName == "resnet101":
        from resnet_copied import resnet101
        model = resnet101(pretrained=None)
        if _weight=="ImageNet":
            state_dict = torch.load("pretrain_model_weights/resnet101-63fe2227.pth", map_location="cpu") # extra
            model.load_state_dict(state_dict) # was url
        kernelCount = model.fc.in_features
        model.fc = nn.Linear(kernelCount, 1)
        print("=> ResNet101 loaded model: " + _weight)
    elif backboneName == "resnet18":
        model = models.resnet18(pretrained=PRETRAINED)
        kernelCount = model.fc.in_features
        model.fc = nn.Linear(kernelCount, 1)
        print("=> ResNet18 loaded model: " + _weight)
    elif backboneName == "senet154":
        model = senet154_()
        print("=> SeNet154 loaded model: " + 'imagenet')
    elif backboneName == "drn_a_50":
        from drn_copiedModel import drn_a_50 as drn_model
        model = drn_model(pretrained=PRETRAINED)
        kernelCount = model.fc.in_features
        model.fc = nn.Linear(kernelCount, 1)
        print("=> DRN-A-50 loaded model: " + _weight)
    elif backboneName == "drn_d_54":
        from drn_copiedModel import drn_d_54 as drn_model
        model = drn_model(pretrained=PRETRAINED)
        model.fc = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
        print("=> DRN-D-54 loaded model: " + _weight)  
    elif backboneName == "resnext50":
        # model = resnext50_(pretrained=PRETRAINED)
        import torchvision.models as models
        num_class=1
        model = models.resnext50_32x4d(num_classes=1000, pretrained=PRETRAINED)
        kernelCount = model.fc.in_features
        # model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
        model.fc = nn.Sequential(nn.Linear(kernelCount, num_class))
        print("=> ResNext50 loaded model " + str(PRETRAINED))
    elif backboneName == "resnext101":
        import torchvision.models as models
        num_class=1
        model = models.resnext101_32x8d(num_classes=1000, pretrained=PRETRAINED)
        kernelCount = model.fc.in_features
        # model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
        model.fc = nn.Sequential(nn.Linear(kernelCount, num_class))
        print("=> ResNext101 loaded model " + str(PRETRAINED))

    return model

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    parser.add_argument("--train_task", type=int, default=1, help="train or not")
    parser.add_argument("--val_task", type=int, default=1, help="val or not")
    parser.add_argument("--manual_load", type=int, default=0, help="model load during val or not")

    parser.add_argument("--backboneName", type=str, default="resnet18", help="resnet18 | resnet50 | densenet121 | xception")
    parser.add_argument("--loadW", type=str, default="ImageNet", help="Random | ImageNet")

    parser.add_argument("--runV", type=str, default="_v0_", help="model load during val or not")

    parser.add_argument("--redu", type=int, default=100, help="Reduced Data")

    parser.add_argument("--BS", type=int, default=20, help="BatchSize")
    parser.add_argument("--numGPU", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--worker", type=int, default=12, help="Number of workers")
    parser.add_argument("--imgSize", type=int, default=576, help="ImageSize")

    args = parser.parse_args()

## ---------------------------- Hyper-parameter ---------------------------- ##  
    train_task = args.train_task
    val_task = args.val_task
    manually_load = args.manual_load
    runV = args.runV
    backboneName = args.backboneName
    loadW = args.loadW
    redu = args.redu
    nWorkers = args.worker
    # numSeed = randrange(2500)

    print("Train Task:", train_task)
    print("Val Task:", val_task)
    print("Model Load:", manually_load)
    print("Run:", runV)
    print("BackBone:", backboneName)
    print("Workers:", nWorkers)
    print("Seed:", numSeed)

    learning_rate = 0.0004 # was 0.0004
    batch_size = args.BS # was 32 | 20
    image_size = args.imgSize # was 576
    num_epoch = 1
    number_of_gpu = args.numGPU # was 4
    num_class = 1

    _model_weight = args.loadW # | "ImageNet"
    gwn =  _model_weight + "_" + str(image_size) + runV 


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
        with open('../process_input/split2/RSNA_PE_shiv/bbox_dict_train.pickle', 'rb') as f:
            bbox_dict_train = pickle.load(f)
        with open('../process_input/split2/RSNA_PE_shiv/image_list_train.pickle', 'rb') as f:
            image_list_train = pickle.load(f) 
        print("[INFO] Training data loaded - Shiv's Data Split")
    else: # for reduced training data
        with open('../process_input/split2/image_list_train_'+ str(redu) + '.pickle', 'rb') as f:
            image_list_train = pickle.load(f) 
        print("[INFO] Training Data loaded: ", redu, "%")


    print("Data is ready...")
    print(len(image_list_train), len(image_dict), len(bbox_dict_train))

    title_name = 'FineTune_TrainData_' + str(redu) + "_"
    out_dir = title_name + backboneName +'_' + gwn + '/'

    if train_task == 1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.device = device

        seed = numSeed # was 2001
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    
        # proxy_idx = 10
        # _exp_name = get_weight_name(os.path.join("pretrain_model_weights", "models.log"), proxy_idx)
        # _model_weight = os.path.join("pretrain_model_weights", _exp_name + ".pth.tar")

        print("Learning Rate: " + str(learning_rate))
        print("Batch Size: " + str(batch_size))
        print("Image Size: " + str(image_size))
        print("Number of Epoch: " + str(num_epoch))
        print("Model Backbone: " + backboneName)
        print("Model Weights: " + gwn)

        # Model Path setup
        
        # out_dir = 'weights_'+ backboneName +'_' + gwn + '/'

        ## Write configurations
        output_text_file_name = title_name + backboneName + '_outputLines' + '_' + gwn + '.txt'
        output_text_file = open(output_text_file_name, 'w')
        output_text_file.write("Learning Rate: " + str(learning_rate) + "\n")
        output_text_file.write("Batch Size: " + str(batch_size) + "\n")
        output_text_file.write("Image Size: " + str(image_size) + "\n")
        output_text_file.write("Number of Epoch: " + str(num_epoch) + "\n")
        output_text_file.write("Model Backbone: " + backboneName + "\n")
        output_text_file.write("Model Weights: " + gwn + "\n")
        output_text_file.write("DataSize: " + str(redu) + "\n")
        output_text_file.write("Run_Version: " + runV + "\n")
        output_text_file.write("--------------------- \n")
        output_text_file.close()


    ## ---------------------------- Model: Training ---------------------------- ##
        # build model
        if args.local_rank != 0:
            torch.distributed.barrier()

        if loadW == 'ImageNet' or loadW == 'Random':
            model = transferLearningModels(backboneName, loadW)
        else:
            _exp_name = loadW
            _model_weight_path = os.path.join("pretrain_model_weights", _exp_name + ".pth.tar")
            model,_ = Classifier_model(backboneName, num_class, conv=None, weight=_model_weight_path, linear_classifier=False, sobel=False, activation=None)

        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(args.device)

        num_train_steps = int(len(image_list_train)/(batch_size*number_of_gpu)*num_epoch)   # 4 GPUs
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
        amp.register_float_function(torch, 'sigmoid') # extra
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        criterion = nn.BCEWithLogitsLoss().to(args.device) # old with Se_resNet50
        # criterion = nn.BCELoss().to(args.device) # training => BCELosswithLogits use

        train_transform = albumentations.Compose([
            albumentations.RandomContrast(limit=0.2, p=1.0),
            albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
            # albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0) # Old
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0, p=1.0) # New
        ])

        datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train, target_size=image_size, transform=train_transform)
        sampler = DistributedSampler(datagen)
        generator = DataLoader(dataset=datagen, sampler=sampler, batch_size=batch_size, num_workers=nWorkers, pin_memory=True)


        # print(model)
        print("Model is ready:", backboneName, _model_weight) 
        output_text_file = open(output_text_file_name, 'a')
        for ep in range(num_epoch):
            losses = AverageMeter()
            model.train()
            for j,(images,labels) in enumerate(generator):
                images = images.to(args.device)
                labels = labels.float().to(args.device)

                logits = model(images)
                loss = criterion(logits.view(-1),labels) # was with BCEwithLogitsLoss
                losses.update(loss.item(), images.size(0))

                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss: # was with BCEwithLogitsLoss
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
                            }, False, filename=out_dir+"__"+backboneName+"__"+gwn)

        print("Model training done...")
        output_text_file.close()
        del datagen, sampler, generator



## ---------------------------- Task: Validation ---------------------------- ##    


    ## ---------------------------- Model: Validation ---------------------------- ##    
    if val_task == 1:        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if manually_load == 1:
            # model building
            if loadW=="ImageNet" or loadW=="Random":
                model = transferLearningModels(backboneName, loadW)
            else:
                _exp_name = loadW
                _model_weight_path = os.path.join("pretrain_model_weights", _exp_name + ".pth.tar")
                model,_ = Classifier_model(backboneName, num_class, conv=None, weight=_model_weight_path, linear_classifier=False, sobel=False, activation=None)

            # checkpoint loading
            path_checkpoint = out_dir+"__"+backboneName+"__"+gwn+"_checkpoint.pth.tar"
            modelCheckpoint = torch.load(path_checkpoint)
            state_dict = modelCheckpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]
            msg = model.load_state_dict(state_dict)
            assert len(msg.missing_keys) == 0
            print("=> loaded checkPoint model '{}'".format(path_checkpoint))

            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)  
            # criterion = nn.BCELoss().to(device)
            criterion = nn.BCEWithLogitsLoss().to(device)

            print('Model loaded manually: ' + gwn)
            output_text_file_name = title_name + backboneName + '_outputLines' + '_' + gwn + '.txt'
        else:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)  
            #criterion = nn.BCELoss().to(device) # testing => BCELoss
            criterion = nn.BCEWithLogitsLoss().to(device)



    ## ---------------------------- Testing Data Loading ---------------------------- ##    
        print("Validation Testing Started")
        import pickle5 as pickle
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
        generator = DataLoader(dataset=datagen, batch_size=batch_size, shuffle=False, num_workers=nWorkers, pin_memory=True)

    ## ---------------------------- Model Testing ---------------------------- ## 
        model.eval()
        losses = AverageMeter()
        for i, (images, labels) in tqdm(enumerate(generator), total=len(generator)):        
            images, labels = images.float().to(device), labels.float().to(device)
            with torch.no_grad():
                start = i*batch_size
                end = start+batch_size
                if i == len(generator)-1:
                    end = len(generator.dataset)
                logits = model(images) # sigmoid comming back should be 0-1
                loss = criterion(logits.view(-1), labels) # was with BCEwithLogitsLoss
                losses.update(loss.item(), images.size(0))
                pred_prob[start:end] = np.squeeze(logits.sigmoid().cpu().data.numpy()) # no need of sigmoid for BCEloss


        label = np.zeros((len(image_list_valid),),dtype=int)        
        for i in range(len(image_list_valid)):
            label[i] = image_dict[image_list_valid[i]]['pe_present_on_image']
        auc = roc_auc_score(label, pred_prob)


        print(backboneName + "_" + gwn)
        print('loss:{}, auc:{}'.format(losses.avg, auc), flush=True)
        print()

        np.save(out_dir + 'groundTruth.npy', label)
        np.save(out_dir + 'predicted_label.npy', pred_prob)


        string_msg = 'Validation => loss:{}, auc:{}\n'.format(losses.avg, auc)
        output_text_file = open(output_text_file_name, 'a')
        string_msg0 = "[INFO] Training Data loaded: " + str(redu) + "%\n"
        output_text_file.write(string_msg0)
        output_text_file.write(string_msg)
        output_text_file.close()

    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


    string_msg = "Time: {:0>2}:{:0>2}:{:05.2f} \n".format(int(hours),int(minutes),seconds) 
    output_text_file = open(output_text_file_name, 'a')
    output_text_file.write("SEED: " + str(numSeed) + "\n")
    output_text_file.write(string_msg)
    output_text_file.close()

    exit()

if __name__ == "__main__":
    main()
