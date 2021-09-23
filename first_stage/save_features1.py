import argparse
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from pretrainedmodels.senet import se_resnext50_32x4d, se_resnet50
import random
from sklearn.metrics import roc_auc_score
import pickle
import pydicom
import time
from model_pytorch import Classifier_model, get_weight_name, ProgressMeter, save_checkpoint

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
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    # X = (X*255.0).astype('uint8')
    return X

class PEDataset(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.target_size=target_size
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,index):
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]
        data1 = pydicom.dcmread('/ocean/projects/bcs190005p/nahid92/Data/RSNA_PE/train/'+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_minus1']+'.dcm')
        data2 = pydicom.dcmread('/ocean/projects/bcs190005p/nahid92/Data/RSNA_PE/train/'+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        data3 = pydicom.dcmread('/ocean/projects/bcs190005p/nahid92/Data/RSNA_PE/train/'+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_plus1']+'.dcm')
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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        feature = x.view(x.size(0), -1)
        x = self.last_linear(feature)
        return feature, x


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
        feature = x.view(x.size(0), -1)
        x = self.last_linear(feature)
        return feature, x


def transferLearningModels(backboneName, _weight):
    if _weight == "ImageNet":
        PRETRAINED = True
    else:
        PRETRAINED = False
        
    if backboneName == "densenet121":
        from densenet_copiedModel import densenet121_f as densenet121
        model = densenet121(pretrained=PRETRAINED)
        model.classifier = nn.Linear(1024, 1)
        print("=> DenseNet121 loaded model: " + _weight)
        return model
    elif backboneName == "vgg16":
        model = models.vgg16(pretrained=PRETRAINED)
        model.classifier = nn.Linear(25088, 1)
        print("=> VGG16 loaded model: " + _weight)
        return model
    elif backboneName == "xception":
        from xception_copiedModel import xception_f as xception
        if _weight == "ImageNet":
            PRETRAINED = 'imagenet'
        else:
            PRETRAINED = None
        model = xception(num_classes=1000, pretrained=PRETRAINED)
        print("=> Xception loaded model: " + _weight)
        kernelCount = model.last_linear.in_features
        model.last_linear = nn.Sequential(nn.Linear(kernelCount, 1))
    elif backboneName == "resnet50":
        from resnet_copiedModel import resnet50_f as resnet50
        model = resnet50(pretrained=False) # true or false
        kernelCount = model.fc.in_features
        model.fc = nn.Linear(kernelCount, 1)
        print("=> ResNet50 loaded model: ", _weight)
    elif backboneName == "resnet18":
        from resnet_copiedModel import resnet18_f as resnet18
        model = resnet18(pretrained=False) # true or false
        kernelCount = model.fc.in_features
        model.fc = nn.Linear(kernelCount, 1)        
        print("=> ResNet18 loaded model: " + _weight)
    else:
        print("Couldn't find model...")
        exit()

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractFeature", type=str, default="valid", help="Extract feature from Train or Valid set")
    parser.add_argument("--backboneName", type=str, default="resnet18", help="resnet18 | resnet50 | densenet121 | xception")
    parser.add_argument("--loadW", type=str, default="ImageNet", help="Random | ImageNet")
    parser.add_argument("--runV", type=str, default="_v0_", help="model load during val or not")
    parser.add_argument("--redu", type=int, default=100, help="Reduced Data")
    parser.add_argument("--batch_size", type=int, default=32, help="BatchSize")
    parser.add_argument("--feature_sSize", type=int, default=512, help="feature_sSize")
    parser.add_argument("--feature_mode", type=int, default=1, help="FunedTune version or nonFinedTune version")
    args = parser.parse_args()

    runV = args.runV
    backboneName = args.backboneName
    loadW = args.loadW
    redu = args.redu
    batch_size = args.batch_size # was 96
    image_size = 576
    feature_sSize = args.feature_sSize # 1024 2048
    extractFeature = args.extractFeature
    feature_mode = args.feature_mode

    # gwn =  loadW + "_" + str(image_size) + runV 
    # title_name = 'TransferLearning' + "_"
    gwn =  str(image_size) + "_" + loadW + runV 
    title_name = 'FineTune_Reduced_' + str(redu) + "_"
    if feature_mode == 1:
        out_dir = "BestModels_100percent_TrainData/" + title_name + backboneName +'_' + gwn + '/'
    else:
        out_dir = "nonFineTuned_100percent_TrainData/" + title_name + backboneName +'_' + gwn + '/'
        title_name = 'nonFineTune_Reduced_' + str(redu) + "_"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    print("Run:", runV)
    print("BackBone:", backboneName)
    print("Batch Size: " + str(batch_size))
    print("Image Size: " + str(image_size))
    print("Output Directory: " + out_dir)


    ## ------------------------------------------------ DataLoader ------------------------------------------------ ##
    # prepare input
    if extractFeature == 'valid': # Valid Data
        import pickle
        if redu == 100:
            with open('../process_input/split2/image_list_valid.pickle', 'rb') as f:
                image_list_valid = pickle.load(f) 
            with open('../process_input/split2/image_dict.pickle', 'rb') as f:
                image_dict = pickle.load(f) 
            with open('../lung_localization/split2/bbox_dict_valid.pickle', 'rb') as f:
                bbox_dict_valid = pickle.load(f)       
        elif redu == 200:
            import pickle5 as pickle
            with open('../process_input/RSNA_PE_shiv/image_list_valid.pickle', 'rb') as f:
                image_list_valid = pickle.load(f) 
            with open('../process_input/RSNA_PE_shiv/image_dict.pickle', 'rb') as f:
                image_dict = pickle.load(f) 
            with open('../process_input/RSNA_PE_shiv/bbox_dict_valid.pickle', 'rb') as f:
                bbox_dict_valid = pickle.load(f)     

        feature = np.zeros((len(image_list_valid), feature_sSize),dtype=np.float32)
        pred_prob = np.zeros((len(image_list_valid),),dtype=np.float32)
        print('Validation Data:', len(image_list_valid), len(image_dict), len(bbox_dict_valid))
        datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=image_list_valid, target_size=image_size)
        generator = DataLoader(dataset=datagen, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)
    else: # Train Data
        import pickle
        if redu == 100:
            with open('../process_input/split2/image_list_train.pickle', 'rb') as f:
                image_list_train = pickle.load(f) 
            with open('../process_input/split2/image_dict.pickle', 'rb') as f:
                image_dict = pickle.load(f) 
            with open('../lung_localization/split2/bbox_dict_train.pickle', 'rb') as f:
                bbox_dict_train = pickle.load(f)
        elif redu == 200:
            import pickle5 as pickle
            with open('../process_input/RSNA_PE_shiv/image_list_train.pickle', 'rb') as f:
                image_list_train = pickle.load(f) 
            with open('../process_input/RSNA_PE_shiv/image_dict.pickle', 'rb') as f:
                image_dict = pickle.load(f) 
            with open('../process_input/RSNA_PE_shiv/bbox_dict_train.pickle', 'rb') as f:
                bbox_dict_train = pickle.load(f)


        feature = np.zeros((len(image_list_train), feature_sSize),dtype=np.float32)
        pred_prob = np.zeros((len(image_list_train),),dtype=np.float32)
        print('Training Data:',len(image_list_train), len(image_dict), len(bbox_dict_train))
        datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train, target_size=image_size)
        generator = DataLoader(dataset=datagen, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)


    ## ------------------------------------------------ Model Loading ------------------------------------------------ ##
    # Validation
    if backboneName == 'seresnext50':
        model = seresnext50()
        if feature_mode == 1:
            model.load_state_dict(torch.load(out_dir + 'epoch2'))
        print("SeResnext50 Model Loaded...") 
    elif backboneName == "seresnet50": # Given - Provided
        model = seresnet50(pretrainedModel="ImageNet")
        if feature_mode == 1:
            model.load_state_dict(torch.load(out_dir + 'epoch0'))
    elif backboneName == "sexception":
        from xception_copiedModel import xception_f as xception
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
        if feature_mode == 1:
            # model.load_state_dict(torch.load(out_dir + 'epoch0'))
            # model.load_state_dict(torch.load("BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_v1403_/_SavedModel_8_sexception_576_ImageNet_v1403__checkpoint.pth.tar"))
            
            # checkpoint loading
            path_checkpoint = "BestModels_100percent_TrainData/FineTune_Reduced_100_seresnext50_576_ImageNet_v1004_/_SavedModel_2_seresnext50_576_ImageNet_v1004__checkpoint.pth.tar"
            modelCheckpoint = torch.load(path_checkpoint)
            state_dict = modelCheckpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]
            msg = model.load_state_dict(state_dict)
            assert len(msg.missing_keys) == 0
            print("=> loaded checkPoint model '{}'".format(path_checkpoint))
        print("SE-Xception Loaded with conversion to SE (newly_created).")
    elif loadW == "ImageNet" or loadW == "Random":
        model = transferLearningModels(backboneName, loadW)
        if feature_mode == 1:
            # checkpoint loading
            # path_checkpoint = out_dir+"__"+backboneName+"__"+gwn+"_checkpoint.pth.tar"
            path_checkpoint = "BestModels_100percent_TrainData/FineTune_Reduced_100_xception_576_ImageNet_v1005_/_SavedModel_1_xception_576_ImageNet_v1005__checkpoint.pth.tar"
            modelCheckpoint = torch.load(path_checkpoint)
            state_dict = modelCheckpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]
            msg = model.load_state_dict(state_dict)
            assert len(msg.missing_keys) == 0
            print("=> loaded checkPoint model '{}'".format(path_checkpoint))
            # fOpen = open(out_dir+"model_arch.txt", "w+")
            # fOpen.write(model)
            # fOpen.close()
            # print(model)
    else:
        _exp_name = loadW
        num_class = 1
        _model_weight_path = os.path.join("pretrain_model_weights", _exp_name + ".pth.tar")
        model,_ = Classifier_model(backboneName, num_class, conv=None, weight=_model_weight_path, linear_classifier=False, sobel=False, activation=None)
        # print(model)
        print(backboneName, _exp_name, "- loaded...")
        if feature_mode == 1: # FineTuned version
            if loadW == "sela-v2":
                path_checkpoint = "BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_sela-v2_576_v110_/__resnet50__sela-v2_576_v110__checkpoint.pth.tar"
                out_dir = "BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_sela-v2_576_v110_/"
            elif loadW == "barlowtwins":
                path_checkpoint = "BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_barlowtwins_576_v106_/__resnet50__barlowtwins_576_v106__checkpoint.pth.tar"
                out_dir = "BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_barlowtwins_576_v106_/"
            elif loadW == "deepcluster-v2":
                path_checkpoint = "BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_deepcluster-v2_576_v110_/__resnet50__deepcluster-v2_576_v110__checkpoint.pth.tar"
                out_dir = "BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_deepcluster-v2_576_v110_/"
            print("Fixed - Out_dir:", out_dir)
            modelCheckpoint = torch.load(path_checkpoint)
            state_dict = modelCheckpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]
            msg = model.load_state_dict(state_dict)
            assert len(msg.missing_keys) == 0
            print("=> loaded checkPoint model '{}'".format(path_checkpoint))

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    criterion = nn.BCEWithLogitsLoss().to(device)  
    print('Model loaded manually: ' + gwn)
    model.eval()


    ## ------------------------------------------------ Feature Extraction Starts ------------------------------------------------ ##
    
    print("Model Feature Extraction Started...")
    losses = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(generator), total=len(generator)):
        with torch.no_grad():
            start = i*batch_size
            end = start+batch_size
            if i == len(generator)-1:
                end = len(generator.dataset)

            images = images.to(device)  
            labels = labels.float().to(device)  

            features, logits = model(images)
            # print("[CHECK] Feature Shape:", features.shape)
            loss = criterion(logits.view(-1),labels)
            losses.update(loss.item(), images.size(0))

            feature[start:end] = np.squeeze(features.cpu().data.numpy())
            pred_prob[start:end] = np.squeeze(logits.sigmoid().cpu().data.numpy())

    if extractFeature == 'valid':
        # Saving features
        np.save(out_dir+'feature_valid', feature)
        np.save(out_dir+'pred_prob_valid', pred_prob)

        label = np.zeros((len(image_list_valid),),dtype=int)        
        for i in range(len(image_list_valid)):
            label[i] = image_dict[image_list_valid[i]]['pe_present_on_image']
        auc = roc_auc_score(label, pred_prob)

        print('loss:{}, auc:{}'.format(losses.avg, auc), flush=True)
        print()
        print("Validation Feature Extraction Done...")

    else: # Train
        np.save(out_dir+'feature_train', feature)
        np.save(out_dir+'pred_prob_train', pred_prob)

        label = np.zeros((len(image_list_train),),dtype=int)        
        for i in range(len(image_list_train)):
            label[i] = image_dict[image_list_train[i]]['pe_present_on_image']
        auc = roc_auc_score(label, pred_prob)

        print('loss:{}, auc:{}'.format(losses.avg, auc), flush=True)
        print()
        print("Training Feature Extraction Done...")


    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    string_msg = "Time: {:0>2}:{:0>2}:{:05.2f} \n".format(int(hours),int(minutes),seconds) 

    f_open = open(out_dir + 'feature_'+extractFeature+'_extraction_AUC.txt', 'w+')
    f_open.write(extractFeature + " Feature Extraction: \n")
    f_open.write("Backbone: " + backboneName + "\n")
    f_open.write("Load Weight: " + loadW + "\n")
    f_open.write("Data Data: " + str(redu) + "% \n")
    f_open.write("ImageSize: " + str(image_size) + "\n")
    f_open.write("Batch Size: " + str(batch_size) + "\n")
    f_open.write("Extract Feature: " + extractFeature + "\n")
    f_open.write("Run Version: " + runV + "\n")
    f_open.write("Output Directory: " + out_dir + "\n")
    f_open.write("AUC: " + str(auc) + "\n")
    f_open.write(string_msg)
    f_open.close()


if __name__ == "__main__":
    main()