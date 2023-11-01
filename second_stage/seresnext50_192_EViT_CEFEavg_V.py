import argparse
import numpy as np
import pandas as pd
import pickle
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from models_vit.modeling_exp import VisionTransformer
from models_vit.modeling_exp import CONFIGS as CONFIGS_model_name
import pandas as pd
import csv 
try:
    from apex import amp
except:
    check=1
from random import randrange
from sklearn.metrics import roc_auc_score, log_loss
from modules_settransformer import ISAB, PMA, SAB

numSeed = randrange(2250) # 2-2-5-0

# https://www.kaggle.com/bminixhofer/a-validation-framework-impact-of-the-random-seed

def step_decay(step, learning_rate, num_epochs, warmup_epochs=15):
    lr = learning_rate
    progress = (step - warmup_epochs) / float(num_epochs - warmup_epochs)
    progress = np.clip(progress, 0.0, 1.0)
    #decay_type == 'cosine':
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    if warmup_epochs:
      lr = lr * np.minimum(1., step / warmup_epochs)
    return lr

def convert2float64(logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic):
    logits_pe = torch.add(logits_pe.type(torch.DoubleTensor).cuda(), 0.00001)
    logits_npe = torch.add(logits_npe.type(torch.DoubleTensor).cuda(), 0.00001)
    logits_idt = torch.add(logits_idt.type(torch.DoubleTensor).cuda(), 0.00001)
    logits_lpe = torch.add(logits_lpe.type(torch.DoubleTensor).cuda(), 0.00001)
    logits_rpe = torch.add(logits_rpe.type(torch.DoubleTensor).cuda(), 0.00001)
    logits_cpe = torch.add(logits_cpe.type(torch.DoubleTensor).cuda(), 0.00001)
    logits_gte = torch.add(logits_gte.type(torch.DoubleTensor).cuda(), 0.00001)
    logits_lt = torch.add(logits_lt.type(torch.DoubleTensor).cuda(), 0.00001)
    logits_chronic = torch.add(logits_chronic.type(torch.DoubleTensor).cuda(), 0.00001)
    logits_acute_and_chronic = torch.add(logits_acute_and_chronic.type(torch.DoubleTensor).cuda(), 0.00001)
    # print("[INFO] Convert2Float64 function called...")
    return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic

def check4NanVal(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10):
    check = torch.any(torch.isnan(a1)).item()
    if check == True:
        print("NAN found in the model output: a1")
        # exit()
    check = torch.any(torch.isnan(a2)).item()
    if check == True:
        print("NAN found in the model output: a2")
        print(a2)
        exit()
    check = torch.any(torch.isnan(a3)).item()
    if check == True:
        print("NAN found in the model output: a3")
        exit()
    check = torch.any(torch.isnan(a4)).item()
    if check == True:
        print("NAN found in the model output: a4")
        exit()
    check = torch.any(torch.isnan(a5)).item()
    if check == True:
        print("NAN found in the model output: a5")
        exit()
    check = torch.any(torch.isnan(a6)).item()
    if check == True:
        print("NAN found in the model output: a6")
        exit()
    check = torch.any(torch.isnan(a7)).item()
    if check == True:
        print("NAN found in the model output: a7")
        exit()
    check = torch.any(torch.isnan(a8)).item()
    if check == True:
        print("NAN found in the model output: a8")
        exit()
    check = torch.any(torch.isnan(a9)).item()
    if check == True:
        print("NAN found in the model output: a9")
        exit()
    check = torch.any(torch.isnan(a10)).item()
    if check == True:
        print("NAN found in the model output: a10")
        exit()

def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    for i in range(classCount):
        outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
    return outAUROC

def printAUC_results(auc_output):
    print("-----")
    print("Negative_Exam_for_PE: " + str(auc_output[0]))
    print("Indeterminate: " + str(auc_output[1]))
    print("Left_PE: " + str(auc_output[2]))
    print("Right_PE: " + str(auc_output[3]))
    print("Central_PE: " + str(auc_output[4]))
    print("RV_LV_ratio_gte_1: " + str(auc_output[5]))
    print("RV_LV_ratio_lt_1: " + str(auc_output[6]))
    print("Chronic_PE: " + str(auc_output[7]))
    print("Acute_and_Chronic_PE: " + str(auc_output[8]))
    print("-----")


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=6144, # was 2048
        num_outputs=1,
        dim_output=1,
        num_inds=32,
        dim_hidden=128,
        num_heads=1,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze(2)

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

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

class PEDataset(Dataset):
    def __init__(self,
                 feature_array,
                 image_to_feature,
                 series_dict,
                 image_dict,
                 series_list,
                 seq_len):
        self.feature_array=feature_array
        self.image_to_feature=image_to_feature
        self.series_dict=series_dict
        self.image_dict=image_dict
        self.series_list=series_list # Validation or training patient list
        self.seq_len=seq_len
    def __len__(self):
        return len(self.series_list)
    def __getitem__(self,index):
        image_list = self.series_dict[self.series_list[index]]['sorted_image_list'] 
        if len(image_list)>self.seq_len: # M > N(512) 
            x = np.zeros((len(image_list), self.feature_array.shape[1]*3), dtype=np.float32) # feature_array.shape[1] = 2048
            y_pe = np.zeros((len(image_list), 1), dtype=np.float32)
            mask = np.ones((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]] 
                y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
            # x = cv2.resize(x, (self.feature_array.shape[1]*3, self.seq_len), interpolation = cv2.INTER_LINEAR)
            # y_pe = np.squeeze(cv2.resize(y_pe, (1, self.seq_len), interpolation = cv2.INTER_LINEAR))
            # tempCenter = x.shape[0] // 2            
            # tempUpX = x[tempCenter:tempCenter+self.seq_len//2, :] # error
            # tempLowX = x[tempCenter-self.seq_len//2:tempCenter, :]
            # x = np.concatenate((tempLowX,tempUpX), axis=0)
            
            # tempUpY = y_pe[tempCenter:tempCenter+self.seq_len//2, :]
            # tempLowY = y_pe[tempCenter-self.seq_len//2:tempCenter, :]
            # y_pe = np.squeeze(np.concatenate((tempLowY,tempUpY), axis=0))

            x = x[x.shape[0]-512:,:]
            y_pe = np.squeeze(y_pe[y_pe.shape[0]-512:,:])
        else: # M < N(512) => Zero-padding
            x = np.zeros((self.seq_len, self.feature_array.shape[1]*3), dtype=np.float32)
            mask = np.zeros((self.seq_len,), dtype=np.float32)
            y_pe = np.zeros((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]]
                mask[i] = 1.  
                y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
        x[1:,self.feature_array.shape[1]:self.feature_array.shape[1]*2] = x[1:,:self.feature_array.shape[1]] - x[:-1,:self.feature_array.shape[1]]
        x[:-1,self.feature_array.shape[1]*2:] = x[:-1,:self.feature_array.shape[1]] - x[1:,:self.feature_array.shape[1]]
        x = torch.tensor(x, dtype=torch.float32)
        y_pe = torch.tensor(y_pe, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        y_npe = self.series_dict[self.series_list[index]]['negative_exam_for_pe']
        y_idt = self.series_dict[self.series_list[index]]['indeterminate']
        y_lpe = self.series_dict[self.series_list[index]]['leftsided_pe']
        y_rpe = self.series_dict[self.series_list[index]]['rightsided_pe']
        y_cpe = self.series_dict[self.series_list[index]]['central_pe']
        y_gte = self.series_dict[self.series_list[index]]['rv_lv_ratio_gte_1']
        y_lt = self.series_dict[self.series_list[index]]['rv_lv_ratio_lt_1']
        y_chronic = self.series_dict[self.series_list[index]]['chronic_pe']
        y_acute_and_chronic = self.series_dict[self.series_list[index]]['acute_and_chronic_pe']
        # print("[INFO] Shape of X:", x.shape)
        # print("[INFO] Shape of Y_PE:", y_pe.shape)
        # exit()
        return x, y_pe, mask, y_npe, y_idt, y_lpe, y_rpe, y_cpe, y_gte, y_lt, y_chronic, y_acute_and_chronic, self.series_list[index]

class PEDataset_2048F(Dataset):
    def __init__(self,
                 feature_array,
                 image_to_feature,
                 series_dict,
                 image_dict,
                 series_list,
                 seq_len):
        self.feature_array=feature_array
        self.image_to_feature=image_to_feature
        self.series_dict=series_dict
        self.image_dict=image_dict
        self.series_list=series_list # Validation or training patient list
        self.seq_len=seq_len
    def __len__(self):
        return len(self.series_list)
    def __getitem__(self,index):
        image_list = self.series_dict[self.series_list[index]]['sorted_image_list'] 
        if len(image_list)>self.seq_len: # M > N(512) 
            x = np.zeros((len(image_list), self.feature_array.shape[1]*1), dtype=np.float32) # feature_array.shape[1] = 2048
            y_pe = np.zeros((len(image_list), 1), dtype=np.float32)
            mask = np.ones((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]] 
                y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
            # x = cv2.resize(x, (self.feature_array.shape[1]*1, self.seq_len), interpolation = cv2.INTER_LINEAR)
            # y_pe = np.squeeze(cv2.resize(y_pe, (1, self.seq_len), interpolation = cv2.INTER_LINEAR))
            x = x[x.shape[0]-512:,:]
            y_pe = np.squeeze(y_pe[y_pe.shape[0]-512:,:])
        else: # M < N(512) => Zero-padding
            x = np.zeros((self.seq_len, self.feature_array.shape[1]*1), dtype=np.float32)
            mask = np.zeros((self.seq_len,), dtype=np.float32)
            y_pe = np.zeros((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]]
                mask[i] = 1.  
                y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
#         x[1:,self.feature_array.shape[1]:self.feature_array.shape[1]*2] = x[1:,:self.feature_array.shape[1]] - x[:-1,:self.feature_array.shape[1]]
#         x[:-1,self.feature_array.shape[1]*2:] = x[:-1,:self.feature_array.shape[1]] - x[1:,:self.feature_array.shape[1]]
        x = torch.tensor(x, dtype=torch.float32)
        y_pe = torch.tensor(y_pe, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        y_npe = self.series_dict[self.series_list[index]]['negative_exam_for_pe']
        y_idt = self.series_dict[self.series_list[index]]['indeterminate']
        y_lpe = self.series_dict[self.series_list[index]]['leftsided_pe']
        y_rpe = self.series_dict[self.series_list[index]]['rightsided_pe']
        y_cpe = self.series_dict[self.series_list[index]]['central_pe']
        y_gte = self.series_dict[self.series_list[index]]['rv_lv_ratio_gte_1']
        y_lt = self.series_dict[self.series_list[index]]['rv_lv_ratio_lt_1']
        y_chronic = self.series_dict[self.series_list[index]]['chronic_pe']
        y_acute_and_chronic = self.series_dict[self.series_list[index]]['acute_and_chronic_pe']
        return x, y_pe, mask, y_npe, y_idt, y_lpe, y_rpe, y_cpe, y_gte, y_lt, y_chronic, y_acute_and_chronic, self.series_list[index]


parser = argparse.ArgumentParser()
parser.add_argument("--backboneName", type=str, default="resnet18", help="resnet18 | resnet50 | densenet121 | xception")
parser.add_argument("--runV", type=str, default="version_1", help="model load during val or not")
parser.add_argument("--featureMode", type=int, default=1, help="fine tuned or non-fine tuned")
parser.add_argument("--ssl_method_name", type=str, default=" ", help="fine tuned or non-fine tuned")

parser.add_argument("--LR", type=float, default=0.001)
parser.add_argument("--BS", type=int, default=32)
parser.add_argument("--EP", type=int, default=50)

parser.add_argument("--LoadViTPreTrainedW", type=str, default="no")
parser.add_argument("--typeTransformer", type=str, default="ViTBase", help=" ")
parser.add_argument("--feature2work", type=int, default=6144, help="6144 or 2048")
parser.add_argument("--mlp_dim", type=int, default=1024)
parser.add_argument("--transOut", type=int, default=512)
parser.add_argument("--numH", type=int, default=2)
parser.add_argument("--numB", type=int, default=2)

parser.add_argument("--transOutLayer", type=str, default='flatten', help="flatten or mean")
parser.add_argument("--typeIntigration", type=str, default='clsToken', help="only classToken or classToken&Rest")
parser.add_argument("--lossAll", type=str, default='yes', help="yes:count loss_pe| no:without loss_pe")
parser.add_argument("--optChoice", type=str, default='SGD', help="ADAM or SGD")
args = parser.parse_args()
backboneName = args.backboneName
runV = args.runV
featureMode = args.featureMode
ssl_method_name = args.ssl_method_name

# prepare input
with open('../process_input/split2/series_list_train.pickle', 'rb') as f:
    series_list_train = pickle.load(f)
with open('../process_input/split2/series_list_valid.pickle', 'rb') as f:
    series_list_valid = pickle.load(f) 
with open('../process_input/split2/image_list_train.pickle', 'rb') as f:
    image_list_train = pickle.load(f)
with open('../process_input/split2/image_list_valid.pickle', 'rb') as f:
    image_list_valid = pickle.load(f) 
with open('../process_input/split2/image_dict.pickle', 'rb') as f:
    image_dict = pickle.load(f) 
with open('../process_input/split2/series_dict.pickle', 'rb') as f:
    series_dict = pickle.load(f)


## Loading Features
if backboneName == "resnet18":
    feature_train = np.load('../seresnext50/TransferLearning_resnet18_ImageNet_576_v206_/feature_train.npy')
    feature_valid = np.load('../seresnext50/TransferLearning_resnet18_ImageNet_576_v206_/feature_valid.npy')
    titleName = "resnet18_512"
    featureSize = 512
elif backboneName == "resnet50":
    if featureMode == 1:
        # Pre-trained from ImageNet
        if ssl_method_name == " ":
            feature_train = np.load('../seresnext50/TransferLearning_resnet50_ImageNet_576_v102_/feature_train.npy')
            feature_valid = np.load('../seresnext50/TransferLearning_resnet50_ImageNet_576_v102_/feature_valid.npy')
            titleName = "resnet50_2048_FT"

        # Pre-trained from SSL method
        if ssl_method_name == "sela-v2":
            feature_train = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_sela-v2_576_v110_/feature_train.npy')
            feature_valid = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_sela-v2_576_v110_/feature_valid.npy')
            titleName = "resnet50_SSL_selav2_2048_ViT"
        elif ssl_method_name == "deepcluster-v2":
            feature_train = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_deepcluster-v2_576_v110_/feature_train.npy')
            feature_valid = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_deepcluster-v2_576_v110_/feature_valid.npy')  
            titleName = "resnet50_SSL_deepclusterv2_2048_ViT"
        elif ssl_method_name == "barlowtwins":
            feature_train = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_barlowtwins_576_v106_/feature_train.npy')
            feature_valid = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_barlowtwins_576_v106_/feature_valid.npy')             
            titleName = "resnet50_SSL_barlowtwins_2048_ViT"
    else:
        # Pre-trained from SSL method
        if ssl_method_name == "sela-v2":
            feature_train = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_sela-v2_SSL_FT_v101_/feature_train.npy')
            feature_valid = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_sela-v2_SSL_FT_v101_/feature_valid.npy')
            titleName = "resnet50_SSL_selav2_2048_nonFT"
        elif ssl_method_name == "deepcluster-v2":
            feature_train = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_deepcluster-v2_SSL_FT_v101_/feature_train.npy')
            feature_valid = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_deepcluster-v2_SSL_FT_v101_/feature_valid.npy')
            titleName = "resnet50_SSL_deepclusterv2_2048_nonFT"  
        elif ssl_method_name == "barlowtwins":                    
            feature_train = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_barlowtwins_SSL_FT_v101_/feature_train.npy')
            feature_valid = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_barlowtwins_SSL_FT_v101_/feature_valid.npy') 
            titleName = "resnet50_SSL_barlowtwins_2048_nonFT"
    featureSize = 2048
elif backboneName == "xception":
    if featureMode == 1:
        dddd = "/mnt/dfs/nuislam/Projects/PE_Detection/code_v7_competitionCodes/1st_Place_RSNA-STR-Pulmonary-Embolism-Detection-main/RSNA-STR-Pulmonary-Embolism-Detection-main/trainval/seresnext50/"
        feature_train = np.load(dddd+'BestModels_100percent_TrainData/TransferLearning_xception_ImageNet_576_v304_/feature_train.npy')
        feature_valid = np.load(dddd+'BestModels_100percent_TrainData/TransferLearning_xception_ImageNet_576_v304_/feature_valid.npy')
        # feature_train = np.load('../seresnext50/BestModels_100percent_TrainData/FineTune_Reduced_100_xception_576_ImageNet_v1005_/feature_train.npy')
        # feature_valid = np.load('../seresnext50/BestModels_100percent_TrainData/FineTune_Reduced_100_xception_576_ImageNet_v1005_/feature_valid.npy')
        titleName = "xception_2048_ViT"
    else:
        feature_train = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_xception_576_ImageNet_vnonFT_101_/feature_train.npy')
        feature_valid = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_xception_576_ImageNet_vnonFT_101_/feature_valid.npy') 
        titleName = "xception_2048_nonFT"
    featureSize = 2048
elif backboneName == "densenet121":
    feature_train = np.load('../seresnext50/TransferLearning_densenet121_ImageNet_576_v406_/feature_train.npy')
    feature_valid = np.load('../seresnext50/TransferLearning_densenet121_ImageNet_576_v406_/feature_valid.npy')
    titleName = "densenet121_1024"
    featureSize = 1024
elif backboneName == "seresnext50":
    if featureMode == 1:
        feature_train = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_seresnext50_576_ImageNet_v1004_/feature_train.npy')
        feature_valid = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_seresnext50_576_ImageNet_v1004_/feature_valid.npy')
        titleName = "seresnext50_2048_ViT"
    else:
        feature_train = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_seresnext50_576_ImageNet_vnonFT_101_/feature_train.npy')
        feature_valid = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_seresnext50_576_ImageNet_vnonFT_101_/feature_valid.npy')    
        titleName = "seresnext50_2048_nonFT"    
    featureSize = 2048
elif backboneName == "sexception":
    if featureMode == 1:
        # feature_train = np.load("../seresnext50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_manual_576_ImageNet_v1005_/feature_train.npy")
        # feature_valid = np.load("../seresnext50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_manual_576_ImageNet_v1005_/feature_valid.npy")
        dddd = "/mnt/dfs/nuislam/Projects/PE_Detection/code_v7_competitionCodes/1st_Place_RSNA-STR-Pulmonary-Embolism-Detection-main/RSNA-STR-Pulmonary-Embolism-Detection-main/trainval"
        feature_train = np.load(dddd+"/seresnext50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_v1007_/feature_train.npy")
        feature_valid = np.load(dddd+"/seresnext50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_v1007_/feature_valid.npy")
        titleName = "sexception_2048_ViT"
    else:
        feature_train = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_vnonFT_101_/feature_train.npy')
        feature_valid = np.load('../seresnext50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_vnonFT_101_/feature_valid.npy')  
        titleName = "sexception_2048_nonFT"
    featureSize = 2048
elif backboneName == "efficientnet-b4":
    if featureMode == 1:
        feature_train = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_efficientnet-b4_ImageNet_576_v1301_/feature_train.npy")
        feature_valid = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_efficientnet-b4_ImageNet_576_v1301_/feature_valid.npy")
        titleName = "efficientnet-b4_1792_ViT"
    featureSize = 1792    
elif backboneName == "efficientnet-b5":
    if featureMode == 1:
        feature_train = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_efficientnet-b5_ImageNet_576_v1507_/feature_train.npy")
        feature_valid = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_efficientnet-b5_ImageNet_576_v1507_/feature_valid.npy")
        titleName = "efficientnet-b5_2048_ViT"
    featureSize = 2048


out_dir = 'predictions_'+titleName+'/' + runV + '/'
out_dir2 = 'predictions_'+titleName+'/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
export_csvFile = pd.DataFrame(columns=['Epoch', 'avgLoss' ,'NPE', 'IND','LPE','RPE','CPE','RLVgte','RLVlt','CHRoPE','AcuAndChroPE', 'Mean', 'AUC_Slice'])
export_csvFile.to_csv(out_dir+'export_csvFile.csv', index=False)


print("Data is ready...")
print(feature_train.shape, feature_valid.shape, len(series_list_train), len(series_list_valid), len(image_list_train), len(image_list_valid), len(image_dict), len(series_dict))

image_to_feature_train = {}
image_to_feature_valid = {}
for i in range(len(feature_train)):
    image_to_feature_train[image_list_train[i]] = i
for i in range(len(feature_valid)):
    image_to_feature_valid[image_list_valid[i]] = i

loss_weight_dict = {
                     'negative_exam_for_pe': 0.0736196319,
                     'indeterminate': 0.09202453988,
                     'chronic_pe': 0.1042944785,
                     'acute_and_chronic_pe': 0.1042944785,
                     'central_pe': 0.1877300613,
                     'leftsided_pe': 0.06257668712,
                     'rightsided_pe': 0.06257668712,
                     'rv_lv_ratio_gte_1': 0.2346625767,
                     'rv_lv_ratio_lt_1': 0.0782208589,
                     'pe_present_on_image': 0.07361963,
                   }


seed = numSeed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# hyperparameters
if args.feature2work == 6144:
    feature_size = featureSize*3 # was 2048*3
else: # 2048
    feature_size = featureSize
seq_len = 512
lstm_size = args.transOut
MD=args.mlp_dim # was 1024
NH=args.numH
NL=args.numB
learning_rate = args.LR # was 0.0005 0.00001 
batch_size = args.BS # was 64
num_epoch = args.EP


print("Learning Rate: " + str(learning_rate))
print("Batch Size: " + str(batch_size))
print("LSTM Size: " + str(lstm_size))
print("Sequence Length: " + str(seq_len))
print("Feature Size: " + str(feature_size))
print("Transformer Output Size: " + str(lstm_size))
print("Number of Epoch: " + str(num_epoch))

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class PENet(nn.Module):
    def __init__(self, input_len, lstm_size):  # 6144, 512
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear_pe = nn.Linear(lstm_size*2, 1)
        self.last_linear_npe = nn.Linear(lstm_size*4, 1)
        self.last_linear_idt = nn.Linear(lstm_size*4, 1)
        self.last_linear_lpe = nn.Linear(lstm_size*4, 1)
        self.last_linear_rpe = nn.Linear(lstm_size*4, 1)
        self.last_linear_cpe = nn.Linear(lstm_size*4, 1)
        self.last_linear_gte = nn.Linear(lstm_size*4, 1)
        self.last_linear_lt = nn.Linear(lstm_size*4, 1)
        self.last_linear_chronic = nn.Linear(lstm_size*4, 1)
        self.last_linear_acute_and_chronic = nn.Linear(lstm_size*4, 1)
        self.attention = Attention(lstm_size*2, seq_len)
    def forward(self, x, mask):
        #x = SpatialDropout(0.5)(x)
        h_lstm1, _ = self.lstm1(x)
        #avg_pool = torch.mean(h_lstm2, 1)
        logits_pe = self.last_linear_pe(h_lstm1)
        max_pool, _ = torch.max(h_lstm1, 1)
        att_pool = self.attention(h_lstm1, mask)
        conc = torch.cat((max_pool, att_pool), 1)  
        logits_npe = self.last_linear_npe(conc)
        logits_idt = self.last_linear_idt(conc)
        logits_lpe = self.last_linear_lpe(conc)
        logits_rpe = self.last_linear_rpe(conc)
        logits_cpe = self.last_linear_cpe(conc)
        logits_gte = self.last_linear_gte(conc)
        logits_lt = self.last_linear_lt(conc)
        logits_chronic = self.last_linear_chronic(conc)
        logits_acute_and_chronic = self.last_linear_acute_and_chronic(conc)
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic

class PENet_ViT(nn.Module):
    def __init__(self, staTe, input_len, MD, NH, NL, lstm_size, seqL):  # 6144, 512
        super().__init__()
        self.lstm1 = VisionTransformer(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, num_classes=lstm_size) # num_classes or outputFeature = 1 to 512
        self.staTe = staTe
        if self.staTe == 'mean':
            seqL = 1
        self.last_linear_pe = nn.Linear(lstm_size*seqL, 512)
        self.last_linear_npe = nn.Linear(lstm_size*seqL, 1)
        self.last_linear_idt = nn.Linear(lstm_size*seqL, 1)
        self.last_linear_lpe = nn.Linear(lstm_size*seqL, 1)
        self.last_linear_rpe = nn.Linear(lstm_size*seqL, 1)
        self.last_linear_cpe = nn.Linear(lstm_size*seqL, 1)
        self.last_linear_gte = nn.Linear(lstm_size*seqL, 1)
        self.last_linear_lt = nn.Linear(lstm_size*seqL, 1)
        self.last_linear_chronic = nn.Linear(lstm_size*seqL, 1)
        self.last_linear_acute_and_chronic = nn.Linear(lstm_size*seqL, 1)
#         self.attention = Attention(lstm_size*2, 512) # seq_len = 512
    def forward(self, x, mask):
        #x = SpatialDropout(0.5)(x)
        h_lstm1,_ = self.lstm1(x)
        check = torch.any(torch.isnan(h_lstm1)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()

        if self.staTe == 'mean':
            h_lstm1 = torch.mean(h_lstm1, 1) # if mean, stays the same as transoutput
        else: # flatten
            h_lstm1 = torch.flatten(h_lstm1, start_dim=1) # if flatten, gets larger

        logits_pe = self.last_linear_pe(h_lstm1)
        logits_npe = self.last_linear_npe(h_lstm1)
        logits_idt = self.last_linear_idt(h_lstm1)
        logits_lpe = self.last_linear_lpe(h_lstm1)
        logits_rpe = self.last_linear_rpe(h_lstm1)
        logits_cpe = self.last_linear_cpe(h_lstm1)
        logits_gte = self.last_linear_gte(h_lstm1)
        logits_lt = self.last_linear_lt(h_lstm1)
        logits_chronic = self.last_linear_chronic(h_lstm1)
        logits_acute_and_chronic = self.last_linear_acute_and_chronic(h_lstm1)
        check4NanVal(logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic)
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic


class PENet_ViT_v2(nn.Module):
    def __init__(self, input_len, MD, NH, NL, lstm_size):  # 6144, 512
        super().__init__()
        self.linear_feature = nn.Linear(2048, 768) # 2048=DataFeatures | 768=forViTinput
        self.lstm1 = VisionTransformer(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, num_classes=lstm_size*2) # ViToutput=512x2=1024

        self.last_linear_pe = nn.Linear(lstm_size*2, 1)
        self.last_linear_npe = nn.Linear(lstm_size*2, 1)
        self.last_linear_idt = nn.Linear(lstm_size*2, 1)
        self.last_linear_lpe = nn.Linear(lstm_size*2, 1)
        self.last_linear_rpe = nn.Linear(lstm_size*2, 1)
        self.last_linear_cpe = nn.Linear(lstm_size*2, 1)
        self.last_linear_gte = nn.Linear(lstm_size*2, 1)
        self.last_linear_lt = nn.Linear(lstm_size*2, 1)
        self.last_linear_chronic = nn.Linear(lstm_size*2, 1)
        self.last_linear_acute_and_chronic = nn.Linear(lstm_size*2, 1)
        
    def forward(self, x, mask):
        logits_features = self.linear_feature(x)
        h_lstm1,_ = self.lstm1(logits_features)
        
        logits_pe = self.last_linear_pe(h_lstm1)    
        
        h_lstm1 = torch.mean(h_lstm1, 1)        
        logits_npe = self.last_linear_npe(h_lstm1)
        logits_idt = self.last_linear_idt(h_lstm1)
        logits_lpe = self.last_linear_lpe(h_lstm1)
        logits_rpe = self.last_linear_rpe(h_lstm1)
        logits_cpe = self.last_linear_cpe(h_lstm1)
        logits_gte = self.last_linear_gte(h_lstm1)
        logits_lt = self.last_linear_lt(h_lstm1)
        logits_chronic = self.last_linear_chronic(h_lstm1)
        logits_acute_and_chronic = self.last_linear_acute_and_chronic(h_lstm1)
        
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic


class PENet_ViT_BASE(nn.Module):
    def __init__(self, staTe, input_len, MD, NH, NL, lstm_size, seqL):  # 6144, 512
        super().__init__()
        self.linear_feature = nn.Linear(lstm_size, 768)
        self.lstm1 = VisionTransformer(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, num_classes=768) # num_classes or outputFeature = 1 to 512
        if args.LoadViTPreTrainedW == "yes":
            self.lstm1.load_from(np.load("/mnt/dfs/nuislam/Projects/PE_Detection/code_v7_competitionCodes/vit_codes/checkpoint/imagenet21k_ViT-B_16.npz"))
            print("Loaded ViT Base 16 pretrained model weights")
        self.staTe = staTe

        self.last_linear_pe = nn.Linear(768, 512)
        self.last_linear_npe = nn.Linear(768, 1)
        self.last_linear_idt = nn.Linear(768, 1)
        self.last_linear_lpe = nn.Linear(768, 1)
        self.last_linear_rpe = nn.Linear(768, 1)
        self.last_linear_cpe = nn.Linear(768, 1)
        self.last_linear_gte = nn.Linear(768, 1)
        self.last_linear_lt = nn.Linear(768, 1)
        self.last_linear_chronic = nn.Linear(768, 1)
        self.last_linear_acute_and_chronic = nn.Linear(768, 1)
    def forward(self, x, mask):
        logits_features = self.linear_feature(x)
        h_lstm1,_ = self.lstm1(logits_features)
        check = torch.any(torch.isnan(h_lstm1)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()

        if self.staTe == 'mean':
            h_lstm1 = torch.mean(h_lstm1, 1) # if mean, stays the same as transoutput
        else: # flatten
            h_lstm1 = torch.flatten(h_lstm1, start_dim=1) # if flatten, gets larger

        logits_pe = self.last_linear_pe(h_lstm1)
        logits_npe = self.last_linear_npe(h_lstm1)
        logits_idt = self.last_linear_idt(h_lstm1)
        logits_lpe = self.last_linear_lpe(h_lstm1)
        logits_rpe = self.last_linear_rpe(h_lstm1)
        logits_cpe = self.last_linear_cpe(h_lstm1)
        logits_gte = self.last_linear_gte(h_lstm1)
        logits_lt = self.last_linear_lt(h_lstm1)
        logits_chronic = self.last_linear_chronic(h_lstm1)
        logits_acute_and_chronic = self.last_linear_acute_and_chronic(h_lstm1)
        check4NanVal(logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic)
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic

class PENet_ViT_BASE_withFCs(nn.Module):
    def __init__(self, staTe, input_len, MD, NH, NL, lstm_size, seqL):  # 6144, 512
        super().__init__()
        self.linear_feature1 = nn.Linear(6144, 3072)
        self.linear_feature2 = nn.Linear(3072, 1536)
        self.linear_feature3 = nn.Linear(1536, 768)
        self.lstm1 = VisionTransformer(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, num_classes=768) # num_classes or outputFeature = 1 to 512
        if args.LoadViTPreTrainedW == "yes":
            self.lstm1.load_from(np.load("/mnt/dfs/nuislam/Projects/PE_Detection/code_v7_competitionCodes/vit_codes/checkpoint/imagenet21k_ViT-B_16.npz"))
            print("Loaded ViT Base 16 pretrained model weights")
        self.staTe = staTe

        self.last_linear_pe = nn.Linear(768, 512)
        self.last_linear_npe = nn.Linear(768, 1)
        self.last_linear_idt = nn.Linear(768, 1)
        self.last_linear_lpe = nn.Linear(768, 1)
        self.last_linear_rpe = nn.Linear(768, 1)
        self.last_linear_cpe = nn.Linear(768, 1)
        self.last_linear_gte = nn.Linear(768, 1)
        self.last_linear_lt = nn.Linear(768, 1)
        self.last_linear_chronic = nn.Linear(768, 1)
        self.last_linear_acute_and_chronic = nn.Linear(768, 1)
    def forward(self, x, mask):
        logits_features = self.linear_feature1(x)
        logits_features = self.linear_feature2(logits_features)
        logits_features = self.linear_feature3(logits_features)

        h_lstm1,_ = self.lstm1(logits_features)
        check = torch.any(torch.isnan(h_lstm1)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()

        if self.staTe == 'mean':
            h_lstm1 = torch.mean(h_lstm1, 1) # if mean, stays the same as transoutput
        else: # flatten
            h_lstm1 = torch.flatten(h_lstm1, start_dim=1) # if flatten, gets larger

        logits_pe = self.last_linear_pe(h_lstm1)
        logits_npe = self.last_linear_npe(h_lstm1)
        logits_idt = self.last_linear_idt(h_lstm1)
        logits_lpe = self.last_linear_lpe(h_lstm1)
        logits_rpe = self.last_linear_rpe(h_lstm1)
        logits_cpe = self.last_linear_cpe(h_lstm1)
        logits_gte = self.last_linear_gte(h_lstm1)
        logits_lt = self.last_linear_lt(h_lstm1)
        logits_chronic = self.last_linear_chronic(h_lstm1)
        logits_acute_and_chronic = self.last_linear_acute_and_chronic(h_lstm1)
        check4NanVal(logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic)
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic

class PENet_ViT_BASE_withAtten_withFCs(nn.Module):
    def __init__(self, staTe, input_len, MD, NH, NL, lstm_size, seqL):  # 6144, 512
        super().__init__()
        self.linear_feature1 = nn.Linear(6144, 3072)
        self.linear_feature2 = nn.Linear(3072, 1536)
        self.linear_feature3 = nn.Linear(1536, 768)
        self.lstm1 = VisionTransformer(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, num_classes=768) # num_classes or outputFeature = 1 to 512
        if args.LoadViTPreTrainedW == "yes":
            self.lstm1.load_from(np.load("/mnt/dfs/nuislam/Projects/PE_Detection/code_v7_competitionCodes/vit_codes/checkpoint/imagenet21k_ViT-B_16.npz"))
            print("Loaded ViT Base 16 pretrained model weights")
        self.staTe = staTe

        self.last_linear_pe = nn.Linear(768*2, 512)
        self.last_linear_npe = nn.Linear(768*2, 1)
        self.last_linear_idt = nn.Linear(768*2, 1)
        self.last_linear_lpe = nn.Linear(768*2, 1)
        self.last_linear_rpe = nn.Linear(768*2, 1)
        self.last_linear_cpe = nn.Linear(768*2, 1)
        self.last_linear_gte = nn.Linear(768*2, 1)
        self.last_linear_lt = nn.Linear(768*2, 1)
        self.last_linear_chronic = nn.Linear(768*2, 1)
        self.last_linear_acute_and_chronic = nn.Linear(768*2, 1)
        self.attention = Attention(768, 512)
    def forward(self, x, mask):
        logits_features = self.linear_feature1(x)
        logits_features = self.linear_feature2(logits_features)
        logits_features = self.linear_feature3(logits_features)

        h_lstm1,_ = self.lstm1(logits_features)
        check = torch.any(torch.isnan(h_lstm1)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()

#         if self.staTe == 'mean':
#             h_lstm1 = torch.mean(h_lstm1, 1) # if mean, stays the same as transoutput
#         else: # flatten
#             h_lstm1 = torch.flatten(h_lstm1, start_dim=1) # if flatten, gets larger
        
        max_pool, _ = torch.max(h_lstm1, 1)
        att_pool = self.attention(h_lstm1, mask)
        conc = torch.cat((max_pool, att_pool), 1)
    
        logits_pe = self.last_linear_pe(conc)
        logits_npe = self.last_linear_npe(conc)
        logits_idt = self.last_linear_idt(conc)
        logits_lpe = self.last_linear_lpe(conc)
        logits_rpe = self.last_linear_rpe(conc)
        logits_cpe = self.last_linear_cpe(conc)
        logits_gte = self.last_linear_gte(conc)
        logits_lt = self.last_linear_lt(conc)
        logits_chronic = self.last_linear_chronic(conc)
        logits_acute_and_chronic = self.last_linear_acute_and_chronic(conc)

        check4NanVal(logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic)
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic

class PENet_ViT_BASE_withMultiAtten_withFCs(nn.Module):
    def __init__(self, staTe, input_len, MD, NH, NL, lstm_size, seqL):  # 6144, 512
        super().__init__()
        from models_vit.modeling_exp import VisionTransformer
        from models_vit.modeling_exp import CONFIGS as CONFIGS_model_name
        
        self.linear_feature1 = nn.Linear(6144, 3072)
        self.linear_feature2 = nn.Linear(3072, 1536)
        self.linear_feature3 = nn.Linear(1536, 768)
        self.lstm1 = VisionTransformer(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, num_classes=768) # num_classes or outputFeature = 1 to 512
        if args.LoadViTPreTrainedW == "yes":
            self.lstm1.load_from(np.load("/ocean/projects/bcs190005p/nahid92/Projects/PE_Detection/vit_codes/checkpoint/imagenet21k_ViT-B_16.npz"))
            print("Loaded ViT Base 16 pretrained model weights")
        self.staTe = staTe

        self.last_linear_pe = nn.Linear(768*2, 512)
        self.last_linear_npe = nn.Linear(768*2, 1)
        self.last_linear_idt = nn.Linear(768*2, 1)
        self.last_linear_lpe = nn.Linear(768*2, 1)
        self.last_linear_rpe = nn.Linear(768*2, 1)
        self.last_linear_cpe = nn.Linear(768*2, 1)
        self.last_linear_gte = nn.Linear(768*2, 1)
        self.last_linear_lt = nn.Linear(768*2, 1)
        self.last_linear_chronic = nn.Linear(768*2, 1)
        self.last_linear_acute_and_chronic = nn.Linear(768*2, 1)
        
        self.attentionPEslice = Attention(768, 512)
        self.attentionNPE = Attention(768, 512)
        self.attentionIDT = Attention(768, 512)
        self.attentionLPE = Attention(768, 512)
        self.attentionRPE = Attention(768, 512)
        self.attentionCPE = Attention(768, 512)
        self.attentionGTE = Attention(768, 512)
        self.attentionLT = Attention(768, 512)
        self.attentionChronic = Attention(768, 512)
        self.attentionAcuteChronic = Attention(768, 512)
        
    def forward(self, x, mask):
        logits_features = self.linear_feature1(x)
        logits_features = self.linear_feature2(logits_features)
        logits_features = self.linear_feature3(logits_features)

        h_lstm1,_ = self.lstm1(logits_features)
        check = torch.any(torch.isnan(h_lstm1)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()
        max_pool, _ = torch.max(h_lstm1, 1)        

        logits_pe = self.last_linear_pe( torch.cat((max_pool, self.attentionPEslice(h_lstm1, mask)), 1) )
        logits_npe = self.last_linear_npe( torch.cat((max_pool, self.attentionNPE(h_lstm1, mask)), 1) )
        logits_idt = self.last_linear_idt( torch.cat((max_pool, self.attentionIDT(h_lstm1, mask)), 1) )
        logits_lpe = self.last_linear_lpe( torch.cat((max_pool, self.attentionLPE(h_lstm1, mask)), 1) )
        logits_rpe = self.last_linear_rpe( torch.cat((max_pool, self.attentionRPE(h_lstm1, mask)), 1) )
        logits_cpe = self.last_linear_cpe( torch.cat((max_pool, self.attentionCPE(h_lstm1, mask)), 1) )
        logits_gte = self.last_linear_gte( torch.cat((max_pool, self.attentionGTE(h_lstm1, mask)), 1) )
        logits_lt = self.last_linear_lt( torch.cat((max_pool, self.attentionLT(h_lstm1, mask)), 1) )
        logits_chronic = self.last_linear_chronic( torch.cat((max_pool, self.attentionChronic(h_lstm1, mask)), 1) )
        logits_acute_and_chronic = self.last_linear_acute_and_chronic( torch.cat((max_pool, self.attentionAcuteChronic(h_lstm1, mask)), 1) )
        
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic


class PENet_ViT_BASE_withAtten_withFCs_clstoken(nn.Module):
    def __init__(self, staTe, input_len, MD, NH, NL, lstm_size, seqL):  # 6144, 512
        super().__init__()
        from models_vit.modeling_exp_clstoken import VisionTransformer as VisionTransformerClsToken
        from models_vit.modeling_exp_clstoken import CONFIGS as CONFIGS_model_name
        
        self.linear_feature1 = nn.Linear(6144, 3072)
        self.linear_feature2 = nn.Linear(3072, 1536)
        self.linear_feature3 = nn.Linear(1536, 768)
        self.lstm1 = VisionTransformerClsToken(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, num_classes=1)
        if args.LoadViTPreTrainedW == "yes":
            self.lstm1.load_from(np.load("/ocean/projects/bcs190005p/nahid92/Projects/PE_Detection/vit_codes/checkpoint/imagenet21k_ViT-B_16.npz"))
            print("Loaded ViT Base 16 pretrained model weights")
        self.staTe = staTe

        self.last_linear_pe = nn.Linear(768*2, 512)
        self.last_linear_npe = nn.Linear(768*2, 1)
        self.last_linear_idt = nn.Linear(768*2, 1)
        self.last_linear_lpe = nn.Linear(768*2, 1)
        self.last_linear_rpe = nn.Linear(768*2, 1)
        self.last_linear_cpe = nn.Linear(768*2, 1)
        self.last_linear_gte = nn.Linear(768*2, 1)
        self.last_linear_lt = nn.Linear(768*2, 1)
        self.last_linear_chronic = nn.Linear(768*2, 1)
        self.last_linear_acute_and_chronic = nn.Linear(768*2, 1)
        self.attention = Attention(768, 512)
    def forward(self, x, mask):
        logits_features = self.linear_feature1(x)
        logits_features = self.linear_feature2(logits_features)
        logits_features = self.linear_feature3(logits_features)

        h_lstm1_clsToken, h_lstm1_restEverything = self.lstm1(logits_features)
        check = torch.any(torch.isnan(h_lstm1_restEverything)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()
        max_pool, _ = torch.max(h_lstm1_restEverything, 1)
        att_pool = self.attention(h_lstm1_restEverything, mask)
        conc = torch.cat((max_pool, att_pool), 1)
        
        logits_pe = self.last_linear_pe(conc)
#         logits_npe_clsToken = np.squeeze(h_lstm1_clsToken)
#         logits_npe_restEverything = h_lstm1_restEverything
        logits_npe = self.last_linear_npe(conc)
        logits_idt = self.last_linear_idt(conc)
        logits_lpe = self.last_linear_lpe(conc)
        logits_rpe = self.last_linear_rpe(conc)
        logits_cpe = self.last_linear_cpe(conc)
        logits_gte = self.last_linear_gte(conc)
        logits_lt = self.last_linear_lt(conc)
        logits_chronic = self.last_linear_chronic(conc)
        logits_acute_and_chronic = self.last_linear_acute_and_chronic(conc)
        
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic


class PENet_ViT_BASE_withAtten_withFCs_clstokenV2(nn.Module):
    def __init__(self, staTe, staTus, input_len, MD, NH, NL, lstm_size, seqL):  # 6144, 512
        super().__init__()
        from models_vit.modeling_exp_clstokenV2 import VisionTransformer as VisionTransformerClsToken
        from models_vit.modeling_exp_clstokenV2 import CONFIGS as CONFIGS_model_name
        
        self.linear_feature1 = nn.Linear(6144, 3072)
        self.linear_feature2 = nn.Linear(3072, 1536)
        self.linear_feature3 = nn.Linear(1536, 768)
        self.lstm1 = VisionTransformerClsToken(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, num_classes=1)
        if args.LoadViTPreTrainedW == "yes":
            self.lstm1.load_from(np.load("/ocean/projects/bcs190005p/nahid92/Projects/PE_Detection/vit_codes/checkpoint/imagenet21k_ViT-B_16.npz"))
            print("Loaded ViT Base 16 pretrained model weights")
        self.staTe = staTe
        self.staTus = staTus

        self.last_linear_pe = nn.Linear(768*2, 512)
        if self.staTus == 'clsToken_rest' or self.staTus == 'rest':
            self.last_linear_npe = nn.Linear(768*2, 1)
            self.last_linear_idt = nn.Linear(768*2, 1)
            self.last_linear_lpe = nn.Linear(768*2, 1)
            self.last_linear_rpe = nn.Linear(768*2, 1)
            self.last_linear_cpe = nn.Linear(768*2, 1)
            self.last_linear_gte = nn.Linear(768*2, 1)
            self.last_linear_lt = nn.Linear(768*2, 1)
            self.last_linear_chronic = nn.Linear(768*2, 1)
            self.last_linear_acute_and_chronic = nn.Linear(768*2, 1)
        self.attention = Attention(768, 512)
    def forward(self, x, mask):
        logits_features = self.linear_feature1(x)
        logits_features = self.linear_feature2(logits_features)
        logits_features = self.linear_feature3(logits_features)

        h_lstm1_clsToken, h_lstm1_restEverything = self.lstm1(logits_features)
        check = torch.any(torch.isnan(h_lstm1_restEverything)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()
            
        # From class token to each exam label    
        logits_npe_clsToken = h_lstm1_clsToken[:, 0, :]
        logits_idt_clsToken = h_lstm1_clsToken[:, 1, :]
        logits_lpe_clsToken = h_lstm1_clsToken[:, 2, :]
        logits_rpe_clsToken = h_lstm1_clsToken[:, 3, :]
        logits_cpe_clsToken = h_lstm1_clsToken[:, 4, :]
        logits_gte_clsToken = h_lstm1_clsToken[:, 5, :]
        logits_lt_clsToken = h_lstm1_clsToken[:, 6, :]
        logits_chronic_clsToken = h_lstm1_clsToken[:, 7, :]
        logits_acute_and_chronic_clsToken = h_lstm1_clsToken[:, 8, :]            
            
        att_pool = self.attention(h_lstm1_restEverything, mask)
        max_pool, _ = torch.max(h_lstm1_restEverything, 1)        
        conc = torch.cat((max_pool, att_pool), 1)     
        logits_pe = self.last_linear_pe(conc)
        if self.staTus == 'clsToken_rest':
            # From embedded 768 features to exam label             
            logits_npe_rest = self.last_linear_npe(conc)
            logits_idt_rest = self.last_linear_idt(conc)
            logits_lpe_rest = self.last_linear_lpe(conc)
            logits_rpe_rest = self.last_linear_rpe(conc)
            logits_cpe_rest = self.last_linear_cpe(conc)
            logits_gte_rest = self.last_linear_gte(conc)
            logits_lt_rest = self.last_linear_lt(conc)
            logits_chronic_rest = self.last_linear_chronic(conc)
            logits_acute_and_chronic_rest = self.last_linear_acute_and_chronic(conc)
            
            logits_npe = (logits_npe_clsToken + logits_npe_rest) / 2
            logits_idt = (logits_idt_clsToken + logits_idt_rest) / 2
            logits_lpe = (logits_lpe_clsToken + logits_lpe_rest) / 2
            logits_rpe = (logits_rpe_clsToken + logits_rpe_rest) / 2
            logits_cpe = (logits_cpe_clsToken + logits_cpe_rest) / 2
            logits_gte = (logits_gte_clsToken + logits_gte_rest) / 2
            logits_lt = (logits_lt_clsToken + logits_lt_rest) / 2
            logits_chronic = (logits_chronic_clsToken + logits_chronic_rest) / 2
            logits_acute_and_chronic = (logits_acute_and_chronic_clsToken + logits_acute_and_chronic_rest) / 2
        elif self.staTus == 'clsToken':
            logits_npe = logits_npe_clsToken
            logits_idt = logits_idt_clsToken
            logits_lpe = logits_lpe_clsToken
            logits_rpe = logits_rpe_clsToken
            logits_cpe = logits_cpe_clsToken
            logits_gte = logits_gte_clsToken
            logits_lt = logits_lt_clsToken
            logits_chronic = logits_chronic_clsToken
            logits_acute_and_chronic = logits_acute_and_chronic_clsToken
        elif self.staTus == 'rest':
            # From embedded 768 features to exam label             
            logits_npe_rest = self.last_linear_npe(conc)
            logits_idt_rest = self.last_linear_idt(conc)
            logits_lpe_rest = self.last_linear_lpe(conc)
            logits_rpe_rest = self.last_linear_rpe(conc)
            logits_cpe_rest = self.last_linear_cpe(conc)
            logits_gte_rest = self.last_linear_gte(conc)
            logits_lt_rest = self.last_linear_lt(conc)
            logits_chronic_rest = self.last_linear_chronic(conc)
            logits_acute_and_chronic_rest = self.last_linear_acute_and_chronic(conc)
            
            logits_npe = logits_npe_rest
            logits_idt = logits_idt_rest
            logits_lpe = logits_lpe_rest
            logits_rpe = logits_rpe_rest
            logits_cpe = logits_cpe_rest
            logits_gte = logits_gte_rest
            logits_lt = logits_lt_rest
            logits_chronic = logits_chronic_rest
            logits_acute_and_chronic = logits_acute_and_chronic_rest
        
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic


class PENet_SetTransformer(nn.Module):
    def __init__(self, input_len, lstm_size):
        super().__init__()

#         self.linear=nn.Linear(input_len, input_len)
        self.last_linear_pe=SetTransformer(dim_output=512)
        
        self.transformer_npe=SetTransformer()
        self.transformer_idt=SetTransformer()
        self.transformer_lpe=SetTransformer()
        self.transformer_rpe=SetTransformer()
        self.transformer_cpe=SetTransformer()
        self.transformer_gte=SetTransformer()
        self.transformer_lt=SetTransformer()
        self.transformer_chronic=SetTransformer()
        self.transformer_acute_and_chronic=SetTransformer()

    def forward(self, x, mask):
#         x=torch.squeeze(x)
#         x=torch.unsqueeze(x,0)
#         x=self.linear(x)

        logits_pe = self.last_linear_pe(x)        
        x_npe=self.transformer_npe(x)
        x_idt=self.transformer_idt(x)
        x_lpe=self.transformer_lpe(x)
        x_rpe=self.transformer_rpe(x)
        x_cpe=self.transformer_cpe(x)
        x_gte=self.transformer_gte(x)
        x_lt=self.transformer_lt(x)
        x_chronic=self.transformer_chronic(x)
        x_acute_and_chronic=self.transformer_acute_and_chronic(x)
        return torch.squeeze(logits_pe), x_npe, x_idt, x_lpe, x_rpe, x_cpe, x_gte, x_lt, x_chronic, x_acute_and_chronic   

class PENet_SetTransformer_v2(nn.Module):
    def __init__(self, input_len, lstm_size=512):  # 6144, 512
        super().__init__()
        self.lstm1 = SetTransformer(dim_input=6144, num_outputs=512, dim_output=768, num_inds=32, dim_hidden=512, num_heads=1) # dim_input=6144, num_outputs=512, dim_output=768, num_inds=32, dim_hidden=512, num_heads=1
        self.last_linear_pe = nn.Linear(768*2, 512)
        self.last_linear_npe = nn.Linear(768*2, 1)
        self.last_linear_idt = nn.Linear(768*2, 1)
        self.last_linear_lpe = nn.Linear(768*2, 1)
        self.last_linear_rpe = nn.Linear(768*2, 1)
        self.last_linear_cpe = nn.Linear(768*2, 1)
        self.last_linear_gte = nn.Linear(768*2, 1)
        self.last_linear_lt = nn.Linear(768*2, 1)
        self.last_linear_chronic = nn.Linear(768*2, 1)
        self.last_linear_acute_and_chronic = nn.Linear(768*2, 1)
        self.attention = Attention(768, 512)
    def forward(self, x, mask):
        h_lstm1 = self.lstm1(x)
        check = torch.any(torch.isnan(h_lstm1)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()        
        max_pool, _ = torch.max(h_lstm1, 1)
        att_pool = self.attention(h_lstm1, mask)
        conc = torch.cat((max_pool, att_pool), 1)
        
        logits_pe = self.last_linear_pe(conc)
        logits_npe = self.last_linear_npe(conc)
        logits_idt = self.last_linear_idt(conc)
        logits_lpe = self.last_linear_lpe(conc)
        logits_rpe = self.last_linear_rpe(conc)
        logits_cpe = self.last_linear_cpe(conc)
        logits_gte = self.last_linear_gte(conc)
        logits_lt = self.last_linear_lt(conc)
        logits_chronic = self.last_linear_chronic(conc)
        logits_acute_and_chronic = self.last_linear_acute_and_chronic(conc)        
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic

# model = PENet(input_len=feature_size, lstm_size=lstm_size)

device = torch.device('cuda')
# model = PENet_ViT(staTe=args.transOutLayer, input_len=feature_size, MD=MD, NH=NH, NL=NL, lstm_size=lstm_size, seqL=seq_len)
# model = PENet_ViT_v2(input_len=768, MD=MD, NH=NH, NL=NL, lstm_size=lstm_size) # input_len=768, MD=3072, NH=1, NL=2, lstm_size=512
if args.typeTransformer == "ViTBase":
    model = PENet_ViT_BASE(staTe=args.transOutLayer, input_len=768, MD=MD, NH=NH, NL=NL, lstm_size=lstm_size, seqL=seq_len)
elif args.typeTransformer == "ViTBase_with_Attn_FCs_CEFEavg":
    from modified_vit_models import PENet_ViT_BASE_withAtten_withFCs_CEFE_avg # This is working !!!
    model = PENet_ViT_BASE_withAtten_withFCs_CEFE_avg(staTe=args.transOutLayer, input_len=768, MD=3072, NH=12, NL=12, lstm_size=6144, seqL=512, LoadViTPreTrainedW=args.LoadViTPreTrainedW, embedding=False)

if args.LoadViTPreTrainedW == "yes":
    print("Loaded ViT Base 16 pretrained model weights")

# print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Parameters:",pytorch_total_params)
print("Total Parameters:",pytorch_total_trainable_params)
print("--------------------------------------------------")
print("Learning Rate:", learning_rate)
print("feature_size =", feature_size)
print("mlp_dim =", MD)
print("num_heads =", NH)
print("num_blocks/layers =", NL)
print("--------------------------------------------------")

class ComulativeBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, head, output_embedding, target):
        loss_head = F.binary_cross_entropy_with_logits(head, target)
        loss_output_embedding = F.binary_cross_entropy_with_logits(output_embedding, target)
        loss = 0.5 * loss_head + 0.5 * loss_output_embedding
        return loss

model = model.cuda()

if args.optChoice == "ADAM":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # was active
elif args.optChoice == "SGD": # good one
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0, momentum=0.9, nesterov=False)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # was active # optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()
# criterion1 = nn.BCEWithLogitsLoss().cuda()
criterion1 = ComulativeBCELoss().cuda()

# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
# model.to(device)

print("Model is ready...")

# training

# iterator for training

if args.feature2work == 6144:
    train_datagen = PEDataset(feature_array=feature_train,
                              image_to_feature=image_to_feature_train,
                              series_dict=series_dict,
                              image_dict=image_dict,
                              series_list=series_list_train,
                              seq_len=seq_len)
else: # 2048
    train_datagen = PEDataset_2048F(feature_array=feature_train,
                          image_to_feature=image_to_feature_train,
                          series_dict=series_dict,
                          image_dict=image_dict,
                          series_list=series_list_train,
                          seq_len=seq_len)

train_generator = DataLoader(dataset=train_datagen,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=24,
                             pin_memory=True)

if args.feature2work == 6144:
    valid_datagen = PEDataset(feature_array=feature_valid,
                              image_to_feature=image_to_feature_valid,
                              series_dict=series_dict,
                              image_dict=image_dict,
                              series_list=series_list_valid,
                              seq_len=seq_len)
else: # 2048
    valid_datagen = PEDataset_2048F(feature_array=feature_valid,
                              image_to_feature=image_to_feature_valid,
                              series_dict=series_dict,
                              image_dict=image_dict,
                              series_list=series_list_valid,
                              seq_len=seq_len)

valid_generator = DataLoader(dataset=valid_datagen,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=24,
                             pin_memory=True)

print("Dataloader is ready...")


print("Model started training - validating...")
for ep in tqdm(range(1, num_epoch + 1)):

    # train
    losses_pe = AverageMeter()
    losses_pe_6144SLICE = AverageMeter()
    losses_npe = AverageMeter()
    losses_idt = AverageMeter()
    losses_lpe = AverageMeter()
    losses_rpe = AverageMeter()
    losses_cpe = AverageMeter()
    losses_gte = AverageMeter()
    losses_lt = AverageMeter()
    losses_chronic = AverageMeter()
    losses_acute_and_chronic = AverageMeter()
    avg_loss_count = []
    model.train()
    for j, (x, y_pe, mask, y_npe, y_idt, y_lpe, y_rpe, y_cpe, y_gte, y_lt, y_chronic, y_acute_and_chronic, series_list) in enumerate(train_generator):

        loss_weights_pe = np.zeros((x.size(0), x.size(1)), dtype=np.float32)
        # print(str(j) + "| Check: Training Series List: " + str(len(series_list)))
        for n in range(len(series_list)):
            image_list = series_dict[series_list[n]]['sorted_image_list']
            num_positive = 0
            for m in range(len(image_list)):
                num_positive += image_dict[image_list[m]]['pe_present_on_image']
            positive_ratio = num_positive / len(image_list)
            adjustment = 0
            if len(image_list)>seq_len:
                adjustment = len(image_list)/seq_len
            else:
                adjustment = 1.
            loss_weights_pe[n,:] = loss_weight_dict['pe_present_on_image']*positive_ratio*adjustment
        loss_weights_pe = torch.tensor(loss_weights_pe, dtype=torch.float32).cuda()

        x = x.cuda()
        y_pe = y_pe.type(torch.DoubleTensor).cuda()
        mask = mask.cuda()
        y_npe = y_npe.type(torch.DoubleTensor).cuda()
        y_idt = y_idt.type(torch.DoubleTensor).cuda()
        y_lpe = y_lpe.type(torch.DoubleTensor).cuda()
        y_rpe = y_rpe.type(torch.DoubleTensor).cuda()
        y_cpe = y_cpe.type(torch.DoubleTensor).cuda()
        y_gte = y_gte.type(torch.DoubleTensor).cuda()
        y_lt = y_lt.type(torch.DoubleTensor).cuda()
        y_chronic = y_chronic.type(torch.DoubleTensor).cuda()
        y_acute_and_chronic = y_acute_and_chronic.type(torch.DoubleTensor).cuda()

        logits_6144SLICE, logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic = model(x, mask)         
        # logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic = model(x, mask)
        

        # loss_pe = criterion1(logits_pe[0].squeeze(), logits_pe[1].squeeze(),y_pe) # was active 29th Sept 2022
        loss_pe = criterion(logits_pe.squeeze(),y_pe) # added 29th Sept 2022 - we don't need class token for slice-level
        loss_pe = loss_pe*mask*1 # excluded mask for experiment 21st Oct 2022
        # loss_pe = loss_pe*mask
        loss_pe = loss_pe.sum()/mask.sum() # excluded mask for experiment 21st Oct 2022    
        # loss_pe = 0    
        logits_6144SLICE_loss_pe = criterion(logits_6144SLICE.squeeze(), y_pe)
        logits_6144SLICE_loss_pe = logits_6144SLICE_loss_pe*mask*1
        logits_6144SLICE_loss_pe = logits_6144SLICE_loss_pe.sum()/mask.sum()

        loss_npe = criterion1(logits_npe[0].view(-1), logits_npe[1].view(-1), y_npe) * loss_weight_dict['negative_exam_for_pe']
        loss_idt = criterion1(logits_idt[0].view(-1),logits_idt[1].view(-1),y_idt) * loss_weight_dict['indeterminate']
        loss_lpe = criterion1(logits_lpe[0].view(-1),logits_lpe[1].view(-1),y_lpe)*loss_weight_dict['leftsided_pe']
        loss_rpe = criterion1(logits_rpe[0].view(-1),logits_rpe[1].view(-1),y_rpe)*loss_weight_dict['rightsided_pe']
        loss_cpe = criterion1(logits_cpe[0].view(-1),logits_cpe[1].view(-1),y_cpe)*loss_weight_dict['central_pe']
        loss_gte = criterion1(logits_gte[0].view(-1),logits_gte[1].view(-1),y_gte)*loss_weight_dict['rv_lv_ratio_gte_1']
        loss_lt = criterion1(logits_lt[0].view(-1),logits_lt[1].view(-1),y_lt)*loss_weight_dict['rv_lv_ratio_lt_1']
        loss_chronic = criterion1(logits_chronic[0].view(-1),logits_chronic[1].view(-1),y_chronic)*loss_weight_dict['chronic_pe']
        loss_acute_and_chronic = criterion1(logits_acute_and_chronic[0].view(-1),logits_acute_and_chronic[1].view(-1),y_acute_and_chronic)*loss_weight_dict['acute_and_chronic_pe']
        
        if args.lossAll == "yes":
            losses_pe.update(loss_pe.item(), mask.sum().item()) # was active # excluded mask for experiment 21st Oct 2022
            losses_pe_6144SLICE.update(logits_6144SLICE_loss_pe.item(), mask.sum().item())

        losses_npe.update(loss_npe.item(), x.size(0))
        losses_idt.update(loss_idt.item(), x.size(0))
        losses_lpe.update(loss_lpe.item(), x.size(0))
        losses_rpe.update(loss_rpe.item(), x.size(0))
        losses_cpe.update(loss_cpe.item(), x.size(0))
        losses_gte.update(loss_gte.item(), x.size(0))
        losses_lt.update(loss_lt.item(), x.size(0))
        losses_chronic.update(loss_chronic.item(), x.size(0))
        losses_acute_and_chronic.update(loss_acute_and_chronic.item(), x.size(0))
        loss = logits_6144SLICE_loss_pe + loss_pe + loss_npe + loss_idt + loss_lpe + loss_rpe + loss_cpe + loss_gte + loss_lt + loss_chronic + loss_acute_and_chronic
        if args.lossAll == "no":
            loss = loss - loss_pe # extraNahid - excluding loss_pe: maybe have a bad effect on the model for other AUCs
        avg_loss_count.append(loss.item())
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    scheduler.step(sum(avg_loss_count)/len(avg_loss_count)) # for scheduler reduceLRonrPla # loss should be the avg loss of an epoch


    # valid
    pred_prob_list = []
    gt_list = []
    loss_weight_list = []
    pred_prob_SliceLevel_list = []
    pred_prob_6144SliceLevel_list = []

    np_predicted_labels = []
    np_predicted_SliceLevel_labels = []
    np_predicted_6144SliceLevel_labels = []
    np_groundTruth_labels = []
    np_groundTruth_SliceLevel_labels = [] 

    losses_pe = AverageMeter()
    losses_pe_6144SLICE = AverageMeter()
    losses_npe = AverageMeter()
    losses_idt = AverageMeter()
    losses_lpe = AverageMeter()
    losses_rpe = AverageMeter()
    losses_cpe = AverageMeter()
    losses_gte = AverageMeter()
    losses_lt = AverageMeter()
    losses_chronic = AverageMeter()
    losses_acute_and_chronic = AverageMeter()
    model.eval()
    for j, (x, y_pe, mask, y_npe, y_idt, y_lpe, y_rpe, y_cpe, y_gte, y_lt, y_chronic, y_acute_and_chronic, series_list) in enumerate(valid_generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(valid_generator)-1:
                end = len(valid_generator.dataset)

            for n in range(len(series_list)):
                gt_list.append(series_dict[series_list[n]]['negative_exam_for_pe'])
                loss_weight_list.append(loss_weight_dict['negative_exam_for_pe'])
                gt_list.append(series_dict[series_list[n]]['indeterminate'])
                loss_weight_list.append(loss_weight_dict['indeterminate'])
                gt_list.append(series_dict[series_list[n]]['chronic_pe'])
                loss_weight_list.append(loss_weight_dict['chronic_pe'])
                gt_list.append(series_dict[series_list[n]]['acute_and_chronic_pe'])
                loss_weight_list.append(loss_weight_dict['acute_and_chronic_pe'])
                gt_list.append(series_dict[series_list[n]]['central_pe'])
                loss_weight_list.append(loss_weight_dict['central_pe'])
                gt_list.append(series_dict[series_list[n]]['leftsided_pe'])
                loss_weight_list.append(loss_weight_dict['leftsided_pe'])
                gt_list.append(series_dict[series_list[n]]['rightsided_pe'])
                loss_weight_list.append(loss_weight_dict['rightsided_pe'])
                gt_list.append(series_dict[series_list[n]]['rv_lv_ratio_gte_1'])
                loss_weight_list.append(loss_weight_dict['rv_lv_ratio_gte_1'])
                gt_list.append(series_dict[series_list[n]]['rv_lv_ratio_lt_1'])
                loss_weight_list.append(loss_weight_dict['rv_lv_ratio_lt_1'])
                image_list = series_dict[series_list[n]]['sorted_image_list']
                num_positive = 0
                for m in range(len(image_list)):
                    num_positive += image_dict[image_list[m]]['pe_present_on_image']
                positive_ratio = num_positive / len(image_list)
                for m in range(len(image_list)):
                    gt_list.append(image_dict[image_list[m]]['pe_present_on_image'])
                    loss_weight_list.append(loss_weight_dict['pe_present_on_image']*positive_ratio)

            loss_weights_pe = np.zeros((x.size(0), x.size(1)), dtype=np.float32)
            for n in range(len(series_list)):
                image_list = series_dict[series_list[n]]['sorted_image_list']
                num_positive = 0
                for m in range(len(image_list)):
                    num_positive += image_dict[image_list[m]]['pe_present_on_image']
                positive_ratio = num_positive / len(image_list)
                adjustment = 0
                if len(image_list)>seq_len:
                    adjustment = len(image_list)/seq_len
                else:
                    adjustment = 1.
                loss_weights_pe[n,:] = loss_weight_dict['pe_present_on_image']*positive_ratio*adjustment
            loss_weights_pe = torch.tensor(loss_weights_pe, dtype=torch.float32).cuda()

            x = x.cuda()
            y_pe = y_pe.type(torch.DoubleTensor).cuda()
            mask = mask.cuda()
            y_npe = y_npe.type(torch.DoubleTensor).cuda()
            y_idt = y_idt.type(torch.DoubleTensor).cuda()
            y_lpe = y_lpe.type(torch.DoubleTensor).cuda()
            y_rpe = y_rpe.type(torch.DoubleTensor).cuda()
            y_cpe = y_cpe.type(torch.DoubleTensor).cuda()
            y_gte = y_gte.type(torch.DoubleTensor).cuda()
            y_lt = y_lt.type(torch.DoubleTensor).cuda()
            y_chronic = y_chronic.type(torch.DoubleTensor).cuda()
            y_acute_and_chronic = y_acute_and_chronic.type(torch.DoubleTensor).cuda()

            logits_6144SLICE, logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic = model(x, mask)
            # logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic = model(x, mask)

            # loss_pe = criterion1(logits_pe[0].squeeze(), logits_pe[1].squeeze(),y_pe) # was active 29th Sept 2022
            loss_pe = criterion(logits_pe.squeeze(),y_pe) # added 29th Sept 2022 - we don't need class token for slice-level
            loss_pe = loss_pe*mask*1 # excluded mask for experiment 21st Oct 2022
            # loss_pe = loss_pe*mask
            loss_pe = loss_pe.sum()/mask.sum() # excluded mask for experiment 21st Oct 2022
            # loss_pe = 0
            logits_6144SLICE_loss_pe = criterion(logits_6144SLICE.squeeze(), y_pe)   
            logits_6144SLICE_loss_pe = logits_6144SLICE_loss_pe*mask*1
            logits_6144SLICE_loss_pe = logits_6144SLICE_loss_pe.sum()/mask.sum()

            loss_npe = criterion1(logits_npe[0].view(-1), logits_npe[1].view(-1), y_npe) * loss_weight_dict['negative_exam_for_pe']
            loss_idt = criterion1(logits_idt[0].view(-1),logits_idt[1].view(-1),y_idt) * loss_weight_dict['indeterminate']
            loss_lpe = criterion1(logits_lpe[0].view(-1),logits_lpe[1].view(-1),y_lpe)*loss_weight_dict['leftsided_pe']
            loss_rpe = criterion1(logits_rpe[0].view(-1),logits_rpe[1].view(-1),y_rpe)*loss_weight_dict['rightsided_pe']
            loss_cpe = criterion1(logits_cpe[0].view(-1),logits_cpe[1].view(-1),y_cpe)*loss_weight_dict['central_pe']
            loss_gte = criterion1(logits_gte[0].view(-1),logits_gte[1].view(-1),y_gte)*loss_weight_dict['rv_lv_ratio_gte_1']
            loss_lt = criterion1(logits_lt[0].view(-1),logits_lt[1].view(-1),y_lt)*loss_weight_dict['rv_lv_ratio_lt_1']
            loss_chronic = criterion1(logits_chronic[0].view(-1),logits_chronic[1].view(-1),y_chronic)*loss_weight_dict['chronic_pe']
            loss_acute_and_chronic = criterion1(logits_acute_and_chronic[0].view(-1),logits_acute_and_chronic[1].view(-1),y_acute_and_chronic)*loss_weight_dict['acute_and_chronic_pe'] 
            
            if args.lossAll == "yes":
                losses_pe.update(loss_pe.item(), mask.sum().item()) # was active # 29th Sept 2022 -> why we don't have this anymore ?  # excluded mask for experiment 21st Oct 2022            
                losses_pe_6144SLICE.update(logits_6144SLICE_loss_pe.item(), mask.sum().item())

            losses_npe.update(loss_npe.item(), x.size(0))
            losses_idt.update(loss_idt.item(), x.size(0))
            losses_lpe.update(loss_lpe.item(), x.size(0))
            losses_rpe.update(loss_rpe.item(), x.size(0))
            losses_cpe.update(loss_cpe.item(), x.size(0))
            losses_gte.update(loss_gte.item(), x.size(0))
            losses_lt.update(loss_lt.item(), x.size(0))
            losses_chronic.update(loss_chronic.item(), x.size(0))
            losses_acute_and_chronic.update(loss_acute_and_chronic.item(), x.size(0))

            # pred_prob_pe = np.squeeze(0.5*logits_pe[0].sigmoid().cpu().data.numpy()+0.5*logits_pe[1].sigmoid().cpu().data.numpy()) # 29th Sept 2022: we don't need cls token [0]
            pred_prob_pe = np.squeeze(logits_pe.sigmoid().cpu().data.numpy()) # 29th Sept 2022: we don't need cls token [0]
            pred_prob_pe_6144Slice = np.squeeze(logits_6144SLICE.sigmoid().cpu().data.numpy())

            pred_prob_npe = np.squeeze(0.5*logits_npe[0].sigmoid().cpu().data.numpy()+0.5*logits_npe[1].sigmoid().cpu().data.numpy())
            pred_prob_idt = np.squeeze(0.5*logits_idt[0].sigmoid().cpu().data.numpy()+0.5*logits_idt[1].sigmoid().cpu().data.numpy())
            pred_prob_lpe = np.squeeze(0.5*logits_lpe[0].sigmoid().cpu().data.numpy()+0.5*logits_lpe[1].sigmoid().cpu().data.numpy())
            pred_prob_rpe = np.squeeze(0.5*logits_rpe[0].sigmoid().cpu().data.numpy()+0.5*logits_rpe[1].sigmoid().cpu().data.numpy())
            pred_prob_cpe = np.squeeze(0.5*logits_cpe[0].sigmoid().cpu().data.numpy()+0.5*logits_cpe[1].sigmoid().cpu().data.numpy())
            pred_prob_chronic = np.squeeze(0.5*logits_chronic[0].sigmoid().cpu().data.numpy()+0.5*logits_chronic[1].sigmoid().cpu().data.numpy())
            pred_prob_acute_and_chronic = np.squeeze(0.5*logits_acute_and_chronic[0].sigmoid().cpu().data.numpy()+0.5*logits_acute_and_chronic[1].sigmoid().cpu().data.numpy())
            pred_prob_gte = np.squeeze(0.5*logits_gte[0].sigmoid().cpu().data.numpy()+0.5*logits_gte[1].sigmoid().cpu().data.numpy())
            pred_prob_lt = np.squeeze(0.5*logits_lt[0].sigmoid().cpu().data.numpy()+0.5*logits_lt[1].sigmoid().cpu().data.numpy())

            # Slice-level
            np_groundTruth_SliceLevel_labels.extend(y_pe.detach().cpu().numpy())
            np_predicted_SliceLevel_labels.append([logits_pe])
            pred_prob_SliceLevel_list.extend(pred_prob_pe)

            np_predicted_6144SliceLevel_labels.append([logits_6144SLICE])
            pred_prob_6144SliceLevel_list.extend(pred_prob_pe_6144Slice)
    
            # For Exam-level
            for n in range(len(series_list)):                
                np_groundTruth_labels.append([y_npe[n].item(),y_idt[n].item(),y_lpe[n].item(),y_rpe[n].item(),y_cpe[n].item(),y_gte[n].item(),y_lt[n].item(),y_chronic[n].item(),y_acute_and_chronic[n].item()])
                np_predicted_labels.append([(logits_npe[0][n].view(-1).item()+logits_npe[1][n].view(-1).item())/2, 
                                            (logits_idt[0][n].view(-1).item()+logits_idt[1][n].view(-1).item())/2,  
                                            (logits_lpe[0][n].view(-1).item()+logits_lpe[1][n].view(-1).item())/2, 
                                            (logits_rpe[0][n].view(-1).item()+logits_rpe[1][n].view(-1).item())/2, 
                                            (logits_cpe[0][n].view(-1).item()+logits_cpe[1][n].view(-1).item())/2, 
                                            (logits_gte[0][n].view(-1).item()+logits_gte[1][n].view(-1).item())/2, 
                                            (logits_lt[0][n].view(-1).item()+logits_lt[1][n].view(-1).item())/2, 
                                            (logits_chronic[0][n].view(-1).item()+logits_chronic[1][n].view(-1).item())/2, 
                                            (logits_acute_and_chronic[0][n].view(-1).item()+logits_acute_and_chronic[1][n].view(-1).item())/2]) 
                
                pred_prob_list.append(pred_prob_npe[n])
                pred_prob_list.append(pred_prob_idt[n])
                pred_prob_list.append(pred_prob_chronic[n])
                pred_prob_list.append(pred_prob_acute_and_chronic[n])
                pred_prob_list.append(pred_prob_cpe[n])
                pred_prob_list.append(pred_prob_lpe[n])
                pred_prob_list.append(pred_prob_rpe[n])
                pred_prob_list.append(pred_prob_gte[n])
                pred_prob_list.append(pred_prob_lt[n])
                num_image = len(series_dict[series_list[n]]['sorted_image_list'])
                # if num_image>seq_len:
                #     pred_prob_list += list(np.squeeze(cv2.resize(pred_prob_pe[n, :], (1, num_image), interpolation = cv2.INTER_LINEAR)))
                # else:
                #     pred_prob_list += list(pred_prob_pe[n, :num_image])
    

    # Slice-level
    np_predicted_SliceLevel_labels = np.array(pred_prob_SliceLevel_list) # 1000 512
    np_groundTruth_SliceLevel_labels = np.array(np_groundTruth_SliceLevel_labels) # 1000 512

    np.save(out_dir+'predicted_logits_SliceLevel_list'+titleName+'_512_'+str(ep), np.array(np_predicted_SliceLevel_labels)) # avg of logits
    np.save(out_dir+'gt_list_SliceLevel'+titleName+'_512_'+str(ep), np.array(np_groundTruth_SliceLevel_labels))

    np_predicted_SliceLevel_labels = np.reshape(np_predicted_SliceLevel_labels,(np_predicted_SliceLevel_labels.shape[0]*512,1))
    np_groundTruth_SliceLevel_labels = np.reshape(np_groundTruth_SliceLevel_labels,(np_groundTruth_SliceLevel_labels.shape[0]*512,1))
    np_predicted_SliceLevel_labels = np.squeeze(np_predicted_SliceLevel_labels)    
    np_groundTruth_SliceLevel_labels = np.squeeze(np_groundTruth_SliceLevel_labels)

    AUC_SliceLevel_Res = roc_auc_score(np_groundTruth_SliceLevel_labels, np_predicted_SliceLevel_labels)


    # Slice-level  6144
    np_predicted_6144SliceLevel_labels = np.array(pred_prob_6144SliceLevel_list) # 1000 512
    # np_groundTruth_SliceLevel_labels = np.array(np_groundTruth_SliceLevel_labels) # 1000 512

    # np.save(out_dir+'predicted_logits_SliceLevel_list'+titleName+'_512_'+str(ep), np.array(np_predicted_SliceLevel_labels)) # avg of logits
    # np.save(out_dir+'gt_list_SliceLevel'+titleName+'_512_'+str(ep), np.array(np_groundTruth_SliceLevel_labels))

    np_predicted_6144SliceLevel_labels = np.reshape(np_predicted_6144SliceLevel_labels,(np_predicted_6144SliceLevel_labels.shape[0]*512,1))
    # np_groundTruth_SliceLevel_labels = np.reshape(np_groundTruth_SliceLevel_labels,(np_groundTruth_SliceLevel_labels.shape[0]*512,1))
    np_predicted_6144SliceLevel_labels = np.squeeze(np_predicted_6144SliceLevel_labels)    
    # np_groundTruth_SliceLevel_labels = np.squeeze(np_groundTruth_SliceLevel_labels)

    AUC_6144SliceLevel_Res = roc_auc_score(np_groundTruth_SliceLevel_labels, np_predicted_6144SliceLevel_labels)
    


    # Exam-level
    np_predicted_labels = np.array(np_predicted_labels)
    np_groundTruth_labels = np.array(np_groundTruth_labels)
    np_predicted_labels = np.reshape(np_predicted_labels,(np_predicted_labels.shape[0],9))
    np_groundTruth_labels = np.reshape(np_groundTruth_labels,(np_groundTruth_labels.shape[0],9))

    AUC_Res = computeAUROC(np_groundTruth_labels, np_predicted_labels, 9)

    

    # print("[INFO -] Slice-level AUC:", AUC_SliceLevel_Res, "| Exam-level AUC:", (AUC_Res[0]+AUC_Res[1]+AUC_Res[2]+AUC_Res[3]+AUC_Res[4]+AUC_Res[5]+AUC_Res[6]+AUC_Res[7]+AUC_Res[8])/9)
    # print("[INFO] Exam-level AUC:", (AUC_Res[0]+AUC_Res[1]+AUC_Res[2]+AUC_Res[3]+AUC_Res[4]+AUC_Res[5]+AUC_Res[6]+AUC_Res[7]+AUC_Res[8])/9)
    print("[INFO] 6144Slice-level AUC:", AUC_6144SliceLevel_Res, "| Slice-level AUC:", AUC_SliceLevel_Res, "| Exam-level AUC:", (AUC_Res[0]+AUC_Res[1]+AUC_Res[2]+AUC_Res[3]+AUC_Res[4]+AUC_Res[5]+AUC_Res[6]+AUC_Res[7]+AUC_Res[8])/9)


    np.save(out_dir+'predicted_logits_list'+titleName+'_512_'+str(ep), np.array(np_predicted_labels)) # avg of logits
    np.save(out_dir+'predicted_prob_list'+titleName+'_512_'+str(ep), np.array(pred_prob_list)) # avg of separate sigmoid(logits)
    np.save(out_dir+'gt_list_'+titleName+'_512_'+str(ep), np.array(np_groundTruth_labels))


    # Store predicted PE labels
    fOPEN = open(out_dir+'runs_result_eachEpoch.txt', 'a+')
    fOPEN.write("Run: " + runV + ": " + titleName + "\n")
    fOPEN.write("Epoch: " + str(ep) + "\n")
    fOPEN.write("feature_size: " + str(feature_size) + "\n")
    fOPEN.write("mlp_dim: " + str(MD) + "\n")
    fOPEN.write("num_heads: " + str(NH) + "\n")
    fOPEN.write("num_blocks: " + str(NL) + "\n")
    fOPEN.write("transformer_output: " + str(lstm_size) + "\n")
    fOPEN.write("batch_size: " + str(batch_size) + "\n")
    fOPEN.write("num_epoch: " + str(num_epoch) + "\n")
    # fOPEN.write("LearningRate: " + str(scheduler.get_last_lr()[0]) + "\n") # for regular scheduler    
    fOPEN.write("LearningRate: " + str(optimizer.param_groups[0]['lr']) + "\n") # for reducelronplateau
    fOPEN.write("Training-Loss: " + str(sum(avg_loss_count)/len(avg_loss_count)) + "\n")    
    fOPEN.write("-----" + "\n")

    fOPEN.write("Negative_Exam_for_PE: " + str(AUC_Res[0]) + "\n")
    fOPEN.write("Indeterminate: " + str(AUC_Res[1]) + "\n")
    fOPEN.write("Left_PE: " + str(AUC_Res[2]) + "\n")
    fOPEN.write("Right_PE: " + str(AUC_Res[3]) + "\n")
    fOPEN.write("Central_PE: " + str(AUC_Res[4]) + "\n")
    fOPEN.write("RV_LV_ratio_gte_1: " + str(AUC_Res[5]) + "\n")
    fOPEN.write("RV_LV_ratio_lt_1: " + str(AUC_Res[6]) + "\n")
    fOPEN.write("Chronic_PE: " + str(AUC_Res[7]) + "\n")
    fOPEN.write("Acute_and_Chronic_PE: " + str(AUC_Res[8]) + "\n")
    temp_mean = (AUC_Res[0]+AUC_Res[1]+AUC_Res[2]+AUC_Res[3]+AUC_Res[4]+AUC_Res[5]+AUC_Res[6]+AUC_Res[7]+AUC_Res[8])/9
    fOPEN.write("MEAN: " + str(temp_mean) + "\n")
    fOPEN.write("Slice_level_AUC: " + str(AUC_SliceLevel_Res) + "\n")
    fOPEN.write("-----" + "\n")
    fOPEN.write(" " + "\n")
    fOPEN.close()

    fields=[ep, sum(avg_loss_count)/len(avg_loss_count), AUC_Res[0], AUC_Res[1], AUC_Res[2], AUC_Res[3], AUC_Res[4], AUC_Res[5], AUC_Res[6], AUC_Res[7], AUC_Res[8], temp_mean, AUC_SliceLevel_Res]
    # fields=[ep, 0, AUC_Res[0], AUC_Res[1], AUC_Res[2], AUC_Res[3], AUC_Res[4], AUC_Res[5], AUC_Res[6], AUC_Res[7], AUC_Res[8], temp_mean, AUC_SliceLevel_Res]
    with open(out_dir+'export_csvFile.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    pred_prob_list = torch.tensor(pred_prob_list, dtype=torch.float32)
    gt_list = torch.tensor(gt_list, dtype=torch.float32)
    loss_weight_list = torch.tensor(loss_weight_list, dtype=torch.float32)
    # print(len(pred_prob_list))
    # kaggle_loss = torch.nn.BCELoss(reduction='none')(pred_prob_list, gt_list)
    # kaggle_loss = (kaggle_loss*loss_weight_list).sum() / loss_weight_list.sum()

    # print()
    # print('epoch: {}, valid_loss_pe: {}, valid_loss_npe: {}, valid_loss_idt: {}, valid_loss_lpe: {}, valid_loss_rpe: {}, valid_loss_cpe: {}, valid_loss_gte: {}, valid_loss_lt: {}, valid_loss_chronic: {}, valid_loss_acute_and_chronic: {}'.format(ep, losses_pe.avg, losses_npe.avg, losses_idt.avg, losses_lpe.avg, losses_rpe.avg, losses_cpe.avg, losses_gte.avg, losses_lt.avg, losses_chronic.avg, losses_acute_and_chronic.avg), flush=True)
    # print("Kaggle_Loss: " + str(kaggle_loss))
    # printAUC_results(AUC_Res)
    # print()

print("Model training-validating done...")

# print("Kaggle_Loss: " + str(kaggle_loss))
printAUC_results(AUC_Res)
print("Slice_Level_AUC:", str(AUC_SliceLevel_Res))

# np.save(out_dir+'pred_prob_list'+titleName+'_512', np.array(np_predicted_labels))
# np.save(out_dir+'gt_list_'+titleName+'_512', np.array(np_groundTruth_labels))
# np.save(out_dir+'loss_weight_list_'+titleName+'_512', np.array(loss_weight_list))

# Store predicted PE labels
fOPEN = open(out_dir2+'runs_result.txt', 'a+')
fOPEN.write("Run: " + runV + ": " + titleName + "\n")
fOPEN.write("-----" + "\n")
# fOPEN.write("Kaggle_Loss: " + str(kaggle_loss) + "\n")
fOPEN.write("Negative_Exam_for_PE: " + str(AUC_Res[0]) + "\n")
fOPEN.write("Indeterminate: " + str(AUC_Res[1]) + "\n")
fOPEN.write("Left_PE: " + str(AUC_Res[2]) + "\n")
fOPEN.write("Right_PE: " + str(AUC_Res[3]) + "\n")
fOPEN.write("Central_PE: " + str(AUC_Res[4]) + "\n")
fOPEN.write("RV_LV_ratio_gte_1: " + str(AUC_Res[5]) + "\n")
fOPEN.write("RV_LV_ratio_lt_1: " + str(AUC_Res[6]) + "\n")
fOPEN.write("Chronic_PE: " + str(AUC_Res[7]) + "\n")
fOPEN.write("Acute_and_Chronic_PE: " + str(AUC_Res[8]) + "\n")
fOPEN.write("Slice-level AUC: " + str(AUC_SliceLevel_Res) + "\n")
fOPEN.write("-----" + "\n")
fOPEN.write(" " + "\n")
fOPEN.close()


print("Output stored...")

# out_dir = 'weights/'
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
# torch.save(model.state_dict(), out_dir+titleName+'_512')

# print("Weights stored...")
