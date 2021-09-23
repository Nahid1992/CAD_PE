import argparse
import numpy as np
import pandas as pd
import pickle
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from random import randrange
from sklearn.metrics import roc_auc_score, log_loss
numSeed = randrange(2500)

# https://www.kaggle.com/bminixhofer/a-validation-framework-impact-of-the-random-seed

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
        if len(image_list)>self.seq_len: # M > N(192) 
            x = np.zeros((len(image_list), self.feature_array.shape[1]*3), dtype=np.float32) # feature_array.shape[1] = 2048
            y_pe = np.zeros((len(image_list), 1), dtype=np.float32)
            mask = np.ones((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]] 
                y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
            x = cv2.resize(x, (self.feature_array.shape[1]*3, self.seq_len), interpolation = cv2.INTER_LINEAR)
            y_pe = np.squeeze(cv2.resize(y_pe, (1, self.seq_len), interpolation = cv2.INTER_LINEAR))
        else: # M < N(192) => Zero-padding
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
        return x, y_pe, mask, y_npe, y_idt, y_lpe, y_rpe, y_cpe, y_gte, y_lt, y_chronic, y_acute_and_chronic, self.series_list[index]



parser = argparse.ArgumentParser()
parser.add_argument("--backboneName", type=str, default="resnet18", help="resnet18 | resnet50 | densenet121 | xception")
parser.add_argument("--runV", type=str, default="version_1", help="model load during val or not")
parser.add_argument("--featureMode", type=int, default=1, help="fine tuned or non-fine tuned")
parser.add_argument("--ssl_method_name", type=str, default=" ", help="SSL method name")
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
    feature_train = np.load('../seresnet50/TransferLearning_resnet18_ImageNet_576_v206_/feature_train.npy')
    feature_valid = np.load('../seresnet50/TransferLearning_resnet18_ImageNet_576_v206_/feature_valid.npy')
    titleName = "resnet18_512"
    featureSize = 512
elif backboneName == "resnet50":
    if featureMode == 1:
        # Pre-trained from ImageNet
        if ssl_method_name == " ":
            feature_train = np.load('../seresnet50/TransferLearning_resnet50_ImageNet_576_v102_/feature_train.npy')
            feature_valid = np.load('../seresnet50/TransferLearning_resnet50_ImageNet_576_v102_/feature_valid.npy')
            titleName = "resnet50_2048_FT"

        # Pre-trained from SSL method
        if ssl_method_name == "sela-v2":
            feature_train = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_sela-v2_576_v110_/feature_train.npy')
            feature_valid = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_sela-v2_576_v110_/feature_valid.npy')
            titleName = "resnet50_SSL_selav2_2048_FT"
        elif ssl_method_name == "deepcluster-v2":
            feature_train = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_deepcluster-v2_576_v110_/feature_train.npy')
            feature_valid = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_deepcluster-v2_576_v110_/feature_valid.npy')  
            titleName = "resnet50_SSL_deepclusterv2_2048_FT"
        elif ssl_method_name == "barlowtwins":
            feature_train = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_barlowtwins_576_v106_/feature_train.npy')
            feature_valid = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_resnet50_barlowtwins_576_v106_/feature_valid.npy')             
            titleName = "resnet50_SSL_barlowtwins_2048_FT"
    else:
        # Pre-trained from SSL method
        if ssl_method_name == "sela-v2":
            feature_train = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_sela-v2_SSL_FT_v101_/feature_train.npy')
            feature_valid = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_sela-v2_SSL_FT_v101_/feature_valid.npy')
            titleName = "resnet50_SSL_selav2_2048_nonFT"
        elif ssl_method_name == "deepcluster-v2":
            feature_train = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_deepcluster-v2_SSL_FT_v101_/feature_train.npy')
            feature_valid = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_deepcluster-v2_SSL_FT_v101_/feature_valid.npy')
            titleName = "resnet50_SSL_deepclusterv2_2048_nonFT"  
        elif ssl_method_name == "barlowtwins":                    
            feature_train = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_barlowtwins_SSL_FT_v101_/feature_train.npy')
            feature_valid = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_resnet50_576_barlowtwins_SSL_FT_v101_/feature_valid.npy') 
            titleName = "resnet50_SSL_barlowtwins_2048_nonFT"
    featureSize = 2048
elif backboneName == "xception":
    if featureMode == 1:
        # feature_train = np.load('../seresnet50/TransferLearning_xception_ImageNet_576_v304_/feature_train.npy')
        # feature_valid = np.load('../seresnet50/TransferLearning_xception_ImageNet_576_v304_/feature_valid.npy')
        feature_train = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_xception_576_ImageNet_v1005_/feature_train.npy')
        feature_valid = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_xception_576_ImageNet_v1005_/feature_valid.npy')
        titleName = "xception_2048_ME"
    else:
        feature_train = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_xception_576_ImageNet_vnonFT_101_/feature_train.npy')
        feature_valid = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_xception_576_ImageNet_vnonFT_101_/feature_valid.npy') 
        titleName = "xception_2048_nonFT"
    featureSize = 2048
elif backboneName == "densenet121":
    feature_train = np.load('../seresnet50/TransferLearning_densenet121_ImageNet_576_v406_/feature_train.npy')
    feature_valid = np.load('../seresnet50/TransferLearning_densenet121_ImageNet_576_v406_/feature_valid.npy')
    titleName = "densenet121_1024"
    featureSize = 1024
elif backboneName == "seresnext50":
    if featureMode == 1:
        # feature_train = np.load('../seresnet50/TransferLearning_seresnext50_ImageNet_576_v004_/feature_train.npy')
        # feature_valid = np.load('../seresnet50/TransferLearning_seresnext50_ImageNet_576_v004_/feature_valid.npy')
        feature_train = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_seresnext50_576_ImageNet_v1004_/feature_train.npy')
        feature_valid = np.load('../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_seresnext50_576_ImageNet_v1004_/feature_valid.npy')
        titleName = "seresnext50_2048_ME"
    else:
        feature_train = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_seresnext50_576_ImageNet_vnonFT_101_/feature_train.npy')
        feature_valid = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_seresnext50_576_ImageNet_vnonFT_101_/feature_valid.npy')    
        titleName = "seresnext50_2048_nonFT"    
    featureSize = 2048
elif backboneName == "sexception":
    if featureMode == 1:
        # feature_train = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_manual_576_ImageNet_v1005_/feature_train.npy")
        # feature_valid = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_manual_576_ImageNet_v1005_/feature_valid.npy")

        # feature_train = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_v1007_/feature_train.npy")
        # feature_valid = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_v1007_/feature_valid.npy")

        feature_train = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_v1403_/feature_train.npy")
        feature_valid = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_v1403_/feature_valid.npy")
        titleName = "sexception_2048_ME"
    else:
        feature_train = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_vnonFT_101_/feature_train.npy')
        feature_valid = np.load('../seresnet50/nonFineTuned_100percent_TrainData/FineTune_Reduced_100_sexception_576_ImageNet_vnonFT_101_/feature_valid.npy')  
        titleName = "sexception_2048_nonFT"
    featureSize = 2048
elif backboneName == "seresnext101":
    if featureMode == 1:
        feature_train = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_seresnext101_ImageNet_576_v1805_/feature_train.npy")
        feature_valid = np.load("../seresnet50/BestModels_100percent_TrainData/FineTune_TrainData_100_seresnext101_ImageNet_576_v1805_/feature_valid.npy")
        titleName = "seresnext101_2048_FT"
    featureSize = 2048



print(backboneName, ssl_method_name)
out_dir = 'predictions_'+titleName+'/' + runV + '/'
out_dir2 = 'predictions_'+titleName+'/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


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
seq_len = 192
feature_size = featureSize*3 # was 2048*3
lstm_size = 512
learning_rate = 0.0005
batch_size = 64 # was 64
num_epoch = 25

print("Learning Rate: " + str(learning_rate))
print("Batch Size: " + str(batch_size))
print("LSTM Size: " + str(lstm_size))
print("Sequence Length: " + str(seq_len))
print("Feature Size: " + str(feature_size))
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


model = PENet(input_len=feature_size, lstm_size=lstm_size)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()
criterion1 = nn.BCEWithLogitsLoss().cuda()

print("Model is ready...")

# training

# iterator for training
train_datagen = PEDataset(feature_array=feature_train,
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

valid_datagen = PEDataset(feature_array=feature_valid,
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
for ep in tqdm(range(num_epoch)):

    # train
    losses_pe = AverageMeter()
    losses_npe = AverageMeter()
    losses_idt = AverageMeter()
    losses_lpe = AverageMeter()
    losses_rpe = AverageMeter()
    losses_cpe = AverageMeter()
    losses_gte = AverageMeter()
    losses_lt = AverageMeter()
    losses_chronic = AverageMeter()
    losses_acute_and_chronic = AverageMeter()
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
        y_pe = y_pe.float().cuda()
        mask = mask.cuda()
        y_npe = y_npe.float().cuda()
        y_idt = y_idt.float().cuda()
        y_lpe = y_lpe.float().cuda()
        y_rpe = y_rpe.float().cuda()
        y_cpe = y_cpe.float().cuda()
        y_gte = y_gte.float().cuda()
        y_lt = y_lt.float().cuda()
        y_chronic = y_chronic.float().cuda()
        y_acute_and_chronic = y_acute_and_chronic.float().cuda()
        logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic = model(x, mask)
        loss_pe = criterion(logits_pe.squeeze(),y_pe)
        loss_pe = loss_pe*mask*loss_weights_pe
        loss_pe = loss_pe.sum()/mask.sum()
        loss_npe = criterion1(logits_npe.view(-1),y_npe)*loss_weight_dict['negative_exam_for_pe']
        loss_idt = criterion1(logits_idt.view(-1),y_idt)*loss_weight_dict['indeterminate']
        loss_lpe = criterion1(logits_lpe.view(-1),y_lpe)*loss_weight_dict['leftsided_pe']
        loss_rpe = criterion1(logits_rpe.view(-1),y_rpe)*loss_weight_dict['rightsided_pe']
        loss_cpe = criterion1(logits_cpe.view(-1),y_cpe)*loss_weight_dict['central_pe']
        loss_gte = criterion1(logits_gte.view(-1),y_gte)*loss_weight_dict['rv_lv_ratio_gte_1']
        loss_lt = criterion1(logits_lt.view(-1),y_lt)*loss_weight_dict['rv_lv_ratio_lt_1']
        loss_chronic = criterion1(logits_chronic.view(-1),y_chronic)*loss_weight_dict['chronic_pe']
        loss_acute_and_chronic = criterion1(logits_acute_and_chronic.view(-1),y_acute_and_chronic)*loss_weight_dict['acute_and_chronic_pe']
        losses_pe.update(loss_pe.item(), mask.sum().item())
        losses_npe.update(loss_npe.item(), x.size(0))
        losses_idt.update(loss_idt.item(), x.size(0))
        losses_lpe.update(loss_lpe.item(), x.size(0))
        losses_rpe.update(loss_rpe.item(), x.size(0))
        losses_cpe.update(loss_cpe.item(), x.size(0))
        losses_gte.update(loss_gte.item(), x.size(0))
        losses_lt.update(loss_lt.item(), x.size(0))
        losses_chronic.update(loss_chronic.item(), x.size(0))
        losses_acute_and_chronic.update(loss_acute_and_chronic.item(), x.size(0))
        loss = loss_pe + loss_npe + loss_idt + loss_lpe + loss_rpe + loss_cpe + loss_gte + loss_lt + loss_chronic + loss_acute_and_chronic

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    scheduler.step()

    # print()
    # print('epoch: {}, train_loss_pe: {}, train_loss_npe: {}, train_loss_idt: {}, train_loss_lpe: {}, train_loss_rpe: {}, train_loss_cpe: {}, train_loss_gte: {}, train_loss_lt: {}, train_loss_chronic: {}, train_loss_acute_and_chronic: {}'.format(ep, losses_pe.avg, losses_npe.avg, losses_idt.avg, losses_lpe.avg, losses_rpe.avg, losses_cpe.avg, losses_gte.avg, losses_lt.avg, losses_chronic.avg, losses_acute_and_chronic.avg), flush=True)

    # del x, y_pe, mask, y_npe, y_idt, y_lpe, y_rpe, y_cpe, y_gte, y_lt, y_chronic, y_acute_and_chronic, series_list

    # valid
    pred_prob_list = []
    gt_list = []
    loss_weight_list = []

    np_predicted_labels = []
    np_groundTruth_labels = []

    losses_pe = AverageMeter()
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
            y_pe = y_pe.float().cuda()
            mask = mask.cuda()
            y_npe = y_npe.float().cuda()
            y_idt = y_idt.float().cuda()
            y_lpe = y_lpe.float().cuda()
            y_rpe = y_rpe.float().cuda()
            y_cpe = y_cpe.float().cuda()
            y_gte = y_gte.float().cuda()
            y_lt = y_lt.float().cuda()
            y_chronic = y_chronic.float().cuda()
            y_acute_and_chronic = y_acute_and_chronic.float().cuda()

            logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic = model(x, mask)
            
            loss_pe = criterion(logits_pe.squeeze(),y_pe)
            loss_pe = loss_pe*mask*loss_weights_pe
            loss_pe = loss_pe.sum()/mask.sum()
            loss_npe = criterion1(logits_npe.view(-1),y_npe)*loss_weight_dict['negative_exam_for_pe']
            loss_idt = criterion1(logits_idt.view(-1),y_idt)*loss_weight_dict['indeterminate']
            loss_lpe = criterion1(logits_lpe.view(-1),y_lpe)*loss_weight_dict['leftsided_pe']
            loss_rpe = criterion1(logits_rpe.view(-1),y_rpe)*loss_weight_dict['rightsided_pe']
            loss_cpe = criterion1(logits_cpe.view(-1),y_cpe)*loss_weight_dict['central_pe']
            loss_gte = criterion1(logits_gte.view(-1),y_gte)*loss_weight_dict['rv_lv_ratio_gte_1']
            loss_lt = criterion1(logits_lt.view(-1),y_lt)*loss_weight_dict['rv_lv_ratio_lt_1']
            loss_chronic = criterion1(logits_chronic.view(-1),y_chronic)*loss_weight_dict['chronic_pe']
            loss_acute_and_chronic = criterion1(logits_acute_and_chronic.view(-1),y_acute_and_chronic)*loss_weight_dict['acute_and_chronic_pe']
            losses_pe.update(loss_pe.item(), mask.sum().item())
            losses_npe.update(loss_npe.item(), x.size(0))
            losses_idt.update(loss_idt.item(), x.size(0))
            losses_lpe.update(loss_lpe.item(), x.size(0))
            losses_rpe.update(loss_rpe.item(), x.size(0))
            losses_cpe.update(loss_cpe.item(), x.size(0))
            losses_gte.update(loss_gte.item(), x.size(0))
            losses_lt.update(loss_lt.item(), x.size(0))
            losses_chronic.update(loss_chronic.item(), x.size(0))
            losses_acute_and_chronic.update(loss_acute_and_chronic.item(), x.size(0))

            pred_prob_pe = np.squeeze(logits_pe.sigmoid().cpu().data.numpy())
            pred_prob_npe = np.squeeze(logits_npe.sigmoid().cpu().data.numpy())
            pred_prob_idt = np.squeeze(logits_idt.sigmoid().cpu().data.numpy())
            pred_prob_lpe = np.squeeze(logits_lpe.sigmoid().cpu().data.numpy())
            pred_prob_rpe = np.squeeze(logits_rpe.sigmoid().cpu().data.numpy())
            pred_prob_cpe = np.squeeze(logits_cpe.sigmoid().cpu().data.numpy())
            pred_prob_chronic = np.squeeze(logits_chronic.sigmoid().cpu().data.numpy())
            pred_prob_acute_and_chronic = np.squeeze(logits_acute_and_chronic.sigmoid().cpu().data.numpy())
            pred_prob_gte = np.squeeze(logits_gte.sigmoid().cpu().data.numpy())
            pred_prob_lt = np.squeeze(logits_lt.sigmoid().cpu().data.numpy())
            for n in range(len(series_list)):
                #np_groundTruth_labels.append(y_cpe[n].item()) # added by Nahid
                #np_predicted_labels.append(logits_cpe[n].view(-1).item()) # added by Nahid

                np_groundTruth_labels.append([y_npe[n].item(),y_idt[n].item(),y_lpe[n].item(),y_rpe[n].item(),y_cpe[n].item(),y_gte[n].item(),y_lt[n].item(),y_chronic[n].item(),y_acute_and_chronic[n].item()])
                np_predicted_labels.append([logits_npe[n].view(-1).item(), logits_idt[n].view(-1).item(), logits_lpe[n].view(-1).item(), logits_rpe[n].view(-1).item(), logits_cpe[n].view(-1).item(), logits_gte[n].view(-1).item(), logits_lt[n].view(-1).item(), logits_chronic[n].view(-1).item(), logits_acute_and_chronic[n].view(-1).item()])

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
                if num_image>seq_len:
                    pred_prob_list += list(np.squeeze(cv2.resize(pred_prob_pe[n, :], (1, num_image), interpolation = cv2.INTER_LINEAR)))
                else:
                    pred_prob_list += list(pred_prob_pe[n, :num_image])
            
    np_predicted_labels = np.array(np_predicted_labels)
    np_groundTruth_labels = np.array(np_groundTruth_labels)
    np_predicted_labels = np.reshape(np_predicted_labels,(np_predicted_labels.shape[0],9))
    np_groundTruth_labels = np.reshape(np_groundTruth_labels,(np_groundTruth_labels.shape[0],9))
    # print("Check: npVersion: ", np_predicted_labels.shape, np_groundTruth_labels.shape)
    AUC_Res = computeAUROC(np_groundTruth_labels, np_predicted_labels, 9)

    pred_prob_list = torch.tensor(pred_prob_list, dtype=torch.float32)
    gt_list = torch.tensor(gt_list, dtype=torch.float32)
    loss_weight_list = torch.tensor(loss_weight_list, dtype=torch.float32)
    # print(len(pred_prob_list))
    kaggle_loss = torch.nn.BCELoss(reduction='none')(pred_prob_list, gt_list)
    kaggle_loss = (kaggle_loss*loss_weight_list).sum() / loss_weight_list.sum()

    # print()
    # print('epoch: {}, valid_loss_pe: {}, valid_loss_npe: {}, valid_loss_idt: {}, valid_loss_lpe: {}, valid_loss_rpe: {}, valid_loss_cpe: {}, valid_loss_gte: {}, valid_loss_lt: {}, valid_loss_chronic: {}, valid_loss_acute_and_chronic: {}'.format(ep, losses_pe.avg, losses_npe.avg, losses_idt.avg, losses_lpe.avg, losses_rpe.avg, losses_cpe.avg, losses_gte.avg, losses_lt.avg, losses_chronic.avg, losses_acute_and_chronic.avg), flush=True)
    # print("Kaggle_Loss: " + str(kaggle_loss))
    # printAUC_results(AUC_Res)
    # print()

print("Model training-validating done...")

print("Kaggle_Loss: " + str(kaggle_loss))
printAUC_results(AUC_Res)

np.save(out_dir+'pred_prob_list'+titleName+'_192', np.array(np_predicted_labels))
np.save(out_dir+'gt_list_'+titleName+'_192', np.array(np_groundTruth_labels))
np.save(out_dir+'loss_weight_list_'+titleName+'_192', np.array(loss_weight_list))

# Store predicted PE labels
fOPEN = open(out_dir2+'runs_result.txt', 'a+')
fOPEN.write("Run: " + runV + ": " + titleName + "\n")
fOPEN.write("-----" + "\n")
fOPEN.write("Kaggle_Loss: " + str(kaggle_loss) + "\n")
fOPEN.write("Negative_Exam_for_PE: " + str(AUC_Res[0]) + "\n")
fOPEN.write("Indeterminate: " + str(AUC_Res[1]) + "\n")
fOPEN.write("Left_PE: " + str(AUC_Res[2]) + "\n")
fOPEN.write("Right_PE: " + str(AUC_Res[3]) + "\n")
fOPEN.write("Central_PE: " + str(AUC_Res[4]) + "\n")
fOPEN.write("RV_LV_ratio_gte_1: " + str(AUC_Res[5]) + "\n")
fOPEN.write("RV_LV_ratio_lt_1: " + str(AUC_Res[6]) + "\n")
fOPEN.write("Chronic_PE: " + str(AUC_Res[7]) + "\n")
fOPEN.write("Acute_and_Chronic_PE: " + str(AUC_Res[8]) + "\n")
fOPEN.write("-----" + "\n")
fOPEN.write(" " + "\n")
fOPEN.close()


print("Output stored...")

out_dir = 'weights/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
torch.save(model.state_dict(), out_dir+titleName+'_192')

print("Weights stored...")
