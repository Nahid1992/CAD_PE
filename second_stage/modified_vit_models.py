import torch
from torch import nn
import numpy as np

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
        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class PENet_ViT_BASE_withAtten_withFCs_CEFE_avg(nn.Module): ## THIS IS WORKING !!!
    def __init__(self, staTe, input_len, MD, NH, NL, lstm_size, seqL, embedding=False, LoadViTPreTrainedW='no'):  # 6144, 512
        super().__init__()
        from models_vit.modeling_exp_CEFE_avg import VisionTransformer as VisionTransformerCEFEavg
        from models_vit.modeling_exp_CEFE_avg import CONFIGS as CONFIGS_model_name

        self.linear_feature1 = nn.Linear(6144, 3072)
        self.linear_feature2 = nn.Linear(3072, 1536)
        self.linear_feature3 = nn.Linear(1536, 768)
        self.lstm1 = VisionTransformerCEFEavg(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, embedding=embedding, num_classes=768) # num_classes or outputFeature = 1 to 512
        print()
        model = self.lstm1
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Inside Total Parameters:",pytorch_total_params)
        print("Inside Total Parameters:",pytorch_total_trainable_params)
        print("--------------------------------------------------")
        print()        
        self.staTus = "clsToken_rest"
        if LoadViTPreTrainedW == "yes":
            self.lstm1.load_from(np.load("/ocean/projects/bcs190005p/nahid92/Projects/PE_Detection/vit_codes/checkpoint/imagenet21k_ViT-B_16.npz"))
            print("Loaded ViT Base 16 pretrained model weights")

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

        h_lstm1_clsToken, h_lstm1_restEverything = self.lstm1(logits_features) # was logits_features
        check = torch.any(torch.isnan(h_lstm1_restEverything)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()

         # From class token to each exam label    
        logits_pe_clsToken = h_lstm1_clsToken[0]
        logits_npe_clsToken = h_lstm1_clsToken[1]
        logits_idt_clsToken = h_lstm1_clsToken[2]
        logits_lpe_clsToken = h_lstm1_clsToken[3]
        logits_rpe_clsToken = h_lstm1_clsToken[4]
        logits_cpe_clsToken = h_lstm1_clsToken[5]
        logits_gte_clsToken = h_lstm1_clsToken[6]
        logits_lt_clsToken = h_lstm1_clsToken[7]
        logits_chronic_clsToken = h_lstm1_clsToken[8]
        logits_acute_and_chronic_clsToken = h_lstm1_clsToken[9]

        att_pool = self.attention(h_lstm1_restEverything, mask)
        max_pool, _ = torch.max(h_lstm1_restEverything, 1)
        conc = torch.cat((max_pool, att_pool), 1)
        # print("[INFO] conc", conc.shape)
        logits_pe = self.last_linear_pe(conc)

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
        return (logits_pe_clsToken,logits_pe), (logits_npe_clsToken, logits_npe_rest), (logits_idt_clsToken, logits_idt_rest), (logits_lpe_clsToken, logits_lpe_rest), \
                (logits_rpe_clsToken, logits_rpe_rest), (logits_cpe_clsToken, logits_cpe_rest), (logits_gte_clsToken, logits_gte_rest), \
                (logits_lt_clsToken, logits_lt_rest), (logits_chronic_clsToken, logits_chronic_rest), \
                (logits_acute_and_chronic_clsToken, logits_acute_and_chronic_rest)


class PENet_ViT_BASE_withAtten_withFCs_CEFE_avgV2(nn.Module): ## THIS IS WORKING !!! -- modified
    def __init__(self, staTe, input_len, MD, NH, NL, lstm_size, seqL, embedding=False, LoadViTPreTrainedW='no'):  # 6144, 512
        super().__init__()
        from models_vit.modeling_exp_CEFE_avg import VisionTransformer as VisionTransformerCEFEavg
        from models_vit.modeling_exp_CEFE_avg import CONFIGS as CONFIGS_model_name

        self.linear_feature1 = nn.Linear(6144, 3072)
        self.linear_feature2 = nn.Linear(3072, 1536)
        self.linear_feature3 = nn.Linear(1536, 768)
        self.lstm1 = VisionTransformerCEFEavg(CONFIGS_model_name["ViT-B_16_NI"], img_size=224, HS=input_len, MD=MD, NH=NH, NL=NL, zero_head=True, embedding=embedding, num_classes=768) # num_classes or outputFeature = 1 to 512
        print()
        model = self.lstm1
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Inside Total Parameters:",pytorch_total_params)
        print("Inside Total Parameters:",pytorch_total_trainable_params)
        print("--------------------------------------------------")
        print()        
        self.staTus = "clsToken_rest"
        if LoadViTPreTrainedW == "yes":
            self.lstm1.load_from(np.load("/ocean/projects/bcs190005p/nahid92/Projects/PE_Detection/vit_codes/checkpoint/imagenet21k_ViT-B_16.npz"))
            print("Loaded ViT Base 16 pretrained model weights")

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

        h_lstm1_clsToken, h_lstm1_restEverything = self.lstm1(logits_features) # was logits_features
        check = torch.any(torch.isnan(h_lstm1_restEverything)).item()
        if check == True:
            print("[INFO] h_lstm1 contains nan element!!")
            exit()

         # From class token to each exam label    
        logits_pe_clsToken = h_lstm1_clsToken[0]
        logits_npe_clsToken = h_lstm1_clsToken[1]
        logits_idt_clsToken = h_lstm1_clsToken[2]
        logits_lpe_clsToken = h_lstm1_clsToken[3]
        logits_rpe_clsToken = h_lstm1_clsToken[4]
        logits_cpe_clsToken = h_lstm1_clsToken[5]
        logits_gte_clsToken = h_lstm1_clsToken[6]
        logits_lt_clsToken = h_lstm1_clsToken[7]
        logits_chronic_clsToken = h_lstm1_clsToken[8]
        logits_acute_and_chronic_clsToken = h_lstm1_clsToken[9]

        att_pool = self.attention(h_lstm1_restEverything, mask)
        max_pool, _ = torch.max(h_lstm1_restEverything, 1)
        conc = torch.cat((max_pool, att_pool), 1)
        # print("[INFO] conc", conc.shape)
        logits_pe = self.last_linear_pe(conc)

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
        return (logits_pe_clsToken,logits_pe), (logits_npe_clsToken, logits_npe_rest), (logits_idt_clsToken, logits_idt_rest), (logits_lpe_clsToken, logits_lpe_rest), \
                (logits_rpe_clsToken, logits_rpe_rest), (logits_cpe_clsToken, logits_cpe_rest), (logits_gte_clsToken, logits_gte_rest), \
                (logits_lt_clsToken, logits_lt_rest), (logits_chronic_clsToken, logits_chronic_rest), \
                (logits_acute_and_chronic_clsToken, logits_acute_and_chronic_rest)

