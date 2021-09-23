import os
import time
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models

import resnet_wider
import densenet
from pytorch3dunet.unet3d.model import UNet3D
from pytorch3dunet.unet3d.losses import flatten
from pytorch3dunet.unet3d.utils import expand_as_one_hot


# ---------------------------------------Classification model------------------------------------
def Classifier_model(arch_name, num_class, conv=None, weight=None, linear_classifier=False, sobel=False,
                     activation=None):
    if weight is None:
        weight = "none"

    if conv is None:
        try:
            model = resnet_wider.__dict__[arch_name](sobel=sobel)
        except:
            model = models.__dict__[arch_name](pretrained=False)
    else:
        if arch_name.lower().startswith("resnet"):
            model = resnet_wider.__dict__[arch_name + "_layerwise"](conv, sobel=sobel)
        elif arch_name.lower().startswith("densenet"):
            model = densenet.__dict__[arch_name + "_layerwise"](conv)

    if arch_name.lower().startswith("resnet"):
        kernelCount = model.fc.in_features
        if activation is None:
            model.fc = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())

        if linear_classifier:
            for name, param in model.named_parameters():
                if name not in ['fc.0.weight', 'fc.0.bias', 'fc.weight', 'fc.bias']:
                    param.requires_grad = False

        # init the fc layer
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        kernelCount = model.classifier.in_features
        if activation is None:
            model.classifier = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())

        if linear_classifier:
            for name, param in model.named_parameters():
                if name not in ['classifier.0.weight', 'classifier.0.bias', 'classifier.weight', 'classifier.bias']:
                    param.requires_grad = False

        # init the classifier layer
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
            model.classifier[0].bias.data.zero_()

    else: # added by Nahid
        if arch_name.lower() == "vgg16":
            kernelCount = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(kernelCount, num_class)
            model = nn.Sequential(model, nn.Sigmoid())
        else:
            kernelCount = model.classifier.in_features
        # model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())        


    def _weight_loading_check(_arch_name, _activation, _msg):
        if len(_msg.missing_keys) != 0:
            if _arch_name.lower().startswith("resnet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {"fc.weight", "fc.bias"}
                else:
                    assert set(_msg.missing_keys) == {"fc.0.weight", "fc.0.bias"}
            elif _arch_name.lower().startswith("densenet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {"classifier.weight", "classifier.bias"}
                else:
                    assert set(_msg.missing_keys) == {"classifier.0.weight", "classifier.0.bias"}

    state_dict = None
    if weight.lower() == "random" or weight.lower() == "none":
        state_dict = model.state_dict()

    if weight.lower() == "imagenet":
        pretrained_model = models.__dict__[arch_name](pretrained=True)
        state_dict = pretrained_model.state_dict()

        # delete fc layer
        for k in list(state_dict.keys()):
            if k.startswith('fc') or k.startswith('classifier'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        _weight_loading_check(arch_name, activation, msg)
        print("=> loaded ImageNet pre-trained model")

    if "imagenet_retrain" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight, map_location="cpu")

            # rename pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.') and \
                    not (k.startswith('module.fc') and k.startswith('module.classifier')):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "ssl_transfer" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "insdis" in weight.lower(): # 0
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "pcl-v1" in weight.lower(): # 3
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "pcl-v2" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "pirl" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "sela-v2" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "infomin" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "byol" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "deepcluster-v2" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "swav" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "moco-v1" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> Check: loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "moco-v2" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> Check: loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "barlowtwins" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> Check: loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "simclr-v1" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> Check: loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    if "simclr-v2" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> Check: loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    # elif "moco" in weight.lower():
    #     if os.path.isfile(weight):
    #         print("=> loading checkpoint '{}'".format(weight))
    #         checkpoint = torch.load(weight, map_location="cpu")

    #         # rename moco pre-trained keys
    #         state_dict = checkpoint['state_dict']
    #         for k in list(state_dict.keys()):
    #             # retain only encoder_q up to before the embedding layer
    #             if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    #                 # remove prefix
    #                 state_dict[k[len("module.encoder_q."):]] = state_dict[k]
    #             # delete renamed or unused k
    #             del state_dict[k]

    #         msg = model.load_state_dict(state_dict, strict=False)
    #         _weight_loading_check(arch_name, activation, msg)
    #         print("=> loaded pre-trained model '{}'".format(weight))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(weight))
    elif "part2whole" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q.encoder'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))
    # elif "simclr" in weight.lower():
    #     if os.path.isfile(weight):
    #         print("=> loading checkpoint '{}'".format(weight))
    #         checkpoint = torch.load(weight, map_location="cpu")

    #         msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
    #         _weight_loading_check(arch_name, activation, msg)
    #         print("=> loaded pre-trained model '{}'".format(weight))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(weight))
    elif "c2l" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['model']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('encoder.module.') and not k.startswith('encoder.module.fc.'):
                    # remove prefix
                    state_dict[k[len("encoder.module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            _weight_loading_check(arch_name, activation, msg)
            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    # reinitialize fc layer again
    if arch_name.lower().startswith("resnet"):
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)

    return model, state_dict


class UNet3D_Encoder(nn.Module):
    def __init__(self, in_channels=1, dense_unit=None, num_classes=1000, activation_name=None, testing=False):
        super(UNet3D_Encoder, self).__init__()

        self.testing = testing
        ### for pytorch3dunet
        unet = UNet3D(in_channels, 1, layer_order='cbr')
        self.encoders = unet.encoders
        latent = self.encoders[3].basic_module.SingleConv2.batchnorm.weight.shape[0]

        # from unet3d import UNet3D
        # unet = UNet3D()
        # self.encoders = nn.ModuleList([unet.down_tr64, unet.down_tr128, unet.down_tr256, unet.down_tr512])
        # latent = self.encoders[3].ops[1].bn1.weight.shape[0]

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if dense_unit is None:
            self.fc = nn.ModuleList([nn.Linear(latent, num_classes)])
        else:
            self.fc = nn.ModuleList([nn.Linear(latent, dense_unit), nn.Linear(dense_unit, num_classes)])
        self.activation_name = activation_name

    def forward(self, x):
        for m in self.encoders:
            x = m(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for fc in self.fc:
            x = fc(x)

        if self.testing and self.activation_name is not None:
            if self.activation_name == "sigmoid":
                x = torch.sigmoid(x)

        return x


# -------------------------------------3D Classification model------------------------------------
def Classifier_model_3D(arch_name="UNet3D", num_class=1, weight=None, dense_unit=1024, activation_name=None,
                        linear_classifier=False, testing=False):
    if weight is None:
        weight = "none"

    if arch_name.lower() == "unet3d":
        model = UNet3D_Encoder(dense_unit=dense_unit, num_classes=num_class, activation_name=activation_name,
                               testing=testing)

    if linear_classifier:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    if "simsiam" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight, map_location="cpu")

            # rename simsiam pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder.encoders'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            if arch_name.lower() == "unet3d":
                assert np.logical_and.reduce(["fc" in key for key in msg.missing_keys])

            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))
    elif "p2w" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight, map_location="cpu")

            # rename moco pre-trained keys
            try:
                state_dict = checkpoint['state_dict']
            except:
                state_dict = checkpoint['state_dict_G']  # adversarial training
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            if arch_name.lower() == "unet3d":
                assert np.logical_and.reduce(["fc" in key for key in msg.missing_keys])

            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))
    # elif "p2w" in weight.lower():
    #     if os.path.isfile(weight):
    #         print("=> loading checkpoint '{}'".format(weight))
    #         checkpoint = torch.load(weight, map_location="cpu")
    #
    #         # rename moco pre-trained keys
    #         try:
    #             state_dict = checkpoint['state_dict']
    #         except:
    #             state_dict = checkpoint['state_dict_G']  # adversarial training
    #         for k in list(state_dict.keys()):
    #             # retain only encoder_q up to before the embedding layer
    #             if k.startswith('module.encoder'):
    #                 # remove prefix
    #                 state_dict[k[len("module."):]] = state_dict[k]
    #             # delete renamed or unused k
    #             del state_dict[k]
    #
    #         msg = model.load_state_dict(state_dict, strict=False)
    #         if arch_name.lower() == "unet3d":
    #             assert np.logical_and.reduce(["fc" in key for key in msg.missing_keys])
    #
    #         print("=> loaded pre-trained model '{}'".format(weight))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(weight))
    elif "moco" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight, map_location="cpu")

            # rename moco pre-trained keys
            try:
                state_dict = checkpoint['state_dict']
            except:
                state_dict = checkpoint['state_dict_G']  # adversarial training
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            if arch_name.lower() == "unet3d":
                assert np.logical_and.reduce(["fc" in key for key in msg.missing_keys])

            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    return model


# -------------------------------------3D Classification model------------------------------------
def Segmentator_model_3D(arch_name="UNet3D", num_class=1, weight=None, activation_name="sigmoid", random_decoder=True,
                         testing=False):
    if weight is None:
        weight = "none"

    if arch_name.lower() == "unet3d":
        model = UNet3D(1, out_channels=num_class, layer_order='cbr',
                       final_sigmoid=(activation_name.lower() == "sigmoid"),
                       testing=testing)

    if "simsiam" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight, map_location="cpu")

            # rename simsiam pre-trained keys
            state_dict = checkpoint['state_dict']
            if random_decoder:
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder.encoders'):
                        # remove prefix
                        state_dict[k[len("module.encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = model.load_state_dict(state_dict, strict=False)
                if arch_name.lower() == "unet3d":
                    assert np.logical_and.reduce(
                        [("decoders" in key or "final_conv" in key) for key in msg.missing_keys])
            else:
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder') and \
                        not k.startswith('module.encoder.proj_fc') and \
                        not k.startswith('module.encoder.pred_fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = model.load_state_dict(state_dict, strict=False)
                if arch_name.lower() == "unet3d":
                    assert (len(msg.missing_keys) == 0)

            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))
    elif "p2w" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight, map_location="cpu")

            if random_decoder:
                # rename moco pre-trained keys
                try:
                    state_dict = checkpoint['state_dict']
                except:
                    state_dict = checkpoint['state_dict_G']  # adversarial training
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q.encoders'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = model.load_state_dict(state_dict, strict=False)
                if arch_name.lower() == "unet3d":
                    assert np.logical_and.reduce(
                        [("decoders" in key or "final_conv" in key) for key in msg.missing_keys])
            else:
                # rename moco pre-trained keys
                try:
                    state_dict = checkpoint['state_dict']
                except:
                    state_dict = checkpoint['state_dict_G']  # adversarial training
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = model.load_state_dict(state_dict, strict=False)
                if arch_name.lower() == "unet3d":
                    assert (len(msg.missing_keys) == 0)

            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))
    # elif "p2w" in weight.lower():
    #     if os.path.isfile(weight):
    #         print("=> loading checkpoint '{}'".format(weight))
    #         checkpoint = torch.load(weight, map_location="cpu")
    #
    #         if random_decoder:
    #             # rename moco pre-trained keys
    #             state_dict = checkpoint['state_dict']
    #             for k in list(state_dict.keys()):
    #                 # retain only encoder_q up to before the embedding layer
    #                 if k.startswith('module.encoders'):
    #                     # remove prefix
    #                     state_dict[k[len("module."):]] = state_dict[k]
    #                 # delete renamed or unused k
    #                 del state_dict[k]
    #
    #             msg = model.load_state_dict(state_dict, strict=False)
    #             if arch_name.lower() == "unet3d":
    #                 assert np.logical_and.reduce(
    #                     [("decoders" in key or "final_conv" in key) for key in msg.missing_keys])
    #         else:
    #             # rename moco pre-trained keys
    #             state_dict = checkpoint['state_dict']
    #             for k in list(state_dict.keys()):
    #                 # retain only encoder_q up to before the embedding layer
    #                 if k.startswith('module.'):
    #                     # remove prefix
    #                     state_dict[k[len("module."):]] = state_dict[k]
    #                 # delete renamed or unused k
    #                 del state_dict[k]
    #
    #             msg = model.load_state_dict(state_dict, strict=False)
    #             if arch_name.lower() == "unet3d":
    #                 assert (len(msg.missing_keys) == 0)
    #
    #         print("=> loaded pre-trained model '{}'".format(weight))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(weight))
    elif "moco" in weight.lower():
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            if arch_name.lower() == "unet3d":
                assert np.logical_and.reduce([("decoders" in key or "final_conv" in key) for key in msg.missing_keys])

            print("=> loaded pre-trained model '{}'".format(weight))
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def computeAUROC(dataGT, dataPRED, classCount=14):
    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC


def save_checkpoint(state, is_best, filename='model'):
    torch.save(state, filename + '_checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_checkpoint.pth.tar', filename + '.pth.tar')


def compute_per_channel_dice(input, target, smooth=1, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = input.sum(-1) + target.sum(-1)
    return (2. * intersect + smooth) / (denominator + smooth)


class DiceLoss(nn.Module):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.register_buffer('weight', weight)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=weight)

    def forward(self, input, target):
        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


def torch_dice_coef_loss(y_true, y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))


def compute_iou(y_true, y_pred):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype("float32")

    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)

    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype("float32")
    return np.mean(IoU)


def mean_iou(y_true, y_pred):
    y_true = y_true.astype("int32")
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = (y_pred > t).astype("int32")
        score = compute_iou(y_true, y_pred_)
        prec.append(score)
    return np.mean(prec)


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, computer_background=False, threshold_list=[0.5], **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels
        self.computer_background = computer_background
        self.threshold_list = threshold_list

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        if n_classes == 1:
            per_thresh_iou = []
            for thresh in self.threshold_list:
                per_batch_iou = []
                for _input, _target in zip(input, target):
                    binary_prediction = self._binarize_predictions(_input, n_classes, thresh)
                    binary_prediction = binary_prediction.byte()
                    binary_target = _target.byte()
                    per_batch_iou.append(self._jaccard_index(binary_prediction[0], binary_target[0]))

                    if self.computer_background:
                        binary_prediction = 1 - self._binarize_predictions(_input, n_classes, thresh)
                        binary_prediction = binary_prediction.byte()
                        binary_target = (1 - _target).byte()
                        per_batch_iou.append(self._jaccard_index(binary_prediction[0], binary_target[0]))

                mean_iou = torch.mean(torch.tensor(per_batch_iou))
                per_thresh_iou.append(mean_iou)
            return torch.mean(torch.tensor(per_batch_iou))

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input, n_classes, thresh=0.5):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > thresh
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


# ----------------------------------Whether Experiment Exist----------------------------------
def experiment_exist(log_file, exp_name):
    if not os.path.isfile(log_file):
        return False

    with open(log_file, 'r') as f:
        line = f.readline()
        while line:
            # print(line)
            # if line.replace('\n', '') == exp_name:
            if line.startswith(exp_name):
                return True
            line = f.readline()

    return False


# ----------------------------------Get Pretrained Weight------------------------------------
def get_weight_name(log_file, idx, wait_time=0):
    weight_name = None

    while (True):
        # print(log_file)
        if os.path.isfile(log_file):
            with open(log_file, 'r') as f:
                for i in range(idx + 1):
                    line = f.readline()
                    if not line:
                        break
                if line:
                    line = line.replace('\n', '')
                    weight_name = line
                    # if line.endswith(str(idx)):
                    #   weight_name = line
                    #   break

        if weight_name is not None or wait_time == 0:
            break
        else:
            time.sleep(wait_time * 60.)

    return weight_name


# ---------------------------Callback function for OptionParser-------------------------------
def vararg_callback_bool(option, opt_str, value, parser):
    assert value is None

    arg = parser.rargs[0]
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        value = True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        value = False

    del parser.rargs[:1]
    setattr(parser.values, option.dest, value)


def vararg_callback_int(option, opt_str, value, parser):
    assert value is None
    value = []

    def intable(str):
        try:
            int(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not intable(arg):
            break
        value.append(int(arg))

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)


def display_args(args):
    """Display Configuration values."""
    print("\nConfigurations:")
    for a in dir(args):
        if not a.startswith("__") and not callable(getattr(args, a)):
            print("{:30} {}".format(a, getattr(args, a)))
    print("\n")


if __name__ == '__main__':
    weight = "/mnt/.nfs/ruibinf/MoCo/chestxray_pretrain/checkpoint_0199.pth.tar"
    model = Classifier_model("resnet18", 14, weight=weight, linear_classifier=False)
    print(model)
