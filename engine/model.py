# coding:utf-8
'''
model file:relation detect
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy, random, sys
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import setting

SMALL_TEST = True
FIX_SEED = True

ROI_MAGNIFY = 1
FEATURE_MAP_SIZE = (16, 16)

FC_DIM = 1000
FEATURE_CHANNEL = 512
EMBED_SIZE = 800

if SMALL_TEST:
    FEATURE_MAP_SIZE = (8, 8)
    FC_DIM = 600
    FEATURE_CHANNEL = 512
    EMBED_SIZE = 500
if FIX_SEED:
    torch.manual_seed(0)

print("FEATURE_MAP_SIZE, FC_DIM, FEATURE_CHANNEL", FEATURE_MAP_SIZE, FC_DIM, FEATURE_CHANNEL)
print("EMBED_SIZE", EMBED_SIZE)
print("SMALL_TEST, FIX_SEED", SMALL_TEST, FIX_SEED)


class Net(nn.Module):
    def __init__(self, device, batch_size, num_classes):
        super(Net, self).__init__()
        self.device = device
        self.batch_size = batch_size

        self.groupnorm1 = nn.GroupNorm(32, 512)

        feature_size = FEATURE_MAP_SIZE[0] * FEATURE_MAP_SIZE[1] * FEATURE_CHANNEL
        self.fc_relation = nn.Sequential(
            nn.Linear(feature_size, FC_DIM),
            nn.ReLU(True),
            nn.Linear(FC_DIM, FC_DIM)
        )

        self.fc_embed = nn.Sequential(
            nn.Linear(FC_DIM + 6, EMBED_SIZE),
            nn.ReLU(True)
        )

        self.fc_predict = nn.Linear(EMBED_SIZE, num_classes)

        pre_model = models.vgg16(pretrained=True)
        self.vgg_layers = nn.Sequential(*list(pre_model.features.children())[:-1])

    def adjust_box(self, box, feat_shape):
        y_start, y_end, x_start, x_end = box
        if y_start == y_end:
            if y_end == feat_shape[2]:# if end equals to height
                y_start -= 1
            else:
                y_end += 1
        if x_start == x_end:
            if x_end == feat_shape[3]:# if end equals to width
                x_start -= 1
            else:
                x_end += 1
        return y_start, y_end, x_start, x_end


    def box_vector(self, feats, box):
        fc_feat = self.fc_relation(feats)

        box_t = torch.from_numpy(box.astype(np.float32)).to(self.device)
        # input also height and width
        box_h = box_t[:, 1] - box_t[:, 0]
        box_w = box_t[:, 3] - box_t[:, 2]

        fc_feat = torch.cat((fc_feat, box_t, box_h.unsqueeze(1), box_w.unsqueeze(1)), 1)

        vector = self.fc_embed(fc_feat)
        return vector


    def forward(self, x, objbox, subbox):
        h_img = x.shape[2]# shape=[1, 3, 600, 800]
        w_img = x.shape[3]

        x = self.vgg_layers(x)

        h_feature = x.shape[2]
        w_feature = x.shape[3]

        feat_scale = h_feature/float(h_img)

        objbox_featmap = np.round(objbox * feat_scale * ROI_MAGNIFY).astype(np.int)
        subbox_featmap = np.round(subbox * feat_scale * ROI_MAGNIFY).astype(np.int)

        features_shape = [self.batch_size, x.shape[1] * FEATURE_MAP_SIZE[0] * FEATURE_MAP_SIZE[1]]
        obj_feats = torch.zeros(features_shape, dtype=torch.float32).to(self.device)
        sub_feats = torch.zeros(features_shape, dtype=torch.float32).to(self.device)

        for idx, (e_objbox, e_subbox) in enumerate(zip(objbox_featmap, subbox_featmap)):
            y_start, y_end, x_start, x_end = self.adjust_box(e_objbox, x.shape)
            x_obj = x[:, :, y_start:y_end, x_start:x_end]

            y_start, y_end, x_start, x_end = self.adjust_box(e_subbox, x.shape)
            x_sub = x[:, :, y_start:y_end, x_start:x_end]

            x_obj = nn.functional.interpolate(x_obj, size=FEATURE_MAP_SIZE, mode='bilinear', align_corners=False)
            x_sub = nn.functional.interpolate(x_sub, size=FEATURE_MAP_SIZE, mode='bilinear', align_corners=False)

            x_obj = self.groupnorm1(x_obj)
            x_sub = self.groupnorm1(x_sub)

            x_obj_flat = x_obj.view(x_obj.size(0), -1)[0]
            x_sub_flat = x_sub.view(x_sub.size(0), -1)[0]
            obj_feats[idx] = x_obj_flat
            sub_feats[idx] = x_sub_flat

        obj_vector = self.box_vector(obj_feats, objbox)
        sub_vector = self.box_vector(sub_feats, subbox)

        relation_vector = obj_vector - sub_vector

        prediction = self.fc_predict(relation_vector)
        return prediction


