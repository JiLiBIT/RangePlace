#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: OverlapTransformer modules for KITTI sequences

import time
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork
from torchvision import transforms
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from models.netvlad import NetVLADLoupe
from tools.utils.utils import *
from einops import rearrange
# from mmcv.cnn import kaiming_init, constant_init
from models.vit import ViT
from models.context_block import ContextBlock
from models.fpn import FPN, Bottleneck
from models.swin import *
from models.GeM import GeM
from models.mixvpr import MixVPR
from typing import Union
import yaml
import math
from functools import partial
"""
    Feature extracter of OverlapTransformer.
    Args:
        height: the height of the range image (64 for KITTI sequences). 
                 This is an interface for other types LIDAR.
        width: the width of the range image (900, alone the lines of OverlapNet).
                This is an interface for other types LIDAR.
        channels: 1 for depth only in our work. 
                This is an interface for multiple cues.
        norm_layer: None in our work for better model.
        use_transformer: Whether to use MHSA.
"""
def get_img_size(down1D,img_size,kernel_size,stride,i):
    if down1D == False:
        for j in range(i):

            img_size[0] = (img_size[0] - kernel_size[j]) // stride[j] + 1
            img_size[1] = (img_size[1] - kernel_size[j]) // stride[j] + 1
    else:
        for j in range(i):
            img_size[0] = (img_size[0] - kernel_size[j]) // stride[j] + 1
        
    return img_size
class RangeEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        height= 64 (int): Image height.  Default: 64.
        width = 900 (int): Image width.  Default: 900.
        patch_size (tuple): Patch token size. Default: 4,1.
        in_chans (int): Number of input image channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 16.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, height= 64, width = 896, patch_size=[2,2], in_chans=1, embed_dim=16, proj_type="mlp",
                 norm_layer=None,drop = 0.):
        super().__init__()
        patches_resolution = [height // patch_size[0] , width // patch_size[1]]
        self.img_size = [height , width]
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.proj_type = proj_type
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_chans,embed_dim//4),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(embed_dim//4,embed_dim//2),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(embed_dim//2,embed_dim),
                nn.GELU(),
                nn.Dropout(drop),)
        elif  proj_type == "conv":
            
            self.embed = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//4, kernel_size=[5,1], stride=[1,1]), #60
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=[3,1], stride=[2,1]), #29
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim//2, embed_dim, kernel_size=[3,1], stride=[2,1]), #14
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=[3,1], stride=[2,1]), #6
                nn.ReLU(inplace=True),) # mixvpr 

            
            # self.proj = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=patch_size, stride=patch_size)

            # self.embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        if self.proj_type == "mlp":
            x = x.view(B,H*W,C)
            x = self.proj(x)
            x = x.view(B,self.embed_dim,H,W)
        elif self.proj_type == "conv":
            x = self.embed(x)

        _, _, Ph, Pw = x.shape 
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return Ph, Pw, x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=False):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)    # B  C X C  C -> B x C

        if self.add_batch_norm:
            gates = self.bn1(gates)                     # B  C -> B  C
        else:
            gates = gates + self.gating_biases          # B  C + C -> B  C

        gates = self.sigmoid(gates)                     # B  C -> B  C

        activation = x * gates                          # B  C X B  C -> B x C

        return activation

class featureExtracter(nn.Module):
    def __init__(self, height=64, width=900, patch_size=[2,2], channels=1,
                 embed_dim=64, depths=[2, 2, 4, 2], num_heads=[1,2,4,4],dim=[64,128,128,256],
                #  embed_dim=16, depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16,
                 window_size=[[1,14],[1,28],[1,56],[1,112]], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 kernel_size=[2,2,2],stride = [2,1,1],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], 
                 token_projection='linear', token_mlp='leff',  down1D = True,
                 use_transformer = True, use_MixVPR = False, use_conv = False, use_fpn = False,
                 gating = True, add_batch_norm=False, **kwargs):
        super(featureExtracter, self).__init__()
        self.resize = transforms.Resize([64,896])
        # self.resize = transforms.Resize([64,896])
        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.num_layers = len(depths)
        self.depths = depths
        self.patch_size = patch_size
        self.embed_dim = embed_dim #C
        self.num_heads = num_heads
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.drop_path = drop_path_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.height = height
        self.width = width
        self.reso = height * width
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.channels = channels
        self.use_conv = use_conv
        self.use_transformer = use_transformer
        self.use_MixVPR =use_MixVPR
        # dim = dim
        self.num_features = int(dim[3])

        self.kernel_size = kernel_size
        self.stride = stride
        self.down1D = down1D
        self.dim = dim
        self.use_fpn = use_fpn
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # split image into non-overlapping patches
        self.patch_embed = RangeEmbed(
            patch_size=patch_size, in_chans=channels, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,proj_type="conv")
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = 6*896 #MIX
        self.layers = self.build_layers()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=256*4, activation='relu', batch_first=False,dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = norm_layer(self.num_features)
    
        self.bnLast1 = nn.BatchNorm2d(4*self.num_features)
        self.bnLast2 = nn.BatchNorm2d(8*self.num_features)
        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.bnLast = nn.BatchNorm1d(256)
        self.lnLast = nn.LayerNorm(256)
        self.gating = gating
        self.hidden_weights = nn.Parameter(torch.randn(1024,256)* 1 / math.sqrt((embed_dim * 2)))
        if self.gating:
            self.context_gating = GatingContext(256, add_batch_norm=True)
        self.norm0 = norm_layer(dim[0])
        self.norm1 = norm_layer(dim[1])
        self.norm2 = norm_layer(dim[2])
        self.norm3 = norm_layer(dim[3])
        self.norms = [self.norm0, self.norm1, self.norm2, self.norm3]




        self.fpn = FeaturePyramidNetwork(dim, 128)

        self.conv01 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1,1),bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv02 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1,1),bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv03 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1,1),bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv04 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1,1),bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv05 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv06 = nn.Conv2d(256, 512, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv07 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bn7 = nn.BatchNorm2d(1024)

        self.conv1 = nn.Sequential(self.conv01,
                                    self.relu
                                    # ,self.bn1
                                    )
        self.conv2 = nn.Sequential(self.conv02,
                                    self.relu
                                    # ,self.bn2
                                    )
        self.conv3 = nn.Sequential(self.conv03,
                                    self.relu
                                    # ,self.bn3
                                    )
        self.conv4 = nn.Sequential(self.conv04,
                                    self.relu
                                    # ,self.bn4
                                    )
        self.conv5 = nn.Sequential(self.conv05,
                                    self.relu
                                    # ,self.bn5
                                    )
        self.conv6 = nn.Sequential(self.conv06,
                                    self.relu
                                    # ,self.bn6
                                    )
        self.conv7 = nn.Sequential(self.conv07,
                                    self.relu
                                    # ,self.bn6
                                    )
        if use_MixVPR:
            
            self.MixVRP1 = MixVPR( in_channels= 4*256,
                                in_h = 6,
                                in_w = 896,
                                out_channels= 64,
                                mix_depth = 4,
                                mlp_ratio = 1,
                                out_rows = 4)

            self.MixVRP2 = MixVPR( in_channels= 4*256,
                                in_h = 3,
                                in_w = 896,
                                out_channels= 64,
                                mix_depth = 4,
                                mlp_ratio = 1,
                                out_rows = 4)
            self.MixVRP3 = MixVPR( in_channels= 4*256,
                                in_h = 2,
                                in_w = 896,
                                out_channels= 64,
                                mix_depth = 4,
                                mlp_ratio = 1,
                                out_rows = 4)
            self.MixVRP4 = MixVPR( in_channels= 4*256,
                                in_h = 1,
                                in_w = 896,
                                out_channels= 64,
                                mix_depth = 4,
                                mlp_ratio = 1,
                                out_rows = 4)
            self.MixVRP=[self.MixVRP1, self.MixVRP2, self.MixVRP3, self.MixVRP4]
            
        else:

            self.vlad0 = NetVLADLoupe(4*256, 6*896, 64,  256, gating=True, add_batch_norm=add_batch_norm,is_training=True)
            self.vlad1 = NetVLADLoupe(4*256, 3*896, 64,  256, gating=True, add_batch_norm=add_batch_norm,is_training=True)
            self.vlad2 = NetVLADLoupe(4*256, 2*896, 64,  256, gating=True, add_batch_norm=add_batch_norm,is_training=True)
            self.vlad3 = NetVLADLoupe(4*256, 1*896, 64,  256, gating=True, add_batch_norm=add_batch_norm,is_training=True)
            sum_cluster_size = 1 + 4 + 16 + 64
            self.vlad=[self.vlad0, self.vlad1, self.vlad2, self.vlad3]
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}
    
    
    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(
                down1D=self.down1D,
                index=i,
                dim=self.dim,
                depths=self.depths,
                embed_dim=self.embed_dim,
                input_resolution = (get_img_size(
                                down1D=self.down1D,img_size=[6,896],
                                kernel_size=self.kernel_size,stride=self.stride, i=i)),
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                win_size=self.window_size[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                token_mlp=self.mlp,
                kernel_size = self.kernel_size,
                stride = self.stride,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers
    
    
    def forward_encoder(self, x):
        B,_,_,_ = x.shape
        Hp,Wp,x = self.patch_embed(x)
        features = []
        for layer in self.layers:
            x,f= layer(x)
            features.append(f)
        return Hp,Wp,features

    def forward_features(self, x):
        # Input Projection
        x = self.resize(x)
        B, _, H, W = x.shape
        C = self.num_features
        
        Hp, Wp, features = self.forward_encoder(x)

        
        x = OrderedDict()
        for i in range(len(features)):
            feature = self.norms[i](features[i])
            feature = features[i]
            B, L, C = feature.shape
            feature = feature.reshape(B,-1,Wp,C).permute(0,3,1,2).contiguous()
            x_name = str(i)
            x[x_name] = feature   
        features = self.fpn(x)
        return Hp, Wp, features
    
    def forward(self, x_l):

        Hp, Wp, features = self.forward_features(x_l) # B L C
        
    
        vlad = []
        if self.use_fpn:
            for i in range(len(features)):
 
                    
                if self.use_MixVPR:
                    feature_l = features[str(i)]
                    feature_l = self.conv5(feature_l)
                    feature_l = self.conv6(feature_l)
                    feature_l = self.conv7(feature_l)
                    temp = self.MixVRP[i](feature_l)
                    temp = F.normalize(temp, dim=1)
                    vlad.append(temp)
                else:
                    # OverlapTransformer
                    feature_l = features[str(i)]
                    feature_l = self.conv1(feature_l)
                    feature_l = self.conv2(feature_l)
                    feature_l = self.conv3(feature_l)
                    feature_l = self.conv4(feature_l)
                    feature_l = self.conv5(feature_l)
                    feature_l = self.conv7(feature_l)

                    feature_l = self.conv6(feature_l)
                    feature_l = F.normalize(feature_l, dim=1)
                    
                    temp = self.vlad[i](feature_l)
                    temp = F.normalize(temp, dim=1)
                    vlad.append(temp)
                        

            pfm = torch.cat((vlad[0], vlad[1], vlad[2], vlad[3]), dim=-1) # B 8*C
            pfm = torch.matmul(pfm, self.hidden_weights)      # B x 8*C X 8*C x 2*C -> B x 2*C
            if self.gating:
                out_l = self.context_gating(pfm)                # B 256 -> B x W256
            out_l = F.normalize(out_l, dim=1)
        else: # w/o FPN
            if self.use_MixVPR:
                feature_l = features["3"]
                feature_l = self.conv5(feature_l)
                feature_l = self.conv6(feature_l)
                feature_l = self.conv7(feature_l)
                out_l = self.MixVRP4(feature_l)
                out_l = F.normalize(out_l, dim=1)
            else:
                feature_l = features["3"]
                feature_l = self.conv5(feature_l)
                feature_l = self.conv6(feature_l)
                feature_l = self.conv7(feature_l)
                # out_l = self.vlad3(feature_l)
                out_l = self.MixVRP4(feature_l)
                #ablation on gem
                feature_l = feature_l.squeeze(2)
                # out_l = self.gem(feature_l)
                out_l = torch.flatten(out_l, 1)

                out_l = F.normalize(out_l, dim=1)
        
        
        return {'global': out_l}
    

if __name__ == '__main__':

    combined_tensor = torch.randn([1,1,64,900]).cuda()
    feature_extracter=featureExtracter(use_transformer=True, channels=1)

    feature_extracter.cuda()
    feature_extracter.eval()

    print("combined_tensor" , combined_tensor.device)
    gloabal_descriptor = feature_extracter(combined_tensor)

    print("size of gloabal descriptor: \n")
    print(gloabal_descriptor["global"].size())
