import argparse
import os
import sys


import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from zoo.DCGRL.util.util import knn, get_graph_feature
from zoo.DCGRL.util.GDANet_util import local_operator, local_operator_withnorm, GDM, SGCAM
from zoo.DCGRL.util.curvenet_util import *

import torch.nn as nn
import torch.nn.functional as F

curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }



class GDAModule_cls(nn.Module):
    def __init__(self, args):
        super(GDAModule_cls, self).__init__()

        self.k = args.k
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn22 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn31 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn32 = nn.BatchNorm1d(128, momentum=0.1)

        self.bn4 = nn.BatchNorm1d(512, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn11)
        self.conv12 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn12)

        self.conv2 = nn.Sequential(nn.Conv2d(67 * 2, 64, kernel_size=1, bias=True),
                                   self.bn2)
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn21)
        self.conv22 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn22)

        self.conv3 = nn.Sequential(nn.Conv2d(131 * 2, 128, kernel_size=1, bias=True),
                                   self.bn3)
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),
                                    self.bn31)
        self.conv32 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=True),
                                    self.bn32)

        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=True),
                                   self.bn4)

        self.SGCAM_1s = SGCAM(64)
        self.SGCAM_1g = SGCAM(64)
        self.SGCAM_2s = SGCAM(64)
        self.SGCAM_2g = SGCAM(64)

        # By default num_of_m is equal to 256 with num points = 1024
        self.num_of_m = round(args.num_points / 4)
        

        
    def forward(self, x):
        B, C, N = x.size()
        ###############
        """block 1"""
        # Local operator:
        
        x1 = local_operator(x, k=self.k)

        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv11(x1))

        x1 = x1.max(dim=-1, keepdim=False)[0]

        # Geometry-Disentangle Module:
        x1s, x1g = GDM(x1, M=self.num_of_m)

        
        # Sharp-Gentle Complementary Attention Module:
        y1s = self.SGCAM_1s(x1, x1s.transpose(2, 1))
        y1g = self.SGCAM_1g(x1, x1g.transpose(2, 1))


        # print("SGCAM sharp shape:",y1s.size(),"SGCAM Gentle shape:",y1g.size())
        z1 = torch.cat([y1s, y1g], 1)
        z1 = F.relu(self.conv12(z1))
        ###############
        """block 2"""
        x1t = torch.cat((x, z1), dim=1)
        x2 = local_operator(x1t, k=self.k)
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv21(x2))
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x2s, x2g = GDM(x2, M=self.num_of_m)

        
        y2s = self.SGCAM_2s(x2, x2s.transpose(2, 1))
        y2g = self.SGCAM_2g(x2, x2g.transpose(2, 1))
        z2 = torch.cat([y2s, y2g], 1)
        z2 = F.relu(self.conv22(z2))
        ###############
        x2t = torch.cat((x1t, z2), dim=1)
        x3 = local_operator(x2t, k=self.k)
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv31(x3))
        x3 = x3.max(dim=-1, keepdim=False)[0]
        z3 = F.relu(self.conv32(x3))
        ###############

        x = torch.cat((z1, z2, z3), dim=1)
        
        x = F.relu(self.conv4(x))
        x_max = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x_max, x_avg), 1)
        

        return x

class CurveModule_cls(nn.Module):
    def __init__(self, args, num_classes=40, k=20, setting='default', additional_channel=32):
        super(CurveModule_cls, self).__init__()

        assert setting in curve_config

        self.args = args
        
        k = args.k
        additional_channel = additional_channel
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=args.num_points, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=args.num_points, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])
        
        self.cic21 = CIC(npoint=args.num_points, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=args.num_points, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        

    def forward(self, xyz, get_flatten_curve_idxs=False):
        flatten_curve_idxs = {}

        l0_points = self.lpfa(xyz, xyz)
        batch_size = xyz.size(0)
        
        l1_xyz, l1_points, flatten_curve_idxs_11 = self.cic11(xyz, l0_points)
        flatten_curve_idxs['flatten_curve_idxs_11'] = flatten_curve_idxs_11
        l1_xyz, l1_points, flatten_curve_idxs_12 = self.cic12(l1_xyz, l1_points)
        flatten_curve_idxs['flatten_curve_idxs_12'] = flatten_curve_idxs_12

        l2_xyz, l2_points, flatten_curve_idxs_21 = self.cic21(l1_xyz, l1_points)
        flatten_curve_idxs['flatten_curve_idxs_21'] = flatten_curve_idxs_21
        l2_xyz, l2_points, flatten_curve_idxs_22 = self.cic22(l2_xyz, l2_points)
        flatten_curve_idxs['flatten_curve_idxs_22'] = flatten_curve_idxs_22

        l3_xyz, l3_points, flatten_curve_idxs_31 = self.cic31(l2_xyz, l2_points)
        flatten_curve_idxs['flatten_curve_idxs_31'] = flatten_curve_idxs_31
        l3_xyz, l3_points, flatten_curve_idxs_32 = self.cic32(l3_xyz, l3_points)
        flatten_curve_idxs['flatten_curve_idxs_32'] = flatten_curve_idxs_32

        l4_xyz, l4_points, flatten_curve_idxs_41 = self.cic41(l3_xyz, l3_points)
        flatten_curve_idxs['flatten_curve_idxs_41'] = flatten_curve_idxs_41
        l4_xyz, l4_points, flatten_curve_idxs_42 = self.cic42(l4_xyz, l4_points)
        flatten_curve_idxs['flatten_curve_idxs_42'] = flatten_curve_idxs_42
        
        x = self.conv0(l4_points)
        
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        
        x = torch.cat((x_max, x_avg), dim=1)
        if get_flatten_curve_idxs:
            return x, flatten_curve_idxs
        else:
            return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, context1, context2, direction='gda_to_curve'):
        # context1 and context2: (B, D)
        if direction == 'curve_to_gda':
            q = context2.unsqueeze(1)
            k = context1.unsqueeze(1)
            v = context1.unsqueeze(1)
        else:
            q = context1.unsqueeze(1)
            k = context2.unsqueeze(1)
            v = context2.unsqueeze(1)

        # Cross-attention: context1 attends to context2
        attended, _ = self.cross_attn(q, k, v)  # (B, 1, D)
        attended = attended.squeeze(1)          # (B, D)

        # Residual + LayerNorm (optional but common)
        fused = self.layernorm(attended + context1)  # (B, D)

        return fused

class DCGRL(nn.Module):
    def __init__(self, args, num_classes=40, k=20, setting='default'):
        super(DCGRL, self).__init__()

        assert setting in curve_config

        self.args = args

        self.gda = GDAModule_cls(args) # output (B, 1024)
        self.curve = CurveModule_cls(args) # output (B, 2048)
        
        self.fusion_type = args.fusion_type
        self.use_residual = args.use_residual

        self.gda_dim = 1024
        self.curve_dim = 2048
        self.hidden_dim = 1024

        if self.fusion_type in ['add', 'gated', 'attention', 'crossattn']:
            self.gda_proj = nn.Linear(self.gda_dim, self.hidden_dim)
            self.curve_proj = nn.Linear(self.curve_dim, self.hidden_dim)

        if self.fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, 1),
                nn.Sigmoid()
            )
            fusion_output_dim = self.hidden_dim

        elif self.fusion_type == 'attention':
            self.att_mlp = nn.Sequential(
                nn.Linear(self.gda_dim + self.curve_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
                nn.Softmax(dim=-1)
            )
            fusion_output_dim = self.hidden_dim

        elif self.fusion_type == 'crossattn':
            self.cross_fusion = CrossAttentionFusion(hidden_dim=self.hidden_dim)
            fusion_output_dim = self.hidden_dim

        elif self.fusion_type == 'add':
            fusion_output_dim = self.hidden_dim

        else:  # concat
            fusion_output_dim = self.gda_dim + self.curve_dim

        if self.fusion_type in ['add', 'gated', 'attention', 'crossattn']:
            residual_input_dim = self.hidden_dim * 2  # 1024 + 1024 = 2048
        else:
            residual_input_dim = self.gda_dim + self.curve_dim  # 1024 + 2048 = 3072

        self.res_proj = nn.Linear(residual_input_dim, fusion_output_dim)
        self.layernorm = nn.LayerNorm(fusion_output_dim)  # for residual + norm
        
        self.fc1 = nn.Linear(fusion_output_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, xyz):
        
        x_gda = self.gda(xyz)
        x_curve = self.curve(xyz)

        if self.fusion_type == 'add':
            x = self.gda_proj(x_gda) + self.curve_proj(x_curve)

        elif self.fusion_type == 'gated':
            x_gda_proj = self.gda_proj(x_gda)
            x_curve_proj = self.curve_proj(x_curve)
            gate_input = torch.cat([x_gda_proj, x_curve_proj], dim= -1)
            alpha = self.gate(gate_input)
            x = alpha * x_gda_proj + (1 - alpha) * x_curve_proj

        elif self.fusion_type == 'attention':
            x_gda_proj = self.gda_proj(x_gda)
            x_curve_proj = self.curve_proj(x_curve)

            # Attention weights
            combined = torch.cat([x_gda, x_curve], dim=1)  # Original features for context
            weights = self.att_mlp(combined)  # (B, 2)

            # Apply attention weights to projected features
            x = weights[:, 0:1] * x_gda_proj + weights[:, 1:2] * x_curve_proj


        elif self.fusion_type == 'crossattn':
            x_gda_proj = self.gda_proj(x_gda)
            x_curve_proj = self.curve_proj(x_curve)
            x = self.cross_fusion(x_gda_proj, x_curve_proj)

        else:
            x = torch.cat((x_gda, x_curve), dim=1)

        if self.use_residual:
            if self.fusion_type in ['add', 'gated', 'attention', 'crossattn']:
                # project x_gda and x_curve first
                resid_input = torch.cat((x_gda_proj, x_curve_proj), dim=1)
            else:
                resid_input = torch.cat((x_gda, x_curve), dim=1)

            residu = self.res_proj(resid_input)
            x = self.layernorm(x + residu)



        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.dp1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--num_points', type=int, default=2048, help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    args = parser.parse_args()

    
    device = torch.device("cuda")

    model = DCGRL(args).to(device)
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn

    print("Num of params:",round(pp/1000000, 3),"m")
    batch_size = args.batch_size
    data = torch.rand(batch_size, 3, args.num_points)
    data = data.to(device)

    

    # # Print initial memory usage
    # initial_memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
    # initial_memory_reserved = torch.cuda.memory_reserved(device) / 1024**2

    # print(f"Initial Memory Allocated: {initial_memory_allocated:.2f} MB")
    # print(f"Initial Memory Reserved: {initial_memory_reserved:.2f} MB")

    
    # Perform a forward pass
    output = model(data)

    # # Print memory usage after model inference
    # post_inference_memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
    # post_inference_memory_reserved = torch.cuda.memory_reserved(device) / 1024**2

    # print(f"Memory Allocated After Inference: {post_inference_memory_allocated:.2f} MB")
    # print(f"Memory Reserved After Inference: {post_inference_memory_reserved:.2f} MB")

    # # Example: Clear cache
    # torch.cuda.empty_cache()

    # # Print memory usage after clearing cache
    # memory_allocated_after_cache_clear = torch.cuda.memory_allocated(device) / 1024**2
    # memory_reserved_after_cache_clear = torch.cuda.memory_reserved(device) / 1024**2

    # print(f"Memory Allocated After Clearing Cache: {memory_allocated_after_cache_clear:.2f} MB")
    # print(f"Memory Reserved After Clearing Cache: {memory_reserved_after_cache_clear:.2f} MB")

    # print(x.size())
