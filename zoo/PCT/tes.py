from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import Pct, RPC, RPCV2, PointTransformerCls
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import rsmix_provider
import time
from modelnetc_utils import eval_corrupt_wrapper, ModelNetC
from tqdm import tqdm
import wandb

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'RPC':
        model = RPC(args).to(device)
    elif args.model == 'RPCV2':
        model = RPCV2(args).to(device)
    elif args.model == 'PT':
        model = PointTransformerCls(args).to(device)
    else:
        model = Pct(args).to(device)
    

    data = torch.randn(8, 1024, 3) # (B, N, 3)
    data = data.to(device)
    data = data.permute(0, 2, 1) # (B, 3, N)
    logits = model(data)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--eval_corrupt', type=bool, default=False,
                        help='evaluate the model under corruption')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--model', type=str, default='PCT', choices=['RPC', 'PCT', 'RPCV2', 'PT'], help='choose model')
    parser.add_argument('--fusion_type', type=str, default='concat',
                    choices=['concat', 'add', 'gated', 'attention', 'crossattn'],
                    help='Fusion strategy')
    parser.add_argument('--use_residual', action='store_true',
                    help='Enable residual connection after fusion')
    parser.add_argument('--pretrain_path', type=str, default='', metavar='N',
                        help='Pretrained model path AdaCROSSNET')
    parser.add_argument('--use_initweight', action='store_true', default=False,
                        help='Use Init Weight')

    # pointwolf
    parser.add_argument('--pw', action='store_true', help='use PointWOLF')
    parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point')
    parser.add_argument('--w_sample_type', type=str, default='fps',
                        help='Sampling method for anchor point, option : (fps, random)')
    parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')

    parser.add_argument('--w_R_range', type=float, default=10, help='Maximum rotation range of local transformation')
    parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
    parser.add_argument('--w_T_range', type=float, default=0.25,
                        help='Maximum translation range of local transformation')

    # rsmix
    parser.add_argument('--rdscale', action='store_true', help='random scaling data augmentation')
    parser.add_argument('--shift', action='store_true', help='random shift data augmentation')
    parser.add_argument('--shuffle', action='store_true', help='random shuffle data augmentation')
    parser.add_argument('--rot', action='store_true', help='random rotation augmentation')
    parser.add_argument('--jitter', action='store_true', help='jitter augmentation')
    parser.add_argument('--rddrop', action='store_true', help='random point drop data augmentation')
    parser.add_argument('--rsmix_prob', type=float, default=0.5, help='rsmix probability')
    parser.add_argument('--beta', type=float, default=0.0, help='scalar value for beta function')
    parser.add_argument('--nsample', type=float, default=512,
                        help='default max sample number of the erased or added points in rsmix')
    parser.add_argument('--knn', action='store_true', help='use knn instead ball-query function')

    args = parser.parse_args()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    train(args, io)