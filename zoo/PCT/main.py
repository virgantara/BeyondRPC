from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ScanObjectNN
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
    wandb.init(project="UnderCorruptionPairedTest", name=args.exp_name)
    
    output_channels = 40
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points, args=args if args.pw else None),
                              num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == 'scanobjectnn':
        output_channels = 15
        train_loader = DataLoader(
            ScanObjectNN(partition='train', num_points=args.num_points),
            num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            ScanObjectNN(partition='test', num_points=args.num_points), 
            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'RPC':
        model = RPC(args, output_channels).to(device)
    elif args.model == 'RPCV2':
        model = RPCV2(args).to(device)
    elif args.model == 'PT':
        model = PointTransformerCls(args, output_channels).to(device)
    else:
        model = Pct(args, output_channels).to(device)
    # print(str(model))

    if not args.use_initweight:
        print("Use Pretrain")
        state_dict = torch.load(args.pretrain_path)

        # optionally: filter only keys that match
        model_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)

    model = nn.DataParallel(model)
    wandb.watch(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    criterion = cal_loss
    best_test_acc = 0

    for epoch in range(args.epochs):
        
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0

        wandb_log = {}
        for data, label in tqdm(train_loader):
            # Data (B, N, 3)
            # label (B, 1)
            '''
            implement augmentation
            '''

            rsmix = False
            r = np.random.rand(1)
            if args.beta > 0 and r < args.rsmix_prob:
                rsmix = True
                n_sample=int(args.nsample)
                data_np = data.cpu().numpy()
                label_np = label.cpu().numpy()
                
                data, lam, label, label_b = rsmix_provider.rsmix(data_np, label_np, beta=args.beta, n_sample=n_sample,
                                                                 KNN=args.knn)

                data = torch.FloatTensor(data)
                label = torch.LongTensor(label)
                label_b = torch.LongTensor(label_b)
                lam = torch.FloatTensor(lam)
            if args.rot or args.rdscale or args.shift or args.jitter or args.shuffle or args.rddrop or (
                    args.beta != 0.0):
                data = torch.FloatTensor(data)
            if rsmix:
                lam = torch.FloatTensor(lam)
                lam, label_b = lam.to(device), label_b.to(device).squeeze()

            data, label = data.to(device), label.to(device).squeeze()

            if rsmix:
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()

                start_time = time.time()
                logits = model(data)
                loss = 0
                for i in range(batch_size):
                    loss_tmp = criterion(logits[i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - lam[i]) \
                               + criterion(logits[i].unsqueeze(0), label_b[i].unsqueeze(0).long()) * lam[i]
                    loss += loss_tmp
                loss = loss / batch_size
            else:
                data = data.permute(0, 2, 1) # (B, 3, N)
                
            
                batch_size = data.size()[0]
                opt.zero_grad()
                start_time = time.time()
                logits = model(data)
                loss = criterion(logits, label)

            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        print('train total time is', total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_accuracy = metrics.accuracy_score(train_true, train_pred)
        train_balanced_accuracy = metrics.balanced_accuracy_score(train_true, train_pred)

        scheduler.step()

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 train_accuracy,
                                                                                 train_balanced_accuracy)
        io.cprint(outstr)
        wandb_log['Train Loss'] = train_loss * 1.0 / count
        wandb_log['Train Acc'] = train_accuracy
        wandb_log['Train AVG Acc'] = train_balanced_accuracy

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        print('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        wandb_log['Test Loss'] = test_loss*1.0/count
        wandb_log['Test Acc'] = test_acc
        wandb_log['Test AVG Acc'] = avg_per_class_acc
        wandb.log(wandb_log)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            print("Best Test Acc: ",best_test_acc)
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
        torch.save(model.state_dict(), 'checkpoints/%s/models/model_final.t7' % args.exp_name)


def test(args, io):
    output_channels = 40
    if args.dataset == 'modelnet40':
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == 'scanobjectnn':
        output_channels = 15
        test_loader = DataLoader(
            ScanObjectNN(partition='test', num_points=args.num_points), 
            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    # test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
    #                          batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'RPC':
        model = RPC(args, output_channels).to(device)
    elif args.model == 'RPCV2':
        model = RPCV2(args).to(device)
    elif args.model == 'PT':
        model = PointTransformerCls(args, output_channels).to(device)
    else:
        model = Pct(args, output_channels).to(device)
        
    # model = Pct(args).to(device)
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1]
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40','scanobjectnn'])
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
    parser.add_argument('--beta', type=float, default=1.0, help='scalar value for beta function')
    parser.add_argument('--nsample', type=float, default=512,
                        help='default max sample number of the erased or added points in rsmix')
    parser.add_argument('--knn', action='store_true', help='use knn instead ball-query function')

    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval and not args.eval_corrupt:
        train(args, io)
    elif args.eval:
        test(args, io)
    elif args.eval_corrupt:
        device = torch.device("cuda" if args.cuda else "cpu")
        if args.model == 'RPC':
            model = RPC(args).to(device)
        elif args.model == 'RPCV2':
            model = RPCV2(args).to(device)
        else:
            model = Pct(args).to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))
        model = model.eval()


        def test_corrupt(args, split, model):
            test_loader = DataLoader(ModelNetC(split=split),
                                     batch_size=args.test_batch_size, shuffle=True, drop_last=False)
            test_true = []
            test_pred = []
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                logits = model(data)
                preds = logits.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            return {'acc': test_acc, 'avg_per_class_acc': avg_per_class_acc}


        eval_corrupt_wrapper(model, test_corrupt, {'args': args})
