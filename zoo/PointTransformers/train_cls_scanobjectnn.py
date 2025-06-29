"""
Author: Oddy
Date: May 2025
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from dataset import ScanObjectNN
import argparse
import numpy as np
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import provider
# import importlib
# import shutil
# import hydra
# import omegaconf
from models.Oddy.model import PointTransformerCls
import wandb
import sklearn.metrics as metrics

# def test(model, loader, num_class=40):
#     mean_correct = []
#     class_acc = np.zeros((num_class,3))
#     for j, data in tqdm(enumerate(loader), total=len(loader)):
#         points, target = data
#         target = target[:, 0]
#         points, target = points.cuda(), target.cuda()
#         classifier = model.eval()
#         pred = classifier(points)
#         pred_choice = pred.data.max(1)[1]
#         for cat in np.unique(target.cpu()):
#             classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
#             class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
#             class_acc[cat,1]+=1
#         correct = pred_choice.eq(target.long().data).cpu().sum()
#         mean_correct.append(correct.item()/float(points.size()[0]))
#     class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
#     class_acc = np.mean(class_acc[:,2])
#     instance_acc = np.mean(mean_correct)
#     return instance_acc, class_acc
def test(model, loader):
    model.eval()
    test_pred = []
    test_true = []
    total_loss = 0.0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            points, target = data
            target = target[:, 0]
            points, target = points.cuda(), target.cuda()

            pred = model(points)
            loss = criterion(pred, target)

            total_loss += loss.item() * points.size(0)
            total += points.size(0)

            pred_choice = pred.data.max(1)[1]
            test_true.append(target.cpu().numpy())
            test_pred.append(pred_choice.cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    instance_acc = metrics.accuracy_score(test_true, test_pred)
    balanced_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    avg_loss = total_loss / total
    return avg_loss, instance_acc, balanced_acc

def main(args):
    wandb.init(project="ScanObjectNN", name=args.exp_name)
    '''DATA LOADING'''
    print('Load dataset ...')

    TRAIN_DATASET = ScanObjectNN(num_points=args.num_points, partition='train')
    TEST_DATASET = ScanObjectNN(num_points=args.num_points, partition='test')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    args.num_class = 15
    args.input_dim = 3
    args.nneighbor = 16
    args.nblocks = 4
    args.transformer_dim = 512
    args.learning_rate = args.lr
    args.weight_decay = 1e-4
    args.optimizer = 'Adam'
    args.name = "Oddy"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")

    classifier = PointTransformerCls(args).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    wandb.watch(classifier)

    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    print('Start training...')
    for epoch in range(start_epoch,args.epochs):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epochs))
        
        wandb_log = {}
        classifier.train()

        train_loss_total = 0.0
        train_total = 0
        train_true = []
        train_pred = []

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

            train_loss_total += loss.item() * points.size(0)
            train_total += points.size(0)

            pred_choice = pred.data.max(1)[1]
            train_true.append(target.cpu().numpy())
            train_pred.append(pred_choice.detach().cpu().numpy())
            
        scheduler.step()

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_accuracy = metrics.accuracy_score(train_true, train_pred)
        train_balanced_accuracy = metrics.balanced_accuracy_score(train_true, train_pred)
        train_avg_loss = train_loss_total / train_total

        # train_instance_acc = np.mean(mean_correct)
        print(f'Train Loss: {train_avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, Balanced Accuracy: {train_balanced_accuracy:.4f}')

        # print('Train Instance Accuracy: %f' % train_instance_acc)

        # wandb_log['Train Instance Acc'] = train_instance_acc
        wandb_log['Train Loss'] = train_avg_loss
        wandb_log['Train Acc'] = train_accuracy
        wandb_log['Train AVG Acc'] = train_balanced_accuracy
        # wandb_log['Train Acc'] = train_accuracy
        # wandb_log['Train AVG Acc'] = train_balanced_accuracy

        test_loss, test_accuracy, test_balanced_accuracy = test(classifier, testDataLoader)
        wandb_log['Test Loss'] = test_loss
        wandb_log['Test Acc'] = test_accuracy
        wandb_log['Test AVG Acc'] = test_balanced_accuracy

        print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Balanced Accuracy: {test_balanced_accuracy:.4f}')

        if test_accuracy >= best_instance_acc:
            best_instance_acc = test_accuracy
            best_balanced_acc = test_balanced_accuracy
            best_epoch = epoch + 1
            print('Saving best model...')
            torch.save({
                'epoch': best_epoch,
                'instance_acc': test_accuracy,
                'balanced_acc': test_balanced_accuracy,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'best_model.pth')

        wandb.log(wandb_log)
        global_epoch += 1

    
    print('Best Acc',best_instance_acc)
    print('End of training...')

if __name__ == '__main__':
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
    parser.add_argument('--knn', action='store_true', help='use knn instead ball-query function')

    args = parser.parse_args()
    main(args)