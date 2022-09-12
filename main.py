from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings("ignore")

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from utils import *
from modules import DynamicWHAR
import sklearn.metrics as metrics
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='DynamicWHAR',
                    help='Model name.')
parser.add_argument('--dataset', type=str, default='opp_24_12',
                    help='Dataset name, e.g. opp_24_12, realworld_40_20, skoda_right_78_39, realdisp_100_50.')
parser.add_argument('--Scheduling_lambda', type=float, default=0.995,
                    help='Scheduling lambda.')
parser.add_argument('--test-user', type=int, default=0, 
                    help='ID of test user.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

args, unknown = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.dataset == "opp_24_12":
    args.window_size = 24
    args.user_num = 4
    args.class_num = 17
    args.node_num = 5
    args.node_dim = 9
    
    args.lr = 0.00005
    args.epochs = 80
    args.batch_size = 64
    
    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128 
    
elif args.dataset == "opp_60_30":
    args.window_size = 60
    args.user_num = 4
    args.class_num = 17
    args.node_num = 5
    args.node_dim = 9
    args.intervals = 10 
    args.window = 6
    
    args.lr = 0.00005
    args.epochs = 80
    args.batch_size = 64
    
    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128     
    
elif args.dataset == "realworld_40_20":
    args.window_size = 40   
    args.user_num = 13
    args.class_num = 8
    args.node_num = 7
    args.node_dim = 9    
    
    args.lr = 0.000001
    args.epochs = 60
    args.batch_size = 128
    
    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128
    
elif args.dataset == "realworld_100_50":
    args.window_size = 100
    args.user_num = 13
    args.class_num = 8
    args.node_num = 7
    args.node_dim = 9    
    
    args.lr = 0.000001
    args.epochs = 60    
    args.batch_size = 128
    
    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128

elif args.dataset == "realdisp_40_20":
    args.window_size = 40   
    args.user_num = 10
    args.class_num = 33
    args.node_num = 9
    args.node_dim = 9
    
    args.lr = 0.0001
    args.epochs = 60
    args.batch_size = 128
    
    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128 

elif args.dataset == "realdisp_100_50":
    args.window_size = 100 
    args.user_num = 10
    args.class_num = 33
    args.node_num = 9
    args.node_dim = 9  
    
    args.lr = 0.0001
    args.epochs = 60
    args.batch_size = 128
    
    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128 

elif args.dataset == "skoda_right_78_39":
    args.window_size = 78
    args.user_num = 1
    args.class_num = 10
    args.node_num = 10
    args.node_dim = 3
    
    args.lr = 0.0001
    args.epochs = 80
    args.batch_size = 64
    
    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128    

elif args.dataset == "skoda_right_196_98":
    args.window_size = 196
    args.user_num = 1
    args.class_num = 10
    args.node_num = 10
    args.node_dim = 3
    
    args.lr = 0.0001
    args.epochs = 80
    args.batch_size = 64
    
    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

rel_rec, rel_send = edge_init(args.node_num, args.cuda)

def train(model, train_loader, test_loader, optimizer, scheduler, epoch):
    model = model
    train_loader = train_loader
    test_loader = test_loader
    optimizer = optimizer
    scheduler = scheduler
    
    t = time.time()    
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    loss_train = []
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        if data.shape[0] == 1:
            continue
    
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)
        optimizer.zero_grad()
        output = model(data, rel_rec, rel_send)
        loss = criterion(output, label) 
        loss.backward()
        optimizer.step()
        loss_train.append(loss.data.item())
        
    scheduler.step()

    correct1 = 0
    size = 0
    predicts = []
    labelss = []  
    loss_val = []
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        if data.shape[0] == 1:
            continue
        
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(
            label, volatile=True)

        output = model(data, rel_rec, rel_send)
        test_loss = criterion(output, label)
        pred1 = output.data.max(1)[1] 
        k = label.data.size()[0]
        correct1 += pred1.eq(label.data).cpu().sum()
        size += k    
        labels = label.cpu().numpy()
        labelss = labelss + list(labels)
        pred1s = pred1.cpu().numpy()
        predicts = predicts + list(pred1s)
        
        loss_val.append(test_loss.data.item())

    print('Epoch: {:04d}'.format(epoch),
          'train_loss: {:.6f}'.format(np.mean(loss_train)),
          'test_loss: {:.6f}'.format(np.mean(loss_val)),
          'test_acc:{:.6f}'.format(1. * correct1.float() / size),
          'test_f1: {:.6f}'.format(metrics.f1_score(labelss, predicts, average='macro')),
          'time: {:.4f}s'.format(time.time() - t))
          
def main():
    model = DynamicWHAR(node_num=args.node_num, 
                    node_dim=args.node_dim, 
                    window_size=args.window_size, 
                    channel_dim=args.channel_dim, 
                    time_reduce_size=args.time_reduce_size, 
                    hid_dim=args.hid_dim, 
                    class_num=args.class_num)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(list(model.parameters()),lr=args.lr)
    lambda1 = lambda epoch: args.Scheduling_lambda ** epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    train_loader, test_loader = load_data(name = args.dataset, batch_size=args.batch_size, test_user=args.test_user)

    for epoch in range(args.epochs):
        train(model, train_loader, test_loader, optimizer, scheduler, epoch)

if __name__ == '__main__':
    main()