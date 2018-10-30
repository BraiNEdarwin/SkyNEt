#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:44:10 2018
@author: ljknoll

Boolean logic gates
I0 I1    AND NAND OR NOR XOR XNOR
0  0     0   1    0  1   0   1
0  1     0   1    1  0   1   0
1  0     0   1    1  0   1   0
1  1     1   0    1  0   0   1
"""

import torch
from Nets.predNNet import predNNet
from Nets.webNNet import webNNet
import matplotlib.pyplot as plt
main_dir = r'/home/lennart/Desktop/nnweb/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net1 = predNNet(main_dir+data_dir)

web = webNNet()
web.add_vertex(net1, 'A', output=True)


N = 10
beta = 0.1
lr = 0.05
maxiterations = 500
nr_epochs = 1
cv_reset = 'rand' # 0.6*torch.ones(5) # None, 'rand', tensor(5)
target_hardcoded = True

# None, mse, l1, bin, softmargin, binmse
training_type = 'bin'
# bin: 6.435
# mse: 3.739
# l1: 8.969
# softmargin: 27.4
# binmse: 7.27

# input data for both I0 and I1
input_data = torch.zeros(N*4,2)
input_data[N:2*N,   1] = 0.9
input_data[2*N:3*N, 0] = 0.9
input_data[3*N:,    0] = 0.9
input_data[3*N:,    1] = 0.9

list_cv = torch.FloatTensor(
      [[387,-387,650,55,-892],[477,-234,-332,-358,827],
       [9,183,714,-313,-416],[514,665,-64,855,846],
       [-771,342,900,-655,-48],[480,149,-900,-2,-450]])
list_cv += 900
list_cv /=1800


# target data for all gates
gates = ['AND','NAND','OR','NOR','XOR','XNOR']
if target_hardcoded:
    # hardcoded target values of logic gates with off->lower and on->upper
    upper = 1.0
    lower = 0.0
    target_data = upper*torch.ones(4*N,6)
    target_data[:3*N, 0] = lower
    target_data[3*N:, 1] = lower
    target_data[:N, 2] = lower
    target_data[N:, 3] = lower
    target_data[:N, 4] = lower
    target_data[3*N:, 4] = lower
    target_data[N:3*N, 5] = lower
else:
    # use output of known cv configurations as targets
    target_data = torch.ones(4*N,6)    
    for (i, cv) in enumerate(list_cv):
        # set parameters of network to cv
        web.reset_parameters(cv)
        # evaluate network
        target_data[:, i] = web.forward(input_data).data[:,0]

# for each logic gate, train and plot difference
optimizer = torch.optim.Adam


# ------------- Different training types -------------
reshape_target = True
# CrossEntropyLoss for 2 class classification
if training_type == 'bin':
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    target_data = target_data.long()
    def custom_loss_fn(y_pred, y):
        y_pred = torch.cat((y_pred, -y_pred), dim=1)
        return loss_fn(y_pred, y)
    reshape_target = False
# L1 norm loss
elif training_type == 'l1':
    loss_fn = torch.nn.L1Loss(reduction='sum')
# default mse loss
elif training_type == 'mse':
    loss_fn = torch.nn.MSELoss(reduction='sum')
# two-class classification logistic loss
elif training_type=='softmargin':
    loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
# combining binary and mse loss
elif training_type=='binmse':
    def custom_loss_fn(y_pred, y):
        # binary
        loss_fn1 = torch.nn.CrossEntropyLoss(reduction='sum')
        y_pred_cross = torch.cat((y_pred, -y_pred), dim=1)
        loss_value1 = loss_fn1(y_pred_cross, y.long())
        # mse
        loss_fn2 = torch.nn.MSELoss(reduction='sum')
        loss_value2 = loss_fn2(torch.sigmoid(y_pred), y.view(-1, 1))
        return 1*loss_value1 + 1*loss_value2
    reshape_target = False
# use default loss function
else:
    custom_loss_fn = None
    reshape_target = False

if reshape_target:
    def custom_loss_fn(y_pred, y):
        return loss_fn(y_pred, y.view(-1,1))

# code snippet to train maximum difference between inputs
#            minv = torch.min(y_pred)
#            diff = torch.max(y_pred)-minv
#            middle = minv+diff/2.
#            y_pred_scaled = (y_pred-middle)/diff+0.5
#            loss_value = loss(y_pred_scaled, y)


# Train each gate
trained_cv = []
for (i,gate) in enumerate(gates):
    print(i, gate)
    web.reset_parameters(list_cv[i])
    cv_output = web.forward(input_data).data
    web.reset_parameters(cv_reset)
    loss, best_cv = web.train(input_data, target_data[:,i], 
                     beta=beta, 
                     maxiterations=maxiterations, 
                     nr_epochs=nr_epochs,
                     optimizer=optimizer,
                     loss_fn=custom_loss_fn,
                     lr = lr)
    trained_cv.append([i.data.tolist() for i in web.parameters()][0])
    
    # print training error
    plt.figure()
#    plt.plot(torch.min(torch.tensor(loss), 30.*torch.ones(len(loss))))
    plt.plot(loss)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title(gate)
    
    cv_loss = web.error_fn(cv_output, target_data[:,i], beta, custom_loss_fn).item()
    print("cv loss:", cv_loss)
    print("loss:", loss[-1])

    output_data = web.forward(input_data).data  
    mseloss = torch.nn.MSELoss(reduction='sum')(output_data, target_data[:,i].view(-1,1).float()).item()
    print("mseloss: ", mseloss)
    
    # print output and sigmoid(output) of network and targets
    plt.figure()
    plt.plot(target_data[:,i])
#    plt.plot(torch.round(torch.sigmoid(output_data)))
    plt.plot(torch.sigmoid(output_data))
    plt.plot(output_data)
    plt.plot(cv_output)
    plt.legend([
            'target'
#            ,'classification'
            ,'sig(network)'
            ,'network '+str(round(loss[-1], 3))
            ,'cv_output '+str(round(cv_loss, 3))
    ])
    plt.title(gate)