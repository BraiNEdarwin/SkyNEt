#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:03:02 2019

@author: ljknoll

Script which searches for control problem solution.
Results are only stored and should be loaded/analyzed with read_control_problem.py

"""

import torch
import torch.tensor as tensor
import numpy as np
import time
import pickle

import SkyNEt.modules.SaveLib as sl
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet
import matplotlib.pyplot as plt
import SkyNEt.experiments.Web.problems as problems

# function to generate train data for control problem
input_data, target_data = problems.get_control_problem(100, mean_I0=-0.3, mean_I1=-0.3, amp_I0=0.9, amp_I1=0.9)

# create folder in which results are stored during searching
result_folder = 'C:/users/lennart/Dropbox/afstuderen/results/'
result_name = 'cp_GD_3-1_v2_beta80-92'
saved_path = sl.createSaveDirectory(result_folder, result_name)
sl.copyFiles(saved_path)

# network import
main_dir = r'C:/users/lennart/Dropbox/afstuderen/search_scripts/'
network_dir = 'MSE_d5w90_500ep_lr1e-3_b2048_b1b2_0.90.75-11-05-21h48m.pt'
net = staNNet(main_dir+network_dir)

def cust_sig(x):
     return torch.sigmoid(x/15*2)

web = webNNet()
web.transfer = cust_sig # always define transfer function before adding vertices & arcs
web.add_vertex(net, 'A', output=True, input_gates=[])
web.add_vertex(net, 'B', input_gates=[1,2])
web.add_vertex(net, 'C', input_gates=[1,2])
web.add_vertex(net, 'D', input_gates=[1,2])

web.add_arc('B', 'A', 1)
web.add_arc('C', 'A', 2)
web.add_arc('D', 'A', 3)

# add custom parameters
web.add_parameters(['scale', 'bias'],                       # parameter names
                   [torch.ones(2), torch.tensor([0., 1.])], # parameter initialization values
#                   lambda : 0.0001*torch.mean(web.scale**2 + web.bias**2)/2, # custom regularization function
                   lr=0.1, betas=(0.9,0.99))                # hyperparameters are applied only to these parameters


# IMPORTANT, depending how many vertices need input data, copy it enough times
input_data = torch.cat((input_data,)*3, dim=1)


batch_size = 300
nr_epochs = 1000
lr = 0.05
beta = 5

cv_reset = 'rand'
#cv_reset = {'A': tensor([0.3040, 0.0888, 0.3488, 0.9963, 0.9983]),
# 'scale': tensor([22.6166, -6.8213]),cp_GD_1_v1_beta70-92
# 'bias': tensor([-2.4433, -1.7016])}

optimizer = torch.optim.Adam
betas=(0.8, 0.92)
eps=1e-08


weights = torch.bincount(target_data[:,0]).float()
weights = 1/weights
weights /= torch.sum(weights)
custom_loss = torch.nn.CrossEntropyLoss(weight = weights)
def loss_fn(y_pred, y):
    temp = web.scale*(y_pred+web.bias)
    return custom_loss(torch.stack((temp[:,0], torch.zeros(y_pred.shape[0]), temp[:,1])).t(), y[:,0])

Softmax = torch.nn.Softmax(dim=1)
def classify(y_pred):
    temp = web.scale*(y_pred+web.bias)
    softmaxoutput = Softmax(torch.stack((temp[:,0], torch.zeros(y_pred.shape[0]), temp[:,1])).t())
    return torch.argmax(softmaxoutput, dim=1).view(-1,1)

def stop_fn(epoch, error_list, best_error):
    # return true if the error has not improved the last 50 epochs or is not under 0.5 after 400 epochs
    if min(error_list[-100:]) > best_error or (len(error_list)>400 and min(error_list[-3:])>0.5):
        return True


session_size = 100
dim_size = len(torch.cat(tuple(web.get_parameters().values())))

# format of list results: param_history, best_parameters, error, classification
for iteration in range(0,10):
    results = [[],[],[],[]]
    results[0] = torch.zeros(session_size, nr_epochs, dim_size)
    for i in range(session_size):
        web.reset_parameters(cv_reset)
        loss, best_cv, all_params = web.train(input_data, target_data, 
                                              beta=beta,
                                              batch_size=batch_size,
                                              max_epochs=nr_epochs,
                                              optimizer=optimizer,
                                              loss_fn=loss_fn,
                                              lr = lr,
                                              betas = betas,
                                              eps = eps,
                                              stop_fn = stop_fn,
                                              verbose=False)
        results[1].append(best_cv)
        classification = classify(web.get_output())
        class_rate = torch.mean((target_data==classification).float())
        results[2].append(loss)
        results[3].append(class_rate.item())
        results[0][i] = all_params
        print("INFO: %i, best error after %i iterations: %i/%i: %f" % (iteration, len(loss), i+1, session_size, min(loss)))
    with open(saved_path+'/single_' + str(iteration) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.p', 'wb') as f:
        pickle.dump(results, f)


def plot_results(best_cv):
    web.reset_parameters(best_cv)
    d0 = -web.bias[0]
    d2 = -web.bias[1]
    web.forward(input_data)
    output_data = web.get_output()
    plt.figure()
    plt.plot(target_data.float().numpy()+0.01)
    plt.plot(output_data.numpy())
    classification = classify(output_data)
    plt.plot(classification.numpy())
    plt.plot((0, len(input_data)), (d0, d0), 'r-', alpha=0.3)
    plt.plot((0, len(input_data)), (d2, d2), 'r-', alpha=0.3)
    plt.legend(['target', 'output', 'classification'])
    plt.title('web output vs target')
    plt.show()
    print('classification rate: %0.3f' % torch.mean((target_data==classification).float()))
plot_results(best_cv)



def plot_param_hist(results, index):
    # generate names for legend
    names = []
    for key,value in results[1][index].items():
        names = names + [key+str(i) for i in range(len(value))]
    
    hist = results[0][index][:len(results[2][index])]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(hist.numpy())
    plt.xlabel('epoch')
    plt.ylabel('param value')
    plt.legend(names)

    plt.subplot(2,1,2)     
    plt.plot(results[2][index])
    plt.ylabel('CrossEntropyLoss')
    plt.xlabel('epoch')
    plt.show()
plot_param_hist(results, 0)


def probs():
    with torch.no_grad():
        y_pred = torch.arange(-30, 30, step=0.01).view(-1,1)
        temp = web.scale*(y_pred+web.bias)
        softmaxoutput = Softmax(torch.stack((temp[:,0], torch.zeros(y_pred.shape[0]), temp[:,1])).t())
        plt.figure()
        plt.plot(y_pred.numpy(), softmaxoutput.numpy())
        plt.legend(['p0', 'p1', 'p2'])
        plt.show()
probs()