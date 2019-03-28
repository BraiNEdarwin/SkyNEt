#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:44:10 2018
@author: ljknoll

Searching for boolean logic functionality using Gradient Descent in web framework

Boolean logic gates
I0 I1    AND NAND OR NOR XOR XNOR
0  0     0   1    0  1   0   1
0  1     0   1    1  0   1   0
1  0     0   1    1  0   1   0
1  1     1   0    1  0   0   1
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.predNNet import predNNet
from SkyNEt.modules.Nets.webNNet import webNNet


# ------------------------ configure ------------------------
# load device simulation
main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_03_14_143310_characterization_7D_t_4days_f_0_1_fs_100\\'
data_dir = 'NN_skip3_MSE_noisefit.pt'

net1 = predNNet(main_dir+data_dir)

# single device web
web = webNNet()
web.add_vertex(net1, 'A', output=True)

# hardcoded target values of logic gates with off->lower and on->upper
upper = 1.
lower = 0.

# if set to false, use output of known cv configurations as targets

N = 100 # number of data points of one of four input cases, total 4*N

batch_size = 50
max_epochs = 300

lr = 0.05
beta = 10
cv_reset = 'rand' #0.4*torch.ones(5)# None, 'rand', tensor(5)

# None, mse, l1, bin, softmargin, binmse, cor, cormse
training_type = 'cormse'


add_noise = True # automatically set to false when using bin/softmargin
sigma = 0.1 # standard deviation of added noise in target


# wether to train scale output and bias before returning
bias=False
scale=False

# define custom stopping criteria
def stop_func(epoch, error_list, best_error):
    # if the error has not improved the last 50 epochs, reset parameters random
    if min(error_list[-50:]) > best_error:
        print("INFO: No improvement after 50 iterations")
        return True

# ------------------------ END configure ------------------------





# input data for both I0 and I1
input_data = -0.5*torch.ones(N*4,2)


input_data[N:2*N,   1] = 0.8
input_data[2*N:3*N, 0] = 0.8
input_data[3*N:,    0] = 0.8
input_data[3*N:,    1] = 0.8

# target data for all gates
gates = ['AND','NAND','OR','NOR','XOR','XNOR']

target_data = upper*torch.ones(6, 4*N)
target_data[0, :3*N] = lower
target_data[1, 3*N:] = lower
target_data[2, :N] = lower
target_data[3, N:] = lower
target_data[4, :N] = lower
target_data[4, 3*N:] = lower
target_data[5, N:3*N] = lower
"""
w = torch.ones(6, 2)
for i in range(6):
    temp = torch.sum(target_data[i].float())/4/N
    weights = torch.bincount(target_data[i].long()).float()
    weights = 1/weights
    weights /= torch.sum(weights)
    w[i] = weights
"""

optimizer = torch.optim.Adam
def cor_loss_fn(x, y):
    corr = torch.mean((x-torch.mean(x))*(y-torch.mean(y)))
    return 1-corr/torch.std(x)/torch.std(y)
mse_loss_fn = torch.nn.MSELoss()

# ------------- Different training types -------------
# CrossEntropyLoss for 2 class classification
if training_type == 'bin':
    add_noise = False
    target_data = target_data.long()
    def loss_fn(y_p, y):
        y_pred = y_p - 0.3
        y_pred = y_pred*10
        y_pred = torch.cat((-y_pred, y_pred), dim=1)
        return cross_fn(y_pred, y[:,0]) # cross_fn is defined below, just before training
# L1 norm loss
elif training_type == 'l1':
    loss_fn = torch.nn.L1Loss()
# default mse loss
elif training_type == 'mse' or training_type == None:
    loss_fn = mse_loss_fn
# two-class classification logistic loss
elif training_type=='softmargin':
    loss_fn = torch.nn.SoftMarginLoss()
    target_data -= 0.5
    target_data *= 2.0
    add_noise = False
# combining binary and mse loss
elif training_type=='binmse':
    def loss_fn(y_pred, y):
        # binary
        loss_fn1 = torch.nn.CrossEntropyLoss()
        y_pred_cross = torch.cat((-y_pred, y_pred), dim=1)
        loss_value1 = loss_fn1(y_pred_cross, y[:,0].long())
        # mse
        loss_fn2 = torch.nn.MSELoss()
        loss_value2 = loss_fn2(torch.sigmoid(y_pred), y)
        return 1*loss_value1 + 1*loss_value2
    add_noise = False
# use default loss function
elif training_type=='cor':
    def loss_fn(x, y):
        return cor_loss_fn(x[:,0], y[:,0])
elif training_type=='cormse':
    alpha = 0.8
    def loss_fn(x_in, y_in):
        x = x_in[:,0]
        y = y_in[:,0]
        cor = cor_loss_fn(x, y)
        mse = mse_loss_fn(x, y)
        return alpha*cor+(1-alpha)*mse/10


if add_noise:
    gauss = torch.distributions.Normal(0.0, sigma)
    target_data += gauss.sample((6, 4*N))



# Train each gate
trained_cv = []
losslist = []
for (i,gate) in enumerate(gates):
    print(i, gate)
    web.reset_parameters(cv_reset)
    if training_type == 'bin':
        cross_fn = torch.nn.CrossEntropyLoss(weight = w[i])

    loss_l, best_cv = web.train(input_data, target_data[i].view(-1,1), 
                     beta=beta,
                     batch_size=batch_size,
                     nr_epochs=max_epochs,

                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     bias=bias,
                     scale=scale,
                     #stop_func = stop_func,

                     lr = lr)
    losslist.append(loss_l)
    trained_cv.append(best_cv)


def print_errors():
    plt.figure()
    for i,gate in enumerate(gates):
        plt.subplot(2, 3 , 1 + i//2 + i%2*3)
        plt.plot(losslist[i])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(gate)
    plt.show()
print_errors()


def print_gates():
    plt.figure()
    for i, gate in enumerate(gates):
        print(gate)
    
        web.reset_parameters(trained_cv[i])
        output_data = web.forward(input_data).data 
        loss = web.error_fn(output_data, target_data[i].view(-1,1), beta, loss_fn).item()
        print("loss:", loss)
        
        mseloss = torch.nn.MSELoss()(output_data, target_data[i].view(-1,1).float()).item()
        print("mseloss: ", mseloss)
        
        # print output network and targets
        plt.subplot(2, 3 , 1 + i//2 + i%2*3)
        plt.plot(target_data[i])
        legend_list = ['target']
        if False: #training_type == 'bin':
            plt.plot(torch.sigmoid(output_data))
            legend_list.append('sig(network) '+str(round(loss, 3)))
            plt.plot(torch.round(torch.sigmoid(output_data)))
            legend_list.append('classification')
        else:
            plt.plot(output_data)
            legend_list.append('network '+str(round(loss, 3)))

        #plt.plot(cv_output.data)
        #legend_list.append('cv_output '+str(round(cv_loss, 3)))

        
        plt.legend(legend_list)
#        plt.title("%s, bias=%s, scale=%s" % (gate, round(web.bias.item(),3), round(web.scale.item()+1,3)))
        plt.title("%s, cv:%s" % (gate, np.round(trained_cv[i]['A'].numpy(), 3)))
    # adjust margins
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # fullscreen plot (only available with matplotlib auto)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

print_gates()