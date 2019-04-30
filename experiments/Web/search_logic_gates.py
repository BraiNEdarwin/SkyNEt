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
from SkyNEt.modules.Nets.lightNNet import lightNNet
from SkyNEt.modules.Nets.webNNet import webNNet

# ------------------------ configure ------------------------
# load device simulation

main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\paper_chip\2019_04_27_115357_train_data_2d_f_0_05\\'
data_dir = 'MSE_d10w90_500ep_lr1e-3_b512_b1b2_0.90.75.pt'
net1 = lightNNet(main_dir+data_dir)
input_gates=[3,4]
input_scaling = False

# single device web
web = webNNet()
web.add_vertex(net1, 'A', output=True, input_gates=input_gates)

# input voltages of boolean inputs (on/upper, off/lower)
input_lower = -0.8
input_upper = 0.2

nr_sessions = 10
# hardcoded target values of logic gates with off->lower and on->upper

upper = 1.0
lower = 0.0

# if set to false, use output of known cv configurations as targets

N = 100 # number of data points of one of four input cases, total 4*N

batch_size = 100
max_epochs = 500
lr = 0.06
beta = 10
cv_reset = 'rand'


training_type = 'cor' # options: None, mse, bin, binmse, cor, cormse

add_noise = False # automatically set to false when using bin
sigma = 0.01 # standard deviation of added noise in target


# define custom stopping criteria
def stop_fn(epoch, error_list, best_error):
    # if the error has not improved the last 50 epochs, reset parameters random
    if min(error_list[-150:]) > best_error:
#        print("INFO: No improvement after 50 iterations")
        return True

# ------------------------ END configure ------------------------



# input data for both I0 and I1
input_data = input_lower*torch.ones(N*4,2)

input_data[N:2*N,   1] = input_upper
input_data[2*N:3*N, 0] = input_upper
input_data[3*N:,    0] = input_upper
input_data[3*N:,    1] = input_upper

# target data for all gates
gates = ['AND','NAND','OR','NOR','XOR','XNOR']

if training_type == 'bin':
    lower = 0
    upper = 1

target_data = upper*torch.ones(6, 4*N)
target_data[0, :3*N] = lower
target_data[1, 3*N:] = lower
target_data[2, :N] = lower
target_data[3, N:] = lower
target_data[4, :N] = lower
target_data[4, 3*N:] = lower
target_data[5, N:3*N] = lower

inp_beta = 10
max_inp = torch.from_numpy(net1.info['amplitude'][input_gates] + net1.info['offset'][input_gates]).to(torch.float)
min_inp = torch.from_numpy(-net1.info['amplitude'][input_gates] + net1.info['offset'][input_gates]).to(torch.float)
def reg_input():
    # Define max and min inputs for the input gates       
    return torch.sum(inp_beta*torch.relu(web.scale + web.bias - max_inp) + inp_beta*torch.relu(-web.scale - web.bias + min_inp))

# Scale the input data if desired  
if input_scaling: 
    # rescale the input data to [-1, 1]
    input_data = input_data / torch.max(torch.abs(input_data))
            
    scale = torch.tensor([0.1],dtype=torch.float)      # Start scale at [-0.1,0.1] V
    bias = torch.tensor([0.,0.],dtype=torch.float)    # Start center of data at [0,0] V
    web.add_parameters(['scale','bias'],[scale, bias], reg_input)


if training_type == 'bin':
    w = torch.ones(6, 2)
    for i in range(6):
        temp = torch.sum(target_data[i].float())/4/N
        weights = torch.bincount(target_data[i].long())
        weights = weights[weights != 0].float()
        weights = 1/weights
        weights /= torch.sum(weights)
        w[i] = weights

optimizer = torch.optim.Adam
#def cor_loss_fn(x, y):
#    corr = torch.mean((x-torch.mean(x))*(y-torch.mean(y)))
#    return 1.0-corr/(torch.std(x,unbiased=False)*torch.std(y,unbiased=False)+1e-16)/()

def cor_loss_fn(x, y):
    corr = torch.mean((x-torch.mean(x))*(y-torch.mean(y)))
    x_high_min = torch.min(x[(y == upper)]).item()
    x_low_max = torch.max(x[(y == lower)]).item()
    return (1.1 - (corr/(torch.std(x,unbiased=False)*torch.std(y,unbiased=False)+1e-10)))/(abs(x_high_min-x_low_max)/4)**.5

mse_loss_fn = torch.nn.MSELoss()

def mse_norm_loss_fn(y_pred, y):
    return mse_loss_fn(y_pred, y)/(upper-lower)**2

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
# default mse loss
elif training_type == 'mse' or training_type == None:
    loss_fn = mse_norm_loss_fn
# combining binary and mse loss
elif training_type=='binmse':
    def loss_fn(y_pred, y):
        # binary
        y_pred_cross = torch.cat((-y_pred, y_pred), dim=1)
        loss_value1 = cross_fn(y_pred_cross, y[:,0].long())
        # mse
        loss_value2 = mse_norm_loss_fn(torch.sigmoid(y_pred), y)
        return 1*loss_value1 + 1*loss_value2
    add_noise = False
# use default loss function
elif training_type=='cor':
    def loss_fn(x, y):
        return cor_loss_fn(x[:,0], y[:,0])
elif training_type=='cormse':
    alpha = 0.3
    def loss_fn(x_in, y_in):
        x = x_in[:,0]
        y = y_in[:,0]
        cor = cor_loss_fn(x, y)
        mse = mse_loss_fn(x, y)
        return alpha*cor+(1-alpha)*mse/(upper-lower)**2


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

    loss_l, best_cv, param_history = web.session_train(input_data, target_data[i].view(-1,1), 
                     beta=beta,
                     batch_size=batch_size,
                     max_epochs=max_epochs,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     stop_fn = stop_fn,
                     lr = lr,
                     nr_sessions = nr_sessions,
                     verbose=False)

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

output_data = torch.zeros((N*4,6))
def print_gates():
    plt.figure()
    for i, gate in enumerate(gates):
        print(gate)
    
        web.reset_parameters(trained_cv[i])
        web.forward(input_data)
        output_data[:,i:i+1] = web.get_output()
        
        loss = web.error_fn(output_data[:,i:i+1], target_data[i].view(-1,1), beta).item()
        print("loss:", loss)
        
        mseloss = mse_norm_loss_fn(output_data[:,i:i+1], target_data[i].view(-1,1).float()).item()
        print("mseloss: ", mseloss)
        
        # print output network and targets
        plt.subplot(2, 3 , 1 + i//2 + i%2*3)
        plt.plot(10*target_data[i].numpy())
        legend_list = ['target']
        if False: #training_type == 'bin':
            plt.plot(torch.sigmoid(output_data))
            legend_list.append('sig(network) '+str(round(loss, 3)))
            plt.plot(torch.round(torch.sigmoid(output_data)))
            legend_list.append('classification')
        else:
            plt.plot(10*output_data[:,i:i+1].numpy())
            legend_list.append('network '+str(round(loss, 3)))
        
        plt.legend(legend_list)
        plt.title("%s, cv:%s" % (gate, np.round(trained_cv[i]['A'].numpy(), 3)))
    # adjust margins
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

print_gates()

CV = np.zeros((6,5))
for i in range(len(trained_cv)):
    CV[i] = trained_cv[i]['A'].numpy()

#saveArrays(r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_04_05_172733_characterization_2days_f_0_05_fs_50\nets\MSE_n_adap_200ep\gates\\', filename="results_NAME",max_epochs = max_epochs, nr_sessions=nr_sessions,sigma=sigma,trained_cv=CV,training_type=training_type,upper=upper,lower=lower,lr=lr,input_upper=input_upper,input_lower=input_lower,input_gates=[4,5],pred=output_data,losslist=losslist)