# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:54:59 2019

@author: Mark Boon
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.lightNNet import lightNNet
from SkyNEt.modules.Nets.webNNet import webNNet
from SkyNEt.modules.SaveLib import saveArrays

# ------------------------ configure ------------------------
# load device simulation

main_dir = r'filepath'
data_dir = 'checkpoint3000_02-07-23h47m.pt'

net1 = staNNet(main_dir+data_dir)
input_gates=[3,4]
input_scaling = True
# single device web
web = webNNet()
web.add_vertex(net1, 'A', output=True, input_gates=input_gates)

nr_sessions = 20
upper = 10.
lower = 0.

batch_size = 100
max_epochs = 800
lr = 0.9
beta = 0.5
cv_reset = 'rand'

training_type = 'cormse' # options: None, mse, bin, binmse, cor, cormse
add_noise = False # automatically set to false when using bin
sigma = -.5 # standard deviation of added noise in target


# define custom stopping criteria
def stop_fn(epoch, error_list, best_error):
    # if the error has not improved the last 50 epochs, reset parameters random
    if min(error_list[-200:]) > best_error:
#        print("INFO: No improvement after 50 iterations")
        return True

# ------------------------ END configure ------------------------

# Load ring data 
ring_file = r'ring_data_path'
input_data = torch.from_numpy(np.load(ring_file)['inp_wvfrm']).to(torch.float)
target_data = torch.from_numpy((np.load(ring_file)['target'] - 1) * (-1) * upper).to(torch.float)[np.newaxis,:] # need to be [1,many] because Boolean logic finder is such that its dims are [# gates, labels]

# Parameters used for regularizing the input
inp_beta = 0.5
max_inp = torch.from_numpy(net1.info['amplitude'][input_gates] + net1.info['offset'][input_gates]).to(torch.float)
min_inp = torch.from_numpy(-net1.info['amplitude'][input_gates] + net1.info['offset'][input_gates]).to(torch.float)
# Regularize the input scaling and bias
def reg_input():
    # Define max and min inputs for the input gates       
    return torch.sum(inp_beta*torch.relu(abs(web.scale) + web.bias - max_inp) + inp_beta*torch.relu( -(-abs(web.scale) + web.bias - min_inp) ))

# Scale the input data if desired  
if input_scaling: 
    # rescale the input data to [-1, 1]
    input_data = input_data / torch.max(torch.abs(input_data))
            
    scale = torch.tensor([0.8],dtype=torch.float)      # Start scale at [-0.1,0.1] V
    bias = torch.tensor([-0.2,-0.2],dtype=torch.float)    # Start center of data at [0,0] V
    web.add_parameters(['scale','bias'],[scale, bias], reg_input)

optimizer = torch.optim.Adam

def cor_loss_fn(x, y):
    corr = torch.mean((x-torch.mean(x))*(y-torch.mean(y)))/ (torch.std(x,unbiased=False)*torch.std(y,unbiased=False)+1e-10)
    return (1.0001 - corr)

def cor_adap_loss_fn(x, y):
    corr = torch.mean((x-torch.mean(x))*(y-torch.mean(y)))/(torch.std(x,unbiased=False)*torch.std(y,unbiased=False)+1e-10)
    x_high_min = torch.min(x[(y >= upper/3)]) #.item()
    x_low_max = torch.max(x[(y <= lower/3)]) #.item()
    return (1.1 - corr)/ torch.sigmoid((x_high_min - x_low_max - 5)/3) 

mse_loss_fn = torch.nn.MSELoss()

def mse_norm_loss_fn(y_pred, y):
    return mse_loss_fn(y_pred, y)/(upper-lower)**2

# ------------- Different training types -------------

# default mse loss
if training_type == 'mse' or training_type == None:
    loss_fn = mse_norm_loss_fn
# use default loss function
elif training_type=='cor':
    def loss_fn(x, y):
        return cor_loss_fn(x[:,0], y[:,0])
elif training_type=='cor_adap':
    def loss_fn(x, y):
        return cor_adap_loss_fn(x[:,0], y[:,0])
elif training_type=='cormse':
    alpha = 0.3
    def loss_fn(x_in, y_in):
        x = x_in[:,0]
        y = y_in[:,0]
        cor = cor_loss_fn(x, y)
        mse = mse_loss_fn(x, y)
        return alpha*cor+(1-alpha)*mse/(upper-lower) # **2


if add_noise:
    gauss = torch.distributions.Normal(0.0, sigma)
    target_data += gauss.sample((1, target_data.shape[1]))



# Train each gate
trained_cv = []
losslist = []

web.reset_parameters(cv_reset)

loss_l, best_cv, param_history = web.session_train(input_data, target_data[0].view(-1,1), 
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
    plt.plot(losslist[0])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Ring')
    plt.show()
print_errors()

output_data = torch.zeros((target_data.shape[1], 1))

def print_output():
    plt.figure()  
    web.reset_parameters(trained_cv[0])
    web.forward(input_data)
    output_data = web.get_output()
    
    loss = web.error_fn(output_data, target_data.view(-1,1), beta).item()
    print("loss:", loss)
    
    mseloss = mse_norm_loss_fn(output_data, target_data.view(-1,1).float()).item()
    print("mseloss: ", mseloss)
    
    # print output network and targets
    plt.plot(target_data[0].numpy())
    legend_list = ['target']
    if False: #training_type == 'bin':
        plt.plot(torch.sigmoid(output_data))
        legend_list.append('sig(network) '+str(round(loss, 3)))
        plt.plot(torch.round(torch.sigmoid(output_data)))
        legend_list.append('classification')
    else:
        plt.plot(output_data.numpy())
        legend_list.append('network '+str(round(loss, 3)))
    
    plt.legend(legend_list)
    plt.title("%s, cv:%s" % ('Ring', np.round(trained_cv[0]['A'].numpy(), 3)))
    # adjust margins
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

print_output()

# Change relevant input data into numpy arrays
CV = best_cv['A'].numpy()[np.newaxis,:]
bias = best_cv['bias'].numpy()
scale = best_cv['scale'].numpy()

#saveArrays(r'filepath', filename="results_ring",max_epochs = max_epochs, nr_sessions=nr_sessions,sigma=sigma,CV=CV,bias=bias,scale=scale,training_type=training_type,lr=lr,t_upper=upper,t_lower=lower,input_gates=input_gates,pred=output_data,losslist=losslist,data_dir=data_dir,main_dir=main_dir)

# Plot input data
plt.figure()
if input_scaling:
    plt.plot( (input_data[:,0]*web.scale.item()+web.bias[0].item()).numpy(), (input_data[:,1]*web.scale.item()+web.bias[1].item()).numpy(),'.')
else:
    plt.plot(input_data[:,0].numpy(), input_data[:,1].numpy, '.')
