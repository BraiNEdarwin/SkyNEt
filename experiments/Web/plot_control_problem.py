#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:30:15 2019

Script to plot the results of the cp solutions loaded with read_control_problem.py

@author: ljknoll
"""

device = 'v2'
#mu = 232.151/2
#std = 304.418*2
beta = 5
n = 100 # integer used to call generate_cp

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet

def plot_param_hist(results, index):
    # generate names for legend
    names = []
    for key,value in results['best'][index].items():
        names = names + [key+str(i) for i in range(len(value))]
    
    hist = results['history'][index][:len(results['error'][index])]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(hist.numpy())
    plt.xlabel('epoch')
    plt.ylabel('param value')
    plt.legend(names)

    plt.subplot(2,1,2)     
    plt.plot(results['error'][index])
    plt.ylabel('CrossEntropyLoss')
    plt.xlabel('epoch')
    plt.show()

# results need to be read from search_scripts/control_sparsity/read_script.py
# history:initial_parameters, best:best parameters of epochs, error:error, class:classification
assert sum(i in results.keys() for i in ['history', 'best', 'error', 'class']) == 4, "results need to be loaded correctly"

# calculate best errors:
best_errors = []
for errors in results['error']:
     best_errors.append(min(errors))

# collect information of best run
best_index = np.argmin(best_errors)
best_subindex = np.argmin(results['error'][best_index])
best_error_cv = results['history'][best_index][best_subindex]

best_class_index = np.argmax(results['class'])
best_class_subindex = np.argmin(results['error'][best_class_index])
best_class_cv = results['history'][best_class_index][best_class_subindex]

best_class_indices= [i for i,e in enumerate(results['class']) if e==1.0]

if True:
    # plot error and parameter propogation of best run
    plot_param_hist(results, best_index)
    
    # histogram of errors
    plt.figure()
    plt.hist(best_errors)
    plt.title('histogram of error values')
    
    # histogram of classification rates
    plt.figure()
    plt.hist(results['class'], bins=100)
    plt.title('histogram of classification rates')

# calculate average error and classification rate:
print("INFO: Average loss: %0.3f with std %0.3f" % (np.mean(best_errors), np.std(best_errors)))
perc = sum(np.array(results['class'])==1.0)/len(results['class'])*100
print("INFO: percentage 100%% classification rate: %0.3f" % perc)
print("INFO: Average classification rate: %0.3f with std %0.3f" % (np.mean(results['class']), np.std(results['class'])))


# ------------------------------- START PLOT OUTPUT --------------------------------------
main_dir = r'/home/lennart/Dropbox/afstuderen/search_scripts/'

# network import
if device == 'v0':
    network_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
elif device == 'v1':
    network_dir = 'NN_skip3_MSE.pt'
elif device == 'v2':
    network_dir = 'MSE_d5w90_500ep_lr1e-3_b2048_b1b2_0.90.75-11-05-21h48m.pt'
elif device == 'v3':
    network_dir = 'MSE_d5w90_500ep_lr3e-3_b2048_largeRange.pt'
else:
    assert False, 'incorrect device'
net = staNNet(main_dir+network_dir)

# function to generate train data for control problem
def generate_cp(n=10, mean_I0=-0.2, mean_I1=-0.2, amp_I0=0.9, amp_I1=0.9):
     values_I0 = [mean_I0-amp_I0+amp_I0*2/2*(i//n//7) for i in range(21*n)]
     values_I1 = [mean_I1-amp_I1+amp_I1*2/6*(i//n%7) for i in range(21*n)]
     input_data = torch.tensor([values_I0, values_I1]).t()
     targets = [0,0,0,1,1,1,1,0,1,1,1,1,2,2,1,1,2,1,2,1,2]
     target_data = torch.tensor([targets]).view(-1,1).repeat(1,n).view(-1,1)
     return input_data, target_data



input_data, target_data = generate_cp(n)

#def cust_sig(x):
#     return torch.sigmoid((x-mu)/std*2)
def cust_sig(x):
     return torch.sigmoid(x/15*2)
 
web = webNNet()
web.transfer = cust_sig
web.add_vertex(net, 'A', output=True, input_gates=[1,2])
#web.add_vertex(net, 'B', input_gates=[1,2])
#web.add_vertex(net, 'C', input_gates=[1,2])
#web.add_vertex(net, 'D', input_gates=[1,2])
#web.add_vertex(net, 'E', input_gates=[1,2])
#web.add_vertex(net, 'F', input_gates=[4,5])
#web.add_vertex(net, 'G', input_gates=[4,5])
#web.add_vertex(net, 'H', input_gates=[4,5])

#web.add_arc('B', 'A', 1)
#web.add_arc('C', 'A', 2)
#web.add_arc('D', 'A', 3)
#web.add_arc('E', 'A', 4)
#web.add_arc('F', 'A', 4)
#web.add_arc('G', 'A', 5)
#web.add_arc('H', 'A', 6)
#input_data = torch.cat((torch.zeros(input_data.shape[0], 2),input_data), dim=1)
#input_data = torch.cat((input_data,)*2, dim=1)

web.add_parameters(['scale', 'bias'],
                   [torch.ones(2), torch.tensor([0., 1.])],
#                   lambda : 0.0001*torch.mean(web.scale**2 + web.bias**2)/2,
                   lr=0.1, betas=(0.9,0.99))

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

del web.loss_fn
web.loss_fn = loss_fn

def plot_output(best_cv, fontsize = 20, extratitle=''):
    web.reset_parameters(best_cv)
    d0 = -web.bias[0]
    d2 = -web.bias[1]
    web.forward(input_data)
    output_data = web.get_output()
    error = web.error_fn(output_data, target_data, beta)
    print('crossentropyloss + regularization: %0.3f' % error)
        
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
    f = plt.figure()
    plt.rcParams.update({'font.size': fontsize})
    plt.subplot(gs[0])
    plt.title('Web output and its classification. '+extratitle)
    plt.plot(output_data.numpy(), 'k')
    plt.plot((0, len(input_data)), (d0, d0), '--k', alpha=0.3)
    plt.plot((0, len(input_data)), (d2, d2), '--k', alpha=0.3)
    plt.ylabel('Output current (nA)')
    my_xticks = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    plt.xticks(np.arange(n//2, 21*n+n//2, n), my_xticks*3)
    
    plt.subplot(gs[1])
    classification = classify(output_data)
    plt.plot(target_data.float().numpy()+0.05, ':k')
    plt.plot(classification.numpy(), 'k')
#    my_xticks = ['E', 'B', 'F', 'A', 'D', 'C', 'G'] 
# previously correct labels, but now cases have redefined to 
# D, B, F, E, A, C, G
    my_xticks = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    my_yticks = ['down', 'stay', 'up']
    plt.xticks(np.arange(n//2, 21*n+n//2, n), my_xticks*3)
    plt.yticks(np.arange(0,3,1), my_yticks)
    plt.legend(['target', 'classification'])
#    plt.show()
    print('classification rate: %0.3f' % torch.mean((target_data==classification).float()))
    
    return f
plot_output(best_error_cv, extratitle='best error')
plot_output(best_class_cv, extratitle='best classification')

# ------------------------------- END PLOT OUTPUT --------------------------------------


#temp_error = []
#temp_class = []
#for temp_cv in results['history'][best_index]:
#    web.reset_parameters(temp_cv)
#    output_data = web.forward(input_data)
#    error = web.error_fn(output_data, target_data, beta)
#    classification = classify(output_data)
#    temp_error.append(error)
#    temp_class.append(torch.mean((target_data==classification).float()))
#
#plt.figure()
#plt.plot(temp_error)
#plt.plot(temp_class)


best_indices = np.where(np.array(results['class'])==1.0)[0]
separations = []
for c, i in enumerate(best_indices):
    best_error_index = np.argmin(results['error'][i])
    temp_cv = results['history'][i][best_error_index]
    
    web.reset_parameters(temp_cv)
    d0 = -web.bias[0]
    d2 = -web.bias[1]
    web.forward(input_data)
    output_data = web.get_output()
    print('best parameters %i/%i:' %(c+1, len(best_indices)))
#    print(web.get_parameters())
    
    if d0>d2:
        output_data = -output_data
    max_zeros = torch.max(output_data[np.where(target_data==0)[0]]).item()
    ones = output_data[np.where(target_data==1)[0]]
    min_twos = torch.min(output_data[np.where(target_data==2)[0]]).item()
    print('max seprations: (0/1, %0.3f), (1/2, %0.3f)' % (torch.min(ones).item()-max_zeros, min_twos-torch.max(ones).item()))
    separations.append([torch.min(ones).item()-max_zeros, min_twos-torch.max(ones).item()])
    
#    f = plot_output(temp_cv, fontsize=8.5)
#    f.savefig("images/"+str(i)+".pdf", bbox_inches='tight')
#    plt.close(f)

if len(best_indices)>0:
    best = 0.0
    best_i = 0
    for i,v in enumerate(separations):
        # only select on separation on class1, 
        # because the switching is harder and class0 separation is always higher
        if v[1]>best:
            best = v[1]
            best_i = i
    best_subindex = np.argmin(results['error'][best_indices[best_i]])
    plot_output(results['history'][best_indices[best_i]][best_subindex])
else:
    print("No solution found, so cannot print solutions")
