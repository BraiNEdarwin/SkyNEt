#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:02:01 2019

@author: ljknoll

generate large list of 7 input points and evaluate them with the device
calculate mean and std of output data
These values can used for the transfer function

IMPORTANT: Results are dumped in working directory!

"""
#%%
import torch
import time
import pickle
import os
cwd = os.getcwd()

from SkyNEt.modules.Nets.staNNet import staNNet


main_dir = r'C:/users/lennart/Dropbox/afstuderen/search_scripts/'

# network import
#network_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
#network_dir = 'NN_skip3_MSE.pt'
network_dir = 'MSE_d5w90_500ep_lr1e-3_b2048_b1b2_0.90.75-11-05-21h48m.pt'
net = staNNet(main_dir+network_dir)

def generate_grid_inputs(net, nsteps, minima, maxima):
    """
    
    CAUTION: do not make nsteps large, it will fill up your RAM
    
    Generates multidimensional grid of input data points
    Returns:
        torch tensor (n, m) with all n combinations of m values, where n = nsteps^m and m = input dimensions
    Example:
        For a network with 7 input dimensions, and resolution of 5 points, you need 5^7=78125 grid points
    """
    dims = minima.shape[0]
    inputs = torch.zeros(nsteps**dims, dims)
    for i in range(dims):
        linspace = torch.linspace(minima[i], maxima[i], nsteps)
        inputs[:,i] = torch.repeat_interleave(linspace, nsteps**i).repeat(nsteps**(dims-1-i))
    return inputs



minima = net.info['offset']-net.info['amplitude']
maxima = net.info['offset']+net.info['amplitude']
nsteps = 10
#input_data = generate_grid_inputs(net, nsteps, minima, maxima)
input_data = torch.rand(nsteps**7, 7)*(torch.tensor(maxima).float()-torch.tensor(minima).float()) + torch.tensor(minima).float()


start = time.time()
with torch.no_grad():
    output_data = net.outputs(input_data, grad=True)

end = time.time()-start
print('Took %.3f seconds' % end)

filename = network_dir[:-3]+'_random.p'
with open(filename, 'wb') as f:
    pickle.dump(output_data, f)
print('output saved in '+cwd+'\\'+filename)




#%%

import pickle
import torch
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt


filename = 'MSE_d5w90_500ep_lr3e-3_b2048_largeRange_random.p'
#filename = 'MSE_d5w90_500ep_lr1e-3_b2048_b1b2_0.90.75-11-05-21h48m_random.p'

with open(filename, 'rb') as f:
    output_data = pickle.load(f)


large_mean = torch.mean(output_data)
large_median = torch.median(output_data)
large_std = torch.std(output_data)
large_min = torch.min(output_data)
large_max = torch.max(output_data)

print('Mean: %.3f nA' % large_mean.item())
print('Median: %.3f nA' % large_median.item())
print('Std: %.3f nA' % large_std.item())
print('Min: %.3f nA' % large_min.item())
print('Max: %.3f nA' % large_max.item())

def func(x, a, b):
    return 1/(1+np.exp(-1*(a*x-b)))


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

plt.figure()
temp = plt.hist(output_data.numpy(), bins=1000, density=True, color='black')
plt.title('Histogram of 1e7 random samples')
plt.xlabel('Output current (nA)')
plt.ylabel('Normalized bincount')



# try to fit sigmoid function on output distribution, but it is not gaussian
if False:
    hist_array = temp[0]
    x_array = temp[1]
    x = x_array[:-1]

    # manually test fitness functions    
    plt.plot(x, func(x, 2/15, 0), label='15') # chosen this one
    plt.plot(x, func(x, 2/20, 0), label='20')
    plt.plot(x, func(x, 2/25, 0), label='25')
    
    cumsum = 0.
    integral = np.zeros(hist_array.shape)
    for i,v in enumerate(hist_array):
        cumsum += v
        integral[i] = cumsum
    # normalize
    integral /= np.max(integral)
    
    popt, pcov = curve_fit(func, x, integral, bounds=([-10, 0], [10,1]))
    
    # 95% interval:
    larger_ind = np.argmax(integral>0.95)
    smaller_ind = np.argmax(integral>0.05)
    
    x1 = x[smaller_ind]
    x2 = x[larger_ind]
    
    c1 = 0.05
    c2 = 0.95
    z1 = 1/c1-1
    z2 = 1/c2-1
    
    a = np.log(z2/z1)/(x1-x2)
    b = a*x1+np.log(z1)
    
    #log_hist = np.log(hist_array+1)
    
    
    plt.figure()
    plt.plot(x, integral, label='integral')
    plt.plot(x, func(x, a, b), label='5% interval')
    plt.plot(x, func(x, *popt), 'r-', label='fit')
    plt.plot(x, hist_array/np.max(hist_array), label='log hist')
    plt.legend()
    