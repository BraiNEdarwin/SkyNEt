#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:28:10 2018

@author: hruiz
"""

import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

dims = 7
Fs = 50
factor = 0.05
skip_points = 2

freq2 = np.array([2,np.pi,5,7,13,17,19])
freq = factor*np.sqrt(freq2[:dims])
c_update = int(24*3600*freq[0])
cycles = int(24*3600*freq[0])
inputs = np.zeros((dims,1))
Vmax =  np.array([0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9])

t = np.arange(0, 2*24*3600, skip_points/Fs)
for i in range(int(cycles/c_update)):
    input_part = freq[:,np.newaxis]*t[np.newaxis]
    phase = np.zeros((dims,1))
    input_part = Vmax[:,np.newaxis] * np.sin(2*np.pi*input_part+phase)
    inputs = np.concatenate((inputs, input_part),axis=1)
inputs = inputs[:,1:]

font = {'font.size': 12}
plt.rcParams.update(font)
"""
plt.figure()
n = 1
buf = list(range(dims))
for i in range(dims):
    buf.remove(i)
    for j in buf:
        plt.subplot(2,dims-1,n)
        plt.plot(inputs[i],inputs[j],linewidth = .5)
        plt.xlabel('freq '+str(i)+' sqrt('+format(freq2[i],'.2f')+')')
        plt.ylabel('freq '+str(j)+' sqrt('+format(freq2[j],'.2f')+')')
#        plt.title('quotient: '+str(freq2[i]/freq2[j]))
        n += 1
plt.tight_layout()

#
#plt.figure()
#plt.hist2d(inputs[5],inputs[2],bins=50)
#plt.colorbar()

def plt_indicator(i,j,bins=500):
    dumf,duma = plt.subplots(1,1)
    counts,_,_,test = duma.hist2d(inputs[i],inputs[j],bins=bins)
    plt.close()
    indicator = np.zeros_like(counts)
    indicator[counts>0] = 1
    plt.imshow(indicator.T)

plt.figure()
n = 1
buf = list(range(dims))
for i in range(dims):
    buf.remove(i)
    for j in buf:
        plt.subplot(3,dims,n)
        #plt.hist2d(inputs[i],inputs[j],bins=50)
        plt_indicator(i,j)
        plt.xlabel('freq '+str(i)+' sqrt('+str(freq2[i])+')')
        plt.ylabel('freq '+str(j)+' sqrt('+str(freq2[j])+')')
#        plt.title('quotient: '+str(freq2[i]/freq2[j]))
#        plt.colorbar()
        n += 1
        
plt.figure()
n = 1
buf = list(range(dims))

for i in range(dims):
    buf.remove(i)
    for j in buf:
        plt.subplot(3,dims,n)
        h = plt.hist2d(inputs[i],inputs[j],bins=50)
        plt.colorbar(h[3])
        plt.xlabel('freq '+str(i)+' sqrt('+str(freq2[i])+')')
        plt.ylabel('freq '+str(j)+' sqrt('+str(freq2[j])+')')
#        plt.title('quotient: '+str(freq2[i]/freq2[j]))
#        plt.colorbar()
        n += 1
      
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = ax.plot(inputs[0,::10],inputs[1,::10],inputs[2,::10]) #, s=1, c=inputs[3,::100]
plt.xlabel('dim 0')
plt.ylabel('dim 1')
#plt.colorbar(sp)
"""

# Determine the smallest distance to the sampled area w.r.t. a uniform grid
radius = 0.15
grid_step = 0.05
grid_range = Vmax - grid_step
#grid = np.mgrid[-grid_range:grid_range:grid_step,
#                -grid_range:grid_range:grid_step,
#                -grid_range:grid_range:grid_step]
                #-grid_range:grid_range:grid_step]
                #-grid_range:grid_range:grid_step]

def distance5D_brute(grid, inputs):
    min_dist = np.zeros(grid.shape)
    min_dist = min_dist[0,:]
    start_block = time.time()
    empty_counter = 0
    for i in range(grid.shape[1]):  
        for j in range(grid.shape[2]):
            for k in range(grid.shape[3]):
                for l in range(grid.shape[4]): 
                    for m in range(grid.shape[5]):
                        dist = np.linalg.norm(inputs - grid[:, i, j, k, l, m ,np.newaxis], axis = 0)
                        min_dist[i, j, k, l, m] = dist.min()    
                    
    end_block = time.time()
    print("time elapsed brute force: " + str(end_block - start_block))
    return min_dist, empty_counter

def distance4D(grid, inputs, radius):
    min_dist = np.zeros(grid.shape)
    min_dist = min_dist[0,:]
    start_block = time.time()
    empty_counter = 0
    for i in range(grid.shape[1]):  
        for j in range(grid.shape[2]):
            for k in range(grid.shape[3]):
                for l in range(grid.shape[4]):
                    # Filter out data dimension per dimension
                    filtered_inputs = inputs[:, np.sum(abs(inputs - grid[:,i,j,k,l,np.newaxis]) <= radius, axis=0) == dims] 
                    if filtered_inputs.size == 0:
                        empty_counter +=1
                    else:                    
                        dist = np.linalg.norm(filtered_inputs - grid[:, i, j, k, l, np.newaxis], axis = 0)
                        min_dist[i,j,k,l] = dist.min()    
                    #dist = np.linalg.norm(inputs - grid[:, i, j, k, l, np.newaxis], axis = 0)
                    #min_dist[i,j,k,l] = dist.min()    
                    
    end_block = time.time()
    print("time elapsed: " + str(end_block - start_block))
    return min_dist, empty_counter



def distance_test(grid, inputs, radius):
    start_block = time.time()
    grid_2d = np.reshape(grid,(dims,grid[0,:].size))
    min_dist = np.zeros((1,grid_2d.shape[1]))
    empty_counter = 0
    for i in range(grid_2d.shape[1]): 
        filtered_inputs = inputs[:, abs(inputs[0,:] - grid_2d[0,i,np.newaxis]) <= radius]
        for j in range(1,dims):
            filtered_inputs = filtered_inputs[:, abs(filtered_inputs[j,:] - grid_2d[j,i,np.newaxis]) <= radius] 
            
        if filtered_inputs.size == 0:
            empty_counter +=1
        else:                    
            dist = np.linalg.norm(filtered_inputs - grid_2d[:,i,np.newaxis], axis = 0)
            min_dist[0,i] = dist.min()    
        #dist = np.linalg.norm(inputs - grid[:, i, j, k, l, np.newaxis], axis = 0)
        #min_dist[i,j,k,l] = dist.min()    
                    
    end_block = time.time()
    print("time elapsed: " + str(end_block - start_block))
    return min_dist, empty_counter

def distanceRandomGrid(gridpoints, inputs, radius, grid_range):
    start_block = time.time()
    #grid = np.random.uniform(-grid_range, grid_range, (inputs.shape[0], gridpoints))
    grid = Vmax[:,np.newaxis] * np.random.uniform(-1, 1, (inputs.shape[0], gridpoints))
    
    min_dist = np.zeros((1, gridpoints))
    empty_counter = 0
    for i in range(gridpoints): 
        filtered_inputs = inputs[:, abs(inputs[0,:] - grid[0,i,np.newaxis]) <= radius]
        for j in range(1,dims):
            filtered_inputs = filtered_inputs[:, abs(filtered_inputs[j,:] - grid[j,i,np.newaxis]) <= radius] 
            
        if filtered_inputs.size == 0:
            empty_counter +=1
        else:                    
            dist = np.linalg.norm(filtered_inputs - grid[:,i,np.newaxis], axis = 0)
            min_dist[0,i] = dist.min()    
        #dist = np.linalg.norm(inputs - grid[:, i, j, k, l, np.newaxis], axis = 0)
        #min_dist[i,j,k,l] = dist.min()    
                    
    end_block = time.time()
    print("time elapsed: " + str(end_block - start_block))
    return min_dist, empty_counter

def distanceRandomGridBrute(gridpoints, inputs, grid_range = 0.25):
    start_block = time.time()
    grid = np.random.uniform(-grid_range, grid_range, (inputs.shape[0], gridpoints))
    min_dist = np.zeros((1, gridpoints))
    empty_counter = 0
    for i in range(gridpoints): 
        dist = np.linalg.norm(inputs - grid[:, i, np.newaxis], axis = 0)
        min_dist[0,i] = dist.min()                     
    end_block = time.time()
    print("time elapsed brute force random grid: " + str(end_block - start_block))
    return min_dist, empty_counter


gridpoints = 10000
#min_dist_brute, empty_counter_brute = distance5D_brute(grid, inputs)
min_dist, empty_counter = distanceRandomGrid(gridpoints, inputs, radius = 0.25, grid_range = Vmax - 0.05)

plt.figure()
#plt.hist(np.reshape(min_dist_brute, (min_dist_brute.size,1)),bins=50, normed = True,label = "Full grid search")
plt.hist(min_dist[0,:],bins=50, normed=False, label = "Random grid, n = " + str(gridpoints))
#plt.legend()
plt.xlabel('Distance (V)')
#plt.ylabel('Sampled points')
#plt.title(str(dims) +"D, " + str(cycles) + " cycles")

# Computation test: 5D, 10 cycles, radius = 0.2
# Brute force: 1875s
# optimized test: 2728s

# random grid, 350 cycles, 10k gridpoints
# random grid: 65s
# random grid brute: 172s

##########################
## Convergence of random grid.
## amount of gridpoints given up to the point where random grid gives same histogram as uniform grid.
## cycles: 200
## factor: 1
## grid_range: 0.85, grid_step: 0.05
## radius: 0.4
##########################
# 3D: 10k
# 4D: 10k
# 5D, 100 cycles, factor = 1, Fs = 50Hz: 10k for near perfect

# 7D 10k cycles, 10k gridpoints, 50 Hz, random grid brute comp time: 658s
# 7D 10k cycles, 10k gridpoints, 50 Hz, radius = 0.6, random grid comp time: 518s
# 7D 10k cycles, 1k gridpoints, 50 Hz, radius = 0.5, random grid comp time: 40s
# 7D 100k cycles, 1k gridpoints, 50 Hz, radius = 0.5, random grid comp time: 449s
# 7D 500k cycles, 1k gridpoints, 50Hz, radius = 0.25, random grid comp time: 1213s

# 7D 1M cycles, 1k gridpoints, 25Hz, radius = 0.25, rand grid comp time: 1230s
# 7D 1M cycles, 10k gridpoints, 25Hz, radius = 0.25, rand grid comp time: 12410s
# 7D 500k cycles, 10k gridpoints, 25Hz, radius = 0.2, rand grid comp time: 12410s
# 7D 2M cycles, 10k gridpoints, 25Hz, radius = 0.25, rand grid comp time: 26156s