#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:28:10 2018

@author: hruiz
"""

import time
import numpy as np
import matplotlib.pyplot as plt

dims = 7
Fs = 50
factor = 0.05
skip_points = 2

freq2 = np.array([2,np.pi,5,7,13,17,19])
freq = factor*np.sqrt(freq2[:dims])
inputs = np.zeros((dims,1))
Vmax =  np.array([0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9])

t = np.arange(0, 2*24*3600, skip_points/Fs)
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


gridpoints = 1000
min_dist, empty_counter = distanceRandomGrid(gridpoints, inputs, radius = 0.25, grid_range = Vmax - 0.05)

plt.figure()
plt.hist(min_dist[0,:],bins=50, normed=False, label = "Random grid, n = " + str(gridpoints))
plt.xlabel('Distance (V)')


