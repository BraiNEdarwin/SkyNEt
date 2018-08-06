#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:46:57 2018

@author: hruiz
"""
import numpy as np
from matplotlib import pyplot as plt
min_laps = 150
nr_laps = 7
I4 = np.zeros((nr_laps*min_laps,1))
for i in range(nr_laps):
    I4[(2*i+1)*min_laps:2*(i+1)*min_laps] = 1.0

I3 = np.ones_like(I4)
I2 = np.copy(I3)
I2[:4*min_laps] = 0.0

for i in range(int(nr_laps/2)):
    I3[(2*i)*(2*min_laps):(2*i+1)*(2*min_laps)] = 0.0

plt.figure()
plt.plot(I4,'-o')
plt.plot(I3,'-x')
plt.plot(I2,'-.')


env_states = np.concatenate((I2,I3,I4),axis=1)
env_states = np.tile(env_states,(3,1))

agent = np.zeros((env_states.shape[0],1))
agent[len(I2):2*len(I2)] = 0.4
agent[2*len(I2):] = 0.8

inputs = np.concatenate((agent,env_states),axis=1)

target = np.zeros((inputs.shape[0],1))
target[:len(I2)] = np.copy(I4)
target[agent[:,0]==0.4,:] = np.copy(I3)
target[-(len(I2)+150):-len(I2)] = -1.0
target[-3*min_laps:] = -1.0 

plt.figure()
plt.subplot(5,1,1)
plt.plot(inputs[:,0])
plt.ylabel('agent state')
for i in range(1,inputs.shape[1]):
    plt.subplot(5,1,i+1)
    plt.plot(inputs[:,i])
    plt.ylabel('env. state '+str(i+1))
plt.subplot(5,1,5)
plt.plot(target,'k')
plt.ylabel('target')
