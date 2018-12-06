#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:28:10 2018

@author: hruiz
"""

import numpy as np
from matplotlib import pyplot as plt

dt = 0.10
t = np.arange(1000)*dt
freq2 = np.array([2,1+np.pi,5,7,13,17,19])
freq = np.sqrt(freq2)
inputs = freq[:,np.newaxis]*t[np.newaxis]
phase = 2*np.pi*np.random.rand(7,1)
inputs = np.sin(inputs+phase)

plt.figure()
n = 1
buf = list(range(7))
for i in range(7):
    buf.remove(i)
    for j in buf:
        plt.subplot(3,7,n)
        plt.plot(inputs[i],inputs[j],'.')
        plt.xlabel('freq '+str(i)+' sqrt('+str(freq2[i])+')')
        plt.ylabel('freq '+str(j)+' sqrt('+str(freq2[j])+')')
#        plt.title('quotient: '+str(freq2[i]/freq2[j]))
        n += 1
plt.tight_layout()
#
#plt.figure()
#plt.hist2d(inputs[5],inputs[2],bins=50)
#plt.colorbar()

def plt_indicator(i,j,bins=50):
    dumf,duma = plt.subplots(1,1)
    counts,_,_,_ = duma.hist2d(inputs[i],inputs[j],bins=bins)
    plt.close()
    indicator = np.zeros_like(counts)
    indicator[counts>0] = 1
    plt.imshow(indicator.T)

plt.figure()
n = 1
buf = list(range(7))
for i in range(7):
    buf.remove(i)
    for j in buf:
        plt.subplot(3,7,n)
        #plt.hist2d(inputs[i],inputs[j],bins=50)
        plt_indicator(i,j)
        plt.xlabel('freq '+str(i)+' sqrt('+str(freq2[i])+')')
        plt.ylabel('freq '+str(j)+' sqrt('+str(freq2[j])+')')
#        plt.title('quotient: '+str(freq2[i]/freq2[j]))
#        plt.colorbar()
        n += 1
        
plt.figure()
n = 1
buf = list(range(7))
for i in range(7):
    buf.remove(i)
    for j in buf:
        plt.subplot(3,7,n)
        plt.hist2d(inputs[i],inputs[j],bins=50)
        plt.xlabel('freq '+str(i)+' sqrt('+str(freq2[i])+')')
        plt.ylabel('freq '+str(j)+' sqrt('+str(freq2[j])+')')
#        plt.title('quotient: '+str(freq2[i]/freq2[j]))
#        plt.colorbar()
        n += 1