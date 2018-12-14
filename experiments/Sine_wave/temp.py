# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:53:19 2018

@author: crazy
"""

import matplotlib.pyplot as plt
import numpy as np



f = 1
Fs = 2 * f_stop
samples = 2000
x = np.arange(samples)
y = np.sin(2 * np.pi * f * x / Fs)
Period = int(Fs/f)
NPeriod = int(Fs/f)
k = 0
y_sweep = [0]*samples
i = 0
k = 0
for i in range(0, samples):
    if i == 0:
        y_sweep[i] = 0
    while i in range((k*Period)+1, ((k+1)*Period)+1):
        y_sweep[i] = np.sin(2 * np.pi * f * i / Fs)
        i = i+1
        if i == NPeriod:
            k = k+1
            f = 2 * f




plt.figure(1)
plt.subplot(211)   
plt.plot(x,y_sweep)
plt.xlabel('frequency')
plt.ylabel('Magniturd')
#plt.subplot(212)
#plt.plot(x,y)
#plt.xlabel('frequency')
#plt.ylabel('Magniturd')
