# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:53:19 2018

@author: crazy
"""

import matplotlib.pyplot as plt
import numpy as np

N_Samples = 2000
Fs = 4000
frequency = 100
Period = int(Fs/frequency)
x = np.arange(N_Samples)
NPeriod = Period
z = x[0:Period]
frequency = 100
y = np.sin(2*np.pi*x*frequency/Fs)
y_new = [0]*N_Samples

k=0
i=0
N_frequency = frequency
for i in range(k*Period, (k+1)*Period):
    y_new[i] = np.sin(2*np.pi*z[i]*N_frequency/Fs)
    if i == Nperiod:
        k = k + 1
        N_frequency = N_frequency + frequency



plt.figure(1)
plt.subplot(211)   
plt.plot(x,y)
plt.xlabel('frequency')
plt.ylabel('Magniturd')
plt.subplot(212)
plt.plot(x,y_new)
plt.xlabel('frequency')
plt.ylabel('Magniturd')
