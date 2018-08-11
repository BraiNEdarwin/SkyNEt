#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:25:11 2018

@author: renhori
"""

import numpy as np
import matplotlib.pyplot as plt

data= np.load('D:\RenDrive\SwiNEt_09_08_2018_212532_RenB7devIV77K\IVDataz.npz')
result =data.f.currentlist

b = 7

plt.figure(1)
j = 421
for a in range(len(result[b])):
    plt.subplot(j)
    plt.plot(result[b][a][0], result[b][a][1])
    plt.ylabel('Amp')
    plt.xlabel('Volt')
    plt.grid(True)
    plt.title('E1 to E' + str(a+2))
    j = j + 1

plt.show()

