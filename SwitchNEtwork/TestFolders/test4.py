#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:25:11 2018

@author: renhori
"""

import numpy as np
import matplotlib.pyplot as plt

voltrange = []
Vabs = 1.5
Vstep = 0.005
steps = Vabs/Vstep + 1
steps = int(steps)

a = np.zeros((4,1,steps))

#Set of voltage range is appended to the voltagerange
first =(np.linspace(0,-Vabs, steps))
a[0] = first
second= (np.linspace(-Vabs, 0,steps))
a[1] = second
third= (np.linspace(0, Vabs, steps))
a[2] = third
fourth = (np.linspace(Vabs, 0, steps))
a[3] = fourth
for b in range(4):
    for c in range(steps):
        voltrange.append(a[b][0][c])
        
test = np.random.rand(8,7,2,1204)
test = np.round(test)

for a in range(len(test)):
    for b in range(len(test[a])):
        i = -1.5
        for c in range(len(test[a][b][0])):
            test[a][b][0][c] = voltrange[c]
            test[a][b][1][c] = i
            i = i+ 0.005




for a in range(len(test)):
    for b in range(len(test[a])):
        plt.figure()
        plt.plot(test[a][b][0], test[a][b][1])