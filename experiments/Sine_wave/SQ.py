# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:34:12 2018

@author: crazy
"""

import matplotlib.pyplot as plt
import numpy as np

v_low = 0
v_high = 1
n_points = 10000
x = np.arange(n_points)
input = [0]*n_points
i = 0

if i in range(0, int((n_points/2)-1)):
    input[i] = v_high
    i = i + 1
if i in range (int(n_points/2), int(n_points-1)):
    input[i] = v_high
    i = i + 1
else:
    input[i] = v_low
        
plt.figure()
plt.plot(x, input)
plt.show()