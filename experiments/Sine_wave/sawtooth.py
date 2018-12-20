# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:25:42 2018

@author: crazy
"""

import matplotlib.pyplot as plt
import numpy as np

v_low = 0
v_high = 0.1
n_points = 1000
x = np.arange(n_points)
Input = [0]*n_points
n_points = n_points/4
  
Input1 = np.linspace(v_low, v_high, n_points)
    
Input[0:len(Input1)] = Input1
Input[len(Input1):(len(Input1)*2)] = Input1
Input[len(Input1)*2:(len(Input1)*3)] = Input1
Input[len(Input1)*3:(len(Input1)*4)] = Input1

    
plt.plot(x, Input)
plt.show()
    