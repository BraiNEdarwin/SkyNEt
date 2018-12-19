# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 17:28:09 2018

@author: crazy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:34:12 2018

@author: crazy
"""

import matplotlib.pyplot as plt
import numpy as np

v_low = 0
v_high = 0.1
v_high1 = 0.2
n_points = 1000
x = np.arange(n_points)
signal = [0]*n_points
i = 0
for i in range(0, n_points):
    for i in range(0, 250):
        signal[i] = v_high
    for i in range(251, 500):
        signal[i] = v_low
    for i in range(501, 750):
        signal[i] = v_high1;
    for i in range(751, 1000):
        signal[i] = v_low
        
plt.figure()
plt.plot(x, signal)
plt.show()