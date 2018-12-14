# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:34:12 2018

@author: crazy
"""

import matplotlib.pyplot as plt
import numpy as np

v_low = 0
v_high = 0.1
n_points = 10000
position = 'high'

i = 0
if position == 'high':
    if i in range(0, int((n_points/2)-1)):
        input = v_high
        i = i+1
    if i in range (int(n_points/2), int(n_points-1)):
        input = 'v_low'
        i = i+1
    else:
        input = 'v_low'