#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:58:43 2019

@author: annefleur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:32:25 2018
This script generates all binary assignments of N elements.
@author: hruiz
"""
import numpy as np
from matplotlib import pyplot as plt

def bintarget(N):
    L = 2**N #length of list, i.e. number of binary targets 
    assignments = []
    list_buf = []
    
    #construct assignments per element i
    print('==='*N)
    print('ALL BINARY LABELS:')
    level = int((L/2))
    while level>=1:
        list_buf = []
        buf0 = [0]*level
        buf1 = [1]*level
        while len(list_buf)<L:
            list_buf += (buf0 + buf1)
        assignments.append(list_buf)
        level = int(level/2)
    
    binary_targets = np.array(assignments).T
    print(binary_targets)
    print('==='*N)
    return binary_targets

if __name__ == '__main__':
    N = 6
    binary = bintarget(N)
    plt.figure()
    plt.imshow(binary)