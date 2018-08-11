#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 18:32:55 2018

@author: renhori
"""

import numpy as np
import matplotlib.pyplot as plt

test = np.random.rand(8,8)
test = np.round(test)

plt.matshow(test,cmap='gray')
plt.title('NAme of the matrix')