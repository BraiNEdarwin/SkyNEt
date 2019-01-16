# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:48:00 2019

@author: crazy
"""

import numpy as np 
import matplotlib.pyplot as plt
import math

#load your NPZ file from the right directory
data=np.load('D:\\62_CH11_15V2019_01_10_164848_test\\data.NPZ') 

# shows which names are linked to different parts of your data
list(data.keys())	


#make the different data sets accesible
input = data['input']
output_2d = data['output']
x = np.arange(len(output_2d[0]))

#reverse the data
output_reversed = output_2d[0]
output = output_reversed[::-1]

#here you can plot the desired data using the names above (in this example I wanted to make 25 plots of different data sets to compare)
plt.figure()
plt.plot(input[0],output)
plt.xlabel('Volt [V]')
plt.ylabel('Current [A]')
plt.show()
