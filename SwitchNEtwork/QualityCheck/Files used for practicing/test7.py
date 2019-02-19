#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:41:00 2018

@author: renhori
"""
#This code should upload the full search result, and will generate analysis results, not fully set

#FROM HERE
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')
#from collections import counter

#write here the path to npz file
data = np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Result/Switch/SwiNEt_21_08_2018_145715_EightDeviceFullSearchRealIV77K/DataArrays.npz')

switch = data.f.genearray
output = data.f.outputarray
fitness = data.f.fitnessarray
success = data.f.successarray
#timearray[m,i,0] = conversiontime
#timearray[m,i,1] = arraytime
#timearray[m,i,2] = outputtime
#timearray[m,i,3] = calculatetime
time = data.f.timearray
#TO HERE SHOULD BE INCLUDED IN EVERY ANALYSIS CODE
#This dev should be read from setup file but for now we automatically select
dev = 8

#quickfix
switch = np.resize(switch,(1,15000,8,8))
output = np.resize(output,(1,15000,8,8))
fitness = np.resize(fitness,(1,15000))
success = np.resize(success,(1,15000))
#FROM HERE

#switch is a pretend genearray
#switch = np.round(np.random.rand(20,8,8))
#array2 shows the number of time it was ON
array2 = np.zeros((8,8))
#array3 shows number of ONs per row (per genome ofc)
array3 = np.zeros((15000,8))
#output is a pretend output array
#output = np.random.rand(20,dev,dev)
transoutput = np.zeros((15000,dev,dev))
#array5and6 are used to help finding the identity matrix
array5 = np.zeros((15000,dev,dev))


#================================================================#
#This part will tell you the population relative to the fitness score
swag = np.transpose(fitness)

#THIS PART DOESNT WORK IM SORRY BUT FOR NOW PLEASE TYPE THE MAX AND MIN NUMBER MANUALLY BY FINDING OUT THE RANGE OF SUCCESS
#minimum = int(round(float(min(swag))/5.0)*5.0)
#maximum = int(round(float(max(swag))/5.0)*5.0)
#swaglord =np.linspace(minimum,maximum,int(1+(maximum-minimum)/5)).tolist()

#fitrange = np.linspace(0,51)
plt.figure()
plt.hist(swag, align = 'left', rwidth=0.6, normed = False)
plt.ylabel('Population', size = 17)
plt.xlabel('Fitness Score', size = 17)
#plt.xticks(range(5), size = 13)
#plt.xticks(range(20))
plt.show()

#This part will print out the population of each success number
unique_elements, counts_elements = np.unique(swag, return_counts = True)
print("Frequency of unique values of the fitness array:")
print(np.asarray((unique_elements, counts_elements)))


#End
#================================================================#

print(fitness[0][3723])

#np.savetxt('/Users/renhori/Desktop/Twente/Year2/Thesis/Result/Switch/SwiNEt_21_08_2018_145715_EightDeviceFullSearchRealIV77K/DataArrays.npz', fitness)