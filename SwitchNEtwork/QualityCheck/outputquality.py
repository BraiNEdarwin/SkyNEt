#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:06:55 2018

@author: renhori
"""

import numpy as np
import matplotlib.pyplot as plt
import math

exec(open("setup.txt").read())

#write here the path to npz file
data = np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Dummy/test_22_05_2018_142449_Test1/DataArrays.npz')

#Use dataextractipm tp find the genome

i = 0
j = 937

o = data.f.outputarray

Outputresult = o[i][j]
TransOutputresult = np.transpose(Outputresult)
threshold = 0
thresholdlist= []
Flist = []
Separatelist = []

while(threshold < 1):
    F = 0
    dist = 0
    #Tolerance. if set 0.5, it considers any output that has more than 50% of the highest current as "non-distinguishable"
    #Criteria 1
    for a in range(len(Outputresult)):
        count = 0
        tempout = Outputresult[a]
        maxi = max(tempout)
        for b in range(len(Outputresult[a])):
            #If the read current is higher than the threshold, add 1 to the count
            if Outputresult[a,b]/maxi > threshold:
                count = count + 1
        #if only one output was HIGH for the given input, that's success!
        if count == 1:
            F = F + 3
            dist = dist + 1
        #if more than 1 output was HIGH for the given input, we give -1 for the number of outputs that were HIGH
        elif count > 1:
            F = F+ -1*count
        #if no output was HIGH for a particular input, we either have to lower the threshold, or just punish the fitness score
        elif count == 0:
            F = F - 10
    #Criteria 2
    #Do exactly the same but transposed matrix. Vertical check
    for a in range(len(Outputresult)):
        count = 0
        tempout = TransOutputresult[a]
        maxi = max(tempout)
        for b in range(len(Outputresult[a])):
            #If the read current is higher than the threshold, add 1 to the count
            if TransOutputresult[a,b]/maxi > threshold:
                count = count + 1
        #if only one output was HIGH for the given input, that's success!
        if count == 1:
            F = F + 3
            dist = dist + 1
            #if more than 1 output was HIGH for the given input, we give -1 for the number of outputs that were HIGH
        elif count > 1:
            F = F+ -1*count
        #if no output was HIGH for a particular input, we either have to lower the threshold, or just punish the fitness score
        elif count == 0:
            F = F - 10
    Flist.append(F)
    Separatelist.append(dist)
    thresholdlist.append(threshold)
    threshold +=0.01
print(threshold)

fig = plt.figure()
plt.rc('grid', linestyle=":", color = 'black')
plt.grid(True)
ax1 = fig.add_subplot(111)
ax1.plot(thresholdlist, Flist)
ax1.set_ylabel('Fitness Score')

ax2 = ax1.twinx()
ax2.plot(thresholdlist, Separatelist, 'r-')
ax2.set_ylabel('Number of distinction', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')

plt.show()