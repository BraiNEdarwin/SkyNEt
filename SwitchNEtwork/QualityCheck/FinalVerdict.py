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

#write here the path to npz file
data = np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Result/Switch/SixDevFS/DataArrays.npz')

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
dev = 6

#FROM HERE
#array1 is a pretend genearray
array1 = np.round(np.random.rand(20,8,8))
#array2 shows the number of time it was ON
array2 = np.zeros((8,8))
#array3 shows number of ONs per row (per genome ofc)
array3 = np.zeros((20,8))
#array4 is a pretend output array
array4 = np.random.rand(20,dev,dev)
transarray4 = np.zeros((20,dev,dev))
#array5and6 are used to help finding the identity matrix
array5 = np.zeros((20,dev,dev))
#TO HERE WILL BE USED FOR RANDOM RESULT TO DEBUG THE PROCESSING WRITEUPS


#================================================================#
#This part of the code will generate an array with number of the times it was ON
for a in range(len(array1)):
    for b in range(len(array1[a])):
        for c in range(len(array1[a][b])):
            if array1[a][b][c] == 1:
                array2[b][c] = array2[b][c] + 1
#End
#================================================================#


#================================================================#
#This part of the code will generate an array for the number of ONs per row
for a in range(len(array1)):
    for b in range(len(array1[a])):
        for c in range(len(array1[a][b])):
            if array1[a][b][c] == 1:
                array3[a][b] = array3[a][b] + 1
arrayyes = np.transpose(array3)
#End
#================================================================#


#================================================================#
#This part for plotting the number of ONs in entire genome
plt.figure()
plt.imshow(array2, cmap = 'winter')
k=0
for i in range(len(array2)):
    for j in range(len(array2[i])):
        temparray = array2.astype(int)
        temparray = np.reshape(array2,64)
        temparray = temparray.astype(int)
        text = plt.text(j, i, temparray[k], ha="center", va="center", color="w")
        k= k+1
plt.show()
#End
#================================================================#


#================================================================#
#This part for plotting the 8 histogram data comparing the number of shared connections and occurance per layers
fig = plt.figure()

#axes = fig.subplots(8,1, sharex = True)
fig, axes = plt.subplots(8, 1, sharex=True, sharey=True)
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)

#Add labels
plt.xlabel("# of shared connections")
plt.ylabel("Occurance")

for n in range(len(arrayyes)):
    #plt.subplot(810+i)
    axes[n].hist(arrayyes[n], bins=range(0,10) ,align = 'left', rwidth=0.95, normed = False)
    axes[n].set_title('Intermediate layer ' + str(n+1))
plt.tight_layout()
plt.show()
##End
#================================================================#


#================================================================#
#This part will find number of output array that corresponded to identity matrix
#First add 1 to the row-wise highest value
for a in range(len(array4)):
    for b in range(len(array4[a])):
        c = np.argmax(array4[a][b])
        array5[a][b][c] = array5[a][b][c]+1
        
#now transpose and add 2 to the row-wise(originally column wise) highest value
for n in range(len(array4)):
    transarray4[n] = np.transpose(array4[n])
for a in range(len(transarray4)):
    for b in range(len(transarray4[a])):
        c = np.argmax(transarray4[a][b])
        array5[a][c][b] = array5[a][c][b]+2
#The processed array5 will have 1 for row highest, 2 for column highest, 3 when together

for a in range(len(array5)):
    counter = 0
    zen = 0
    for b in range(len(array5[a])):
        if array5[a][b][b]==3 :
            counter = counter + 1
    if counter == dev:
        print("Genome " + str(a+1) + " is an identity matrix")
        zen = zen + 1
        
if zen ==0:
    print("swag")


#End
#================================================================#


#================================================================#
#This part will plot the timing analysis
timenew = np.zeros((1,4,3000))

for a in range(len(timenew[0])):
	for b in range(len(timenew[0][a])):
		timenew[0][a][b] = time[0][b][a]

i = np.linspace(1,3000,3000)
plt.subplot(221)
plt.plot(i,timenew[0][0])
plt.ylabel('Time (s)')
plt.xlabel('Genomes')
plt.grid(True)
plt.title('Conversion Time')

plt.subplot(222)
plt.plot(i,timenew[0][1])
plt.ylabel('Time (s)')
plt.xlabel('Genomes')
plt.grid(True)
plt.title('Array Time')

plt.subplot(223)
plt.plot(i,timenew[0][2])
plt.ylabel('Time (s)')
plt.xlabel('Genomes')
plt.grid(True)
plt.title('Output evaluation Time')

plt.subplot(224)
plt.plot(i,timenew[0][3])
plt.ylabel('Time (s)')
plt.xlabel('Genomes')
plt.grid(True)
plt.title('Fitness calculation Time')

plt.show()

#End
#================================================================#

#================================================================#
#This part will just find out how successful the FS results were in terms of classifying
swiggity = np.transpose(success)
plt.figure()
plt.hist(swiggity, bins=range(0,dev+2) ,align = 'left', rwidth=0.95, normed = False)
plt.ylabel('Population')
plt.xlabel('Successrate out of ' + str(dev))
plt.show()

#End
#================================================================#


#================================================================#
#This part will tell you the population relative to the fitness score
swag = np.transpose(fitness)
minimum = int(round(float(min(swag))/5.0)*5.0)
maximum = int(round(float(max(swag))/5.0)*5.0)
swaglord =np.linspace(minimum,maximum,int(1+(maximum-minimum)/5)).tolist()
plt.figure()
plt.hist(swag, bins = swaglord ,align = 'left', rwidth=0.95, normed = False)
plt.ylabel('Population')
plt.xlabel('Fitness Score')
plt.show()

#int(round(b/5.0)*5.0)

#End
#================================================================#


