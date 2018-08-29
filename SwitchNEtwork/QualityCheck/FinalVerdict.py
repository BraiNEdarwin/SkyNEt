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
data = np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Result/Switch/SwiNEt_22_08_2018_140242_FINALEightDeviceFullSearchRealDigitIV77K/DataArrays.npz')

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

#TO HERE WILL BE USED FOR RANDOM RESULT TO DEBUG THE PROCESSING WRITEUPS


#================================================================#
#This part of the code will generate an array with number of the times it was ON
#a because it still has generations
for a in range(len(switch)):
    #per genome
    for b in range(len(switch[a])):
        #per row
        for c in range(len(switch[a][b])):
            for d in range(len(switch[a][b][c])):
                if switch[a][b][c][d] == 1:
                    array2[c][d] = array2[c][d] + 1
#End
#================================================================#


#================================================================#
#This part of the code will generate an array for the number of ONs per row
for a in range(len(switch)):
    for b in range(len(switch[a])):
        for c in range(len(switch[a][b])):
            for d in range(len(switch[a][b][c])):
                if switch[a][b][c][d] == 1:
                    array3[b][c] = array3[b][c] + 1
arrayyes = np.transpose(array3)
#End
#================================================================#


#================================================================#
#This part for plotting the number of ONs in entire genome
plt.figure()
fig, ax1 = plt.subplots(1,1)
ax1.imshow(array2, cmap= 'winter')
ax1.set_xticklabels(['Device', 1,2,3,4,5,6,7,8])
ax1.set_yticklabels(['Electrode', 5,6,7,8,1,2,3,4])
plt.xlabel('Device')
plt.ylabel('Electrode')
#plt.imshow(array2, cmap = 'winter')
k=0
for i in range(len(array2)):
    for j in range(len(array2[i])):
        temparray = array2.astype(int)
        temparray = np.reshape(array2,64)
        temparray = temparray.astype(int)
        text = plt.text(j, i, temparray[k], ha="center", va="center", color="k")
        k= k+1
plt.show()
#End
#================================================================#


#================================================================#
#This part for plotting the 8 histogram data comparing the number of shared connections and occurance per layers
fig = plt.figure()

#axes = fig.subplots(8,1, sharex = True)
fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)

#Add labels
plt.xlabel("# of shared connections")
plt.ylabel("Occurance")

for n in range(4):
    #plt.subplot(810+i)
    axes[n].hist(arrayyes[n], bins=range(0,10) ,align = 'left', rwidth=0.95, normed = False)
    axes[n].set_title('Intermediate layer ' + str(n+1))

plt.tight_layout()
plt.show()

fig = plt.figure()

#axes = fig.subplots(8,1, sharex = True)
fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)

#Add labels
plt.xlabel("# of shared connections")
plt.ylabel("Occurance")

for n in range(4):
    #plt.subplot(810+i)
    axes[n].hist(arrayyes[n+4], bins=range(0,10) ,align = 'left', rwidth=0.95, normed = False)
    axes[n].set_title('Intermediate layer ' + str(n+5))

plt.tight_layout()
plt.show()
##End
#================================================================#


#================================================================#
#This part will find number of output array that corresponded to identity matrix
#First add 1 to the row-wise highest value
for a in range(len(output)):
    for b in range(len(output[a])):
        for c in range(len(output[a][b])):
            d = np.argmax(output[a][b][c])
            array5[b][c][d] = array5[b][c][d] + 1
        
#now transpose and add 2 to the row-wise(originally column wise) highest value
for n in range(len(output[0])):
    transoutput[n] = np.transpose(output[0][n])
for a in range(len(transoutput)):
    for b in range(len(transoutput[a])):
        c = np.argmax(transoutput[a][b])
        array5[a][c][b] = array5[a][c][b]+2
#The processed array5 will have 1 for row highest, 2 for column highest, 3 when together

zen = 0
checklist = []
for a in range(len(array5)):
    counter = 0
    for b in range(len(array5[a])):
        if array5[a][b][b]==3 :
            counter = counter + 1
    if counter == 8:
        print("Genome " + str(a+1) + " is an identity matrix")
        checklist.append(a)
        zen = zen + 1
        
if zen ==0:
    print("swag")


#End
#================================================================#


#================================================================#
#This part will plot the timing analysis
timenew = np.zeros((1,4,7000))

for a in range(len(timenew[0])):
    for b in range(len(timenew[0][a])):
        timenew[0][a][b] = time[0][b][a]
plt.figure()
i = np.linspace(1,7000,7000)
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

plt.tight_layout()
plt.show()

#End
#================================================================#

#================================================================#
#This part will just find out how successful the FS results were in terms of classifying
swiggity = np.transpose(success)
plt.figure()
plt.hist(swiggity, bins=np.arange((9)-0.5) ,align = 'left', rwidth=0.5, normed = False)
plt.ylabel('Population')
plt.xlabel('Successrate out of ' + str(dev))
plt.xticks(range(10))
plt.show()

unique_elements, counts_elements = np.unique(swiggity, return_counts = True)
print("Frequency of unique values of the success array:")
print(np.asarray((unique_elements, counts_elements)))

#End
#================================================================#


#================================================================#
#This part will tell you the population relative to the fitness score
swag = np.transpose(fitness)

#THIS PART DOESNT WORK IM SORRY BUT FOR NOW PLEASE TYPE THE MAX AND MIN NUMBER MANUALLY BY FINDING OUT THE RANGE OF SUCCESS
#minimum = int(round(float(min(swag))/5.0)*5.0)
#maximum = int(round(float(max(swag))/5.0)*5.0)
#swaglord =np.linspace(minimum,maximum,int(1+(maximum-minimum)/5)).tolist()
fitrange = np.linspace(0,51)
plt.figure()
plt.hist(swag ,bins = np.arange((45)-0.5), align = 'left', rwidth=0.5, normed = False)
plt.ylabel('Population', size = 17)
plt.xlabel('Fitness Score', size = 17)
plt.xticks(range(49), size = 13)
#plt.xticks(range(20))
plt.show()

#This part will print out the population of each success number
unique_elements, counts_elements = np.unique(swag, return_counts = True)
print("Frequency of unique values of the fitness array:")
print(np.asarray((unique_elements, counts_elements)))


#End
#================================================================#


#================================================================#
#This part will list the successful evaluation
winnerlist = []
for a in range(len(success[0])):
    if success[0][a] == 8:
        print('Genome ' + str(a+1) + 'is a winner. With a fitness score of ' + str(fitness[0][a]))
        winnerlist.append(a)
        #print(switch[0][a])
        #print(output[0][a])
#End
#================================================================#


#================================================================#
#this snippet is to compare winnerlist and checklist, please comment out unless u rly need it
thesislist =[]
for a in range(len(winnerlist)):
    counter = 0
    for b in range(len(checklist)):
        if winnerlist[a] == checklist[b]:
            counter = 1
    if counter == 0:
       print("Genome" + str(winnerlist[a]+1) + "is non identity but high success genome")
       thesislist.append(winnerlist[a])


#================================================================#


#================================================================#
#This part will list the fitness evaluation
fitlist = []
for a in range(len(fitness[0])):
    if fitness[0][a] == 3*2*dev:
        #print('Genome ' + str(a+1) + 'is a fitness winner. With a fitness score of ' + str(fitness[0][a]))
        fitlist.append(a)
        #print(switch[0][a])
        #print(output[0][a])
#End
#================================================================#


#================================================================#
#This part will make a gradual change in the tolerance value, and see the change in the successrate

success = 0
tolerance = 0.8
#Tolerance. if set 0.5, it considers any output that has more than 50% of the highest current as "non-distinguishable"
threshold = tolerance
#list of tolerance values
tolist = np.linspace(0.01,0.99,num = 99)
graphlist = np.linspace(0,1, num = 11)
#new fitness array for 10 different tolerance values
fitrange = np.zeros((99,15000))
successrange = np.zeros((99,15000))
for i in range(len(fitrange)):
    threshold = tolist[i]
    for j in range(len(fitrange[i])):
        F = 0
        for k in range(len(output[0][j])):
            count = 0
            tempout = output[0][j][k]
            maxi = max(tempout)
            for l in range(len(output[0][j][k])):
                if output[0][j][k][l]/maxi >threshold:
                    count = count + 1
            if count ==1:
                F = F + 3
            if count >1:
                F = F + -1*count
        
        Transoutput = np.copy(output[0][j])
        Transoutput = np.transpose(Transoutput)
        for m in range(len(Transoutput)):
            count = 0
            tempout = Transoutput[m]
            maxi = max(tempout)
            for n in range(len(Transoutput[m])):
                if Transoutput[m,n]/maxi > threshold:
                    count = count + 1
            if count ==1:
                F = F + 3
            if count >1:
                F = F + -1*count
        fitrange[i][j] = F
        #This part is not complete because success does not currently involve the tolerance value
        successful = 0
        for o in range(len(output[0][j])):
            tempout = output[0][j][o]
            maxi = max(tempout)
            for p in range(len(output[0][j][o])):
                if output[0][j][o][p] == maxi:
                    tempx = p
                    tempy = o
            tempout = Transoutput[tempx]
            maxi = max(tempout)
            if Transoutput[tempx][tempy] == maxi:
                successful = successful + 1
        successrange[i][j] = successful
'''
'''
plt.figure()
plt.imshow(fitrange, cmap= 'summer', aspect = 'auto')
#ax1.set_xticklabels(['Device', 1,2,3,4,5,6,7,8])
#ax1.set_yticklabels(['Electrode', 5,6,7,8,1,2,3,4])
plt.xlabel('Genomes')
plt.ylabel('Tolerance value')
plt.yticks(np.linspace(0,99,num = 11),[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
cb = plt.colorbar()
cb.set_label("Fitness score")
plt.show()



#winner1 = fitrange[:,359]
#winner2 = fitrange[:,1116]
'''
plt.figure()
for a in range(len(winnerlist)):
    plt.plot(fitrange[:,a], label =str(winnerlist[a]))
plt.xticks(np.linspace(0,99,num = 11),[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.legend(prop = fontP, loc = 2)
plt.xlabel('Tolerance value')
plt.ylabel('Fitness score')
plt.show()
'''
#End
#================================================================#

'''
#================================================================#
#Ignore here, this is for dealing with thesislist
for a in range(len(thesislist)):
    print(switch[0][thesislist[a]])
    print(output[0][thesislist[a]])
    print(fitness[0][thesislist[a]])
    print(success[0][thesislist[a]])



#================================================================#
'''