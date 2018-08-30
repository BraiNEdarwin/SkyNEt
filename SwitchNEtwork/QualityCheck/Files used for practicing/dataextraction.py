'''
This code should be duplicated and be thrown into the same folder as the npz file you want to analyze

Change "choice" in the winning value you want to find, if you choose 1, you find the gold medalist
if you choose 2, you find all the silver medalist

the output information is genes as well as the output current that was obtained from the fitness evaluation

Takes into account of all the generations
'''

import numpy as np
import matplotlib.pyplot as plt
import math

exec(open("setup.txt").read())

#write here the path to npz file
data = np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Dummy_18_05_2018_174716_Test1/DataArrays.npz')
#identify the winning position
choice = 1

f = data.f.fitnessarray
g = data.f.genearray
o = data.f.outputarray

sort = f
#Flatten the two dimensional data into 1 dimensional float
sortarray = np.unravel_index(np.argsort(sort, axis=None), sort.shape)
#sort them in the winning order
yo = np.flip(sort[sortarray],0)
#say we want the second best, choose choice = 2
i = choice - 1
k = 0

#locate which fitness score is the Xth winner
while(i>=0):
    if i ==0:
        value = yo[k]
    a = yo[k]
    if a > yo[k+1]:
        i-=1
    k +=1

#print the gene configuration for that particular winner
for i in range(len(f)):
    for j in range(len(f[i])):
        if f[i][j] == value:
            print(i,j)
            print(g[i][j])
            print(o[i][j])
