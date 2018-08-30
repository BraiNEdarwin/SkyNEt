

#FROM HERE
import numpy as np
import matplotlib.pyplot as plt
import math

#write here the path to npz file
data1 = np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Result/Switch/SwiNEt_15_08_2018_145350_RenB7devIV77K(3DevFSREAL)/DataArrays.npz')

switch1 = data1.f.genearray
output1 = data1.f.outputarray
fitness1 = data1.f.fitnessarray
success1 = data1.f.successarray

data2 = np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Result/Switch/BeforeFix/ThreeDevFullSearchv2/DataArrays.npz')

switch2 = data2.f.genearray
output2 = data2.f.outputarray
fitness2 = data2.f.fitnessarray
success2 = data2.f.successarray

newswitch1= np.zeros((1,3000,3,8))
newswitch2= np.zeros((1,999,3,8))
newswitch3= np.zeros((1,3000,8,3))
newswitch4= np.zeros((1,999,8,3))
#print(switch1[0][5])
#print(switch2[0][5])
#print(switch1[0][5][:,2])

for a in range(len(switch2[0])):
    switch2[0][a][0]=[0,0,0,0,0,0,0,0]
    switch2[0][a][4]=[0,0,0,0,0,0,0,0]
for a in range(len(switch1[0])):
	newswitch1[0][a][0] = switch1[0][a][:,0]
	newswitch1[0][a][1] = switch1[0][a][:,2]
	newswitch1[0][a][2] = switch1[0][a][:,5]	

for a in range(len(switch2[0])):
	newswitch2[0][a][0] = switch2[0][a][:,2]
	newswitch2[0][a][1] = switch2[0][a][:,4]
	newswitch2[0][a][2] = switch2[0][a][:,6]	
    
for n in range(len(newswitch1[0])):
    newswitch3[0][n] = np.transpose(newswitch1[0][n])    
for n in range(len(newswitch2[0])):
    newswitch4[0][n] = np.transpose(newswitch2[0][n])
#These are actual effective lines

newswitch6 = np.copy(newswitch4)

targetlist = []
for a in range(len(newswitch4[0])):
    flag = False
    for b in range(len(newswitch4[0][a])):
        tally = 0
        for c in range(len(newswitch4[0][a][b])):
            if newswitch4[0][a][b][c] == 1:
                tally = tally + 1
        if tally == 1:
            flag = True
            newswitch6[0][a][b] = [0,0,0]
    if flag == True:
        print("Genome " + str(a+1))
        targetlist.append(a)

#Now we have newswitch6, which is removed sole 1 from new4

for a in range(len(newswitch3[0])):
    for b in range(len(newswitch6[0][b])):
        if np.array_equal(newswitch3[0][a],newswitch6[0][b]):
            print("Genome " + str(a+1) + "from 3. AND Genome " + str(b+1)+ " matches")
