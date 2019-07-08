#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:16:27 2019

@author: Darwin
"""
def generate_cp(n=47, mean_I0=-0.3, mean_I1=-0.3, amp_I0=0.9, amp_I1=0.9):
     values_I0 = [mean_I0-amp_I0+amp_I0*2/2*(i//n//7) for i in range(21*n)]
     values_I1 = [mean_I1-amp_I1+amp_I1*2/6*(i//n%7) for i in range(21*n)]
     input_data = np.array([[values_I0],[values_I1]])
     targets = np.array([0,0,0,1,1,1,1,0,1,1,1,1,2,2,1,1,2,1,2,1,2])
     target_data = np.zeros((2100))
     for i in range(len(targets)):
         target_data[i*100:i*100+100] = np.ones(100)*targets[i]
     return input_data, target_data

import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
import numpy as np
import os
import config_pseudoinverse as config

# Load the information from the config class.
config = config.experiment_config()

# Initialize save directory.
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)
# Define the device input using the function in the config class.
#Input = config.pseudoInput(config.vpoints, config.vc, config.numberinputs, config.points_per_input, config.slopesize)
Data = generate_cp()
input1 = np.asarray(Data[0][0][0])
input2 = np.asarray(Data[0][1][0])
print(len(input1))
print(input2)
Input = np.zeros((2,len(input1)))
Input[0] = input1
Input[1] = input2
print(Input)
print(Input[0])
print(Input[1])
#plt.figure()
#plt.plot(Data[1])
#plt.plot(Input[0])
#plt.plot(Input[1])
#plt.show()


#make yteach
#random outputs
#Output=np.random.rand(6,config.n_points)
#for i in range(6):
#    Output[i,:] = np.linspace(i/6,1-i/6,config.n_points)

# Measure using the device specified in the config class.
if config.device == 'nidaq':
    Output = InstrumentImporter.nidaqIO.IO_cDAQ9132(Input, config.fs)
elif config.device == 'adwin':
    adwin = InstrumentImporter.adwinIO.initInstrument()
    Output = InstrumentImporter.adwinIO.IO(adwin, Input, config.fs)
else:
    print('specify measurement device')
   
#print output for fun
plt.figure()
plt.plot(Input[0,:], '--', label='input 1')
plt.plot(Input[1,:], '--', label='input 2')
plt.legend()
plt.title('Boolean measurement 1')
plt.xlabel('time (ms)')
plt.ylabel('current (10nA)')
plt.figure()
plt.plot(Output[0,:], label ='output 1')
plt.plot(Output[1,:], label ='output 2')
plt.plot(Output[2,:], label ='output 3')
plt.plot(Output[3,:], label ='output 4')
plt.plot(Output[4,:], label ='output 5')
plt.plot(Output[5,:], label ='output 6')
plt.legend()
plt.title('Boolean measurement 1')
plt.xlabel('time (ms)')
plt.ylabel('current (10nA)')
plt.show()

#calculate average out for each input 00,01,10,11 all hardcoded babyyy
#avgout=np.zeros((Output.shape[0],config.vc))
#for i in range(avgout.shape[0]):
#    for j in range(avgout.shape[1]):
#        avgout[i,j]=np.average(Output[i,j*config.points_per_input:(j+1)*config.points_per_input-config.slopesize])
#print('avgout')
#print(avgout)
#
##calculate the pseudoinverse of avgout
#pseudoinverse = np.linalg.pinv(avgout)
#print('pseudoinverse')
#print(pseudoinverse.shape)
#print(pseudoinverse)
##pseudoinversecheck
#checker = np.dot(pseudoinverse,avgout)
#print('unity check')
#print(checker.shape)
#print(checker)
##calculate the weight matrix for yteach
#weights = np.dot(yteach,pseudoinverse)
#print('weights')
#print(weights.shape)
#print(weights)
#
#
#y=np.dot(weights,avgout)
#print('y')
#print(y.shape)
#print(y)
##calculate fitness
#fitness = np.zeros(y.shape[0])
#
#
##for i in range(1,y.shape[0]-1):
##    trueval = []
##    falseval = []
##    for j in range(y.shape[1]):
##        if yteach[i][j] !=0:
##            trueval.append(y[i,j])
##        else:
##            falseval.append(y[i,j])
##    #check for correct gate
##    if (not trueval or not falseval):
##        fitness[i]=0
##    else:
##        fitness[i]= min(trueval)-max(falseval)
##    
#print('fitness')
#print(fitness)
#
#InstrumentImporter.reset(0, 0)

