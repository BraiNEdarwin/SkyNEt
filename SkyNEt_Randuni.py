import time
from instruments.DAC import IVVIrack
import modules.SaveLib as SaveLib
from instruments.niDAQ import nidaqIO
import os
# temporary imports
import numpy as np
import random
import datetime


# config
name = 'Distribution'
filepath = 'D:\data\Hans\\'
basepoints = 100
switchpoints = 100
fs = 1000
measurementtime = 7.5 
measurepoints = fs*measurementtime
voltagerange = np.array([900,-900])


data = np.zeros([int(basepoints*2*switchpoints),int(measurepoints)])
CVdata = np.zeros([int(basepoints*2*switchpoints),7])
measurepoints = np.zeros(int(measurepoints))
bpoints = (np.random.rand(basepoints*7)*2*voltagerange[0]+voltagerange[1])
counter = 0
saveDirectory = SaveLib.createSaveDirectory(filepath, name)

ivvi = IVVIrack.initInstrument()

print(datetime.datetime.now())

for i in range(basepoints):

    controlvoltagesbase = bpoints[i*7:i*7+7]
    IVVIrack.setControlVoltages(ivvi, controlvoltagesbase)
    

    output = nidaqIO.IO(measurepoints, fs)
    data[counter] = output
    CVdata[counter] = controlvoltagesbase
    
    counter = counter + 1
    for j in range(switchpoints):

        controlvoltagesswitch = controlvoltagesbase
        electrode = random.randint(0,6)
        controlvoltagesswitch[electrode] = np.random.rand(1)[0]*2*voltagerange[0]+voltagerange[1]
        IVVIrack.setControlVoltages(ivvi, controlvoltagesswitch)
        

        output = nidaqIO.IO(measurepoints, fs)
        data[counter] = output
        CVdata[counter] = controlvoltagesswitch
        counter = counter + 1

        IVVIrack.setControlVoltages(ivvi, controlvoltagesbase)
      

        output = nidaqIO.IO(measurepoints, fs)
        data[counter] = output
        CVdata[counter] = controlvoltagesbase
       
        counter = counter+1
        
        np.savez(os.path.join(saveDirectory, 'nparrays'),data=data, CVdata=CVdata)

print(datetime.datetime.now())
IVVIrack.setControlVoltages(ivvi, np.zeros([16]))
inp = np.zeros((2,20))
nidaqIO.IO_2D(inp, 1000)