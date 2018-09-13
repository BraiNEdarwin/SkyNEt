import time
# from instruments.DAC import IVVIrack
# import modules.SaveLib as SaveLib
# from instruments.niDAQ import nidaqIO
# temporary imports
import numpy as np
import random

# config
name
filepath
basepoints = 20
switchpoints = 50
fs = 1000
measurementtime = 7.5 
measurepoints = fs*measurementtime
voltagerange = np.array([900,-900])


data = np.zeros([2020,measurepoints])
measurepoints = np.zeros(int(measurepoints))
bpoints = (np.random.rand(basepoints*7)*2*voltagerange[0]+voltagerange[1])
counter = 0
saveDirectory = SaveLib.createSaveDirectory(filepath, name)

# ivvi = IVVIrack.initInstrument()

for i in range(basepoints):

    controlvoltagesbase = bpoints[i*7:i*7+7]
    IVVIrack.setControlVoltages(ivvi, controlVoltagesbase)
    time.sleep(1)

    output = nidaqIO.IO(measurepoints, fs)
    data[counter] = output
    counter = counter + 1
    for j in range(switchpoints):

        controlvatagesswitch = controlvoltagesbase
        electrode = random.randint(0,6)
        controlvatagesswitch[electrode] = np.random.rand(1)[0]*2*voltagerange[0]+voltagerange[1]
        IVVIrack.setControlVoltages(ivvi, controlVoltagesswitch)
        time.sleep(1)

        output = nidaqIO.IO(measurepoints, fs)
        data[counter] = output
        counter = counter + 1

        IVVIrack.setControlVoltages(ivvi, controlVoltagesbase)
        time.sleep(1)

        output = nidaqIO.IO(measurepoints, fs)
        data[counter] = output
        counter = counter+1

SaveLib.saveArrays(filepath,data)