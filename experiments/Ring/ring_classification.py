'''
This is a template for evolving the NN based on the boolean_logic experiment. 
The only difference to the measurement scripts are on lines where the device is called.

'''

from matplotlib import pyplot as plt
import ring_evolve as re
import numpy as np
import sys
import config_ring as config
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.niDAQ import nidaqIO

steps = 3

with np.load('/home/Darwin/SkyNEt/experiments/Ring/Class_data_0.20.npz') as data:
    inputs = data['inp_wvfrm'][::steps,:].T
    inputs[0,0:46]=inputs[0,0:46]*0.2
    inputs[1,0:46]=inputs[1,0:46]*0.2
    print('Input shape: ', inputs.shape)
    labels = data['target'][::steps]
    print('Target sgape ', labels.shape)

mask0 = labels==0
mask1 = labels==1
labels[mask0] = 1
labels[mask1] = 0
cf = config.experiment_config(inputs, labels)
target_wave = cf.TargetGen
t, inp_wave, weights = cf.InputGen

print(len(t))
#plt.figure()
#plt.plot(t,inp_wave.T)
#plt.plot(t,target_wave,'k')
#plt.show()
print(len(inp_wave[0]))
Output = nidaqIO.IO_cDAQ9132(inp_wave, 1000)
print(len(Output[0]))
plt.figure()
plt.plot(t, Output[0])
plt.show()