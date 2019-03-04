'''
This is a template for evolving the NN based on the boolean_logic experiment. 
The only difference to the measurement scripts are on lines where the device is called.

'''

from matplotlib import pyplot as plt
import ring_evolve as re
import numpy as np
import sys
import config_ring as config

steps = 3

with np.load(r'experiments\Ring\Class_data_0.40.npz') as data:
    inputs = data['inp_wvfrm'][::steps,:].T
    print('Input shape: ', inputs.shape)
    labels = data['target'][::steps]
    print('Target shape ', labels.shape)

mask0 = labels==0
mask1 = labels==1
labels[mask0] = 1
labels[mask1] = 0
cf = config.experiment_config(inputs, labels)
target_wave = cf.TargetGen
t, inp_wave, weights = cf.InputGen
plt.figure()
plt.plot(t,inp_wave.T)
plt.plot(t,target_wave,'k')
plt.show()

best_genome, best_output, max_fitness, accuracy = re.evolve(inputs, labels, filepath = r'D:\\data\\Hans\\Ring_testing\\', hush=False)

best_cv = [g[0] + best_genome[i]*(g[1]-g[0]) for i,g in enumerate(cf.generange)]

try:
    re.reset(0, 0)
except:
    print('Not resetted')