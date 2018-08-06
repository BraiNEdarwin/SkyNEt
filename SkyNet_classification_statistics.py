''''
Measurement script to perform an evolution experiment of a selected
gate. This will initially be tested on the Heliox (with nidaq) setup.
'''

# Import packages
import modules.ReservoirFull as Reservoir
import modules.PlotBuilder as PlotBuilder
import modules.GenerateInput as GenerateInput
import modules.Evolution as Evolution
import modules.PostProcess as PostProcess
import modules.SaveLib as SaveLib
from instruments.niDAQ import nidaqIO
from instruments.DAC import IVVIrack
import time

# temporary imports
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

exec(open("config.txt").read())

a = [0, 1]
b = [-1, 1, 0]
win = np.array(list(itertools.product(*[b,a,a,a])))
# print(win)

# win = np.delete(win,23,axis = 0)
# win = np.delete(win,15,axis = 0)
# win = np.delete(win,7,axis = 0)

c = len(win)

wtest = 100
fs = 1000
vin= 1500.0

y = np.zeros(1000)

winout = np.zeros([c,wtest])
winoutsd = np.zeros([2,c])

# CV = np.array([0,0,0,0,-8.52494202e+02, -9.06303617e+02,  6.74293612e+01])
# inputScaling =  np.array([1.00000000e+00,8.82277001e-01,  2.35952653e-01,  7.45336604e-01])

# CV = np.array([0,0,0,0,-8.45118912e+02, -9.01618798e+02,  1.49925643e+02])
# inputScaling =  np.array([1.00000000e+00, 8.82277001e-01,  2.24716813e-01,  7.84564847e-01])

# CV = np.array([0,0,0,0,-1319.44,-582.86,-365.72])
# inputScaling = np.array([0.95])

# CV = np.array([-1040.03176, -1250.11642, 755.219511])
# inputScaling = np.array([0.924523497])

# CV = nparray([-1.29694157e+03, -5.50238107e+02, -4.24061608e+02])
# inputScaling = np.array([1.00000000e+00])

CV = np.array([-1.29694157e+03,-5.02750012e+02,-4.24061608e+02])
inputScaling = np.array([1.00000000e+00])



# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(filepath, name)

# initialize instruments
ivvi = IVVIrack.initInstrument()
IVVIrack.setControlVoltages(ivvi,CV)
for i in range(4):
    win[:,i] = win[:,i] * inputScaling*vin

print("Setting up is done")
for m in range(c):
    print('')
    print("We are at " + str(m) + " out of 24")
    inputVoltages = win[m]
    IVVIrack.setControlVoltages(ivvi,inputVoltages)
    time.sleep(0.2)
    print("loading")
    for n in range(wtest):
        if n%int(wtest/10) == 0:
            print('.', end = '', flush = True)
        measureddata = np.asarray(nidaqIO.IO(y, fs)) * 10
        winout[m,n] = np.average(measureddata)
IVVIrack.setControlVoltages(ivvi,np.zeros([10]))
saveDirectory = SaveLib.createSaveDirectory(filepath,name)
np.savez(os.path.join(saveDirectory, 'nparrays'), outputArray=winout, win=win,CV = CV, inputScaling = inputScaling)

plt.figure()
y = np.zeros(len(winout))
for i in range(len(winout)):
    y[i] = np.average(winout[i])
    a = plt.plot(winout[i],np.arange(0,len(winout[i])),dashes = [1,1], label = win[i])
    plt.plot(winout[i],np.arange(0,len(winout[i])),'x',color = a[0].get_color())
    plt.plot([y[i],y[i]],[0,len(winout[0])],dashes = [5,5],color = a[0].get_color())
plt.legend()
plt.pause(0.01)
plt.show()