import time
import numpy as np
import itertools
import modules.PlotBuilder as PlotBuilder
import modules.PostProcess as PostProcess
import modules.SaveLib as SaveLib
from instruments.ADwin import adwinIO
import matplotlib.pyplot as plt


amplification = 0.5
numberinputs = 100
samplefrequency = 1000
plateasupoints = 10


a = [0, 1]
c = [1]
win = np.array(list(itertools.product(*[a,a,c])))
print(win)
adwininput = np.zeros(numberinputs)
output = np.zeros([len(win),numberinputs])
axis = np.linspace(0, numberinputs-1, numberinputs)
print(np.shape(win)[1])

# adwin = adwinIO.initInstrument() 

for i in range(len(win)):
    for j in range(np.shape(win)[1]):
        adwininput[plateasupoints*j:plateasupoints*j+plateasupoints]= win[i,j]
    scaledadwininput = adwininput*amplification
    print(scaledadwininput)
    plt.figure()
    plt.plot(axis, scaledadwininput)
    plt.show()
    # output[i] = adwinIO.IO(adwin, scaledadwininput, samplefrequency)

np.savetxt('.txt', output)

# f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
# ax1.plot(axis, output[0])
# ax2.plot(axis, output[1]) 
# ax3.plot(axis, output[2])
# ax4.plot(axis, output[3])
# plt.show()

