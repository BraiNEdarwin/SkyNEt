import time
import numpy as np
import itertools
import modules.PlotBuilder as PlotBuilder
import modules.PostProcess as PostProcess
import modules.SaveLib as SaveLib
from instruments.ADwin import adwinIO
import matplotlib.pyplot as plt



pulseamount = 2
amplification = 0.5
samplefrequency = 1000
plateasupoints = 10
numberinputs = plateasupoints*3


a = [0, 1]
c = [1]
d = [0]
win = np.array(list(itertools.product(*[d,c,d])))
singlepuls = np.zeros(numberinputs)
output = np.zeros([len(win),numberinputs])
axis = np.linspace(0, numberinputs-1, numberinputs)
axis2 =  np.linspace(0, numberinputs, numberinputs*pulseamount)
print(axis2)
print(np.shape(win)[1])

# adwin = adwinIO.initInstrument() 

for i in range(len(win)):
    for j in range(np.shape(win)[1]):
        singlepuls[plateasupoints*j:plateasupoints*j+plateasupoints]= win[i,j]
    scaledsinglepulse = singlepuls
    plt.figure()
    plt.plot(axis, scaledsinglepulse)
    plt.show()

adwininput = np.zeros(pulseamount*len(scaledsinglepulse))
for n in range(pulseamount):
    adwininput[n*len(scaledsinglepulse):n*len(scaledsinglepulse)+len(scaledsinglepulse)] = scaledsinglepulse

plt.figure()
plt.plot(axis2, adwininput)
plt.show()

outputpulsescheme= adwinIO.IO(adwin, scaledadwininput, samplefrequency)

singlevoltage = np.array([0.01])
current = adwin.IO(adwin,singlevoltage,samplefrequency)

np.savetxt('.txt', output)

# f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
# ax1.plot(axis, output[0])
# ax2.plot(axis, output[1]) 
# ax3.plot(axis, output[2])
# ax4.plot(axis, output[3])
# plt.show()

 