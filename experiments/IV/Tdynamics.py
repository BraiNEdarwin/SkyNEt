import time
import matplotlib.pyplot as plt
from instruments.niDAQ import nidaqIO
from instruments.ADwin import adwinIO
from instruments.DAC import IVVIrack
import numpy as np
import time

Sourcegain = 1
Igain = 1			#use to make output in nA

Fs = 10000 						#change sample frequency
filepath = 'D:/data/Bram/IV'			
name = 'IVtest.txt'
instrument = 1  #choose between nidaq (1) and adwin (0)


# Generate the disred input sequence

inputs = np.array([[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0]])
inputsc = 0.8
lens = 1000
lene = 500
lent = 10000
vb = 600
ivvi = IVVIrack.initInstrument()
IVVIrack.setControlVoltages(ivvi, np.array([vb]))
time.sleep(1)  # Wait after setting DACs
outsave = np.zeros([int((np.shape(inputs)[1]-1)*lens+(np.shape(inputs)[1]-1)*lene+lent-1),int(len(inputs)*2)])
ic = 0
for ii in inputs:
    x = np.array([])
    for i in range(len(ii)-1):
        x1 = np.linspace(ii[i],ii[i],lens)
        x2 = np.linspace(ii[i],ii[i+1],lene)
        x = np.append(x,x1)
        x = np.append(x,x2)
    x = np.append(x,np.zeros(lent))*inputsc


    if  instrument == 0:   
        adwin = adwinIO.initInstrument()
        output = adwinIO.IO(adwin, Inputadwin, Fs)
        output = np.array(output) * Igain
    elif instrument == 1:
        output = nidaqIO.IO(x, Fs)
        output = np.array(output) * Igain
    else:
        print('specify measurement device')
    outsave[:,ic] = x[0:len(output)]
    ic = ic+1
    outsave[:,ic] = output
    ic = ic+1

    plt.figure()
    plt.plot(range(len(output)), output)
    

IVVIrack.setControlVoltages(ivvi, np.array([0]))

# datetime = time.strftime("%d_%m_%Y_%H%M%S")
# filepath = filepath + '\\' + datetime + '_' + name
# np.savetxt(filepath, outsave)


plt.show()
