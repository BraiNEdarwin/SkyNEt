import time
import matplotlib.pyplot as plt
from instruments.niDAQ import nidaqIO
from instruments.ADwin import adwinIO
import numpy as np
import time

Sourcegain = 1
Igain = 1			#use to make output in nA
V_low = -0.8			#needs to be 0 or negative
V_high = 0.8		#needs to be 0 or positive
V_steps = 10000*(V_high-V_low) 	#change stepsize 
Fs = 10000 						#change sample frequency
filepath = r'D:\Tao\TCSPnoise\\'			
name = 'IVtest.txt'
instrument = 1  #choose between nidaq (1) and adwin (0)


# Generate the disred input sequence
Input1 = np.linspace(0, V_low, int((V_steps*V_low)/(V_low-V_high)))
Input2 = np.linspace(V_low, V_high, V_steps)
Input3 = np.linspace(V_high, 0, int((V_steps*V_high)/(V_high-V_low)))

Input = np.zeros(len(Input1)+len(Input2)+len(Input3))
Input[0:len(Input1)] = Input1
Input[len(Input1):len(Input1)+len(Input2)] = Input2
Input[len(Input1)+len(Input2):len(Input1)+len(Input2)+len(Input3)] = Input3
Inputadwin = Input/Sourcegain

print(Inputadwin)   
if  instrument == 0:
    adwin = adwinIO.initInstrument()
    output = adwinIO.IO(adwin, Inputadwin, Fs)
    output = np.array(output) * Igain
elif instrument == 1:
    output = nidaqIO.IO(Inputadwin, Fs)
    output = np.array(output) * Igain
else:
    print('specify measurement device')

plt.figure()
plt.plot(Input[0:len(output)], output)
plt.show()

datetime = time.strftime("%d_%m_%Y_%H%M%S")
filepath = filepath + '\\' + datetime + '_' + name
np.savetxt(filepath, (Input ,output))



