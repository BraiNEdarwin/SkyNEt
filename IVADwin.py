import time
import matplotlib.pyplot as plt
from instruments.ADwin import adwinIO
import numpy as np
import time

V_low = -1						#needs to be 0 or negative
V_high = 0						#needs to be 0 or positive
V_steps = 1000*(V_high-V_low) 	#change stepsize 
Fs = 1000 						#change sample frequency
filepath = 'directory'			
name = 'name.txt'


# Generate the disred input sequence
Input1 = np.linspace(0, V_low, (V_steps*V_low)/(V_low-V_high))
Input2 = np.linspace(V_low, V_high, V_steps)
Input3 = np.linspace(V_high, 0, (V_steps*V_high)/(V_high-V_low))

Input = np.zeros(len(Input1)+len(Input2)+len(Input3))
Input[0:len(Input1)] = Input1
Input[len(Input1):len(Input1)+len(Input2)] = Input2
Input[len(Input1)+len(Input2):len(Input1)+len(Input2)+len(Input3)] = Input3
    
adwin = adwinIO.initInstrument()
Output = adwinIO.IO(adwin, Input, Fs)

plt.figure()
plt.plot(Input, Output)
plt.show()

datetime = time.strftime("%d_%m_%Y_%H%M%S")
filepath = filepath + '\\' + datetime + '_' + name
np.savetxt(filepath, output)



