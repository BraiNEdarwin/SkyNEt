import time
import matplotlib.pyplot as plt
from instruments.niDAQ import nidaqIO
from instruments.ADwin import adwinIO
import numpy as np
import time
import modules.SaveLib as SaveLib

Sourcegain = 1
Igain = 10			#use to make output in nA
V_low = -0.7			#needs to be 0 or negative
V_high = 0.7		#needs to be 0 or positive
Vg_low = -1.5
Vg_high = 1.5
V_steps = 10000*(V_high-V_low) 	#change stepsize 
Fs = 10000 						#change sample frequency
filepath = r'D:/Tao/TCSPnoiseSR/'
name = 'IVtest.txt'
instrument = 1  #choose between nidaq (1) and adwin (0)
siglen = 4

# Generate the disred input sequence

for vin1 in np.linspace(V_low,V_high,100):
	for vin2 in np.linspace(Vg_low,Vg_high,100):
		x=np.ones([2,siglen*Fs])
		t=np.arange(len(x[0,:]))*1/Fs
		x[0,:]=x[0,:]*vin1+np.random.normal(0, 1, size=len(x[0,:]))*0.01+0.01*np.sin(2*np.pi*23*t)
		x[1,:]=x[1,:]*vin2
		output = nidaqIO.IO_2D(x, Fs)
		datetime = time.strftime("%d_%m_%Y_%H%M%S")
		fp = filepath + '/' + datetime + '_' + name
		np.savetxt(fp,np.append([vin1,vin2,Fs],np.asarray(output)))