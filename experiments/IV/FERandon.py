import time
import matplotlib.pyplot as plt
from instruments.niDAQ import nidaqIO
from instruments.ADwin import adwinIO
import numpy as np
import time
import modules.SaveLib as SaveLib
from SkyNEt.instruments import InstrumentImporter
Sourcegain = 1
Igain = 10			#use to make output in nA
Fs = 1000 						#change sample frequency
filepath = r'D:/Tao/DN-BdW190206-evolution/FERandom3/'
name = 'FERandom2.txt'
siglen = 100


adw = adwinIO.initInstrument()
ivvi = InstrumentImporter.IVVIrack.initInstrument()
Features=np.array([[0,0,0,0],
					[0,0,0,1],
					[0,0,1,0],
					[0,0,1,1],
					[0,1,0,0],
					[0,1,0,1],
					[0,1,1,0],
					[0,1,1,1],
					[1,0,0,0],
					[1,0,0,1],
					[1,0,1,0],
					[1,0,1,1],
					[1,1,0,0],
					[1,1,0,1],
					[1,1,1,0],
					[1,1,1,1]])

InstrumentImporter.reset(0,0, exit=False)
nG = 50000
inG = 0
# in1=dac7, in2=dac8, in3=dac3, in4=dac4, CV1=dac1, CV2=dac2, CV3=dac5 first run
#in1=dac2, in2=dac7, in3=dac8, in4=dac2, CV1=dac1, CV2=dac4, CV3=dac5 first run

while inG < nG:
	CV = (np.random.rand(3)-0.5)*2*1000
	CV1 = CV[0]
	CV2 = CV[1]
	CV3 = CV[2]
	data=np.array([])
	data = np.append(data,[CV1,CV2,CV3])
	print('Generation '+str(inG)+'...')
	
	for Ft in Features:
		in1 = (Ft[0]-0.5)*2*1000
		in2 = (Ft[1]-0.5)*2*1000
		in3 = (Ft[2]-0.5)*2*1000
		in4 = (Ft[3]-0.5)*2*1000
		InstrumentImporter.IVVIrack.setControlVoltages(ivvi,[CV1,CV2,in3,in4,CV3,0,in1,in2])
		time.sleep(0.05)
		x = np.zeros((1,siglen))
		output = adwinIO.IO(adw, x, Fs)
		data = np.append(data,output)
	datetime = time.strftime("%d_%m_%Y_%H%M%S")
	fp = filepath + '/' + datetime +'_nG'+str(inG)+'_' + name
	np.savetxt(fp,data)
	inG = inG+1

InstrumentImporter.reset(0,0)