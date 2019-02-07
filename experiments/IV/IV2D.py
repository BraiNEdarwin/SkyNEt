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
Fs = 10000 						#change sample frequency
filepath = r'D:/Tao/DN-BdW190206-evolution/Stability diagrams/'
name = 'stability.txt'
siglen = 200
Vsteps = 81
Vin1dac=6
Vin2dac=7

adw = adwinIO.initInstrument()
ivvi = InstrumentImporter.IVVIrack.initInstrument()
ControlVoltages=np.array([[106.9,-401.5,166.32,-774.6,-116.1,-250,250], 
							[106.9,-401.5,166.3,-774.6,-116.1,-250,250],
							[106.9,-688.7,166.3,-646.8,-76,-250,250],
							[106.9,-728.7,166.3,-606.8,-116.1,-250,250],
							[-134.1,241.4,159.1,-114.1,-116.6,-250,250],
							[-164.1,248.6,146.3,189.8,-136.6,-250,250],
							[-194.1,268.6,166.3,209.8,-166.6,-250,250],
							[-254.1,268.6,126.3,249.8,-116.6,-250,250],
							[-804.7,398.5,189.6,2.4,-176.8,-250,250],
							[-290.6,35.5,-649.1,223.2,-390.7,0,500],
							[-3.6,347.6,-649.1,552.3,-804.1,0,500],
							[-3.6,347.6,-649.1,552.3,-804.5,0,500],
							[26.4,327.6,-669.1,572.3,-784.5,0,500],
							[26.4,327.6,-669.1,572.2,-784.5,0,500],
							[-826.9,-838.7,77.9,423.4,-46.6,0,500],
							[-806.1,-377.4,534.7,282.6,-29.4,-250,250],
							[-856.8,-551.4,-236.9,443.4,-66.6,0,500],
							[-856.9,-511.4,-23.2,443.4,-128.9,0,500],
							[-891.8,-236.8,556.9,-49.6,244,0,500],
							[-11.3,308.4,470.4,-777.9,385.7,0,500],
							[-724.6,-771.3,-141.7,-508.1,-284.7]])

iCV=0
for CV in ControlVoltages:
	V1_low = CV[5]-150			#needs to be 0 or negative
	V1_high = CV[6]+150		#needs to be 0 or positive
	V2_low = CV[5]-150
	V2_high = CV[6]+150
	V1range = np.append(np.linspace(V1_low,V1_high,Vsteps),np.linspace(V1_high,V1_low,Vsteps))
	V2range = np.linspace(V2_low,V2_high,Vsteps)
	InstrumentImporter.IVVIrack.setControlVoltages(ivvi, CV[0:5])
	data = np.array([])
	for Vin2 in V2range:
		print(Vin2)
		InstrumentImporter.IVVIrack.setControlVoltage(ivvi,Vin2,Vin2dac)
		time.sleep(0.05)
		for Vin1 in V1range:
			time.sleep(0.03)
			x = np.zeros((1,siglen))
			InstrumentImporter.IVVIrack.setControlVoltage(ivvi,Vin2,Vin1dac)
			output = adwinIO.IO(adw, x, Fs)
			data = np.append(data,[Vin2,Vin1,np.average(output)])
	datetime = time.strftime("%d_%m_%Y_%H%M%S")
	fp = filepath + '/' + datetime +'_CV'+str(iCV)+'_'+ '_' + name
	iCV=iCV+1
	np.savetxt(fp,data)

InstrumentImporter.reset(0,0)