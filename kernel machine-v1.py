# from instruments.DAC import IVVIrack
# initially written on 15/05/2018
import time
# temporary imports
import numpy as np
from instruments.DAC import IVVIrack
from instruments.niDAQ import nidaqIO

import matplotlib.pyplot as plt


ivvi = IVVIrack.initInstrument()

# win is the convolution window, 16 possiblities
win = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])
wtest = 10
winout = np.zeros([wtest,16])
winoutsd = np.zeros(2,16)
# vba, and vaa are the bias voltage and input voltages
vba = np.arange(400,800,20)
vaa = np.arange(300,1000,100)

# winsdr is defined as maximum span divided by maximum standard deviation
winsdr = np.zero(np.len(vba)*np.len(vaa),3) 
winsdrm = np.zero(np.len(vba),np.len(vbb))
#x1 = np.array([[0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]])
digits = np.loadtxt('Y:\Tao\PN laptop\Digits recognition experiment\resources\digitscondensed.txt')
 
fs = 100
# dout is the output for digits convolution
dout = np.zeros([80,10])

# below, test the mapped output of all windows and find best va and vb
k=0
kb=0
ka=0
for vb in vba:
	for va in vaa:
		for i in range(wtest):
			for j in range(16):
				inputVoltages = [win[j,:]*va,vb]
				print(inputVoltages)
				IVVIrack.setControlVoltages(ivvi, inputVoltages)
				time.sleep(1)

				measureddata = nidaqIQ.IO(y, fs)
				measureddata = np.average(measureddata)
				print(measureddata)
				winout[i,j]=measureddata
	

		for i in range(16):
			winoutsd[1,i] = winout[:,i].average()
			winoutsd[2,i] = np.std(winout[:,i])
		sdr = (winoutsd[1,:].max() - winoutsd[1,:].min())/winoutsd[2,:].max()
		winsdr[k,:]=[vb,va,sdr]
		winsdrm[kb,ka]
		k=k+1
		ka=ka+1
	kb=kb+1					
	np.savetxt('Y:\Tao\PN laptop\Digits recognition experiment\convNet_shatter_va'+va+'_vb'+vb+'.txt', winout)
	np.savetxt('Y:\Tao\PN laptop\Digits recognition experiment\convNet_shatter_winsdr.txt', winsdr)

winsdrbest=winsdr(np.argmax(winsdr[:,3]),:)
vb=winsdrbest(0)
va=winsdrbest(1)
print(winsdrbest)
plt.figure()
plt.contour(kb,ka,sdr)
plt.show()


# # pattern recognition with 50 handcrafted digits
# for i in range(10):
# 	q=0
# 	for j in range(5):
		
# 		for k in range(4):
# 			for l in range(4):
# 				inputVoltages = [[digits[i*5+k,j*5+l],digits[i*5+k,j*5+l+1],digits[i*5+k+1,j*5+l],digits[i*5+k+1,j*5+l+1]]*va,vb,vb]
# 				print(inputVoltages)
# 				IVVIrack.setControlVoltages(ivvi, inputVoltages)
# 				time.sleep(1)

# 				measureddata = nidaqIQ.IO(y, fs)
# 				measureddata = np.average(measureddata)
# 				print(measureddata)
# 				dout[q,i]=measureddata
# 				q=q+1

# np.savetxt('Y:\Tao\PN laptop\Digits recognition experiment\convNet_digits_va'+va+'_vb'+vb+'.txt', dout)

# childdigits=np.zeros(500,50)
# childout=np.zeros(1600,10)
# # pattern recognition with 1000 digits
# for i in range(10):
# 	parentdigit = digits[0:4,i*5:i*5+5]
# 	q=0
# 	for j in range(100):
# 		an=1*j/99
# 		matrixnoise = np.random.rand(5,5)-0.5
# 		childdigit = parentdigit + matrixnoise*an
# 		# matrixth = np.ones((5,5))*pm
# 		# matrixmask=matrixpm < matrixth
# 		# matrixmaskinv=np.logical_not(matrixmask)
# 		# matrixrand = np.random.rand(5,5)
# 		# matrixmutation=np.multiply(matrixrand,matrixmask)
# 		# childdigit = np.multiply(parentdigit,matrixmaskinv)+matrixmutation
# 		childdigits(i*5:i*5+5,j*5:j*5+5)=childdigit
# 		for k in range(4):
# 			for l in range(4):
# 				inputVoltages = [[childdigit[i*5+k,j*5+l],childdigit[i*5+k,j*5+l+1],childdigit[i*5+k+1,j*5+l],childdigit[i*5+k+1,j*5+l+1]]*va,vb,vb]
# 				print(inputVoltages)
# 				IVVIrack.setControlVoltages(ivvi, inputVoltages)
# 				time.sleep(1)

# 				measureddata = nidaqIQ.IO(y, fs)
# 				measureddata = np.average(measureddata)
# 				print(measureddata)
# 				childout[q,i]=measureddata
# 				q=q+1

# np.savetxt('Y:\Tao\PN laptop\Digits recognition experiment\convNet_child_va'+va+'_vb'+vb+'.txt', childout)
# np.savetxt('Y:\Tao\PN laptop\Digits recognition experiment\convNet_child.txt', childdigits)