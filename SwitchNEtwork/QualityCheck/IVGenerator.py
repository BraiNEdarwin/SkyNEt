#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:25:11 2018

@author: renhori
"""

import numpy as np
import matplotlib.pyplot as plt

data= np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Result/Switch/(verified)SwiNEt_24_08_2018_141106_FINALEightDeviceAllIV77K/IVDataz.npz')
result =data.f.currentlist

j = 421
#for b in range(len(result)):
    #plt.subplot(j)
b = 0
a = 0
'''
for a in range(len(result)):
	for b in range(len(result[a])):
		for c in range(len(result[a][b])):
			for d in range(len(result[a][b][c])):

				result[a][b][c][d] = result[a][b][c][d]*(10.0**9)
'''
'''
for a in range(len(result)):
	plt.figure(1)
	#plt.plot(result[a][0][0], result[a][0][1], label = "Electrode2")
	plt.plot(result[a][1][0], result[a][1][1], label = "Electrode3")
	plt.plot(result[a][2][0], result[a][2][1], label = "Electrode4")
	plt.plot(result[a][3][0], result[a][3][1], label = "Electrode5") 	
	plt.plot(result[a][4][0], result[a][4][1], label = "Electrode6")
	plt.plot(result[a][5][0], result[a][5][1], label = "Electrode7")
	#plt.plot(result[a][6][0], result[a][6][1], label = "Electrode8")
	#plt.plot(result[a][7][0], result[a][7][1], label = "hi8")
	plt.ylabel('Amp (nA)', size = 17)
	plt.xlabel('Volt (V)', size = 17)
	plt.grid(True)
	plt.title('Inter-device IV: Device = '+ str(a + 1) +' Input = Electrode 1')
	plt.legend()
	plt.show()
	#plt.savefig('Inter-device IV: Device = '+ str(a + 1) +' Input = Electrode 1.png')
'''
'''
for b in range(len(result[0])):
	plt.figure(1)
	plt.plot(result[0][b][0], result[0][b][1], label = "Vin @ Device 1")
	plt.plot(result[1][b][0], result[1][b][1], label = "Vin @ Device 2")
	plt.plot(result[2][b][0], result[2][b][1], label = "Vin @ Device 3")
	plt.plot(result[3][b][0], result[3][b][1], label = "Vin @ Device 4") 	
	plt.plot(result[4][b][0], result[4][b][1], label = "Vin @ Device 5")
	plt.plot(result[5][b][0], result[5][b][1], label = "Vin @ Device 6")
	plt.plot(result[6][b][0], result[6][b][1], label = "Vin @ Device 7")
	plt.plot(result[7][b][0], result[7][b][1], label = "Vin @ Device 8")
	plt.ylabel('Amp (nA)', size = 17)
	plt.xlabel('Volt (V)', size = 17)
	plt.grid(True)
	plt.title('Intra-device IV: Input = Electrode 1, Output = Electrode 5, Output device = ' + str(b + 1))
	plt.legend()
	plt.show()
	plt.savefig('Intra-device IV: Input = Electrode 1, Output = Electrode 5, Output device = ' + str(b + 1) + '.png')

'''
'''
for a in range(len(result)):
	plt.figure(1)
	plt.plot(result[a][0][0], result[a][0][1], label = "Input @D1")
	plt.plot(result[a][1][0], result[a][1][1], label = "Input @D2")
	plt.plot(result[a][2][0], result[a][2][1], label = "Input @D3")
	plt.plot(result[a][3][0], result[a][3][1], label = "Input @D4") 	
	plt.plot(result[a][4][0], result[a][4][1], label = "Input @D5")
	plt.plot(result[a][5][0], result[a][5][1], label = "Input @D6")
	plt.plot(result[a][6][0], result[a][6][1], label = "Input @D7")
	plt.plot(result[a][7][0], result[a][7][1], label = "Input @D8")
	plt.ylabel('Amp (nA)', size = 17)
	plt.xlabel('Volt (V)', size = 17)
	plt.grid(True)
	plt.title('Input = E5, Output = E1 of D' + str(a+1))
	plt.legend()
	plt.show()
	#plt.savefig('Inter-device IV: Device = '+ str(a + 1) +' Input = Electrode 1.png')


for b in range(len(result[0])):
	plt.figure(1)
	plt.plot(result[0][b][0], result[0][b][1], label = "Iout @ Device 1")
	plt.plot(result[1][b][0], result[1][b][1], label = "Iout @ Device 2")
	plt.plot(result[2][b][0], result[2][b][1], label = "Iout @ Device 3")
	plt.plot(result[3][b][0], result[3][b][1], label = "Iout @ Device 4") 	
	plt.plot(result[4][b][0], result[4][b][1], label = "Iout @ Device 5")
	plt.plot(result[5][b][0], result[5][b][1], label = "Iout @ Device 6")
	plt.plot(result[6][b][0], result[6][b][1], label = "Iout @ Device 7")
	plt.plot(result[7][b][0], result[7][b][1], label = "Iout @ Device 8")
	plt.ylabel('Amp (nA)', size = 17)
	plt.xlabel('Volt (V)', size = 17)
	plt.grid(True)
	plt.title('Input = E5 of D'+str(b+1)+', Output = E1')
	plt.legend()
	plt.show()
	#plt.savefig('Intra-device IV: Input = Electrode 1, Output = Electrode 5, Output device = ' + str(b + 1) + '.png')

'''
plt.figure(1)
for a in range(len(result)):
	
	#plt.plot(result[a][0][0], result[a][0][1], label = "Electrode2")
	plt.plot(result[a][3][0], result[a][3][1], label = "Device " + str(a+1))
	#plt.plot(result[a][2][0], result[a][2][1], label = "Electrode4")
	#plt.plot(result[a][3][0], result[a][3][1], label = "Electrode5") 	
	#plt.plot(result[a][4][0], result[a][4][1], label = "Electrode6")
	#plt.plot(result[a][5][0], result[a][5][1], label = "Electrode7")
	#plt.plot(result[a][6][0], result[a][6][1], label = "Electrode8")
	#plt.plot(result[a][7][0], result[a][7][1], label = "hi8")
	
plt.ylabel('Amp (nA)', size = 17)
plt.xlabel('Volt (V)', size = 17)
plt.grid(True)
plt.title('IV relation of Vin = E5 Iout = E1')
plt.legend()
plt.show()