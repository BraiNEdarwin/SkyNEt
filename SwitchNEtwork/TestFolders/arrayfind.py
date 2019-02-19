import serial
import numpy as np

import struct

#ser = serial.Serial(port='COM3', baudrate=9600, bytesize=8, parity='N',dsrdtr=True)

import time
from time import sleep

#ser = serial.Serial(port='COM3', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 1)
#ser = serial.Serial(port='/dev/cu.usbmodem1411', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 1)



array1 = np.random.rand(8,8)
#said arrays contain random value from 0 to 1, round it so it's a same array with binary bits
Start = np.around(array1)

bytelist=[]

for k in range(len(Start)):
	tempbits= 0
	for j in range(len(Start[k])):
		if Start[k][j] == 1:
			tempbits += 2**(7-j)
			#print(tempbits)
	bytelist.append(tempbits)

a = "<100,200,300>"
b = str(100)
c = str(200)
d = str(300)
e = "<"+ b +","+ c +","+ d + ">"