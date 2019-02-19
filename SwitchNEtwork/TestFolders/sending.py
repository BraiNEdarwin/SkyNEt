import serial
import numpy as np
import time
from time import sleep

ser = serial.Serial(port='COM3', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 1, dsrdtr = True)
#ser = serial.Serial(port='/dev/cu.usbmodem1411', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 1, dsrdtr = False)


array1 = np.random.rand(8,8)
#said arrays contain random value from 0 to 1, round it so it's a same array with binary bits
Start = np.around(array1)
allzero = np.zeros((8,8))
allhigh = np.zeros((8,8))
for i in range(len(allhigh)):
	for j in range(len(allhigh[i])):
		allhigh[i][j] = 1

bytelist=[]
bytelist2=[]
bytelist3=[]
for k in range(len(Start)):
	tempbits= 0
	for j in range(len(Start[k])):
		if Start[k][j] == 1:
			tempbits += 2**(7-j)
			#print(tempbits)
	bytelist.append(tempbits)

for k in range(len(allhigh)):
	tempbits= 0
	#print(allhigh[k])
    #process every lines
	for j in range(len(allhigh[k])):
		if allhigh[k][j] == 1:
			tempbits += 2**(7-j)
			#print(tempbits)
	bytelist2.append(tempbits)

for k in range(len(allzero)):
	tempbits= 0
	#print(allzero[k])
    #process every lines
	for j in range(len(allzero[k])):
		if allzero[k][j] == 1:
			tempbits += 2**(7-j)
			#print(tempbits)
	bytelist3.append(tempbits)

#ex = [0, 0, 0, 0, 0, 0, 0, 0]
#for n in range(0, 255, 1):
#	x =str(n)
#	ser.write(x.encode('ascii'))
#	print(n)
#	time.sleep(5)

ser.write("<".encode())
ser.write("0".encode())
ser.write(">".encode())

'''
print("Sending ALL")
ser.write("<0,0,0,0,0,0,0,0>".encode())
print("DID IT WORK?!?")
'''