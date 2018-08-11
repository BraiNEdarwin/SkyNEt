#=====================================================================

#This code does not reflect the actual 

#=====================================================================

import serial
import numpy as np
from instrument import Keith2400
import SaveLibrary as SaveLib
import time
from time import sleep

exec(open("setup.txt").read())
#Initialize the directory to save the files
#savedirectory = SaveLib.createSaveDirectory(filepath, name)
ser = serial.Serial(port='COM3', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 1, dsrdtr = True)
keithley = Keith2400.Keithley_2400('keithley', 'GPIB0::11')
#set the current limit, in Amp
keithley.compliancei.set(50E-6)
#set the voltage limit in volts just in case something goes wrong from the set up file. DO NOT CHANGE THIS UNLESS YOU KNOW WHAT YOU'RE DOING
keithley.compliancev.set(2)

'''
For this experiment, we suppose all 8 devices are connected, and the electrode 5 of all the devices are to the battery and the remains are to the output.
'''
#Necessary for the IV curve
voltrange = []
steps = int(Vabs/Vstep + 1)
a = np.zeros((4,1,steps))

#Set of voltage range is appended to the voltagerange
first =(np.linspace(0,-Vabs, steps))
a[0] = first
second= (np.linspace(-Vabs, 0,steps))
a[1] = second
third= (np.linspace(0, Vabs, steps))
a[2] = third
fourth = (np.linspace(Vabs, 0, steps))
a[3] = fourth
for b in range(4):
	for c in range(steps):
		voltrange.append(a[b][0][c])

print('Ready')

#initialize the IV numpy array and the bytelist to control switch configs
#currentlist = np.zeros((devs,genes-1,2,4*steps))

bytelist = [0,0,0,2,0,0,0,2]
sendlist = []

for i in range(len(bytelist)):
	sendlist.append(str(bytelist[i]))

print("Sending")
ser.write("<".encode())
time.sleep(0.25)
ser.write(sendlist[0].encode() + ",".encode() + sendlist[1].encode() + ",".encode() + sendlist[2].encode() + ",".encode() +
sendlist[3].encode() + ",".encode() + sendlist[4].encode() + ",".encode() + sendlist[5].encode() + ",".encode() +
 sendlist[6].encode() + ",".encode() +sendlist[7].encode())
time.sleep(0.25)
ser.write(">".encode())

for c in range(len(voltrange)):
	#take money get rich
			
	keithley.volt.set(voltrange[c])
	time.sleep(0.5)
	current = keithley.curr()
	print(str(voltrange[c]) + 'V' + '        ' + str(current) + 'A')
	#currentlist[a][b][0][c] = voltrange[c]
	#urrentlist[a][b][1][c] = current


'''
for a in range(len(currentlist)):
	print('Device '+ str(a+1))
	#b corresponds to the connection from electrode 1 to the electrode (b+1)
	for b in range(len(currentlist[a])):
		print('To electrode ' + str(b+2))
		#Initialize bytelist (reset the bytelist everytime new scheme is examined)
		#Initialize sendlist
		bytelist=[0,0,0,0,0,0,0,0]
		sendlist = []
		#turn on the battery(switch on the electrode 1 of the selected device)
		bytelist[0] = 2**a
		#then only turn on the switch that corresponds to the electrode output we want. a controls the x axis of the matrix config(device number) and b controls the y axis of matrix config(electrodes)
		bytelist[b+1] = 2**a
		#Convert to the arduino readable format
		for i in range(len(bytelist)):
			sendlist.append(str(bytelist[i]))
		
		#Send
		print("Sending StartMark")
		ser.write("<".encode())
		time.sleep(0.5)

		ser.write(sendlist[0].encode() + ",".encode() + sendlist[1].encode() + ",".encode() + sendlist[2].encode() + ",".encode() +
sendlist[3].encode() + ",".encode() + sendlist[4].encode() + ",".encode() + sendlist[5].encode() + ",".encode() +
 sendlist[6].encode() + ",".encode() +sendlist[7].encode())
		print("Sending the array")

		ser.write(">".encode())
		print("Sending EndMark")
		ser.write(">".encode())
		time.sleep(0.5)

		#Switch configuration is set

		#for the range of voltage
		for c in range(len(voltrange)):
			#take money get rich
			
			keithley.volt.set(voltrange[c])
			time.sleep(0.01)
			current = keithley.curr()
			print(str(voltrange[c]) + 'V' + '        ' + str(current) + 'A')
			currentlist[a][b][0][c] = voltrange[c]
			currentlist[a][b][1][c] = current
			

#save the file
SaveLib.SaveMainIV(savedirectory, currentlist)
'''