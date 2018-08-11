#============================================================================================================================================================================================================================================

#This code is there to check just one connection between two switches(by making IV)
#For the explanation please refer to MEGA IV curve code

#It won't save a file, so just keep an eye on the keythley :p

#============================================================================================================================================================================================================================================

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
#Change bytelist corresponding to the two switch(es) that you want to open, do not include spaces
#so no [0, 0, 0, 0, 0, 0, 0, 0]
bytelist = [255,0,0,0,0,0,0,255]

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
	time.sleep(0.1)		
	keithley.volt.set(voltrange[c])
	time.sleep(0.1)
	current = keithley.curr()
	print(str(voltrange[c]) + 'V' + '        ' + str(current) + 'A')
	#currentlist[a][b][0][c] = voltrange[c]
	#urrentlist[a][b][1][c] = current
