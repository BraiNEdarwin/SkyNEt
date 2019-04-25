#=====================================================================
#This script is purely for generating many IV curves with random switch configuration
#=====================================================================

#Import necessary stuff
import serial
import numpy as np
from instrument import Keith2400
import SaveLibrary as SaveLib
import time
from time import sleep

#Open the set up
exec(open("setup.txt").read())

#Initialize the directory to save the files
savedirectory = SaveLib.createSaveDirectory(filepath, name)

#Initialize Arduino and keithley
ser = serial.Serial(port='COM3', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 1, dsrdtr = True)
keithley = Keith2400.Keithley_2400('keithley', 'GPIB0::11')

#set the current limit, in Amp
keithley.compliancei.set(1E-6)
#set the voltage limit in volts just in case something goes wrong from the set up file. DO NOT CHANGE THIS UNLESS YOU KNOW WHAT YOU'RE DOING
keithley.compliancev.set(4)

#Necessary for the IV curve
voltrange = []
steps = int(Vabs/Vstep + 1)
a = np.zeros((4,1,steps))

#Set of voltage range is appended to the voltagerange, formatting of "linespace" forces this part to be split into 4 quadrants, 0 to neg, neg to 0, 0 to pos, pos to 0
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

randomarray =[
[0,0,0,0,0,0,0,0],
[1,0,0,1,0,1,0,0],
[1,0,1,1,0,1,1,1],
[0,1,0,0,1,1,0,1],
[0,0,0,0,0,0,0,0],
[0,0,1,1,0,0,1,0],
[1,0,1,0,1,0,1,0],
[0,1,0,1,0,1,0,1]]

evaluateinput=[1,2,4,8,16,32,64,128]
evaluateoutput=[1,2,4,8,16,32,64,128]
#Let's go
print('Ready')

#initialize the IV numpy array and the bytelist to control switch configs
currentlist = np.zeros((genes,genes,2,4*steps))

#a corresponds to the device
for a in range(len(currentlist)):
	print('Device '+ str(a+1))
	#b corresponds to the connection from the electrode 5 to the electrode (b+1)
	for b in range(len(currentlist[a])):
		print('To electrode ' + str(b+2) + ' of Device ' + str(a+1))
		#Initialize bytelist (reset the bytelist everytime new scheme is examined)
		#Initialize sendlist, the array used for actually sending
		bytelist=[0,0,0,0,0,0,0,0]
		sendlist = []
		for k in range(len(randomarray)):
			tempbits = 0
			#for 1 in every location of the matrix, add it to the bit
			for l in range(len(randomarray[k])):
				if randomarray[k][l] == 1:
					tempbits += 2**k
			#Bytelist will contain 8 byte(64 bits), starting from the top of the matrix
			bytelist.append(tempbits)
			
		#turn on the battery(switch on the input electrode of the selected device)
		bytelist[0] = evaluateinput[a]
		#then only turn on the switch that corresponds to the electrode output we want. a controls the x axis of the matrix config(device number) and b controls the y axis of matrix config(electrodes)
		bytelist[4] = evaluateoutput[a]
		#Convert to the arduino readable format
		for i in range(len(bytelist)):
			sendlist.append(str(bytelist[i]))
		
		#Send
		print("Sending StartMark")
		ser.write("<".encode())

		time.sleep(0.2)

		ser.write(sendlist[0].encode() + ",".encode() + sendlist[1].encode() + ",".encode() + sendlist[2].encode() + ",".encode() +
sendlist[3].encode() + ",".encode() + sendlist[4].encode() + ",".encode() + sendlist[5].encode() + ",".encode() +
 sendlist[6].encode() + ",".encode() +sendlist[7].encode())
		print("Sending the array")

		time.sleep(0.2)


		print("Sending EndMark")
		ser.write(">".encode())

		time.sleep(1)
		item = ser.readline()
		item2 = item.strip()
		item3 = item2.split()

		print(item3)

		print("READY")

		#Switch configuration is set

		#for the range of voltage
		for c in range(len(voltrange)):
			#make money get rich
			time.sleep(0.01)
			keithley.volt.set(voltrange[c])

			#This waiting time between setting the voltage and reading current, as well as setting of voltage, can be optimized maybe
			#If huge hysterisis (and suppose we don't want that), increase this second sleep time to something longer
			time.sleep(0.04)
			current = keithley.curr()
			showcurrent = current * 1000000000

			#Maybe I should change this to nA representation?
			print(str(voltrange[c]) + '	V' + '        ' + str(showcurrent) + '	nA')
			currentlist[a][b][0][c] = voltrange[c]
			currentlist[a][b][1][c] = current
			

#save the file
SaveLib.SaveMainIV(savedirectory, currentlist)