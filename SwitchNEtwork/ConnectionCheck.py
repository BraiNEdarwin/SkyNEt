#=======================================================================

#This connection check code is dedicated to checking if the switching works or not

#Every contact pads are connected to the common terminal thus they are all connected

#At the room temperature, with 1mV applied, ,ON state should give >2uA and OFF state should give 100nA> current, depending on the noise level
#Otherwise there's sometin wong with the inhibitive resistance of the switch, call Martin


#This is incomplete, don't test yet

#=======================================================================
#Import necessary libraries
import serial
import numpy as np
from instrument import Keith2400
import SaveLibrary as SaveLib
from time import sleep

#Initialize Arduino AND Keithley
ser = serial.Serial(port='/dev/cu.usbmodem1411', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 1, dsrdtr = True)
keithley = Keith2400.Keithley_2400('keithley', 'GPIB0::11')

#first row won't be used
ConnectionCheck = np.zeros((8,8))
OutputCurrentON = np.zeros((8,8))
OutputCurrentOFF = np.zeros((8,8))

#set the compliance current, 100uA
keithley.compliancei.set(100E-6)
#set the voltage (in volts), 1 mV
keithley.volt.set(1E-3)

for z in range(len(OutputCurrentON)):
	for y in range(len(OutputCurrentON[z])):
		#Turn on the corresponding switch in terms of 8 by 8 array
		ConnectionCheck[z][y] = 1
		bytelist = []
		sendlist = []
		for x in range(len(ConnectionCheck)):
			tempbits = 0
			for w in range(len(ConnectionCheck[x])):
				if ConnectionCheck[x][w] == 1:
					tempbits += 2**w
			bytelist.append(tempbits)
		#This results in bytelist equivalent to only one port ON and remainder off

		#Convert the first row into ON, This is the electrode 5
		bytelist[0] = 255
		#Send the connection configuration
		for i in range(len(bytelist)):
			sendlist.append(str(testlist[i]))
		
		#Send the start mark to let Arduino know the serial transmission happens
		ser.write("<".encode())
		print("Sending <")

		time.sleep(0.1)

		ser.write(sendlist[0].encode() + ",".encode() + sendlist[1].encode() + ",".encode() + sendlist[2].encode() + ",".encode() +
sendlist[3].encode() + ",".encode() + sendlist[4].encode() + ",".encode() + sendlist[5].encode() + ",".encode() +
 sendlist[6].encode() + ",".encode() +sendlist[7].encode())
		print("Sending Array")
		time.sleep(0.1)

		ser.write(">".encode())
		print("Sending >")

		#current = keithley.curr()
		#OutputCurrent[z][y] = current
		ConnectionCheck[z][y] = 0

#All the variables except the first row should give a finite current value.
print(OutputCurrent)
