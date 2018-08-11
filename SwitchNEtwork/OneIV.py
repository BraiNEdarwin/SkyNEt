import serial
import numpy as np
from instrument import Keith2400
import SaveLibrary as SaveLib
import time
from time import sleep
import matplotlib.pyplot as plt

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
keithley.compliancev.set(2)


#Necessary for the IV curve
voltrange = []
steps = int(Vabs/Vstep + 1)
a = np.zeros((4,1,steps))
currentlist = np.zeros((2,4*steps))

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

#Change bytelist accordingly
bytelist = [4,4,4,4,4,4,4,4]
sendlist = []

for i in range(len(bytelist)):
	sendlist.append(str(bytelist[i]))

ser.write("<".encode())
ser.write(sendlist[0].encode() + ",".encode() + sendlist[1].encode() + ",".encode() + sendlist[2].encode() + ",".encode() +
sendlist[3].encode() + ",".encode() + sendlist[4].encode() + ",".encode() + sendlist[5].encode() + ",".encode() +
 sendlist[6].encode() + ",".encode() +sendlist[7].encode())
ser.write(">".encode())
time.sleep(1)
print("Array Sent")

for c in range(len(voltrange)):
	time.sleep(0.01)
	keithley.volt.set(voltrange[c])
	time.sleep(0.04)
	current = keithley.curr()
	showcurrent = current * 1000000000
	print(str(voltrange[c]) + '	V' + '        ' + str(showcurrent) + '	nA')
	currentlist[0][c] = voltrange[c]
	currentlist[1][c] = current

print("DONE")
time.sleep(1)

plt.figure(1)
plt.plot(currentlist[0],currentlist[1])
plt.ylabel('Amp')
plt.xlabel('Volt')
plt.grid(True)

plt.show()


