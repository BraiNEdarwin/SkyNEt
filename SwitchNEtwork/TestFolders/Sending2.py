import serial
import numpy as np

import struct
import time
from time import sleep

ser = serial.Serial(port='COM3', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 1, dsrdtr = True)
#ser = serial.Serial(port='/dev/cu.usbmodem1411', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 1, dsrdtr = True)
sendlist=[]
testlist = [16,16,16,16,16,16,16,16]
for i in range(len(testlist)):
	sendlist.append(str(testlist[i]))
time.sleep(1)
ser.write("<".encode())
print("Sending <")

time.sleep(0.2)

x = 100

#ser.write("0, 0, 0, 0, 32, 32, 0, 0".encode())
ser.write(sendlist[0].encode()+ ",".encode() +sendlist[1].encode()+ ",".encode() +sendlist[2].encode()+ 
	",".encode() +sendlist[3].encode()+ ",".encode() +sendlist[4].encode()+ ",".encode() +
	sendlist[5].encode()+ ",".encode() +sendlist[6].encode()+ ",".encode() +sendlist[7].encode())

print("Sending")

time.sleep(0.2)

ser.write(">".encode())
print("Sending >")

time.sleep(0.2)
item = ser.readline()
item2 = item.strip()
item3 = item2.split()

print(item3)
