from time import sleep
import serial
import numpy as np
#is this better in another file?
ser = serial.Serial(port='/dev/cu.usbmodem1411', baudrate=9600, bytesize=8, timeout=10000)

#counter = 32 # Below 32 everything in ASCII is gibberish
genomes = 10
genes = 8
bigdaddy = np.random.rand(genomes,genes,genes)
bigmommy = np.around(bigdaddy)
print(bigmommy)
#convert array into string?
#if 10000000=1
#if 10100000=4


#convert the list of 8 bits into a byte
#testlist = [[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 0]]
#print(testlist)

bytelist=[]


tempbits= 0
for j in range(len(bigmommy[0])):
    tempbits = 0
    for i in range(len(bigmommy[0][j])):
        if bigmommy[0][j][i] == 1:
            tempbits += 2**(7-i)
    print(tempbits)
    bytelist.append(tempbits)

print(bytelist)
#testbyte = bytes(testlist)
#print(testbyte)

#print(testarray2)
while False:
     #counter +=1
     #ser.write(str(chr(counter))) # Convert the decimal number to ASCII then send it to the Arduino
     ser.write(bytelist)
     print (ser.readline()) # Read the newest output from the Arduino
     sleep(.8) # Delay for one tenth of a second
     #if counter == 255:
         #counter = 32