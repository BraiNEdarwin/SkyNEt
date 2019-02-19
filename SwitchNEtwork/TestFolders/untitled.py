import serial
#make dsrdtr true to not lose the program compiled on the arduino
ser = serial.Serial(port='COM3', baudrate=9600, bytesize=8, parity='N',dsrdtr=True)

#value = 0 should correspond to closing all the switches
Value = 0

#encode the value into ascii byte
Sending = Value.encode('ascii')

#send
ser.write(Sending)