__author__ = 'RenH'

#import necessary libraries
import time
import Evo as Evo
import numpy as np
import operator
from time import sleep

import serial


#Initialize the serial connection to the arduino
#ser = serial.Serial(port='/dev/cu.usbmodem1411', baudrate=9600, bytesize=EIGHTBITS, parity=PARITY_NONE, stopbits=STOPBITS_TEN, timeout=10000, xonxoff=False, rtscts=False, write_timeout=None, dsrdtr=False, inter_byte_timeout=None, exclusive=None)

#open the set up files, this is not fully incorporated yet (needs to figure out how to sort out the each procedures)
exec(open("setup.txt").read())
run = 0
newGenConfigs=np.zeros((genomes,genes,genes))
#genepool = Evo.GeneMatrix(genes,genomes)

#generate random 10 (genomes) arrays of 8 by 8 (genes)
bigdaddy = np.random.rand(genomes,genes,genes)

#said arrays contain random value from 0 to 1, round it so it's a same array with binary bits
bigmommy = np.around(bigdaddy)

#sanity check
#print(bigdaddy)
#print(bigmommy)

for m in range(generations):
  #Fitness scores of each genomes
  Fitnessscores =[]

  #Start here for each genome

  for k in range(len(bigmommy)):
    #Serial can send 1 byte info, so convert the 8 bits into a byte
    bytelist=[]

    tempbits= 0
    #process every lines
    for j in range(len(bigmommy[k])):
      tempbits = 0
      for i in range(len(bigmommy[k][j])):
        if bigmommy[k][j][i] == 1:
            tempbits += 2**(7-i)
        #print(tempbits)
        bytelist.append(tempbits)
      

      #Sanity check
      print(bytelist)

      #USe the serial connection to send 8 byte (64 bits) datas, setting the switch config
      #this SHOULD send it according to the specified baud rate
      #ser.write(bytelist)

     #Switch config set, run the input output relation check
      #in the bytelist, only the first and the last string will be altered since they are connected to the battery and sensor respectively

      evaluateinput = [128,64,32,16,8,4,2,1]

      evaluateoutput = [128,64,32,16,8,4,2,1]

      Outputresult = np.random.rand(genes,genes)

      for i in range(len(evaluateinput)):
       for j in range(len(evaluateoutput)):
        #set the first byte(input) into only one port opening
        bytelist[0] = evaluateinput[i]
        #set the last byte(output) into only one port opening
        bytelist[7] = evaluateoutput[j]
      #sleep
      #read the current value(STILL NEEDS TO FIND OUT HOW!!!)
      #append the read current value into the output array, 8 by 8



     
      '''
     for N number of devices, N number of array which is a Iout for a particular input being HIGH.
      Ideally, the Iout in a single output array should have one HIGH and remaining LOW
      so [LOW,HIGH, HIGH] or [HIGH, LOW, HIGH] is considered insufficient.
     Further more, the location of HIGH should not be in the same index of array, 
      so the N=3 output array of [LOW,HIGH,LOW], [LOW,LOW,HIGH], [LOW,HIGH,LOW] is insufficient.
      Calculate the fitness of each genome(determined by how well the above two criterias are satisfied)
     it may be better to normalize the current from the highest one, then check if any exceeds more than 50% of that?
      '''

     #Normalize the current values, STILL NEED WORKING

      F = 0
  
      threshold = 2
      #Check for Criteria 1
      for i in range(len(Outputresult)):
     #Reser the counter
        count = 0
	
        for j in range(len(Outputresult[i])):
       #arbitrary value
        #If higher than certain voltage, add 1 to the count
          if Outputresult[i,j]>threshold:
            count = count + 1
		        #At the end, a row will be chcked and count will equal the number of output electrode that was HIGH for this configuration

	         #if only one output was HIGH for the given input, that's success!
          if count == 1:
    	       F = F + 3
      

           #if more than 1 output was HIGH for the given input, we give -1 for the number of outputs that were HIGH
          elif count > 1:
    	       F = F + -1*count

            #if no output was HIGH for a particular input, we either have to lower the threshold, or just punish the fitness score
          elif count == 0:
             F = F - 10


      #Do the exact same, but for the criteria 2
      for i in range(len(Outputresult)):
        #Reset the counter
        count = 0
	
        for j in range(len(Outputresult[i])):
        #arbitrary value
        #If higher than certain voltage, add 1 to the count
          if Outputresult[j,i]>threshold:
            count = count + 1
		
		        #At the end, a column will be chcked and count will equal the number of input that gave HIGH to the particular output electrode
		        #This criteria may need revision, as later we may move on to more than 1 set of input pixels

	         #if a unique input output relation is met, that's success!
        if count == 1:
    	     F = F + 3

     #if the output was HIGH for the more than 1 input, we give -1 for the number of inputs that were corresponded
        elif count > 1:
    	    F = F + -1*count
    	
      #if no input gave HIGH state for the particular output, we either have to lower the threshold, or just punish the fitness score
        elif count == 0:
          F = F - 10
      #sanity check
      #print(F)

      Fitnessscores.append(F)

  #Up until here in the for loop for each genomes

  #decide the winner
  winner = Fitnessscores.index(max(Fitnessscores))
  #Sanity check, winner is an index number of the array
  #print (winner)

  #Find out who won the second place
  '''
  count = 0
  m1 = m2 = float('-inf')
  for x in Fitnessscores:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
  if count>=2:
      silver = m2

  secondwinner = Fitnessscores.index(silver)
  print(secondwinner)
  '''
  #identify the winner and the second

  print('The winner is:')
  print(bigmommy[winner])
  run = run + 1
  print('with a fitness score of ' + str(F))


  #Mutation
  #Winner remains = 1
  newGenConfigs[0] = bigmommy[winner]

  #Mutate with 10% chance
  for i in range(1, 8):
    templist = bigmommy[winner]
    for j in range(genes):
        for k in range(genes):
            if(np.random.rand() < 0.1):
              if templist[j, k] == 1:
                templist[j, k] = 0
              elif templist[j, k] == 0:
                  templist[j, k] = 1
    newGenConfigs[i] = templist

  #Is it possible to utilize the second winner for the breeding?

  #complete random 
  for i in range(8,10):
    templist = np.random.rand(genes,genes)
    newGenConfigs[i] = np.around(templist)

#Do this for however many generations