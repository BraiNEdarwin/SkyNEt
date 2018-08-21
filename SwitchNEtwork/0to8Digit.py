__author__ = 'RenH'
'''
This is a final model of the MSc project. Most of the stuff is the same as the switch code, but when evaluating the output we use different information
For this set up, two input rows are necessary, electrode 5 and 6 (row 1 and 2 of the matrix) will be inputs, while the electrode 1(row 5) will be the output
'''

#import necessary libraries
import time
import numpy as np
import operator
from time import sleep
from instrument import Keith2400
#Until plot building memory issue is solved, this remains commented
#from instrument import PlotBuilding as PlotBuilder
import serial
import SaveLibrary as SaveLib

#open the set up file
exec(open("setup.txt").read())

#setup keithley
keithley = Keith2400.Keithley_2400('keithley', 'GPIB0::11')
#set the compliance current
keithley.compliancei.set(CompI)


#in case the compliance voltage is set way too high, scratch the whole process
if (CompV > 4):
	generations = 0
	genes = 0
	devs = 0
	genomes = 0

#Set the compliance voltage
keithley.compliancev.set(CompV)

#set the voltage from the input (in volts)
keithley.volt.set(Volts)

#Initialize the serial connection to the arduino
ser = serial.Serial(port='COM3', baudrate=9600, bytesize=8, parity='N', stopbits=1, write_timeout = 0, dsrdtr = True)

#I don't even know if you need this, maybe wait time for the port to open
#Also gotta implement the try and catch code for if the port is available
time.sleep(1)

#Initialize the directory to save the files
savedirectory = SaveLib.createSaveDirectory(filepath, name)

#generate necessary arrays to save the datas
genearray = np.zeros((generations, genomes, genes, genes))
outputarray = np.zeros((generations, genomes, devs, devs))
fitnessarray = np.zeros((generations, genomes))
successarray = np.zeros((generations, genomes))
timearray = np.zeros((generations,genomes,4))

#define the initial switches, array has to be 8 by 8 because it'll be converted to bytes
array1 = np.random.rand(genomes,genes,genes)
#said arrays contain random value from 0 to 1, round it so it's a same array with binary bits
NewGenConfigs = np.around(array1)

#convert the unused device column into 0s, newgenconfig = (genomes, gene, dev) so for all the genomes, convert dev 
#make a list of numbers (from 0 to 7) where the devices are not installed, please refer to the guideline of which device spot is which
nullist = []
for a in range(len(nullist)):
	NewGenConfigs[:,:,nullist[a]] = 0

#And then, we will convert the row that corresponds to input and output to 0
#This is done so when duplicate check is run, we don't allow two copies with same switch config upon changing input and output byte
for a in range(len(NewGenConfigs)):
	NewGenConfigs[a][0] = [0,0,0,0,0,0,0,0]
	NewGenConfigs[a][1] = [0,0,0,0,0,0,0,0]
	NewGenConfigs[a][4] = [0,0,0,0,0,0,0,0]

#Time checker 1 to see when duplicate check starts
start=time.time()

#Check if duplicate exist AFTER converting the unused column to 0
#For all the genomes

#NewGenConfigs = NewGenConfigs.astype(np.int64)

for a in range(len(NewGenConfigs)):
	#set the boolean keywords
	flag = True
	duplicate = False
	connect = True
	remake = False
	#flag becomes false ONLY when the target array satisfies every condition
	while(flag == True):
		#connection checking. This code looks for a scenario where only 1 item in a row is ON.
		#Such configuration is meaningless because only one bridge is made to the intermediate layer, thus no overall connection
		#Therefore we would like to have 0 or more than 2 1's per row to ensure that the ON switch leads to a current flow
		while(remake == False and connect == True):
			for b in range(len(NewGenConfigs[a])):
				connection = 0
				for c in range(len(NewGenConfigs[a][b])):
					if NewGenConfigs[a][b][c] == 1:
						connection = connection + 1
				if connection !=1:
					connect = False
					duplicate = True
				elif connection ==1:
					remake = True
					connect = False
		while(remake == False and duplicate == True):
			stack = 0
			templist = np.copy(NewGenConfigs[a])
			for d in range(len(NewGenConfigs)):
				if np.array_equal(templist, NewGenConfigs[d]):
					stack = stack + 1
			if stack == 1:
				NewGenConfigs[a] = NewGenConfigs[a]
				duplicate = False
				flag = False
			if stack >1:
				remake = True
				duplicate = False
		while(remake == True):
			NewGenConfigs[a] = np.random.rand(8,8)
			NewGenConfigs[a] = np.round(NewGenConfigs[a])
			for e in range(len(nullist)):
				NewGenConfigs[a,:,nullist[e]] = 0
			NewGenConfigs[a][0] = [0,0,0,0,0,0,0,0]
			NewGenConfigs[a][1] = [0,0,0,0,0,0,0,0]
			NewGenConfigs[a][4] = [0,0,0,0,0,0,0,0]
			remake = False
			connect = True
			duplicate = False



#Time checker 2 to see how long the duplicate elimination takes, this can take some motherfucking time
end = time.time()

print("Eliminating the duplicate from " + str(genomes) +" genomes took %f s" % ((end - start)))

#generate the plot figure, this is untested and can seriously influence evolution as their update speed may significantly hinder the process tempo of the GA
#mainFig = PlotBuilder.MainfigInitforFullSearch()

time.sleep(0.1)

#start the process, per generation
for m in range(generations):
	print("Generation " + str(m+1) + " begins")
	#Define the array to insert the fitness scores
	Fitnessscore = []
	successrate = []
	#per genomes
	for i in range(len(NewGenConfigs)):
		#Check the starting marker for the processing time
		start1 = time.time()
		print("Genome " + str(i+1) + "begins")
		bytelist = []
		sendlist = []
		
		for j in range(len(NewGenConfigs[i])):
			tempbits = 0
			for k in range(len(NewGenConfigs[i][j])):
				#for 1 in every location of the matrix, add it to the bit
				if NewGenConfigs[i][j][k] == 1:
					tempbits += 2**k
			#Bytelist will contain 8 byte(64 bits), starting from the top of the matrix
			bytelist.append(tempbits)

		#at this point, the bytelist is made, so convert to sendlist
		
		for l in range(len(bytelist)):
			sendlist.append(str(bytelist[l]))

		#Time checker 1 to see how long the byte and string conversion takes
		end1=time.time()

		#Send 8 byte info to the switch, it is configured in a certain interconnectivity
		#PlotBuilder.UpdateCurrentSwitchFullSearch(mainFig, array = NewGenConfigs[i])
		#maybe you need time for plot to update?
		#Plotbuilding takes too long, to avoid potential memory leak, this part will be excluded
		time.sleep(0.01)

		ser.write("<".encode())
		ser.write(sendlist[0].encode()+ ",".encode() +sendlist[1].encode()+ ",".encode() +sendlist[2].encode()+ 
	",".encode() +sendlist[3].encode()+ ",".encode() +sendlist[4].encode()+ ",".encode() +
	sendlist[5].encode()+ ",".encode() +sendlist[6].encode()+ ",".encode() +sendlist[7].encode())
		ser.write(">".encode())
		
		print ("Array sent")

		time.sleep(0.01)

		receivemessage = ser.readline()
		receivemessage = receivemessage.strip()
		receivemessage = receivemessage.split()

		#Print out the message received from Arduino
		print(receivemessage)

		#Instead of plotting, now we print out 8 by 8 array
		print(NewGenConfigs[i])
		
		#Timechecker2 to see how long it takes to send array, highly doubtful this slows done over the time
		end2 = time.time()

		evaluateinput =[]
		evaluateoutput = []

		#Make list = [1,2,4,8,16,32,64,128], corresponds to one port being open from a row

		#for a in range(devs):
			#These lines are for doing all 8
			#evaluateinput.append(2**(a))
			#evaluateoutput.append(2**(a))


		#give number from the makelist that corresponds to the digit
		inputlist1 =[146,231,167,237,207,207,39,239]
		inputlist2 = [36,114,121,73,57,59,73,123]

		#output port will be opened one at a time
		outputlist=[1,2,4,8,16,32,64,128]

		#For the current mode, stick to 1 dev per 1 evaliate
		Outputresult = np.zeros((devs, devs))


		#Evaluate output
		for a in range(len(evaluateinput)):
			for b in range(len(evaluateoutput)):
				time.sleep(0.1)
				#set the byte(input) into only one port opening
				bytelist[0] = inputlist1[a]
				bytelist[1] = inputlist2[a]
				#set the last byte(output) into only one port opening
				bytelist[4] = evaluateoutput[b]
				#send a bytelist where the input and output path are modified
				#reinitialize sendlist
				sendlist = []
				for l in range(len(bytelist)):
					sendlist.append(str(bytelist[l]))

				ser.write("<".encode())
				ser.write(sendlist[0].encode()+ ",".encode() +sendlist[1].encode()+ ",".encode() +sendlist[2].encode()+
					",".encode() +sendlist[3].encode()+ ",".encode() +sendlist[4].encode()+ ",".encode() +
					sendlist[5].encode()+ ",".encode() +sendlist[6].encode()+ ",".encode() +sendlist[7].encode())
				ser.write(">".encode())

				#print ("Array sent")

				time.sleep(0.3)

				receivemessage = ser.readline()
				receivemessage = receivemessage.strip()
				receivemessage = receivemessage.split()

				#Print out the message received from Arduino
				print(receivemessage)

				#Read current values, store into an output array
				current = keithley.curr.get()
				Outputresult[a][b] = current
				#This printing may slow down the whole thing, comment for now
				#print("Current recorded " + str(p) + " out of " + str(devs*devs))
				
				#p = p + 1


		#After the forloop with a, you should acquire dev by dev output array
		#PlotBuilder.UpdateIoutFullSearch(mainFig, array = Outputresult, devs = devs)
		#PlotBuilder.UpdateLastSwitch(mainFig, array = NewGenConfigs[i])
		#give time to update
		#print(Outputresult)

		#From end2, this gives time it takes for output evaluation, this is likely to slow down over the time
		print("Evaluation finished")
		end3 = time.time()
		#print("Genome " + str(i) + " took %f ms" % ((end - start) * 1000))
		time.sleep(0.1)

		F = 0
		success = 0
		#Tolerance. if set 0.5, it considers any output that has more than 50% of the highest current as "non-distinguishable"
		threshold = tolerance
		#Criteria 1
		for a in range(len(Outputresult)):
			count = 0
			tempout = Outputresult[a]
			maxi = max(tempout)
			for b in range(len(Outputresult[a])):
				#If the read current is higher than the threshold, add 1 to the count
				if Outputresult[a,b]/maxi >= threshold:
					count = count + 1
			#if only one output was HIGH for the given input, that's success!
			if count == 1:
				F = F + 3
			#if more than 1 output was HIGH for the given input, we give -1 for the number of outputs that were HIGH
			elif count > 1:
				F = F+ -1*count
			#if no output was HIGH for a particular input, we either have to lower the threshold, or just punish the fitness score
			#This is not implemented yet since count will be minimum 1, as we are determining by normalizing outputs with respect to the highest current.
			elif count == 0:
				F = F - 10

		#Criteria 2
		#Do exactly the same but transposed matrix. Vertical check
		TransOutputresult = np.copy(Outputresult)
		TransOutputresult = np.transpose(TransOutputresult)
		
		for a in range(len(Outputresult)):
			count = 0
			tempout = TransOutputresult[a]
			maxi = max(tempout)
			for b in range(len(Outputresult[a])):
				#If the read current is higher than the threshold, add 1 to the count
				if TransOutputresult[a,b]/maxi >= threshold:
					count = count + 1
			#if only one output was HIGH for the given input, that's success!
			if count == 1:
				F = F + 3
			#if more than 1 output was HIGH for the given input, we give -1 for the number of outputs that were HIGH
			elif count > 1:
				F = F+ -1*count
			#if no output was HIGH for a particular input, we either have to lower the threshold, or just punish the fitness score
			elif count == 0:
				F = F - 10
		#Append the fitness score of that genome to the fitnessscore
		Fitnessscore.append(F)

		#count how many distinctions it made
		for a in range(len(Outputresult)):
			#Assign a row as a temporary array
			tempout = Outputresult[a]
			#set maxi as the highest of that row
			maxi = max(tempout)
			for l in range(len(Outputresult[a])):
				#Identify the location of maximum in the matrix, call it Output[a][l] ,reverse the x and y order
				if Outputresult[a][l] == maxi:
					tempx = l
					tempy = a
			#now that the locaiton of max is found, transpose and check
			tempout = TransOutputresult[tempx]
			maxi = max(tempout)

			#If the transposed row, original column, and the original row had the same value as the maximum, it is "classifiable"
			if TransOutputresult[tempx][tempy] == maxi:
				success = success + 1

		successrate.append(success)
		#from end 3 gives the time it takes to analyze the fitness score, matrix process might take longer over the genomes
		end4 = time.time()

		#time calculation
		#byte and string conversion per genome
		conversiontime = end1 - start1
		#array sending duration per genome
		arraytime = end2 - end1
		#Output evaluation time per genome
		outputtime = end3 - end2
		#fitness score analysis per genome
		calculatetime = end4 - end3

		#Genome operation done, save the result in numpy array
		genearray[m,i, :, :] = NewGenConfigs[i,:]
		outputarray[m,i, :, :]=Outputresult
		fitnessarray[m,i] = F
		successarray[m,i] = success

		#This time array might not be necessary in the future, so ill put here instead
		timearray[m,i,0] = conversiontime
		timearray[m,i,1] = arraytime
		timearray[m,i,2] = outputtime
		timearray[m,i,3] = calculatetime

		#Add save library here so just in case, program can terminate midway
		SaveLib.saveMain(savedirectory, genearray, outputarray, fitnessarray, successarray, timearray)



	#Identify the winner from that generation by finding the INDEX of the winner
	winner = Fitnessscore.index(max(Fitnessscore))

	#Announcement
	print('The winner is the index' + str(winner))
	print(NewGenConfigs[winner])

	print('with a fitness score of ' + str(F))
	#Hermafrodite is the sole winner of the generation
	Hermafrodite = np.copy(NewGenConfigs[winner])

	#Save the generation result
	SaveLib.saveMain(savedirectory, genearray, outputarray, fitnessarray, successarray, timearray)

	#Mutation
	#Winner remains = 1
	NewGenConfigs[0] = np.copy(Hermafrodite)
	#PlotBuilder.UpdateBestConfig(mainFig, array = Hermafrodite)

	#Mutate with 10% chance for 2-8
	for i in range(1, int(genome*4/5)):
		#copy the winner
		templist = np.copy(Hermafrodite)
		#while dupilcate is true, it won't allow the mutation to complete
		duplicate = True
		restart = True
		while(duplicate):
			for j in range(genes):
				for k in range(devs):
					#By both rows and columns, roll a die, with a certain chance, if yes, change 1 to 0 and 0 to 1
					if(np.random.rand() < 0.2):
						if templist[j, k] == 1:
							templist[j, k] = 0
						elif templist[j, k] == 0:
							templist[j, k] = 1
			#After the change has been made to the temporary array, set stack variable
			stack = 0
			for a in range(len(genearray)):
				for b in range(len(genearray[a])):
					#Go through all the genearray, IF the mutated temporary array is identical to the previously existing genome, add a point to the stack
					if np.array_equal(templist, genearray[a][b]):
						stack = stack + 1
			#If stack is increased (at all), stay in the duplicate, remutate the temporary array until no duplicate is found
			if stack < 1:
				NewGenConfigs[i] = np.copy(templist)
				#set duplicate to the false, go out of the while loop, and we can go back to the for loop for mutation
				duplicate = False

	#complete random for 9-10, but with the duplicate check
	for i in range(int(genome*4/5),genome):
		duplicate = True
		while(duplicate):
			templist = np.random.rand(genes,devs)
			templist = np.around(templist)
			stack = 0
			for a in range(len(genearray)):
				for b in range(len(genearray[a])):
					if np.array_equal(templist, genearray[a][b]):
						stack = stack + 1
			if stack < 1:
				NewGenConfigs[i] = np.copy(templist)
				duplicate = False		

#In the current version, there is a problem with this duplicate checker, we use 8 by 8 array as a genome identification (DNA) 
#but we really only utilize the switches that are not connected to the input.
#our duplicate checker will check every bits, therefore, the actual switches used to evolve the connectivity may be the same. 
#But if the first row, input byte, is different, they'll think it's a different genome, even though the real functionality part is the same.
#As such, two different genomes with an identical row 2,3,4,6,7,8 and nonidentical row 1 and 5, will be saved, despite them being practically the same

#Repeat until all generations meet 
#don't know why I need this. But the window disappears without this command.
#PlotBuilder.finalMain(mainFig)

#========================================================================================================================================================================
#This space used for the digit conversion of 3 by 5 digit to binary
#
#0:					1:					2:					3:			
#	1	1	1			0	1	0			1	1	1			1	1	1
# 	1	0	1			0	1	0			0	0	1			0	0	1
#	1	0	1			0	1	0			1	1	1			0	1	1
#	1	0	1			0	1	0			1	0	0			0	0	1
#	1	1	1			0	1	0			1	1	1			1	1	1
#
#4:					5:					6:					7:
#	1	0	1			1	1	1			1	1	1			1	1	1
#	1	0	1			1	0	0			1	0	0			0	0	1
#	1	1	1			1	1	1			1	1	1			0	0	1
#	0	0	1			0	0	1			1	0	1			0	0	1
#	0	0	1			1	1	1			1	1	1			0	0	1
#
#8:					9:
#	1	1	1			1	1	1
#	1	0	1			1	0	1
#	1	1	1			1	1	1
#	1	0	1			0	0	1
#	1	1	1			1	1	1
#
#For all 10 digit, I guess we have to use two output row, which will heavily limit the switch configuration
#
#For conversion purpose
#Suppose matrix looked like								We will allocate our digit as 
#	|1	|2	|3	|4	|5	|6	|7	|8	|						
#------------------------------------						A1 A2 A3 	  
#A  |   |   |   |   |   |   |   |   |						A4 A5 A6
#------------------------------------						A7 A8 B1
#B  |   |   |   |   |   |   |   |   |						B2 B3 B4
#------------------------------------						B5 B6 B7
#C  |   |   |   |   |   |   |   |   |	
#------------------------------------					And B8 will be permanently closed (otherwise there's gonna be a connection to the battery where it's not intended)
#D  |   |   |   |   |   |   |   |   |
#------------------------------------
#E  |   |   |   |   |   |   |   |   |
#------------------------------------
#F  |   |   |   |   |   |   |   |   |
#------------------------------------
#G  |   |   |   |   |   |   |   |   |
#------------------------------------
#H  |   |   |   |   |   |   |   |   |
#------------------------------------
#
#As such							binary
# digit 0 is 11110110, 11011010		111		91
#		1 is 01001001, 00100100		146		36
#		2 is 11100111, 01001110		231		114
#		3 is 11100101, 10011110		167		121
#		4 is 10110111, 10010010		237		73
#		5 is 11110011, 10011110		207		57
#		6 is 11110011, 11011110		207		59
#		7 is 11100100, 10010010		39		73
#		8 is 11110111, 11011110		239		123
#		9 is 11110111, 10011110		239		121
#========================================================================================================================================================================