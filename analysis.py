import numpy as np 
import matplotlib.pyplot as plt
import math

# turn this part into a config file.
# range of genome voltages.
low_range = -2
high_range = 2
# determine wich file to plot.
Big_daddy_gen = 65
# loads .npz file.
data = np.load('D:/data/Ren/Evolution/19_04_2018_164956_AND/nparrays.npz')



Fitness_Array = np.zeros(Big_daddy_gen)
#save the fitness matrix as f.
f = data.f.fitnessArray

n = 0
while n <= Big_daddy_gen-1:
	#find position of highest value and save it in the fitness_Array.
	fitness = f.argmax()
	Array_length = np.shape(f)
	Array_highest = int(math.floor(fitness/Array_length[1]))
	Positon_highest = f[Array_highest].argmax()
	Fitness_Array[n] = f[Array_highest,Positon_highest]

	#overwrite the highest fitness such that the second highest etc. can be found.
	f[Array_highest,Positon_highest] = 0.1
	n = n+1

#show the fitness of the chosen big daddy.
print('Fitness =') 
print( Fitness_Array[n-1])

print('V1-V5')
print(data.f.geneArray[Array_highest,:,Positon_highest]*(high_range-low_range)+low_range)
#plot different parts of the measurement.
# plt.plot(data.f.t,data.f.o)
plt.plot(data.f.t,data.f.outputArray[Array_highest,:,Positon_highest])
plt.plot(data.f.t,data.f.inp[0])
plt.plot(data.f.t,data.f.inp[1])
plt.show()
