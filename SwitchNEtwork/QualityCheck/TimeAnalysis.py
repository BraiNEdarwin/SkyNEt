'''
This code does the timing analysis, the genome numbers need to be manually fixed

>>3018-8-30
This part of the code is included in the final verdict. SCRATCH

'''


import numpy as np
import matplotlib.pyplot as plt
import math

#write here the path to npz file
data = np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Result/Switch/SixDevFS/DataArrays.npz')

switch = data.f.genearray
output = data.f.outputarray
fitness = data.f.fitnessarray
success = data.f.successarray
#timearray[m,i,0] = conversiontime
#timearray[m,i,1] = arraytime
#timearray[m,i,2] = outputtime
#timearray[m,i,3] = calculatetime
time = data.f.timearray

timenew = np.zeros((1,4,3000))

for a in range(len(timenew[0])):
	for b in range(len(timenew[0][a])):
		timenew[0][a][b] = time[0][b][a]

i = np.linspace(1,3000,3000)
plt.subplot(221)
plt.plot(i,timenew[0][0])
plt.ylabel('Time (s)')
plt.xlabel('Genomes')
plt.grid(True)
plt.title('Conversion Time')

plt.subplot(222)
plt.plot(i,timenew[0][1])
plt.ylabel('Time (s)')
plt.xlabel('Genomes')
plt.grid(True)
plt.title('Array Time')

plt.subplot(223)
plt.plot(i,timenew[0][2])
plt.ylabel('Time (s)')
plt.xlabel('Genomes')
plt.grid(True)
plt.title('Output evaluation Time')

plt.subplot(224)
plt.plot(i,timenew[0][3])
plt.ylabel('Time (s)')
plt.xlabel('Genomes')
plt.grid(True)
plt.title('Fitness calculation Time')

plt.show()
