import numpy as np 
import matplotlib.pyplot as plt
import math

data=np.load('D:\\Evolutioncheck\\2018_10_11_105529_XOR\\nparrays.npz') #load your NPZ file from the right directory

print(data.keys())	# shows which names are linked to different parts of your data


#Mmake the different data sets accesible
farray = data['fitnessArray']
garray = data['geneArray']
outputArray = data['outputArray']
outp = data['outp']
t = data['t']


#here you can plot the desired data using the names above (in this example I wanted to make 25 plots of different data sets to compare)
for i in range(1):
    plt.figure()
    plt.plot(t,outputArray[0,2]*10)
    plt.show()