import numpy as np 
import matplotlib.pyplot as plt
import math

data=np.load('C:\\Users\\VenB\\Desktop\\Evolution\\2018_11_11_144633_AND_77%_high\\nparrays.npz')

print(data.keys())

farray = data['fitnessArray']
garray = data['geneArray']
outputArray = data['outputArray']
outp = data['outp']
t = data['t']

for i in range(25):
    plt.figure()
    plt.plot(t,outputArray[0,i])