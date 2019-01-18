import numpy as np 
import matplotlib.pyplot as plt
import math
import config_SWOG as config

#load your NPZ file from the right directory
data=np.load('D:\\Useful\\62_CH4_01V_warm2019_01_10_174758_test\\data.NPZ') 

# shows which names are linked to different parts of your data
print(list(data.keys()))	


#make the different data sets accesible
input1_2d = data['input']
output1_2d = data['output']
x = np.arange(len(output1_2d[0]))

#reverse the data
output1_reversed = output1_2d[0]
output1_unscaled = output1_reversed
input1 =  input1_2d[0]

#scale the current (0.069)
output1 = (output1_unscaled/0.069)*(-1)

#comparison with ideal situation
#Load the information from the config class.
#config = config.experiment_config()
#Input = config.SquareWave( config.v_high, config.v_low, config.n_points)
#y = np.arange(len(Input))

#plot the data
plt.figure()
plt.plot(input1,output1)
#plt.plot(x, Input, label = 'ideal situation')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [nA]')
#plt.legend( loc = 'best')
plt.show()

