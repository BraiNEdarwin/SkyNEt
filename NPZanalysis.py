import numpy as np 
import matplotlib.pyplot as plt
import math
import config_SWOG as config

#load your NPZ file from the right directory
data=np.load('D:\\R_W2019_01_15_153517_test\\data.NPZ') 

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
#here you can plot the desired data using the names above (in this example I wanted to make 25 plots of different data sets to compare)

#load your second NPZ file from the right directory
data=np.load('D:\\test2019_01_10_161600_testing\\data.NPZ')

# shows which names are linked to different parts of your data
print(list(data.keys()))

#make the different data sets accesible
input2_2d = data['input']
output2_2d = data['input']
y = np.arange(len(output2_2d[0]))

#reverse the data
output2_reversed = output2_2d[0]
output2_unscaled = output2_reversed
input2 = input2_2d[0]

#scale the current (0.069)
output2 = (output2_unscaled/0.069)


# Plot the Square wave
plt.figure()
plt.plot(x, output1)
plt.plot(x, output2)
plt.xlabel('Samples [n]')
plt.ylabel('Current [nA]')
plt.show()

#comparison with ideal situation
# Load the information from the config class.
#config = config.experiment_config()
#Input = config.SquareWave( config.v_high, config.v_low, config.n_points)
#y = np.arange(len(Input))