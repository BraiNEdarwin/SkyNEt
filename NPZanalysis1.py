import numpy as np 
import matplotlib.pyplot as plt
import math
import config_SWOG as config

#load your NPZ file from the right directory
data=np.load('D:\\test2019_01_10_161600_testing\\data.NPZ') 

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

#plot the data
plt.figure()
plt.plot(x,output1)
plt.show()

