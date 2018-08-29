#================================================================================================
#
#This code is written specifically for the switch involved IV curve result.
#
#================================================================================================
import numpy as np
import matplotlib.pyplot as plt

data= np.load('/Users/renhori/Desktop/Twente/Year2/Thesis/Result/Switch/SwiNEt_21_08_2018_123133_EightDeviceSwitchIV77K/IVDataz.npz')
result =data.f.currentlist

b = 1

plt.figure(1)
for a in range(len(result[b])):
    plt.plot(result[b][a][0], result[b][a][1]*1000000000, label = 'Output ' + str(a+1))
    plt.ylabel('Amp (nA)')
    plt.xlabel('Volt (V)')
    plt.grid(True)
    plt.legend(loc = 'lower right')

plt.show()