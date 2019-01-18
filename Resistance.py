import numpy as np 
import matplotlib.pyplot as plt
import math

data = np.load('D:\\R_cold2019_01_15_152759_test\\data.NPZ')

print(list(data.keys()))

input_2d = data['input']
output_2d = data['output']

input = input_2d[0]
output_b = (output_2d[0]*(-1))/0.069
output = output_b*10E-9
resistance = input/output

plt.figure()
plt.plot(input,resistance)
plt.show()