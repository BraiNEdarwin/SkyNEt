import numpy as np 
import matplotlib.pyplot as plt
import math

# Read config.txt file
#exec(open("config_software.txt").read())

#loads .npz file
def generate_cp(n=100, mean_I0=-0.3, mean_I1=-0.3, amp_I0=0.9, amp_I1=0.9):
     values_I0 = [mean_I0-amp_I0+amp_I0*2/2*(i//n//7) for i in range(21*n)]
     values_I1 = [mean_I1-amp_I1+amp_I1*2/6*(i//n%7) for i in range(21*n)]
     input_data = np.array([[values_I0],[values_I1]])
     targets = np.array([0,0,0,1,1,1,1,0,1,1,1,1,2,2,1,1,2,1,2,1,2])
     target_data = np.zeros((2100))
     for i in range(len(targets)):
         target_data[i*100:i*100+100] = np.ones(100)*targets[i]
     return input_data, target_data


input = generate_cp(n=100, mean_I0=-0.3, mean_I1=-0.3, amp_I0=0.9, amp_I1=0.9)
print(input[1])
print(input[0][1])
plt.figure()
plt.plot(input[1])
plt.plot(input[0][1][0])
plt.plot(input[0][0][0])
plt.show()

