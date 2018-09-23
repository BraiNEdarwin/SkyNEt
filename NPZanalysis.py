import numpy as np 
import matplotlib.pyplot as plt

#data = np.load('D:\data\Hans\\2018_09_14_135519_Distribution\\nparrays.npz')
#
#print(data.keys())
#
#data = data['data']


input = np.array([[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0]])
inputsc = 1
lens = 1000
lene = 100
lent = 10000
for ii in input:
    x = np.array([])
    for i in range(len(ii)-1):
        x1 = np.linspace(ii[i],ii[i],lens)
        x2 = np.linspace(ii[i],ii[i+1],lene)
        x = np.append(x,x1)
        x = np.append(x,x2)
    x = np.append(x,np.zeros(lent))
    plt.plot(range(len(x)),x)
    plt.show()