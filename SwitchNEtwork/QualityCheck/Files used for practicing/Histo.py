import numpy as np
import matplotlib.pyplot as plt
import math

#exec(open("setup.txt").read())
data = np.load('/Users/renhori/Desktop/SwiNEt_07_08_2018_173315_FullSearchTry1/DataArrays.npz')

x = data.f.fitnessarray

'''
rng = np.random.RandomState(10)  # deterministic random data
a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
'''
#plt.figure()
#b = np.random.rand(100)
#plt.hist(x)
#plt.show()

print(x)
plt.hist(x)
plt.show()