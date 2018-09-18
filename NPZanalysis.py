import numpy as np 


data = np.load('D:\data\Hans\\2018_09_14_135519_Distribution\\nparrays.npz')

print(data.keys())

data = data['data']