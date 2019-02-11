import numpy as np 
import matplotlib.pyplot as plt

data = np.load(r'D:\Tao\DN-BdW190206-evolution\XOR\2019_02_06_163135_XOR\\nparrays.npz')

print(data.keys())

fitnessArray = data['fitnessArray']
outputArray = data['outputArray']
geneArray = data['geneArray']
inp = data['inp']