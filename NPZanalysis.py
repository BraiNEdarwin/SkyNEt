import numpy as np 
import matplotlib.pyplot as plt

data = np.load(r'D:\Tao\TCSP2NAND\2018_09_26_164623_NAND\\nparrays.npz')

print(data.keys())

fitnessArray = data['fitnessArray']
outputArray = data['outputArray']
geneArray = data['geneArray']
inp = data['inp']