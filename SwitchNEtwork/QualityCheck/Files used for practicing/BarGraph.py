import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

#Generate 3d plot of one genome or entire generation

genomeplot = False
genplot = False

#Random generate for checking
generations = 10
genomes = 8
genes = 8
Outputresult = np.random.rand(generations, genomes, genes, genes)

#choose the generation or genome

gen = 0
geno = 0
print(Outputresult[gen][geno])

xposr = []
yposr = []
zposr = []
dzr = []
dxr = np.ones(genes*genes)
dxr = dxr/2
dyr = np.ones(genes*genes)
dyr = dyr/2


for m in range(len(Outputresult[gen][geno])):
    for n in range(len(Outputresult[gen][geno][m])):
        xposr.append(m+1)
        yposr.append(n+1)
        zposr.append(0)
        dzr.append(Outputresult[gen][geno][m][n])


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
'''
xpos = [1,2,3,4,5,6,7,8,9,10]
ypos = [2,3,4,5,1,6,2,1,7,2]
num_elements = len(xpos)
zpos = [0,0,0,0,0,0,0,0,0,0]
dx = np.ones(10)
dy = np.ones(10)
dz = [1,2,3,4,5,6,7,8,9,10]
'''

ax1.bar3d(xposr, yposr, zposr, dxr, dyr, dzr, color='#00ceaa')
plt.show()