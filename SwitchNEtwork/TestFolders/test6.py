from instrument import PlotBuilding as PlotBuilder
import time
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import math

#open the figure
mainFig = PlotBuilder.MainfigInit(genes = 8, generations = 200)
currentgen = 1
genes = 8
time.sleep(3)
k = 0
genearray = np.zeros((200,10, genes,genes), dtype = int)
#array3 = np.array([[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16],[1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,1],[2,3,4,5,6,7,8,9],[10,9,8,7,6,5,4,3],[6,5,3,4,7,5,2,1],[9,7,5,3,1,2,4,6]])

for i in range(100):
	array = np.random.rand(8,8)
	array = np.round(array)
	genearray[i][0] = np.copy(array)
	PlotBuilder.UpdateSwitchConfig(mainFig, array)
	time.sleep(1)
	#only need the best result which is the first genome of each gen
	PlotBuilder.UpdateSwitchHistory(mainFig, array, genes, currentgen, genearray)
	time.sleep(10)
	currentgen = currentgen+1


PlotBuilder.finalMain(mainFig)