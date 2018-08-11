import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math

def MainfigInit(genes, generations):
	plt.ioff()
	mainFig = plt.figure()
	figManager = plt.get_current_fig_manager()
	#figManager.window.showMaximized()
	plt.pause(0.01)
	spec = gridspec.GridSpec(ncols=16, nrows=genes)
	for i in range(genes):
		for j in range(genes):
			ax = mainFig.add_subplot(spec[i, j])
			ax.set_xlim(1, generations)
			ax.set_ylim(0, 1)
			ax.grid()
			#ax.set_title(genelabel[i][j])
			plt.rc('xtick', labelsize = 2)
			plt.rc('ytick', labelsize = 2)

	axBestConfig = mainFig.add_subplot(spec[0:int(genes/2), 16-genes:16-int(genes/2)])
	#axBestConfig.grid()
	axBestConfig.set_title('Best configuration so far')

	axSwitchConfig = mainFig.add_subplot(spec[0:int(genes/2), 16-int(genes/2):16])
	#axSwitchConfig.grid()
	axSwitchConfig.set_title('Switch configuration')

	axIout = mainFig.add_subplot(spec[int(genes/2):genes, 16-int(genes/2):16])
	#axIout.grid()
	axIout.set_title('CurrentOutput')	

	
	plt.tight_layout()
	mainFig.subplots_adjust(hspace = 0.1, wspace = 0.1)
	return mainFig

def MainfigInitforFullSearch():
	plt.ioff()
	mainFig = plt.figure()
	figManager = plt.get_current_fig_manager()
	#figManager.window.showMaximized()
	plt.pause(0.01)
	spec = gridspec.GridSpec(ncols=16, nrows=10)

	axBestConfig = mainFig.add_subplot(spec[8:16,3:8])
	#axBestConfig.grid()
	axBestConfig.set_title('Currently testing Switch Config')

	axSwitchConfig = mainFig.add_subplot(spec[0:7, 6:10])
	#axSwitchConfig.grid()
	axSwitchConfig.set_title('Switch configuration for the IOut')

	axIout = mainFig.add_subplot(spec[0:7, 0:5])
	#axIout.grid()
	axIout.set_title('CurrentOutput')	

	
	plt.tight_layout()
	mainFig.subplots_adjust(hspace = 0.1, wspace = 0.1)
	return mainFig

def UpdateSwitchConfig(mainFig, array):
	mainFig.axes[-2].imshow(array, cmap = 'gray')
	plt.pause(0.01)

def UpdateBestConfig(mainFig, array):
	mainFig.axes[-3].imshow(array, cmap = 'gray')
	plt.pause(0.01)

def UpdateIout(mainFig, array, devs):
	mainFig.axes[-1].imshow(array)
	k = 0
	array4 = np.copy(array)

	for i in range(len(array)):
		for j in range(len(array[i])):
			array4 = array4.astype(int)
			array4 = np.reshape(array4,devs*devs)
			text = mainFig.axes[-1].text(j, i, array4[k], ha="center", va="center", color="w")
			k= k+1
	plt.pause(0.01)

def UpdateCurrentSwitchFullSearch(mainFig, array):
	mainFig.axes[-3].imshow(array, cmap = 'gray')
	plt.pause(0.01)

def UpdateIoutFullSearch(mainFig, array, devs):
	mainFig.axes[-1].imshow(array)
	k = 0
	array4 = np.copy(array)

	for i in range(len(array)):
		for j in range(len(array[i])):
			array4 = array4.astype(int)
			array4 = np.reshape(array4,devs*devs)
			newI = array4[k] * 100000000
			text = mainFig.axes[-1].text(j, i, newI, ha="center", va="center", color="w")
			k= k+1
	plt.pause(0.01)

def UpdateLastSwitch(mainFig, array):
	mainFig.axes[-2].imshow(array, cmap = 'gray')
	plt.pause(0.01)

def UpdateSwitchHistory(mainFig, array, genes, currentgen,genearray):
	for i in range(genes):
		for j in range(genes):
			mainFig.axes[-4 - 8*i - j].plot(range(genearray[0][0][i][j], genearray[currentgen][0][i][j]), range(0,currentgen), 'r-x')


def finalMain(mainFig):
	plt.show()