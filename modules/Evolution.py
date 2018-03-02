'''
Handles evolution using fitness from post-process
'''
import numpy as np

class GenePool(object):

	def __init__(self, genes, genomes, generange):

		self.pool = np.empty(genes + 1, genomes)  #last row is used for fitness score

		for i in range(genes):
			for j in range(genomes):
				self.pool[i,j] = generange[i][0] + (generange[i][1] - generange[i][0]) * np.random.rand()

	def nextGen(self):
		indices = range(genes + 1)
		indices.sort()
		fitnessArray = self.pool[genes]
