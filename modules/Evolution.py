'''
Handles evolution using fitness from post-process
'''
import numpy as np


class GenePool(object):

    def __init__(self, genes, genomes):

        self.genes = genes
        self.genomes = genomes
        self.pool = np.random.rand(genes, genomes)
        self.fitness = np.empty(genomes)

    def nextGen(self):
        indices = np.argsort(self.fitness)
        indices = indices[::-1]
        quarter = int(self.genomes / 4)
        newPool = self.pool
        # promote fittest quarter of the pool
        for i in range(quarter):
            newPool[:, i] = self.pool[:, indices[i]]

        # generate second quarter
        for i in range(quarter):
            parent1 = newPool[:, i] * 1.01
            parent2 = newPool[:, i] * 0.99

            # check that genes are in [0,1]
            for j in range(self.genes):
                if(parent1[j] > 1):
                    parent1[j] = 1
                if(parent1[j] < 0):
                    parent1[j] = 0
                if(parent2[j] > 1):
                    parent2[j] = 1
                if(parent2[j] < 0):
                    parent2[j] = 0

            for j in range(self.genes):
                # crossover
                if(np.random.rand() < 0.5):
                    newPool[j, i + quarter] = parent1[j]
                else:
                    newPool[j, i + quarter] = parent2[j]

                # mutation
                if(np.random.rand() < 0.1):
                    newPool[j, i + quarter] = np.random.rand()

        # generate third quarter
        for i in range(quarter):
            for j in range(self.genes):
                if(np.random.rand() < 0.5):
                    newPool[j, i + 2 * quarter] = self.pool[j, indices[i]]
                else:
                    newPool[j, i + 2 * quarter] = self.pool[j, indices[i + 1]]

                # mutation
                if(np.random.rand() < 0.1):
                    newPool[j, i + 2 * quarter] = np.random.rand()

        # generate fourth quarter
        for i in range(quarter):
            for j in range(self.genes):
                if(np.random.rand() < 0.5):
                    newPool[j, i + 3 * quarter] = newPool[j, i]
                else:
                    newPool[j, i + 3 * quarter] = self.pool[j,
                                                            np.random.randint(self.genomes)]

                # mutation
                if(np.random.rand() < 0.1):
                    newPool[j, i + 3 * quarter] = np.random.rand()
 
        # replace pool
        self.pool = newPool   
        
        #empty fitness
        self.fitness = np.empty(self.genomes) 

    def returnPool(self):
        return self.pool
