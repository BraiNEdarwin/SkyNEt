'''
Handles evolution using fitness from post-process
'''
import numpy as np


def mapGenes(generange, gene):
    return generange[0] + gene * (generange[1] - generange[0])


class GenePool(object):

    def __init__(self, genes, genomes):

        self.genes = genes
        self.genomes = genomes
        self.pool = np.random.rand(genes, genomes)
        self.fitness = np.empty(genomes)

    def nextGen(self):
        indices = np.argsort(self.fitness)
        indices = indices[::-1]
        fifth = int(self.genomes / 5)
        newPool = self.pool.copy()
        # promote fittest fifth of the pool
        for i in range(fifth):
            newPool[:, i] = self.pool[:, indices[i]]

        # generate second fifth
        for i in range(fifth):
            parent1 = newPool[:, i] * 1.05
            parent2 = newPool[:, i] * 0.95

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
                    newPool[j, i + fifth] = parent1[j]
                else:
                    newPool[j, i + fifth] = parent2[j]

                # mutation
                if(np.random.rand() < 0.1):
                    newPool[j, i + fifth] = np.random.rand()

        # generate third fifth
        for i in range(fifth):
            for j in range(self.genes):
                if(np.random.rand() < 0.5):
                    newPool[j, i + 2 * fifth] = self.pool[j, indices[i]]
                else:
                    newPool[j, i + 2 * fifth] = self.pool[j, indices[i + 1]]

                # mutation
                if(np.random.rand() < 0.1):
                    newPool[j, i + 2 * fifth] = np.random.rand()

        # generate fourth fifth
        for i in range(fifth):
            for j in range(self.genes):
                if(np.random.rand() < 0.5):
                    newPool[j, i + 3 * fifth] = newPool[j, i]
                else:
                    newPool[j, i + 3 * fifth] = self.pool[j,
                                                            np.random.randint(self.genomes)]

                # mutation
                if(np.random.rand() < 0.1):
                    newPool[j, i + 3 * fifth] = np.random.rand()

        # generate fifth fifth
        for i in range(fifth):
            for j in range(self.genes):
                newPool[j, i + 4 * fifth] = np.random.rand()

        # replace pool
        self.pool = newPool.copy()   
        
        #empty fitness
        self.fitness = np.empty(self.genomes) 

    def returnPool(self):
        return self.pool
