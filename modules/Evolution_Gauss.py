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
            parent1 = newPool[:, i] + 0.02*np.random.randn(newPool[:, i].shape[0])
            parent2 = newPool[:, i] + 0.02*np.random.randn(newPool[:, i].shape[0])

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
                    newPool[j, i + fifth] = self.mutation(newPool[j, i + fifth])

        # generate third fifth
        for i in range(fifth):
            for j in range(self.genes):
                if(np.random.rand() < 0.5):
                    newPool[j, i + 2 * fifth] = self.pool[j, indices[i]]
                else:
                    newPool[j, i + 2 * fifth] = self.pool[j, indices[i + 1]]

                # mutation
                if(np.random.rand() < 0.1):
                    newPool[j, i + 2 * fifth] = self.mutation(newPool[j, i + 2 * fifth])

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
                    newPool[j, i + 3 * fifth] = self.mutation(newPool[j, i + 3 * fifth])

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
    
    def mutation(self,mode):
        mutant = np.random.triangular(0,mode,1)
        return mutant