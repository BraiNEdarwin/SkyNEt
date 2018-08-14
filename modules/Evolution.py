'''
Handles evolution using fitness from post-process
'''
import numpy as np


class GenePool(object):

    def __init__(self, config_obj):

        self.genes = len(config_obj.generange)
        self.genomes = config_obj.genomes
        self.pool = np.random.rand(self.genomes, self.genes)
        self.fitness = np.zeros(self.genomes)
        self.partition = config_obj.partition
        self.mutationrate = config_obj.mutationrate

    def nextGen(self):
        indices = np.argsort(self.fitness)
        indices = indices[::-1]
        self.pool = self.pool[indices]
        self.newpool = self.pool.copy() # promote fittest partition gene configurations

        # generate second partition by adding Gaussian noise to the fittest partition
        self.AddNoise()
        # mutation
        self.newpool[self.partition[0]:self.partition[1]] = self.mutation(
                    self.newpool[self.partition[0]:self.partition[1]])
         
        ##TODO: encapsulate in crossover methods 
        # generate third partition by mixing fittest n with fittest n+1
        for i in range(fifth):
            for j in range(self.genes):
                if(np.random.rand() < 0.5):
                    newpool[j, i + 2 * fifth] = self.pool[j, indices[i]]
                else:
                    newpool[j, i + 2 * fifth] = self.pool[j, indices[i + 1]]

                # mutation
                newpool[j, i + 2 * fifth] = self.mutation(newpool[j, i + 2 * fifth])

        # generate fourth partition by mixing fittest with randomly selected 
        for i in range(fifth):
            for j in range(self.genes):
                if(np.random.rand() < 0.5):
                    newpool[j, i + 3 * fifth] = newpool[j, i]
                else:
                    newpool[j, i + 3 * fifth] = self.pool[j,
                                                            np.random.randint(self.genomes)]

                # mutation
                  newpool[j, i + 3 * fifth] = self.mutation(newpool[j, i + 3 * fifth])

        # generate fifth partition by uniform sampling
        for i in range(fifth):
            for j in range(self.genes):
                newpool[j, i + 4 * fifth] = np.random.rand()

        # replace pool
        self.pool = newpool   
        
        #empty fitness
        self.fitness = np.empty(self.genomes)

    
    def Mutation(self,mode):
        if(np.random.rand() < self.mutationrate):
            mutant = np.random.triangular(0,mode,1)
        else:
            mutant = mode
        return mutant
    
    def MapGenes(self,generange, gene):
        return generange[0] + gene * (generange[1] - generange[0])
    
    def AddNoise(self):
        self.newpool[self.partition[0]:self.partition[1]] += 0.02*np.random.randn(
                self.partition[1],self.newpool.shape[1])

        # check that genes are in [0,1]
        buff = self.newpool[self.partition[0]:self.partition[1]] > 1.0
        self.newpool[self.partition[0]:self.partition[1]][buff] = 1.0
        
        buff = self.newpool[self.partition[0]:self.partition[1]] < 0.0
        self.newpool[self.partition[0]:self.partition[1]][buff] = 0.0

