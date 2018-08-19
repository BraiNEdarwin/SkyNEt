'''
Handles evolution using fitness from post-process
'''
import numpy as np

#TODO All methods need description and comments
#TODO Check for duplicate genomes
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
        self.pool = self.pool[indices]  # Sort genepool on fitness
        self.newpool = self.pool.copy() # promote fittest partition[0] gene configurations

        # generate second partition by adding Gaussian noise to the fittest partition
        self.AddNoise()

        # generate third partition by mixing fittest :partition[2] with fittest 1:partition[2]
        self.CrossoverFitFit()

        # generate fourth partition by mixing fittest with randomly selected
        self.CrossoverFitRandom()

        # generate fifth partition by uniform sampling
        self.AddRandom()

        # Mutation over all new partitions
        self.newpool[self.partition[0]:self.partition[1]] = self.Mutation(
                    self.newpool[self.partition[0]:self.partition[1]])
                    
        # replace pool
        self.pool = self.newpool.copy()

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
        #TODO I believe this is wrong, it should add noise to the fittest partition[1]
        # genomes, but now it adds noise to fittest partition[0]-partition[1] genes.
        self.newpool[self.partition[0]:self.partition[1]] += 0.02*np.random.randn(
                self.partition[1],self.newpool.shape[1])

        # check that genes are in [0,1]
        buff = self.newpool[self.partition[0]:self.partition[1]] > 1.0
        self.newpool[self.partition[0]:self.partition[1]][buff] = 1.0

        buff = self.newpool[self.partition[0]:self.partition[1]] < 0.0
        self.newpool[self.partition[0]:self.partition[1]][buff] = 0.0

    def CrossoverFitFit(self):
        '''Perform crossover between the fittest :partition[2] genomes and the
        fittest 1:partition[2]+1 genomes'''
        filter = np.random.randint(2, size=(self.partition[2], self.genes))
        self.newpool[sum(self.partition[:2]):sum(self.partition[:3])]
            = filter * self.pool[:self.partition[2]]
                + (np.ones(filter.shape) - 1) * self.pool[1:self.partition[2]+1]

    def CrossoverFitRandom(self):
        '''Perform crossover between the fittest :partition[3] genomes and random
        genomes'''
        filter = np.random.randint(2, size=(self.partition[3], self.genes))
        self.newpool[sum(self.partition[:3]):sum(self.partition[:4])]
            = filter * self.pool[:self.partition[3]]
                + (np.ones(filter.shape) - 1) * self.pool[np.random.randint(self.genomes, size=(self.partition[3], self.genes))]

    def AddRandom():
        '''Generate partition[4] random genomes'''
        self.newpool[sum(self.partition[:4]):] = np.random.rand((partition[4], self.genes))
