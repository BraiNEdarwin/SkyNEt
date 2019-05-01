'''
A Class definition that defines a genetic algorithm. 
'''
import numpy as np


class GenePool(object):

    def __init__(self, config_obj):

        self.config_obj = config_obj
        self.genes = config_obj.genes
        self.genomes = config_obj.genomes
        
        self.generange = config_obj.generange
        self.pool = np.zeros((self.genomes, self.genes))
        for i in range(0,self.genes):
            self.pool[:,i] = np.random.uniform(self.generange[i][0], self.generange[i][1], size=(self.genomes,))
        #input scaling between 0 and 1 
        self.fitness = np.zeros(self.genomes)
        self.partition = config_obj.partition
        self.mutationrate = config_obj.mutationrate

    def NextGen(self):
        indices = np.argsort(self.fitness)
        indices = indices[::-1]
        self.pool = self.pool[indices]  # Sort genepool on fitness
        self.newpool = self.pool.copy() # promote fittest partition[0] gene configurations

        # Generate second partition by adding Gaussian noise to the fittest partition
        self.AddNoise()

        # Generate third partition by mixing fittest :partition[2] with fittest 1:partition[2]
        self.CrossoverFitFit()

        # Generate fourth partition by mixing fittest with randomly selected
        self.CrossoverFitRandom()

        # Generate fifth partition by uniform sampling
        self.AddRandom()

        # Mutation over all new partitions
        self.Mutation()
        
        # Check for duplicate genomes
        self.RemoveDuplicates()
            
        # Replace pool
        self.pool = self.newpool.copy()

        # Reset fitness
        self.fitness = np.zeros(self.genomes)

    def Mutation(self):
        '''Mutate all genes but the first partition[0] with a triangular 
        distribution between 0 and 1 with mode=gene. The chance of mutation is 
        config_obj.mutationrate'''
        mask = np.random.choice([0, 1], size=self.pool[self.partition[0]:].shape, 
                                  p=[1-self.config_obj.mutationrate, self.config_obj.mutationrate])
        mutatedpool = np.zeros((self.genomes-self.partition[0], self.genes))
    
        for i in range(0,self.genes):
          
            mutatedpool[:,i] = np.random.triangular(self.generange[i][0], self.newpool[self.partition[0]:,i], self.generange[i][1])
            
        self.newpool[self.partition[0]:] = ((np.ones(self.newpool[self.partition[0]:].shape) - mask)*self.newpool[self.partition[0]:] 
                                            + mask * mutatedpool)
        
        #check if its is in the range 
        for i in range(0,self.genes):
            buff = self.newpool[self.partition[0]:] < self.generange[i][0]
            self.newpool[self.partition[0]:][buff] = self.generange[i][0]
            buff = self.newpool[self.partition[1]:] > self.generange[i][1]
            self.newpool[self.partition[1]:][buff] = self.generange[i][1]
        

    def MapGenes(self,generange, gene):
        '''Convert the gene [0,1] to the appropriate value set by generange [a,b]'''
        return generange[0] + gene * (generange[1] - generange[0])

    def AddNoise(self):
        '''Add Gaussian noise to the fittest partition[1] genes'''
        for i in range(0,self.genes):
            self.newpool[sum(self.partition[:1]):sum(self.partition[:2]), i] = self.pool[:self.partition[1],i] 
            + 0.02 * np.random.uniform(self.generange[i][0], self.generange[i][1], size=(self.partition[1],))
            #check if genes are in the generange 
            buff = self.newpool[sum(self.partition[:1]):sum(self.partition[:2]), i] < self.generange[i][0]
            self.newpool[sum(self.partition[:1]):sum(self.partition[:2]), i][buff] = self.generange[i][0]
            
            buff = self.newpool[sum(self.partition[:1]):sum(self.partition[:2]), i] > self.generange[i][1]
            self.newpool[sum(self.partition[:1]):sum(self.partition[:2]), i][buff] = self.generange[i][1]
            

    def CrossoverFitFit(self):
        '''Perform crossover between the fittest :partition[2] genomes and the
        fittest 1:partition[2]+1 genomes'''
        mask = np.random.randint(2, size=(self.partition[2], self.genes))
        self.newpool[sum(self.partition[:2]):sum(self.partition[:3])] = (mask * self.pool[:self.partition[2]]
                + (np.ones(mask.shape) - mask) * self.pool[1:self.partition[2]+1])

    def CrossoverFitRandom(self):
        '''Perform crossover between the fittest :partition[3] genomes and random
        genomes'''
        mask = np.random.randint(2, size=(self.partition[3], self.genes))
        self.newpool[sum(self.partition[:3]):sum(self.partition[:4])] = (mask * self.pool[:self.partition[3]]
                + (np.ones(mask.shape) - mask) * self.pool[np.random.randint(self.genomes, size=(self.partition[3],))])

    def AddRandom(self):
        '''Generate partition[4] random genomes'''
        for i in range(0,self.genes):
            self.newpool[sum(self.partition[:4]):,i] = np.random.uniform(self.generange[i][0], self.generange[i][1], 
                     size=(self.partition[4],))
        
    def RemoveDuplicates(self):
        '''Check the entire pool for any duplicate genomes and replace them by 
        the genome put through a triangular distribution'''
        for i in range(self.genomes):
            for j in range(self.genomes):
                if(j != i and np.array_equal(self.newpool[i],self.newpool[j])):
                    for k in range(0,self.genes):
                       self.newpool[j][k] = np.random.triangular(self.generange[k][0], self.newpool[j][k], self.generange[k][1])
        
            
