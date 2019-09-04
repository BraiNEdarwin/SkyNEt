'''
A Class definition that defines a genetic algorithm. 
'''
import numpy as np
import time
import random 


class GenePool(object):
    '''This is the evolution file that defiend the genetic algorithm. 
    The constructor obtains the necessary parameters from the config_obj for evolving 
    the next generation. In addition, it randomly initializes the genePool.
    ----------------------------------------------------------------------------
    Description of methods
    ----------------------------------------------------------------------------
    For a more detailed description of all genetic operators and the corresponding 
    parameters, we will refer to TBA report 

    '''

    def __init__(self, config_obj):

        self.config_obj = config_obj
        self.genes = config_obj.genes
        self.genomes = config_obj.genomes
        self.generange = config_obj.generange
        
        self.fitness = np.zeros(self.genomes)
        self.partition = config_obj.partition
        self.mutationrate = config_obj.mutationrate
        self.generations = config_obj.generations
        self.pool = np.zeros((self.genomes, self.genes))
        self.opposite_pool = np.zeros((self.genomes, self.genes))
        for i in range(0,self.genes):
            self.pool[:,i] = np.random.uniform(self.generange[i][0], self.generange[i][1], size=(self.genomes,))
        np.random.seed(seed=None)
        
    
    def NextGen(self, gen):
        # Sort genePool based on fitness 
        indices = np.argsort(self.fitness)
        indices = indices[::-1]
        self.pool = self.pool[indices]  
        self.fitness = self.fitness[indices]
        # Copy the current pool 
        self.newpool = self.pool.copy()
        # Determine which genomes are chosen to generate offspring 
        # Note: twice as much parents are selected as there are genomes to be generated
        chosen = self.Universal_sampling()
        # Generate offspring by means of crossover. 
        # The crossover method returns 1 genome from 2 parents       
        for i in range(0,len(chosen),2):
            index_newpool = int(i/2 +sum(self.partition[:1]))
            if chosen[i] == chosen[i+1]:
                if chosen[i] == 0: 
                    chosen[i] = chosen[i] + 1 
                else: 
                    chosen[i] = chosen[i] - 1 
            #The individual with the highest fitness score is given as input first
            if chosen[i] < chosen[i+1]:
                self.newpool[index_newpool,:] = self.Crossover_BLXab(self.pool[chosen[i],:], self.pool[chosen[i+1],:])
            else:
                self.newpool[index_newpool,:] = self.Crossover_BLXab(self.pool[chosen[i+1],:], self.pool[chosen[i],:])
        # The mutation rate is updated based on the generation counter     
        self.UpdateMutation(gen)
        # Every genome, except the partition[0] genomes are mutated 
        self.Mutation()
        self.RemoveDuplicates()
        self.pool = self.newpool.copy()
        self.fitness = np.zeros(self.genomes)
        
        
 #------------------------------------------------------------------------------       
 
    def Universal_sampling(self):
        '''
        Sampling method: Stochastic universal sampling returns the chosen 'parents' 
        '''
        no_genomes = 2 * self.partition[1]    
        chosen = []
        probabilities = self.Linear_rank()
        for i in range(1, len(self.fitness)):
            probabilities[i] = probabilities[i] + probabilities[i-1]
        distance = 1/(no_genomes)
        start = random.random() * distance
        for n in range(no_genomes):
            pointer = start+n*distance
            for i in range(len(self.fitness)):
                if pointer < probabilities[0]:
                    chosen.append(0)
                    break 
                elif pointer< probabilities[i] and pointer >= probabilities[i-1]:
                    chosen.append(i)
                    break 
        chosen = random.sample(chosen, len(chosen))    
        return chosen
     
    
    def Linear_rank(self):
        '''
        Assigning probabilities: Linear ranking scheme used for stochastic universal sampling method. 
        It returns an array with the probability that a genome will be chosen. 
        The first probability corresponds to the genome with the highest fitness etc. 
        '''
        maximum = 1.5
        rank = np.arange(self.genomes) + 1 
        minimum = 2 - maximum 
        probability =  (minimum +( (maximum-minimum) * (rank -1)/ (self.genomes - 1)))/self.genomes
        return probability[::-1]
    
 
 #------------------------------------------------------------------------------    
   
    def Crossover_BLXab(self, parent1, parent2):
        '''
        Crossover method: Blend alpha beta crossover returns a new genome (voltage combination)
        from two parents. Here, parent 1 has a higher fitness than parent 2
        '''
        
        alpha = 0.6
        beta = 0.4
        maximum = np.maximum(parent1, parent2)
        minimum = np.minimum(parent1, parent2)
        I = (maximum - minimum) 
        offspring= np.zeros((parent1.shape))
        for i in range(len(parent1)):
            if parent1[i] > parent2[i]:
                offspring[i] = np.random.uniform(minimum[i]-I[i]*beta, maximum[i]+I[i]*alpha)
            else: 
                offspring[i] = np.random.uniform(minimum[i] - I[i]*alpha, maximum[i]+I[i]*beta)
        for i in range(0,self.genes):
            if offspring[i] < self.generange[i][0]: 
                offspring[i] = self.generange[i][0]
            if offspring[i]  > self.generange[i][1]: 
                offspring[i] = self.generange[i][1]
        return offspring
    
    
 #------------------------------------------------------------------------------    
  
    def UpdateMutation(self,gen):
        '''
        Dynamic parameter control of mutation rate: This formula updates the mutation 
        rate based on the generation counter
        '''
        pm_inv = 2+5/(self.generations-1)* gen
        self.mutationrate = 0.625/pm_inv

 #------------------------------------------------------------------------------ 
 
    def Mutation(self):
        '''
        Mutate all genes but the first partition[0] with a triangular 
        distribution in generange with mode=gene to be mutated.
        '''
        np.random.seed(seed=None)
        mask = np.random.choice([0, 1], size=self.pool[self.partition[0]:].shape, 
                                p=[1-self.mutationrate, self.mutationrate])
        mutatedpool = np.zeros((self.genomes-self.partition[0], self.genes))
    
        for i in range(0,self.genes):
            if self.generange[i][0] == self.generange[i][1]:
                mutatedpool[:,i] = self.generange[i][0]*np.ones(mutatedpool[:,i].shape)
            else:
                mutatedpool[:,i] = np.random.triangular(self.generange[i][0], self.newpool[self.partition[0]:,i], self.generange[i][1])
        self.newpool[self.partition[0]:] = ((np.ones(self.newpool[self.partition[0]:].shape) - mask)*self.newpool[self.partition[0]:]  + mask * mutatedpool)
    
    
  #------------------------------------------------------------------------------   
     
    def RemoveDuplicates(self):
        np.random.seed(seed=None)
        '''
        Check the entire pool for any duplicate genomes and replace them by 
        the genome put through a triangular distribution
        '''
        for i in range(self.genomes):
            for j in range(self.genomes):
                if(j != i and np.array_equal(self.newpool[i],self.newpool[j])):
                    for k in range(0,self.genes):
                        if self.generange[k][0] != self.generange[k][1]:
                            self.newpool[j][k] = np.random.triangular(self.generange[k][0], self.newpool[j][k], self.generange[k][1])
                        else: 
                            self.newpool[j][k] = self.generange[k][0]
                            
                        
#------------------------------------------------------------------------------
    #Methods required for evaluating the opposite bool                        
    def Opposite(self):
        '''
        Define opposite pool  
        '''
        opposite_pool = np.zeros((self.genomes, self.genes))
        for i in range(0,self.genes):
            opposite_pool[:,i] = self.generange[i][0] + self.generange[i][1] - self.pool[:,i]
        self.opposite_pool = opposite_pool
    def setNewPool(self, indices):
        '''
        After evaluating the opposite pool, set the new pool. 
        '''
        for k in range(len(indices)):
            if indices[k][0]:
                self.pool[k,:] = self.opposite_pool[k,:]
            
        
   