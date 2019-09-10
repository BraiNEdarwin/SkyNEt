# -*- coding: utf-8 -*-
""" Contains class implementing the Genetic Algorithm for all SkyNEt platforms.
Created on Thu May 16 18:16:36 2019
@author: HCRuiz and A. Uitzetter
"""

import numpy as np
import time
import random 
#import pdb
#import logging
#import sys
from SkyNEt.modules.GenWaveform import GenWaveform 
import SkyNEt.modules.Grabber as Grabber
from SkyNEt.modules.Classifiers import perceptron
from SkyNEt.modules.Observers import God as Savior
#TODO: Implement Plotter
class GA:
    '''This is a class implementing the genetic algorithm (GA).
    The methods implement the GA regardless of the platform being optimized, 
    i.e. it can be used with the chip, the model or the physical simulations. 

    ---------------------------------------------------------------------------
    
    Argument : config dictionary. 
    
    The configuration dictionary must contain the following:    
      - genes : number of genes in each individual 
      - generange
      - genomes : number of individuals in the population
      - partition : a list with the partition for the different operations on the population
      - mutation_rate : rate of mutation applied to genes
      - Arguments for GenWaveform:
          o lengths : defines the input lengths (in ms) of the targets
          o slopes : defines the slopes (in ms) between targets
        
    Notes:
        * All other methods serve as default settings but they can be modified as well by
        the user via inheritance.
        * Requires args to GenWaveform because the method Evolve requires targets, 
        so one single instance of GA can handle multiple tasks.
        *   The get_platform() function in Grabber gets an instance of the platform used
        to evaluate the genomes in the population
        * The get_fitness() function in Grabber gets the fitness function used as a score in GA

    '''
    def __init__(self,config_dict):   
        
        #Define GA hyper-parameters
        self.genes = config_dict['genes']           # Nr of genes include CVs and affine trafo of input 
        self.generange = config_dict['generange']   # Voltage range of CVs
        self.genomes = config_dict['genomes']       # Nr of individuals in population
        self.partition = config_dict['partition']   # Partitions of population
        self.mutationrate = config_dict['mutationrate']
        #Parameters to define target waveforms
        self.lengths = config_dict['lengths']       # Length of data in the waveform
        self.slopes = config_dict['slopes']         # Length of ramping from one value to the next
        #Parameters to define task
        self.platform = config_dict['platform']     # Dictionary containing all variables for the platform
        self.fitness_function = config_dict['fitness'] # String determining fitness funtion
        #Define platform and fitness function from Grabber
        self.Platform = Grabber.get_platform(self.platform)
        self.Fitness = Grabber.get_fitness(self.fitness_function)
        
        # Internal parameters and variables
        self.stop_thr = 0.9
        #----- observers ------#
        self._observers = set()
        self._next_state = None
        self.savior = Savior(config_dict)
        self.attach(self.savior)

#%% Methods implementing observer pattern for Saver and Plotter
    def attach(self, observer):
        #register subject in observer
        observer.subject = self
        #add observer to observers list
        self._observers.add(observer)
        
    def detach(self, observer): #not needed but for completeness
        observer.subject = None
        self._observers.discard(observer)
        
    def _notify(self):
        for obs in self._observers:
            obs.update(self._next_state)
            
    @property
    def next_state(self):
        return self._next_state
    
    @next_state.setter
    def next_state(self,arg):
        self._next_state = arg
        self._notify()
            
#%% Method implementing evolution
    def optimize(self, inputs, targets, 
                 epochs=100, 
                 savepath=r'../test/evolution_test/NN_testing/',
                 dirname = 'TEST',
                 seed=None):
        
        assert len(inputs[0]) == len(targets), f'No. of input data {len(inputs)} does not match no. of targets {len(targets)}'
        np.random.seed(seed=seed)
        
        self.generations = epochs
        # Initialize target
        self.target_wfm = self.waveform(targets)
        # Initialize target
        self.inputs_wfm, self.filter_array = self.input_waveform(inputs)
        # Generate filepath and filename for saving
        self.savepath = savepath
        self.dirname = dirname
        #reset placeholder arrays and filepath in saviour
        self.savior.reset()
        
        self.pool = np.zeros((self.genomes, self.genes))
        self.opposite_pool = np.zeros((self.genomes, self.genes))
        for i in range(0,self.genes):
            self.pool[:,i] = np.random.uniform(self.generange[i][0], self.generange[i][1], size=(self.genomes,))
        
        #Evolution loop
        for gen in range(self.generations):
            start = time.time()
            #-------------- Evaluate population (user specific) --------------#
            self.outputs = self.Platform.evaluatePopulation(self.inputs_wfm,
                                                  self.pool, 
                                                  self.target_wfm)
            self.fitness = self.Fitness(self.outputs, self.target_wfm)
            #-----------------------------------------------------------------#
            # Status print
            max_fit = max(self.fitness)
            print(f"Highest fitness: {max_fit}")
            
            self.next_state = {'generation':gen, 'genes':self.pool, 
                               'outputs':self.outputs, 'fitness': self.fitness}
            

            end = time.time()
            print("Generation nr. " + str(gen + 1) + " completed; took "+str(end-start)+" sec.")
            if self.StopCondition(max_fit):
                print('--- final saving ---')
                self.savior.save()
                break
            #Evolve to the next generation
            self.NextGen(gen)
                        
        #Get best results
        max_fitness, best_genome, best_output = self.savior.judge()
#        print(best_output.shape,self.target_wfm.shape)
        best_corr = self.corr(best_output)
        print(f'\n========================= BEST SOLUTION =======================')
        print('Fitness: ', max_fitness)
        print('Correlation: ', best_corr)
        print(f'Genome:\n {best_genome}')
        y = best_output[self.filter_array][:,np.newaxis]
        trgt = self.target_wfm[self.filter_array][:,np.newaxis]
        accuracy, _, _ = perceptron(y,trgt)
        print('Accuracy: ', accuracy)
        print('===============================================================')
        return best_genome, best_output, max_fitness, accuracy
    
#%% Step to next generation    
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
        
#%% ##########################################################################
    ###################### Methods defining evolution ########################
    ##########################################################################
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
    #Methods required for evaluating the opposite pool                        
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
            
#%% #################################################
    ########### Helper Methods ######################
    #################################################
    
    def StopCondition(self, max_fit):
        best = self.outputs[self.fitness==max_fit][0]
        corr = self.corr(best)
        print(f"Correlation of fittest genome: {corr}")
        if corr >= self.stop_thr:
            print(f'Very high correlation achieved, evolution will stop! \
                  (correlaton threshold set to {self.stop_thr})')
        return corr >= self.stop_thr
    
    def corr(self,x):
        x = x[self.filter_array][np.newaxis,:]
        y = self.target_wfm[self.filter_array][np.newaxis,:]
        X = np.concatenate((x, y), axis=0)
        return np.corrcoef(X)[0,1]
        
    def waveform(self, data):
        data_wfrm = GenWaveform(data, self.lengths, slopes=self.slopes)
        return np.asarray(data_wfrm)
    
    def input_waveform(self, inputs):
        nr_inp = len(inputs)
        print(f'Input is {nr_inp} dimensional')
        inp_list = [self.waveform(inputs[i]) for i in range(nr_inp)]
        inputs_wvfrm = np.asarray(inp_list) 
#        print('Size of input', inputs_wvfrm.shape)
        samples = inputs_wvfrm.shape[-1]
        print(f'Input signals have length {samples}')
#        time_arr = np.linspace(0, samples/self.fs, samples)
        w_ampl = [1,0]*len(inputs[0])
        w_lengths = [self.lengths[0],self.slopes[0]]*len(inputs[0])       
        weight_wvfrm = GenWaveform(w_ampl, w_lengths)
        bool_weights = [x==1 for x in weight_wvfrm[:samples]]
        
        return inputs_wvfrm, bool_weights#, time_arr
    
#%% MAIN
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    # Define platform
    platform = {}
    platform['modality'] = 'nn'
    platform['path2NN'] = r'D:\UTWENTE\PROJECTS\DARWIN\Data\Mark\MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
#    platform['path2NN'] = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/April_2019/MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
    platform['amplification'] = 10.
    
    config_dict = {}    
    config_dict['partition'] =  [5]*5 # Partitions of population
    # Voltage range of CVs in V
    config_dict['generange'] = [[-1.2,0.6], [-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3], [-0.7, 0.3],[1,1]]
    config_dict['genes'] = len(config_dict['generange'])    # Nr of genes
    config_dict['genomes'] = sum(config_dict['partition'])  # Nr of individuals in population   
    config_dict['mutationrate'] = 0.1
    
    #Parameters to define target waveforms
    config_dict['lengths'] = [80]     # Length of data in the waveform
    config_dict['slopes'] = [0]        # Length of ramping from one value to the next
    #Parameters to define task
    config_dict['fitness'] = 'corrsig_fit'#'corr_fit'
    
    config_dict['platform'] = platform    # Dictionary containing all variables for the platform
    
    ga = GA(config_dict)
    
    inputs = [[-1.,0.4,-1.,0.4,-0.8, 0.2],[-1.,-1.,0.4,0.4, 0., 0.]]
    targets = [1,1,0,0,1,1]
    
    best_genome, best_output, max_fitness, accuracy = ga.optimize(inputs,targets)
    
    plt.figure()
    plt.plot(best_output)
    plt.title(f'Best output for target {targets}')
    plt.show()