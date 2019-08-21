# -*- coding: utf-8 -*-
""" Contains class implementing the Genetic Algorithm for all SkyNEt platforms.
Created on Thu May 16 18:16:36 2019
@author: HCRuiz (some code taken from contributions of other SkyNEt members)
"""

import numpy as np
import time
#import pdb
#import logging
#import sys
from SkyNEt.modules.GenWaveform import GenWaveform 
import SkyNEt.modules.Grabber

class GA:
    '''This is a class implementing the genetic algorithm (GA).
    The methods implement the GA regardless of the platform being optimized, 
    i.e. it can be used with the chip, the model or the physical simulations. 

    ---------------------------------------------------------------------------
    
    Argument : config object (config_GA). 
    
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
        self.mutation_rate = config_dict['mutation_rate']
        #Parameters to define target waveforms
        self.lengths = config_dict['lengths']       # Length of data in the waveform
        self.slopes = config_dict['slopes']         # Length of ramping from one value to the next
        #Parameters to define task
        self.platform = config_dict['platform']     # Dictionary containing all variables for the platform
        self.fitness_function = config_dict['fitness'] # String determining fitness funtion
        #Define platform and fitness function from Grabber
        self.Platform = Grabber.get_platform(self.platform)
        self.Fitness = Grabber.get_fitness(self.fitness_function)

    
    ##########################################################################
    ###################### Methods defining evolution ########################
    ##########################################################################
    def Evolve(self, inputs, targets, generations=100):
        assert len(inputs[0]) == len(targets), f'No. of input data {len(inputs)} does not match no. of targets {len(targets)}'
        # Initialize target
        self.target_wfm = self.waveform(targets)
        # Initialize target
        self.inputs_wfm, self.filter_array = self.input_waveform(inputs)

        #Initialize population
        self.pool = np.random.rand(self.genomes, self.genes)
        
        #Define placeholders
        self.geneArray = np.zeros((generations, self.genomes, self.genes))
        self.outputArray = np.zeros((generations, self.genomes, len(self.target_wfm)))
        self.fitnessArray = -np.inf*np.ones((generations, self.genomes))
        
        #%% Evolution loop
        for gen in range(generations):
            start = time.time() 
            #-------------- Evaluate population (user specific) --------------#
            self.output = self.Platform.evaluate(self.inputs_wfm,
                                                  self.pool, 
                                                  self.target_wfm)
            self.fitness = self.Fitness(self.output, self.target_wfm)
            #-----------------------------------------------------------------#
            # Status print
            max_fit = max(self.fitness)
            print(f"Highest fitness: {max_fit}")
            if self.StopCondition(max_fit):
                break
            # Evolve to the next generation
            self.NextGen()
            end = time.time()
            print("Generation nr. " + str(gen + 1) + " completed; took "+str(end-start)+" sec.")
                        
            #%% Save generation data (USE OBSERVER PATTERN?)
            #TODO: Implemanet SAVER
#            self.geneArray[i, :, :] = self.pool
#            self.outputArray[i, :, :] = self.output
#            self.fitnessArray[i, :] = self.fitness
#            # Save generation
#            SaveLib.saveExperiment(saveDirectory,
#                                   genes = self.geneArray,
#                                   output = self.outputArray,
#                                   fitness = self.fitnessArray,
#                                   target = target[w][:,np.newaxis],
#                                   weights = w, time = t)
            
        #%%Get best results
        max_fitness = np.max(self.fitnessArray)
        ind = np.unravel_index(np.argmax(self.fitnessArray, axis=None), self.fitnessArray.shape)
        best_genome = self.geneArray[ind]
        best_output = self.outputArray[ind]
        print('Max. Fitness: ', max_fitness)
        print('Best genome: ', best_genome)
        return best_genome, best_output, max_fitness
#%%    
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
#%%
    def Mutation(self):
        '''Mutate all genes but the first partition[0] with a triangular 
        distribution between 0 and 1 with mode=gene. The chance of mutation is 
        config_dict['mutationrate']'''
        mask = np.random.choice([0, 1], size=self.pool[self.partition[0]:].shape, 
                                  p=[1-self.mutationrate, self.mutationrate])
        mutatedpool = np.random.triangular(0, self.newpool[self.partition[0]:], 1)
        self.newpool[self.partition[0]:] = ((np.ones(self.newpool[self.partition[0]:].shape) - mask)*self.newpool[self.partition[0]:] 
                                            + mask * mutatedpool)
        
#%%
#    def MapGenes(self,generange, gene):
#        '''Convert the gene [0,1] to the appropriate value set by generange [a,b]'''
#        return generange[0] + gene * (generange[1] - generange[0])
#%%
    def AddNoise(self):
        '''Add Gaussian noise to the fittest partition[1] genes'''
        self.newpool[sum(self.partition[:1]):sum(self.partition[:2])] = (self.pool[:self.partition[1]] +
                0.02*np.random.randn(self.partition[1],self.newpool.shape[1]))

        # check that genes are in [0,1]
        buff = self.newpool[sum(self.partition[:1]):sum(self.partition[:2])] > 1.0
        self.newpool[sum(self.partition[:1]):sum(self.partition[:2])][buff] = 1.0

        buff = self.newpool[sum(self.partition[:1]):sum(self.partition[:2])] < 0.0
        self.newpool[sum(self.partition[:1]):sum(self.partition[:2])][buff] = 0.0
#%%
    def CrossoverFitFit(self):
        '''Perform crossover between the fittest :partition[2] genomes and the
        fittest 1:partition[2]+1 genomes'''
        mask = np.random.randint(2, size=(self.partition[2], self.genes))
        self.newpool[sum(self.partition[:2]):sum(self.partition[:3])] = (mask * self.pool[:self.partition[2]]
                + (np.ones(mask.shape) - mask) * self.pool[1:self.partition[2]+1])
#%%
    def CrossoverFitRandom(self):
        '''Perform crossover between the fittest :partition[3] genomes and random
        genomes'''
        mask = np.random.randint(2, size=(self.partition[3], self.genes))
        self.newpool[sum(self.partition[:3]):sum(self.partition[:4])] = (mask * self.pool[:self.partition[3]]
                + (np.ones(mask.shape) - mask) * self.pool[np.random.randint(self.genomes, size=(self.partition[3],))])
#%%
    def AddRandom(self):
        '''Generate partition[4] random genomes'''
        self.newpool[sum(self.partition[:4]):] = np.random.rand(self.partition[4], self.genes)
#%%        
    def RemoveDuplicates(self):
        '''Check the entire pool for any duplicate genomes and replace them by 
        the genome put through a triangular distribution'''
        for i in range(self.genomes):
            for j in range(self.genomes):
                if(j != i and np.array_equal(self.newpool[i],self.newpool[j])):
                    self.newpool[j] = np.random.triangular(0, self.newpool[j], 1)

#%% ########### Helper Methods ######################
    def StopCondition(self, max_fit, corr_thr=0.9):
        best = self.output[self.fitness==max_fit][:,np.newaxis]
        y = self.target_wfm[self.filter_array][:,np.newaxis]
        X = np.stack((best, y), axis=0)[:,:,0]
        corr = np.corrcoef(X)[0,1]
        print(f"Correlation of fittest genome: {corr}")
        if corr >= corr_thr:
            print(f'Very high correlation achieved, evolution will stop! \
                  (correlaton threshold set to {corr_thr})')
        return corr >= corr_thr
    
    def waveform(self, data):
        data_wfrm = GenWaveform(data, self.lengths, slopes=self.slopes)
        return np.asarray(data_wfrm)
    
    def input_waveform(self, inputs):
        nr_inp = len(inputs)
        print(f'Input is {nr_inp} dimensional')
        inp_list = [waveform(inputs[i]) for i in range(nr_inp)]
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
    



