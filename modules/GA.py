# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:16:36 2019

@author: HCRuiz (some code taken from contributions of other SkyNEt members)

This is an abstract class implementing the genetic algorithm. The abstract methods and attributes are
experiment specific, so they must be implemented by the user.

Abstract methods:
    - Fitness : Defines fitness landscape for a fixed target. Takes only one argument outputs as numpy arrays
    - EvaluatePopulation : Implements the evaluation of genomes. 
        It takes pool as argument and returns outputs and fitness as numpy arrays

Abstract attributes:
    - genes : number of genes in each individual 
    - genomes : number of individuals in the population
    - partition : a list with the partition for the different operations on the population
    - mutation_rate : rate of mutation applied to genes

All other methods are concreate and serve as default setting; they can be modified as well by user via inheritance

"""

import numpy as np
import time
from abc import ABC, abstractmethod
#import pdb
#import logging
#import sys
from SkyNEt.modules.GenWaveform import GenWaveform

# Define abstract attribute
class abstract_attribute(object):
    def __get__(self, obj, type):
        raise NotImplementedError("Attribute was not set in your experiment subclass!")

class GA(ABC):
    
    def __init__(self):
        super().__init__()
    ###########################################################################
    ###################### Define abstract attributes and methods #############
    ###########################################################################
    ### Abstract attributes
    #TODO: how to implement separationof CV evolution and input affine transformation?
    genes = abstract_attribute() #Nr of genes 
    generange = abstract_attribute()
    genomes = abstract_attribute() #Nr of individuals in population
    partition = abstract_attribute()
    mutation_rate = abstract_attribute()
    lengths = abstract_attribute()
    slopes = abstract_attribute()
    ### Abstract methods    
    @abstractmethod
    def Fitness(self, output):
        pass   
    @abstractmethod
    def EvaluatePopulation(self, pool):
        pass
    
    ##########################################################################
    ################ Concrete methods defining evolution #####################
    ##########################################################################
    def Evolve(self, targets, generations):
        #TODO: check if no. of targets is equal to no. of inputs!!
        # Initialize target
        self.target_wfm = self.Target(targets)  # Target signal
        #Initialize population
        self.pool = np.random.rand(self.genomes, self.genes)
        self.fitness = np.zeros(self.genomes)
        self.output = np.zeros((self.genomes, len(self.target_wfm)))
        
        #Define placeholders
        self.geneArray = np.zeros((generations, self.genomes, self.genes))
        self.outputArray = np.zeros((generations, self.genomes, len(self.target_wfm)))
        self.fitnessArray = np.zeros((generations, self.genomes))
        
        #%% Evolution loop
        for gen in range(generations):
            start = time.time() 
            ####### Evaluate population (user specific) #######
            self.output, self.fitness = self.EvaluatePopulation(self.pool)
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
        config_obj.mutationrate'''
        mask = np.random.choice([0, 1], size=self.pool[self.partition[0]:].shape, 
                                  p=[1-self.config_obj.mutationrate, self.config_obj.mutationrate])
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
    def StopCondition(self, max_fit):
        return False
    
    def Target(self, targets):
        target_wfrm = GenWaveform(targets, self.lengths, slopes=self.slopes)
        return np.asarray(target_wfrm)
    
#%% MAIN
if __name__=='__main__':
    
    #Define concrete class
    class test(GA):
        def __init__(self):
            super().__init__()
            self.w = True
            
        def Fitness(self, output):
            pass
        def EvaluatePopulation(self, pool):
            pass
        def StopCondition(self, max_fit, corr_thr=0.85):
            out = self.output[self.fitness==max_fit][:,np.newaxis]
            y = self.target_wfm[self.w][:,np.newaxis]
            X = np.stack((out, y), axis=0)[:,:,0]
            corr = np.corrcoef(X)[0,1]
            print(f"Correlation of fittest genome: {corr}")
            if corr >= corr_thr:
                print(f'Very high correlation achieved, evolution will stop! \
                      (correlaton threshold set to {corr_thr})')
            return corr >= corr_thr

    # Instantiate
    t = test()
    # Evolve
    t.Evolve([0,0,1,1],100)

