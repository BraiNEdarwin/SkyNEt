# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:38:36 2019

@author: HCRuiz
"""

import numpy as np
import SkyNEt.modules.GA as base
from SkyNEt.modules.GenWaveform import GenWaveform

class experiment_template(base.GA):
    
    def __init__(self, inputs):
        super().__init__()
        # Set necessary attibutes enforced by base.GA
        self.genes = 5 #Nr of genes 
        self.generange = [[-600,600], [-900, 900], [-900, 900], [-900, 900], [-600, 600]]
        #TODO: how to implement separation of CV evolution and input affine transformation?
        self.genomes = 25 #Nr of individuals in population
        self.partition = [5]*5
        self.mutation_rate = 0.1
        self.lengths = [80]
        self.slopes = [0]
        # Set experiment specific attributes, e.g. fs, comport, etc...
        self.input_wfrm, self.w = self.InputWfrm2D(inputs)
        
    #Define necessary methods enforced by base.GA (Fitness and EvaluatePopulation)
    def Fitness(self, output):
        '''Gets outputs as arguments and calculates elementwise the fitness 
        of each member in the population.
        Returns: numpy array of fitness values corresponding to each individal/ 
        '''
        pass
    
    def EvaluatePopulation(self, pool):
        '''Gets pool of genes as argument and obtains the response of the system 
        given the control voltages and the inputs.
        This method handles all the necessary components to get the output needed in the GA,
        including the distinction between the modality of the experiment (device, NN or simulator)
        and the separation of the affine transformation and control voltages.
        Retruns: numpy arrays of output (waveforms) and fitness values.
        '''
        outputs = self.verycomplicatedfunction(pool)
        fitness = np.zeros_like(outputs)
        return outputs, np.asarray(fitness)
    
    ######################## Other helper methods #############################
    def verycomplicatedfunction(self, pool):
        pass
    
    def InputWfrm2D(self, inputs): 
        #Put this in GenWaveform module??
        assert len(inputs) == 2, 'Input must be 2 dimensional!'
        inp_wvfrm0 = GenWaveform(inputs[0], self.lengths, slopes=self.slopes)
        inp_wvfrm1 = GenWaveform(inputs[1], self.lengths, slopes=self.slopes)
        samples = len(inp_wvfrm0)
        inputs_wvfrm = np.asarray([inp_wvfrm0,inp_wvfrm1])
#        print('Size of input', inputs_wvfrm.shape)
        w_ampl = [1,0]*len(inputs[0])
        w_lengths = [self.lengths[0],self.slopes[0]]*len(inputs[0])       
        weight_wvfrm = GenWaveform(w_ampl, w_lengths)
        bool_weights = [x==1 for x in weight_wvfrm[:samples]]        
        return inputs_wvfrm, bool_weights
    
    def InpTrafo(self, x):
        return x
    
    def StopCondition(self, max_fit):
        return False
    
#%%MAIN
if __name__=='__main__':
    inputs = [[-0.7,0.7,-0.7,0.7,-1,1],[-0.7,-0.7,0.7,0.7,0,0]]
    binary_labels = [1,0,1,1,0,1]
    experiment = experiment_template(inputs)
    experiment.Evolve(binary_labels, 80)
    