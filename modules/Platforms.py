# -*- coding: utf-8 -*-
"""Contains the platforms used in all SkyNEt experiments to be optimized by Genetic Algo.

The classes in Platform must have a method self.evaluate() which takes as arguments 
the inputs inputs_wfm, the gene pool and the targets target_wfm. It must return
outputs as numpy array of shape (len(pool), inputs_wfm.shape[-1]).
    
Created on Wed Aug 21 11:34:14 2019

@author: HCRuiz
"""

import numpy as np
import importlib

#TODO: Add chip platform
#TODO: Add simulation platform
#TODO: Target wave form as argument can be left out if output dimension is known internally

#%% Chip platform to measure the current output from voltage configurations of disordered NE systems
class chip:
    def __init__(self, platform_dict):
        pass
    
    def evaluatePopulation(self,inputs_wfm, gene_pool, target_wfm):
        pass

#%% NN platform using models loaded form staNNet
class nn:

    def __init__(self, platform_dict):
        #Import required packages
        self.torch = importlib.import_module('torch') 
        self.staNNet = importlib.import_module('SkyNEt.modules.Nets.staNNet').staNNet
        
        # Initialize NN
        self.net = self.staNNet(platform_dict['path2NN'])
        self.dtype = self.torch.FloatTensor #torch.cuda.FloatTensor
        
        # Set parameters 
        self.amplification = self.net.info['amplification']
        self.input_indx = platform_dict['in_list']
        self.nn_input_dim = len(self.net.info['amplitude'])
        
    def evaluatePopulation(self,inputs_wfm, genePool, target_wfm):
        genomes = len(genePool)
        outputPopul = np.zeros((genomes,target_wfm.shape[-1]))
        
        for j in range(genomes):
            # Set the input scaling
            x_trafo = self.trafo(inputs_wfm, genePool[j, -1])
        
            # Feed input to NN
            #target_wfm.shape, genePool.shape --> (time-steps,) , (nr-genomes,nr-genes)
            g = np.ones_like(target_wfm)[:,np.newaxis]*genePool[j,:5,np.newaxis].T
            g_index = np.delete(np.arange(self.nn_input_dim),self.input_indx)
            #g.shape,x_trafo.shape --> (time-steps,nr-CVs) , (input-dim, time-steps)
            x_dummy = np.empty((g.shape[0],self.nn_input_dim)) # dims of input (time-steps)xD_in
            x_dummy[:,self.input_indx] = x_trafo.T
            x_dummy[:,g_index] = g 
            inputs = self.torch.from_numpy(x_dummy).type(self.dtype)
            output = self.net.outputs(inputs)
            outputPopul[j] = output
        
        return self.amplification*np.asarray(outputPopul)
    
    def trafo(self,inputs_wfm,params_trafo):
        # inputs_wfm.shape -> (nr-inputs,nr-time-steps)
        # params_trafo.shape -> ()
        return inputs_wfm*params_trafo

#%% Simulation platform for physical MC simulations of devices 
class kmc:
    def __init__(self, platform_dict):
        pass
    
    def evaluatePopulation(self,inputs_wfm, gene_pool, target_wfm):
        pass

#%% MAIN function (for debugging)    
if __name__ == '__main__':
    
    # Define platform
    platform = {}
    platform['path2NN'] = r'../test/NN_test/checkpoint3000_02-07-23h47m.pt'
    platform['in_list'] = [0,5,6]
    nn = nn(platform)
    
    out = nn.evaluatePopulation(np.array([[0.3,0.5,0],[0.3,0.5,0],[0.3,0.5,0]]),
                                np.array([[0.1,-0.5,0.33,-1.2],[0.1,-0.5,0.33,-1.2]]),
                                np.array([1,1,1]))
    
    print(f'nn output: \n {out}')