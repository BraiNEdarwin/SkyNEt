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
        self.amplification = platform_dict['amplification']
        
    def evaluatePopulation(self,inputs_wfm, genePool, target_wfm):
        genomes = len(genePool)
        outputPopul = np.zeros((genomes,target_wfm.shape[-1]))
        
        for j in range(genomes):
            # Set the input scaling
            x_trafo = self.trafo(inputs_wfm, genePool[j, -1])
        
            # Feed input to NN
            #target_wfm.shape, genePool.shape --> (time-steps,) , (nr-genomes,nr-genes)
            g = np.ones_like(target_wfm)[:,np.newaxis]*genePool[j,:5,np.newaxis].T
            #g.shape,x_trafo.shape --> (time-steps,nr-CVs) , (input-dim, time-steps)
            x_dummy = np.concatenate((x_trafo.T,g),axis=1) # First input then genes; dims of input TxD
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
    platform['path2NN'] = r'D:\UTWENTE\PROJECTS\DARWIN\Data\Mark\MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
    platform['amplification'] = 10.
    
    nn = nn(platform)