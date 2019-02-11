#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:31:48 2019

@author: hruiz
"""

import numpy as np
from SkyNEt.config.config_class import config_class
from SkyNEt.modules.GenWaveform import GenWaveform
from SkyNEt.modules.Classifiers import perceptron

class experiment_config(config_class):
    '''This is the experiment configuration file to classify center-ring classes
    This experiment_config class inherits from config_class default values that are known to work well with boolean logic.
    You can define user-specific parameters in the construction of the object in __init__() or define
    methods that you might need after, e.g. a new fitness function or input and output generators.
    Remember if you define a new fitness function or generator, you have to redefine the self.Fitness,
    self.Target_gen and self.Input_gen in __init__()

    ----------------------------------------------------------------------------
    Description of general parameters
    ----------------------------------------------------------------------------
    comport; the COM port to which the ivvi rack is connected.
    amplification; specify the amount of nA/V. E.g. if you set the IVVI to 100M,
        then amplification = 10
    generations; the amount of generations for the GA
    generange; the range that each gene ([0, 1]) is mapped to. E.g. in the Boolean
        experiment the genes for the control voltages are mapped to the desired
        control voltage range.
    partition; this tells the GA how it will evolve the next generation.
        In order, this will make the GA evolve the specified number with
        - promoting fittest partition[0] genomes unchanged
        - adding Gaussian noise to the fittest partition[1] genomes
        - crossover between fittest partition[2] genomes
        - crossover between fittest partition[3] and randomly selected genes
        - randomly adding parition[4] genomes
    genomes; the amount of genomes in the genepool, speficy this parameter instead
        of partition if you don't care about the specific partition.
    genes; the amount of genes per genome
    mutationrate; the probability of mutation for each gene (between 0 and 1)
    fitnessavg; the amount of times the same genome is tested to obtain the fitness
        value.
    fitnessparameters; the parameters for FitnessEvolution (see its doc for
        specifics)
    filepath; the path used for saving your experiment data
    name; name used for experiment data file (date/time will be appended)

    ----------------------------------------------------------------------------
    Description of method parameters
    ----------------------------------------------------------------------------
    signallength; the length in s of the Boolean P and Q signals
    edgelength; the length in s of the edge between 0 and 1 in P and Q
    fs; sample frequency for niDAQ or ADwin

    ----------------------------------------------------------------------------
    Description of methods
    ----------------------------------------------------------------------------
    TargetGen; specify the target function you wish to evolve
    Fitness; specify the fitness function, as the accuracy of a perceptron separating the data
    '''

    def __init__(self, inputs, labels,filepath=r'../../test/evolution_test/Ring_testing/'):
        super().__init__() #DO NOT REMOVE!
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM3'  # COM port for the ivvi rack

        # Define experiment
        self.lengths, self.slopes = [120], [10] # in 1/fs
        self.InputGen = self.input_waveform(inputs)
        self.amplification = 1
        self.TargetGen = np.asarray(GenWaveform(labels, self.lengths, slopes=self.slopes))
        self.generations = 2
        self.generange = [[-900,900], [-900, 900], [-900, 900], [-900, 900], [-900, 900]]
        self.input_scaling = 0.5
        print('INPUT will be SCALED with',self.input_scaling)  

        self.Fitness = self.corr_fit

#        #Specify either partition or genomes
#        self.genomes = 25
#        self.partition = [2, 6, 6, 6, 5]

        # Documentation
        self.genelabels = ['CV1','CV2','CV3','CV4','CV5']

        # Save settings
        self.filepath = filepath
        self.name = 'Ring-'

        ################################################
        ################# OFF-LIMITS ###################
        ################################################
        # Check if genomes parameter has been changed
        if(self.genomes != sum(self.default_partition)):
            if(self.genomes%5 == 0):
                self.partition = [self.genomes%5]*5  # Construct equally partitioned genomes
            else:
                print('WARNING: The specified number of genomes is not divisible by 5.'
                      + ' The remaining genomes are generated randomly each generation. '
                      + ' Specify partition in the config instead of genomes if you do not want this.')
                self.partition = [self.genomes//5]*5  # Construct equally partitioned genomes
                self.partition[-1] += self.genomes%5  # Add remainder to last entry of partition

        self.genomes = sum(self.partition)  # Make sure genomes parameter is correct
        self.genes = len(self.generange)  # Make sure genes parameter is correct

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
    
    def input_waveform(self, inputs):
        assert len(inputs) == 2, 'Input must be 2 dimensional!'
        inp_wvfrm0 = GenWaveform(inputs[0], self.lengths, slopes=self.slopes)
        inp_wvfrm1 = GenWaveform(inputs[1], self.lengths, slopes=self.slopes)
        samples = len(inp_wvfrm0)
        time_arr = np.linspace(0, samples/self.fs, samples)
        inputs_wvfrm = np.asarray([inp_wvfrm0,inp_wvfrm1])
        
#        print('Size of input', inputs_wvfrm.shape)
        w_ampl = [1,0]*len(inputs[0])
        w_lengths = [self.lengths[0],self.slopes[0]]*len(inputs[0])
        
        weight_wvfrm = GenWaveform(w_ampl, w_lengths)
        bool_weights = [x==1 for x in weight_wvfrm[:samples]]
        
        return time_arr, inputs_wvfrm, bool_weights
    
    def accuracy_fit(self, output, target, w):
#        print(w)
#        print('shape of target = ', target.shape)
        x = output[w][:,np.newaxis]
        y = target[w][:,np.newaxis]
#        print('shape of x,y: ', x.shape,y.shape)
        acc, _, _ = perceptron(x,y)
        X = np.stack((x, y), axis=0)[:,:,0]
        corr = np.corrcoef(X)[0,1]
        return acc*corr
    
    def corr_fit(self, output, target, w):
        x = output[w][:,np.newaxis]
        y = target[w][:,np.newaxis]
        X = np.stack((x, y), axis=0)[:,:,0]
        corr = np.corrcoef(X)[0,1]
#        print('corr_fit')
        return corr
    
if __name__ is '__main__':
    
    from matplotlib import pyplot as plt
    with np.load('Class_data.npz') as data:
        print(data.keys())
        inputs = data['inp_wvfrm'].T
        labels = data['target']
        
    cf = experiment_config(inputs, labels)
    target_wave = cf.TargetGen
    t, inp_wave, weights = cf.InputGen
    print('Max jump for y-input: ', np.diff(inputs[1]).max())
    plt.figure()
    plt.plot(t,inp_wave.T)
    plt.plot(t,target_wave,'k')
    plt.show()
    