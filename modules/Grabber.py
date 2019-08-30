# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:12:34 2019

@author: HCRuiz
"""
import SkyNEt.modules.Platforms as Platforms
import SkyNEt.modules.FitnessFunctions as FitF

def get_platform(platform):
    '''Gets an instance of the determined class from Platforms.
    The classes in Platform must have a method self.evaluate() which takes as 
    arguments the inputs inputs_wfm, the gene pool and the targets target_wfm. 
    It must return outputs as numpy array of shape (self.genomes, len(self.target_wfm))
    '''
    if platform['modality'] == 'chip':
        return Platforms.chip(platform)
    elif platform['modality'] == 'nn':
        return Platforms.nn(platform)
    elif platform['modality'] == 'kmc':
        return Platforms.kmc(platform)
    else:
        raise NotImplementedError(f"Platform {platform['modality']} is not recognized!")

def get_fitness(fitness):
    '''Gets the fitness function used in GA from the module FitnessFunctions
    The fitness functions must take two arguments, the outputs of the black-box and the target
    and must return a numpy array of scores of size len(outputs).
    '''
    if fitness == 'corr_fit':
        return FitF.corr_fit
    elif fitness == 'accuracy_fit':
        return FitF.accuracy_fit
    elif fitness == 'corrsig_fit':
        return FitF.corrsig_fit
    else:
        raise NotImplementedError(f"Fitness function {fitness} is not recognized!")