#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:38:07 2019

@author: hruiz
"""
import Platforms 
import FitnessFunctions

class config_GA:
    def __init__(self, config_dict):   
        self.genes = config_dict['genes']           # Nr of genes include CVs and affine trafo of input 
        self.generange = config_dict['generange']   # Voltage range of CVs
        self.genomes = config_dict['genomes']       # Nr of individuals in population
        self.partition = config_dict['partition']
        self.mutation_rate = config_dict['mutation_rate']
        #Parameters to define target waveforms
        self.lengths = config_dict['lengths']
        self.slopes = config_dict['slopes']
        #Define methods for fitness and platform from config_obj
        self.fitness = config_dict['fitness']
        self.platform = config_dict['platform']
    
    def get_platform(self):
        if self.platform == 'chip':
            return Platforms.chip
        elif self.platform == 'nn':
            return Platforms.nn
        elif self.platform == 'simulation':
            return Platforms.simulation
        else:
            raise NotImplementedError(f"Platform {self.platform} is not recognized!")
    
    def get_fitness(self):
        if self.fitness == 'corr_fit':
            return FitnessFunctions.corr_fit
        elif self.fitness == 'sig_corr':
            return FitnessFunctions.sig_corr
        elif self.fitness == 'max_delta_corr':
            return FitnessFunctions.max_delta_corr
        else:
            raise NotImplementedError(f"Fitness function {self.fitness} is not recognized!")
            
        