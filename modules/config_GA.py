#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:38:07 2019

@author: hruiz
"""


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
        #Define platform from config_obj
        self.platform = config_dict['platform']