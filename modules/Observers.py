# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:14:23 2019

@author: HCRuiz
"""
import numpy as np 
import os
import pickle
import SkyNEt.modules.SaveLib as SaveLib

class God:
    
    def __init__(self,config_dict):
        self.subject = None
        self.config_dict = config_dict
        
        
    def update(self, next_sate):
        gen = next_sate['generation']
        self.geneArray[gen, :, :] = next_sate['genes']
        self.outputArray[gen, :, :] = next_sate['outputs']
        self.fitnessArray[gen, :] = next_sate['fitness']
        if gen%5 == 0:
            #Save generation
            print('--- checkpoint ---')
            self.save()
        
    def reset(self):
        #Define placeholders
        self.geneArray = np.zeros((self.subject.generations, self.subject.genomes, self.subject.genes))
        self.outputArray = np.zeros((self.subject.generations, self.subject.genomes, len(self.subject.target_wfm)))
        self.fitnessArray = -np.inf*np.ones((self.subject.generations, self.subject.genomes))
        # Initialize save directory
        self.saveDirectory = SaveLib.createSaveDirectory(
                                        self.subject.savepath,
                                        self.subject.dirname)
        # Save experiment configurations
        self.config_dict['target'] = self.subject.target_wfm
        self.config_dict['inputs'] = self.subject.inputs_wfm
        self.config_dict['mask'] = self.subject.filter_array
        with open(os.path.join(self.saveDirectory, 'configs.pkl'), 'wb') as f:
            pickle.dump(self.config_dict,f)
        
    def judge(self):
        max_fitness = np.max(self.fitnessArray)
        ind = np.unravel_index(np.argmax(self.fitnessArray, axis=None), self.fitnessArray.shape)
        best_genome = self.geneArray[ind]
        best_output = self.outputArray[ind]
        
        return max_fitness, best_genome, best_output
    
    def save(self):
        SaveLib.saveExperiment(self.saveDirectory, filename='Results_GA',
                                                   geneArray = self.geneArray,
                                                   outputArray = self.outputArray,
                                                   fitnessArray = self.fitnessArray,
                                                   mask = self.subject.filter_array)