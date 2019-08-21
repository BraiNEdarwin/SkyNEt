# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:12:34 2019

@author: HCRuiz
"""
import SkyNEt.modules.Platforms as Platforms
import SkyNEt.modules.FitnessFunctions as FitF

def get_platform(platform):
    if platform['modality'] == 'chip':
        return Platforms.chip(platform)
    elif platform['modality'] == 'nn':
        return Platforms.nn(platform)
    elif platform['modality'] == 'simulation':
        return Platforms.simulation(platform)
    else:
        raise NotImplementedError(f"Platform {platform['modality']} is not recognized!")

def get_fitness(fitness):
    if fitness == 'corr_fit':
        return FitF.corr_fit
    elif fitness == 'sig_corr':
        return FitF.sig_corr
    elif fitness == 'max_delta_corr':
        return FitF.max_delta_corr
    else:
        raise NotImplementedError(f"Fitness function {fitness} is not recognized!")