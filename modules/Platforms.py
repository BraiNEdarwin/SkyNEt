# -*- coding: utf-8 -*-
"""Contains the platforms used in all SkyNEt experiments to be optimized by Genetic Algo.

The classes in Platform must have a method self.evaluate() which takes as arguments 
the inputs inputs_wfm, the gene pool and the targets target_wfm. It must return
outputs as numpy array of shape (len(pool), inputs_wfm.shape[-1]).
    
Created on Wed Aug 21 11:34:14 2019

@author: HCRuiz
"""

import numpy as np

class chip:
    def __init__(self, platform_dict):
        pass


class nn:
    def __init__(self, platform_dict):
        pass


class simulation:
    def __init__(self, platform_dict):
        pass