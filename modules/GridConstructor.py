# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:36:16 2018
Creates a grid which can be used to obtain datapoints.
If for the controlRange only one list is specified, the function assumes that
every control voltage uses these gridpoints. To use different gridpoints for
different control voltages, use a list of lists: [[...],[...],...]
@author: Mark Boon
"""

import numpy as np


def gridConstructor(controls, controlRange):
    # If only one CV range is given, it is assumed that all the controls use this range:
    if not isinstance(controlRange[0],list):
        controlRange = [controlRange for i in range(controls)]
    if len(controlRange) != controls:
        print('\nWarning: number of controls does not equal amount of control ranges\n')
    
    steps = [len(controlRange[i]) for i in range(len(controlRange))] # List of number of steps per CV
    controlVoltages = np.zeros((2 * np.prod(steps), controls))
    
    for j in range(controls):            
        for i in range(steps[j]):
            controlVoltages[i * int(np.prod(steps[:j]))  : (i + 1) * int(np.prod(steps[:j])), j] = controlRange[j][i]
        a = controlVoltages[0:  int(np.prod(steps[:j + 1])), j] 
        controlVoltages[int(np.prod(steps[:j + 1])): 2 *  int(np.prod(steps[:j + 1])), j] = a[::-1]
        
        sublength = 2 * int(np.prod(steps[:j + 1]))
        a = controlVoltages[0:sublength, j] # Copy the voltage block for CV(j)
        
        for i in range(int((2 * np.prod(steps)) / sublength)):
            controlVoltages[i * sublength : (i+1) * sublength, j] = a # Fill the rest of the column with the CV block
        
    controlVoltages = controlVoltages[0 : np.prod(steps), :]

    return controlVoltages
