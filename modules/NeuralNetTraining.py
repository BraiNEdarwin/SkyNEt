# This module implements functions utilized in obtaining data to train for a neural net
import numpy as np

def initTraj(controls, controlRange):
    steps = len(controlRange)
    controlVoltages = np.zeros((2 * steps ** controls, controls))
    for j in range(controls):
        for i in range(steps):
            controlVoltages[i * steps ** j : (i+1) * steps ** j  , j] = controlRange[i]
        a = controlVoltages[0:  steps ** (j + 1), j]
        controlVoltages[steps ** (j + 1): 2 *  steps ** (j + 1), j] = a[::-1]
    
        sublength = 2 * steps ** (j + 1)
        a = controlVoltages[0:sublength, j]

        for i in range(int(2 * steps ** controls / sublength)):
            controlVoltages[i * sublength : (i+1) * sublength, j] = a

    controlVoltages = controlVoltages[0 : steps ** controls, :]

    return controlVoltages


def gridMaker(controls, controlRange):
    # If only one CV range is given, it is assumed that all the controls use this range:
    if not isinstance(controlRange[0],list):
        controlRange = [controlRange for i in range(controls)]
    
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