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