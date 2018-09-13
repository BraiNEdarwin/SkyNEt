#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:56:09 2018

@author: hruiz
"""
import modules.NeuralNetTraining as NN
import numpy as np

def Boolean_Grid(blockSize, controls, voltageGrid):
    controlVoltages = np.empty((4 * blockSize, 7))

    controlVoltages[0:blockSize,0] = 0
    controlVoltages[blockSize:2*blockSize,0] = 1
    controlVoltages[2*blockSize:3*blockSize,0] = 0
    controlVoltages[3*blockSize:4*blockSize,0] = 1
    
    controlVoltages[0:blockSize,1] = 0
    controlVoltages[blockSize:2*blockSize,1] = 0
    controlVoltages[2*blockSize:3*blockSize,1] = 1
    controlVoltages[3*blockSize:4*blockSize,1] = 1
    
    controlVoltages[0:blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
    controlVoltages[blockSize:2*blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
    controlVoltages[2*blockSize:3*blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
    controlVoltages[3*blockSize:4*blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
    
    return controlVoltages

def CP_Grid(nr_blocks, blockSize, smallest_div, controls, voltageGrid):
    controlVoltages = np.zeros((nr_blocks * blockSize, 7))

    controlVoltages[0:smallest_div*blockSize,0] = -900 #voltageGrid[0]
    controlVoltages[smallest_div*blockSize:2*smallest_div*blockSize,0] = 0.0
    controlVoltages[2*smallest_div*blockSize:3*smallest_div*blockSize,0] = 900 #voltageGrid[-1]
    #controlVoltages[3*blockSize:4*blockSize,0] = 1
    
    for i in range(smallest_div): 
        controlVoltages[i*blockSize:(i+1)*blockSize,1] = -900 + i*0.16666*1800 #voltageGrid[0] + i*0.16666*(voltageGrid[-1]-voltageGrid[0])
        controlVoltages[i*blockSize:(i+1)*blockSize,2:] = NN.initTraj(controls, voltageGrid)
        # WARNING: Scale CV properly!!
    controlVoltages[smallest_div*blockSize:2*smallest_div*blockSize,1] = controlVoltages[0:smallest_div*blockSize,1]
    controlVoltages[2*smallest_div*blockSize:3*smallest_div*blockSize,1] = controlVoltages[0:smallest_div*blockSize,1]
    controlVoltages[smallest_div*blockSize:2*smallest_div*blockSize,2:] = controlVoltages[0:smallest_div*blockSize,2:]
    controlVoltages[2*smallest_div*blockSize:3*smallest_div*blockSize,2:] = controlVoltages[0:smallest_div*blockSize,2:]
    
    return controlVoltages
        