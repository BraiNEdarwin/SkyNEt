
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from create_binary import bintarget
import numpy as np
from matplotlib import pyplot as plt
import os
import evolve_VCdim5_Final as vcd 
"""
Wrapper to measure the VC dimension of a device using the measurement script measure_VCdim.py
This wrapper creates the binary labels for N points and for each label it finds the control voltages.
If successful (measured by a threshold on the correlation and by the perceptron accuracy), the entry 1 is set in a vector corresponding to all labellings.
User specific parameters can be defined 
----------------------------------------------------------------------------
User specific parameters
----------------------------------------------------------------------------
- inputs: Input data 
- bad_gates: Gates you want to look for
- save: Boolean variable indicating whether you want to save the data 
- filepath_folder: Folder where you want to save the data
- no_measurements: Number of times you want to look for the 'badgates'. 
- below one can find a piece of code to plot the outputs. Dependent on the number of bad_gates, subplot_no must be adapted
"""

#inputs 
inputs = [[-1.1,0.5,-1.1,0.5,-0.6],[-1.1,-1.1,0.5,0.5,0]]
bad_gates = [3,11,19,22,23,30]
#bad_gates = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
save =False
filepath_folder = r'../../test/evolution_test/VCdim_testing5/Verification_data_ramp_new2/'
no_measurements = 1



dat_list = []
for i in range(no_measurements):
    dat_list.append('meas'+str(i)+'/')


for i in range(len(dat_list)): 
    name = dat_list[i] 
    N=len(inputs[0])
    #Create save directory
    if save:
            filepath = filepath_folder + name 
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            saveDirectory = filepath 
    binary_labels = bintarget(N)[bad_gates].tolist()  
    threshold = (1-0.5/N)
    print('Threshold for acceptance is set at: ',threshold)
    
    #Initialize container variables
    genes_classifier = []
    output_classifier = []
    fitness_classifier = []
    accuracy_classifier = []
    target_classifier = []
    end_classifier = []   
    found_classifier = []
    #Call the evolve script to evolve the NN
    for bl in binary_labels:
        if len(set(bl))==1:
            print('Label ',bl,' ignored')
            genes, output, fitness, accuracy, target = np.nan, np.nan, np.nan, np.nan, np.nan
            found_classifier.append(1)
        else:
            print('Finding classifier ',bl)
            genes, output, fitness, accuracy, target,end, = vcd.evolve(i, threshold, inputs, bl)
            if accuracy>threshold:
                found_classifier.append(1)
            else:
                found_classifier.append(0)
        genes_classifier.append(genes)
        output_classifier.append(output)
        fitness_classifier.append(fitness)
        accuracy_classifier.append(accuracy)
        target_classifier.append(target)    
        end_classifier.append(end)  
    end_classifier = np.asarray(end_classifier)   
    fitness_classifier = np.asarray(fitness_classifier)
    accuracy_classifier = np.asarray(accuracy_classifier)
    found_classifier = np.asarray(found_classifier)
    capacity = np.mean(found_classifier)
    
    for i in range(len(genes_classifier)):
        if genes_classifier[i] is np.nan:
            genes_classifier[i] = np.nan*np.ones_like(genes_classifier[1])
            output_classifier[i] = np.nan*np.ones_like(output_classifier[1])
    output_classifier = np.asarray(output_classifier)
    genes_classifier = np.asarray(genes_classifier)
    #Save data returned from the evolve script
    if save:
        np.savez(filepath+'Results',
             inputs = inputs,
             binary_labels = binary_labels,
             capacity = capacity,
             found_classifier = found_classifier,
             fitness_classifier = fitness_classifier,
             accuracy_classifier = accuracy_classifier,
             output_classifier = output_classifier,
             genes_classifier = genes_classifier,
             target_classifier = target_classifier,
             end_classifier = end_classifier,
             threshold = threshold)
#only show output if you did 1 run  

#if no_measurements == 1:
#    output_f = output_classifier
#    plt.figure()
#    subplot_no = 230
#    output_f = output_classifier
#    for i in range(0, len(binary_labels)): 
#        subplot_no = subplot_no + 1 
#        ax = plt.subplot(subplot_no)
#        ax.plot(output_f[i][w].T,label=binary_labels[i])
#        ax.legend()
#        plt.title('Accuracy: '+str(accuracy_classifier[i]), fontsize=20)
#        plt.rc('xtick', labelsize=20) 
#        plt.rc('ytick', labelsize=20) 
#        plt.show()
#        
        
