#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:18:29 2018
Wrapper to measure the VC dimension of a device using the measurement script measure_VCdim.py
This wrapper creates the binary labels for N points and for each label it finds the control voltages.
If successful (measured by a threshold on the correlation and by the perceptron accuracy), the entry 1 is set in a vector corresponding to all labellings.
@author: hruiz and ualegre
"""

import numpy as np
from matplotlib import pyplot as plt
from create_binary import bintarget

try:
    import instruments.InstrumentImporter
except ModuleNotFoundError:
    print(r'No module named instruments')

import evolve_VCdim as vcd
# import measure_VCdim as vcd


class VCDimensionTest():

    def __init__(self):
        self.inputs = [[-1., 0.4, -1., 0.4], [-1., -1., 0.4, 0.4]]
        # [[-0.7,0.7,-0.7,0.7,-1.,1.],[-0.7,-0.7,0.7,0.7,0.,0.]]
        self.N = len(self.inputs[0])
        # Create save directory
        # @todo improve the way in which directories are handled
        self.dirname = r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/checkpoint3000_02-07-23h47m.pt'

        # Create binary labels for N samples
        # bad_gates = # for N=6 on model [51]
        # ###### On Device ########
        # [55]#[22,23,48,52,53,55,57,60,61] for N=6 w. large range
        # for N=6 with (+/-0.35, 0.) as inputs 5 & 6 w. range +/-[1.2,1.0]: [6,33,37,41,45,53,57,60,61]
        # --> bad gates for N=6 w. range +/-0.9 and lower: [1,3,6,7,9,12,14,17,19,22,23,24,25,28,30,33,35,36,37,38,39,41,44,45,46,47,49,51,52,53,54,55,56,57,60,61,62]
        # binary_labels = bintarget(N)[bad_gates].tolist()
        self.binary_labels = bintarget(self.N).tolist()

        self.threshold = (1-0.5/self.N)  # 1-(0.65/N)*(1+1.0/N)
        print('Threshold for acceptance is set at: ', self.threshold)
        # Initialize container variables
        self.fitness_classifier = []
        self.genes_classifier = []
        self.output_classifier = []
        self.accuracy_classifier = []
        self.found_classifier = []

    def test(self):
        for label in self.binary_labels:
            self.__test_label(label)

        for i in range(len(self.genes_classifier)):
            if self.genes_classifier[i] is np.nan:
                self.genes_classifier[i] = np.nan*np.ones_like(self.genes_classifier[1])
                self.output_classifier[i] = np.nan*np.ones_like(self.output_classifier[1])

        self.__to_numpy_array()
        self.__save()
        self.plot()

    def __test_label(self, label):
        if len(set(label)) == 1:
            print('Label ', label, ' ignored')
            genes, output, fitness, accuracy = np.nan, np.nan, np.nan, np.nan
            self.found_classifier.append(1)
        else:
            print('Finding classifier ', label)

            genes, output, fitness, accuracy =\
                vcd.evolve(self.inputs, label, path_2_NN=self.dirname, hush=True)
            if accuracy > self.threshold:
                self.found_classifier.append(1)
            else:
                self.found_classifier.append(0)

        self.genes_classifier.append(genes)
        self.output_classifier.append(output)
        self.fitness_classifier.append(fitness)
        self.accuracy_classifier.append(accuracy)

    def __to_numpy_array(self):
        self.fitness_classifier = np.array(self.fitness_classifier)
        self.accuracy_classifier = np.array(self.accuracy_classifier)
        self.found_classifier = np.array(self.found_classifier)
        self.capacity = np.mean(self.found_classifier)
        self.output_classifier = np.array(self.output_classifier)
        self.genes_classifier = np.array(self.genes_classifier)

    def __save(self):
        np.savez(self.dirname+'Summary_Results',
                 inputs=self.inputs,
                 binary_labels=self.binary_labels,
                 capacity=self.capacity,
                 found_classifier=self.found_classifier,
                 fitness_classifier=self.fitness_classifier,
                 accuracy_classifier=self.accuracy_classifier,
                 output_classifier=self.output_classifier,
                 genes_classifier=self.genes_classifier,
                 threshold=self.threshold)

    def plot(self):
        try:
            vcd.reset(0, 0)
        except AttributeError:
            print(r'module evolve_VCdim has no attribute reset')

        plt.figure()
        plt.plot(self.fitness_classifier, self.accuracy_classifier, 'o')
        plt.plot(np.linspace(np.nanmin(self.fitness_classifier),
                                np.nanmax(self.fitness_classifier)),
                                self.threshold*np.ones_like(np.linspace(0, 1)), '-k')
        plt.xlabel('Fitness')
        plt.ylabel('Accuracy')
        plt.show()

        try:
            not_found = self.found_classifier == 0
            print('Classifiers not found: %s' %
                        np.arange(len(self.found_classifier))[not_found])
            binaries_nf = np.array(self.binary_labels)[not_found]
            print('belongs to : \n', binaries_nf)
            output_nf = self.output_classifier[not_found]
            # plt output of failed classifiers
            plt.figure()
            plt.plot(output_nf.T)
            plt.legend(binaries_nf)
            # plt gnes with failed classifiers
            plt.figure()
            plt.hist(self.genes_classifier[not_found, :5], 30)
            plt.legend([1, 2, 3, 4, 5])

            plt.show()
        except:
            #@todo improve the exception management
            #@warning bare exception is not recommended
            print('Error in plotting output!')


if __name__ == "__main__":
    test = VCDimensionTest()
    test.test()
