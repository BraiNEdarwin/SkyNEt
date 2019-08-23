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


try:
    import instruments.InstrumentImporter
except ModuleNotFoundError:
    print(r'No module named instruments')

import evolve_VCdim as vcd
# import measure_VCdim as vcd


class VCDimensionTest():

    def __init__(self, configs):
        self.configs = configs

    def run_test(self, inputs, binary_labels, threshold):
        for label in binary_labels:
            self.__test_label(inputs, label, threshold)

        for i in range(len(self.configs.genes_classifier)):
            if self.configs.genes_classifier[i] is np.nan:
                self.configs.genes_classifier[i] = np.nan * np.ones_like(self.configs.genes_classifier[1])
                self.configs.output_classifier[i] = np.nan * np.ones_like(self.configs.output_classifier[1])

        self.__to_numpy_array()

        not_found = found_classifier==0
        check_not_found(not_found)
        if self.configs.save:
            self.__save(inputs, binary_labels, threshold)
        if self.configs.plot:
            self.plot(binary_labels, threshold)
        return self.oracle()

    def oracle(self):
        print(self.configs.capacity)
        return self.configs.capacity == 1

    def check_not_found(not_found):
        if not_found.size > 0:
            try:
                indx_nf = np.arange(2**N)[not_found]
            except IndexError as error:
                print(f'{error} \n Trying indexing bad_gates')
                indx_nf = bad_gates[not_found]
            print('Classifiers not found: %s' % indx_nf)
            binaries_nf = np.array(binary_labels)[not_found]
            print('belongs to : \n', binaries_nf)
        else:
            print('All classifiers found!')
            indx_nf = None

    def __test_label(self, inputs, label, threshold):
        if len(set(label)) == 1:
            print('Label ', label, ' ignored')
            genes, output, fitness, accuracy = np.nan, np.nan, np.nan, np.nan
            self.configs.found_classifier.append(1)
        else:
            print('Finding classifier ', label)

            genes, output, fitness, accuracy =\
                vcd.evolve(inputs, label, path_2_NN=self.configs.dirname, hush=True)
            if accuracy > threshold:
                self.configs.found_classifier.append(1)
            else:
                self.configs.found_classifier.append(0)

        self.configs.genes_classifier.append(genes)
        self.configs.output_classifier.append(output)
        self.configs.fitness_classifier.append(fitness)
        self.configs.accuracy_classifier.append(accuracy)

    def __to_numpy_array(self):
        self.configs.fitness_classifier = np.array(self.configs.fitness_classifier)
        self.configs.accuracy_classifier = np.array(self.configs.accuracy_classifier)
        self.configs.found_classifier = np.array(self.configs.found_classifier)
        self.configs.capacity = np.mean(self.configs.found_classifier)
        self.configs.output_classifier = np.array(self.configs.output_classifier)
        self.configs.genes_classifier = np.array(self.configs.genes_classifier)

    def __save(self, inputs, binary_labels, threshold):
        np.savez(self.configs.dirname + 'Summary_Results',
                 inputs=inputs,
                 binary_labels=binary_labels,
                 capacity=self.configs.capacity,
                 found_classifier=self.configs.found_classifier,
                 fitness_classifier=self.configs.fitness_classifier,
                 accuracy_classifier=self.configs.accuracy_classifier,
                 output_classifier=self.configs.output_classifier,
                 genes_classifier=self.configs.genes_classifier,
                 threshold=threshold)

    def plot(self, binary_labels, threshold):
        try:
            vcd.reset(0, 0)
        except AttributeError:
            print(r'module evolve_VCdim has no attribute reset')

        plt.figure()
        plt.plot(self.configs.fitness_classifier, self.configs.accuracy_classifier, 'o')
        plt.plot(np.linspace(np.nanmin(self.configs.fitness_classifier),
                             np.nanmax(self.configs.fitness_classifier)),
                 threshold*np.ones_like(np.linspace(0, 1)), '-k')
        plt.xlabel('Fitness')
        plt.ylabel('Accuracy')
        plt.show()

        try:
            not_found = self.configs.found_classifier == 0
            print('Classifiers not found: %s' %
                  np.arange(len(self.configs.found_classifier))[not_found])
            binaries_nf = np.array(binary_labels)[not_found]
            print('belongs to : \n', binaries_nf)
            output_nf = self.configs.output_classifier[not_found]
            # plt output of failed classifiers
            plt.figure()
            plt.plot(output_nf.T)
            plt.legend(binaries_nf)
            # plt gnes with failed classifiers
            plt.figure()
            plt.hist(self.configs.genes_classifier[not_found, :5], 30)
            plt.legend([1, 2, 3, 4, 5])

            plt.show()
        except:
            # @todo improve the exception management
            # @warning bare exception is not recommended
            print('Error in plotting output!')


class VCDimensionTestConfigs():

    def __init__(self,
                 # inputs = [[-1., 0.4, -1., 0.4], [-1., -1., 0.4, 0.4]],
                 dirname=r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/checkpoint3000_02-07-23h47m.pt',
                 save='True',
                 plot='False',
                 vc_dim=4):

        # Create save directory
        # @todo improve the way in which directories are handled
        self.dirname = dirname
        self.plot = plot
        self.save = save

        # Initialize container variables
        self.fitness_classifier = []
        self.genes_classifier = []
        self.output_classifier = []
        self.accuracy_classifier = []
        self.found_classifier = []


if __name__ == "__main__":
    configs = VCDimensionTestConfigs()
    inputs = [[-1., 0.4, -1., 0.4], [-1., -1., 0.4, 0.4]]
    test = VCDimensionTest(configs)
    print(test.run_test())
