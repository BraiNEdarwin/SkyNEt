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
# import measure_VCdim as vcd


class VCDimensionTest():

    def __init__(self, algorithm,
                 dirname=r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/checkpoint3000_02-07-23h47m.pt',
                 save='True'
                 ):
        self.dirname = dirname
        self.filename = 'summary_results'
        self.algorithm = algorithm
        self.init_containers()

    def init_containers(self):
        # Initialise container variables
        self.fitness_classifier = []
        self.genes_classifier = []
        self.output_classifier = []
        self.accuracy_classifier = []
        self.found_classifier = []

    def run_test(self, inputs, binary_labels, threshold):
        for label in binary_labels:
            self.__test_label(inputs, label, threshold)

        for i in range(len(self.genes_classifier)):
            if self.genes_classifier[i] is np.nan:
                self.genes_classifier[i] = \
                    np.nan * np.ones_like(self.genes_classifier[1])
                self.output_classifier[i] = \
                    np.nan * np.ones_like(self.output_classifier[1])

        self.__to_numpy_array()
        not_found = self.found_classifier == 0
        indx_nf = self.check_not_found(len(inputs[0]), binary_labels, not_found)
        # if self.save is True:
        #    self.__save(inputs, binary_labels, threshold, indx_nf)
        # if self.plot is True:
        #   self.plot(binary_labels, threshold)
        return binary_labels, indx_nf

    def oracle(self):
        print(self.capacity)
        return self.capacity == 1,

    def check_not_found(self, current_dimension, binary_labels, not_found):
        if not_found.size > 0:
            try:
                indx_nf = np.arange(2**current_dimension)[not_found]
            except IndexError as error:
                print(f'{error} \n Trying indexing bad_gates')
                indx_nf = binary_labels[not_found]
            print('Classifiers not found: %s' % indx_nf)
            binaries_nf = np.array(binary_labels)[not_found]
            print('belongs to : \n', binaries_nf)
        else:
            print('All classifiers found!')
            indx_nf = None
        return indx_nf

    def __test_label(self, inputs, label, threshold):
        if len(set(label)) == 1:
            print('Label ', label, ' ignored')
            genes, output, fitness, accuracy = np.nan, np.nan, np.nan, np.nan
            self.found_classifier.append(1)  # append
        else:
            print('Finding classifier ', label)

            genes, output, fitness, accuracy =\
                self.algorithm.optimize(inputs, label)
            if accuracy > threshold:
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

    def save(self, inputs, binary_labels, threshold, indx_nf, dimension):
        np.savez(self.dirname + "_" + self.filename + "_VCDIM" + dimension,
                 inputs=inputs,
                 binary_labels=binary_labels,
                 capacity=self.capacity,
                 found_classifier=self.found_classifier,
                 fitness_classifier=self.fitness_classifier,
                 accuracy_classifier=self.accuracy_classifier,
                 output_classifier=self.output_classifier,
                 genes_classifier=self.genes_classifier,
                 threshold=threshold,
                 indx_nf=indx_nf)

    def plot(self, binary_labels, threshold):  # pylint: disable=E0202
        plt.figure()
        plt.plot(self.fitness_classifier, self.accuracy_classifier, 'o')
        plt.plot(np.linspace(np.nanmin(self.fitness_classifier),
                             np.nanmax(self.fitness_classifier)),
                 threshold * np.ones_like(np.linspace(0, 1)), '-k')
        plt.xlabel('Fitness')
        plt.ylabel('Accuracy')
        plt.show()

        try:
            not_found = self.found_classifier == 0
            print('Classifiers not found: %s' %
                  np.arange(len(self.found_classifier))[not_found])
            binaries_nf = np.array(binary_labels)[not_found]
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

        except Exception:
            # @todo improve the exception management
            # @warning bare exception is not recommended
            print('Error in plotting output!')
