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
import pandas as pd
from matplotlib import pyplot as plt


class VCDimensionTest():

    def __init__(self, algorithm,
                 dirname=r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/checkpoint3000_02-07-23h47m.pt',
                 save='True'
                 ):
        self.dirname = dirname
        self.filename = 'summary_results'
        self.algorithm = algorithm
        name = self.dirname + "_" + self.filename + '.xlsx'
        self.writer = pd.ExcelWriter(name, engine='openpyxl')  # pylint: disable=abstract-class-instantiated

    def init_data(self, inputs, binary_labels, threshold):
        column_names = ['gate', 'found', 'accuracy', 'fitness', 'output', 'genes']
        self.data = pd.DataFrame(index=pd.Series(map(str, binary_labels)), columns=column_names)
        self.threshold = threshold
        self.inputs = inputs
        self.binary_labels = binary_labels

    def run_test(self, binary_labels=np.array([])):
        if binary_labels.size == 0:
            binary_labels = self.binary_labels
        for label in binary_labels:
            self.__test_label(self.inputs, label, self.threshold)

        return self.data

    # def format_genes(self):
    #     for i in range(len(self.genes_classifier)):
    #         if self.genes_classifier[i] is np.nan:
    #             self.genes_classifier[i] = np.nan * np.ones_like(self.genes_classifier[1])
    #             self.output_classifier[i] = np.nan * np.ones_like(self.output_classifier[1])

    # def check_not_found(self, current_dimension, binary_labels, not_found):
    #     if not_found.size > 0:
    #         try:
    #             indx_nf = np.arange(2**current_dimension)[not_found]
    #         except IndexError as error:
    #             print(f'{error} \n Trying indexing bad_gates')
    #             indx_nf = binary_labels[not_found]
    #         print('Classifiers not found: %s' % indx_nf)
    #         binaries_nf = np.array(binary_labels)[not_found]
    #         print('belongs to : \n', binaries_nf)
    #     else:
    #         print('All classifiers found!')
    #         indx_nf = None
    #     return indx_nf

    def __test_label(self, inputs, label, threshold):

        if len(set(label)) == 1:
            print('Label ', label, ' ignored')
            genes, output, fitness, accuracy = np.nan, np.nan, np.nan, np.nan
            found = True
        else:
            print('Finding classifier ', label)

            genes, output, fitness, accuracy =\
                self.algorithm.optimize(inputs, label)
            found = (accuracy > threshold)

        row = {'gate': label, 'found': found, 'genes': np.array(genes), 'output': np.array(output), 'fitness': np.array(fitness), 'accuracy': np.array(accuracy)}
        self.data.loc[str(label)] = pd.Series(row)

    def close_test(self, threshold, dimension):
        self.save_tab(threshold, dimension)
        return self.oracle()

    def oracle(self):
        return self.data.loc[self.data['found'] == False].size == 0  # noqa: E712

    def save_tab(self, threshold, dimension):
        aux = self.data.copy()
        aux.index = range(len(aux.index))
        name = 'VC Dimension ' + str(dimension) + ' Threshold ' + str(threshold)
        aux.to_excel(self.writer, sheet_name=name)

    def plot(self, threshold):  # pylint: disable=E0202
        plt.figure()
        fitness_classifier = self.data['fitness'].to_numpy()
        plt.plot(fitness_classifier, self.data['accuracy'].to_numpy(), 'o')
        plt.plot(np.linspace(np.nanmin(fitness_classifier),
                             np.nanmax(fitness_classifier)),
                 threshold * np.ones_like(np.linspace(0, 1)), '-k')
        plt.xlabel('Fitness')
        plt.ylabel('Accuracy')
        plt.show()

        # try:  # solo hacerlo en las primeras dimensions
        # not_found = self.found_classifier == 0
        # print('Classifiers not found: %s' %
        #       np.arange(len(self.found_classifier))[not_found])
        # binaries_nf = np.array(binary_labels)[not_found]  # labels que no encontro
        # print('belongs to : \n', binaries_nf)
        # output_nf = self.output_classifier[not_found]
        # # plt output of failed classifiers
        # plt.figure()
        # plt.plot(output_nf.T)
        # plt.legend(binaries_nf)
        # # plt gnes with failed classifiers
        # plt.figure()
        # plt.hist(self.genes_classifier[not_found, :5], 30)
        # plt.legend([1, 2, 3, 4, 5])
        # plt.show()

        # except Exception:
        # @todo improve the exception management
        # @warning bare exception is not recommended
        #     print('Error in plotting output!')
