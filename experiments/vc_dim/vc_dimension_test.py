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

    def __init__(self, algorithm, output_dir, surrogate_model_name):
        self.algorithm = algorithm
        self.output_dir = output_dir
        self.init_filenames(surrogate_model_name)
        self.init_excel_writer()

    def init_filenames(self, surrogate_model_name):
        self.surrogate_model_name = surrogate_model_name
        self.test_data_name = '_capacity_test_data.xlsx'
        self.test_data_plot_name = '_plot.png'

    def init_excel_writer(self):
        path = self.output_dir + self.surrogate_model_name + self.test_data_name
        self.writer = pd.ExcelWriter(path, engine='openpyxl')  # pylint: disable=abstract-class-instantiated

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

    def close_test(self, threshold, dimension, show_plot):
        self.save_tab(threshold, dimension)
        self.save_plot(threshold, dimension, show_plot)
        return self.oracle()

    def oracle(self):
        return self.data.loc[self.data['found'] == False].size == 0  # noqa: E712

    def save_tab(self, threshold, dimension):
        aux = self.data.copy()
        aux.index = range(len(aux.index))
        tab_name = 'VC Dimension ' + str(dimension) + ' Threshold ' + str(threshold)
        aux.to_excel(self.writer, sheet_name=tab_name)

    def save_plot(self, threshold, dimension, show_plot):  # pylint: disable=E0202
        plt.figure()
        fitness_classifier = self.data['fitness'].to_numpy()
        plt.plot(fitness_classifier, self.data['accuracy'].to_numpy(), 'o')
        plt.plot(np.linspace(np.nanmin(fitness_classifier),
                             np.nanmax(fitness_classifier)),
                 threshold * np.ones_like(np.linspace(0, 1)), '-k')
        plt.xlabel('Fitness')
        plt.ylabel('Accuracy')
        plt.savefig(self.output_dir + self.surrogate_model_name + '_dimension_' + str(dimension) + self.test_data_plot_name)
        if show_plot:
            plt.show()

        # try:
        # not_found = self.found_classifier == 0
        # print('Classifiers not found: %s' %
        #       np.arange(len(self.found_classifier))[not_found])
        # binaries_nf = np.array(binary_labels)[not_found]  # labels not found
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
        #     @todo improve the exception management
        #     print('Error in plotting output!')
