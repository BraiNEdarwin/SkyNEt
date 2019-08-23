#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:32:25 2018
This script generates all binary assignments of N elements.
@author: hruiz and ualegre
"""
import numpy as np
from vc_dimension_test import VCDimensionTest
from vc_dimension_test import VCDimensionTestConfigs


class CapacityTest():

    def __init__(self, configs):
        self.configs = configs
        self.current_dimension = configs.from_dimension
        self.threshold = self.__calculate_threshold()
        self.vcdimension_test = VCDimensionTest(configs.vcdimension_configs)

    def run_test(self):
        veredict = True
        while veredict is True:
            print('Generating inputs for VC Dimension %d: ' % self.current_dimension)
            inputs, binary_labels = self.generate_test_inputs(self.current_dimension)
            veredict = self.vcdimension_test.run_test(inputs, binary_labels, self.threshold)
            if self.__next_vcdimension() is False:
                break

    def __calculate_threshold(self):
        return self.configs.threshold_numerator / self.current_dimension

    def __next_vcdimension(self):
        if self.current_dimension + 1 > self.configs.to_dimension:
            return False
        else:
            self.current_dimension = + 1
            self.__calculate_threshold()

    # @todo change generation of inputs to differetn vc dimensions
    def generate_test_inputs(self, vc_dim):
        # ###### On Device ########
        # [55]#[22,23,48,52,53,55,57,60,61] for N=6 w. large range
        # for N=6 with (+/-0.35, 0.) as inputs 5 & 6 w. range +/-[1.2,1.0]: [6,33,37,41,45,53,57,60,61]
        # --> bad gates for N=6 w. range +/-0.9 and lower: [1,3,6,7,9,12,14,17,19,22,23,24,25,28,30,33,35,36,37,38,39,41,44,45,46,47,49,51,52,53,54,55,56,57,60,61,62]
        # binary_labels = bintarget(N)[bad_gates].tolist()
        inputs = []
        for i in range(2):
            inputs.append(
                np.random.choice(
                    a=[self.configs.voltage_true, self.configs.voltage_false],
                    size=vc_dim))
        binary_labels = self.__generate_binary_target(vc_dim).tolist()
        return inputs, binary_labels

    def __generate_binary_target(self, target_dim):
        # length of list, i.e. number of binary targets
        binary_target_no = 2**target_dim
        assignments = []
        list_buf = []

        # construct assignments per element i
        print('===' * target_dim)
        print('ALL BINARY LABELS:')
        level = int((binary_target_no / 2))
        while level >= 1:
            list_buf = []
            buf0 = [0] * level
            buf1 = [1] * level
            while len(list_buf) < binary_target_no:
                list_buf += (buf0 + buf1)
            assignments.append(list_buf)
            level = int(level / 2)

        binary_targets = np.array(assignments).T
        print(binary_targets)
        print('===' * target_dim)
        return binary_targets


class CapacityTestConfigs():
    def __init__(self, vcdimension_configs=VCDimensionTestConfigs(), from_dimension=1, to_dimension=4, voltage_false=-1., voltage_true=0.4, threshold_numerator=1-0.5):
        self.vcdimension_configs = vcdimension_configs
        self.from_dimension = from_dimension
        self.to_dimension = to_dimension
        self.voltage_false = voltage_false
        self.voltage_true = voltage_true
        # The threshold is calculated as: threshold_numerator/vc_dimension
        # Create binary labels for N samples
        # bad_gates = # for N=6 on model [51]
        self.threshold_numerator = threshold_numerator
        # self.threshold = (1-0.5/self.N)  # 1-(0.65/N)*(1+1.0/N)
        # print('Threshold for acceptance is set at: ', self.threshold)


if __name__ == '__main__':
    test = CapacityTest(CapacityTestConfigs(from_dimension=4))
    test.generate_test_inputs
    test.run_test()
