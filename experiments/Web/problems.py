#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ljknoll
"""

import torch

def get_class_weights(target_data):
    """
    returns normalized weights for each class in target_data
    Note target_data should only have integer values corresponding to each class
    """
    weights = torch.bincount(target_data.long()).float()
    weights = 1/weights
    weights /= torch.sum(weights)
    return weights


def boolean(N, input_values=[0., 1.], target_values=[0., 1.], sigma=None):
    """
    Defines input and target data for all of the boolean logic gates:
    Boolean logic gates
    I0 I1    AND NAND OR NOR XOR XNOR
    0  0     0   1    0  1   0   1
    0  1     0   1    1  0   1   0
    1  0     0   1    1  0   1   0
    1  1     1   0    1  0   0   1
    
    Arguments:
        N                   Number of points each boolean case is copied
        input_values        lower and upper values used as inputs
        target_values       lower and upper values used as target
        (optional) sigma    standard deviation of gaussian noise added to target
    Returns:
        gates               list of all 6 boolean logic gates
        input_data          4*N by 2 tensor, index with [signal, which input]
        target_data         6 by 4*N tensor, index with [gate, signal]
    """
    gates = ['AND','NAND','OR','NOR','XOR','XNOR']
    
    # each of the 4 boolean input cases is N points long
    input_data = torch.zeros(N*4,2)
    input_data[N:2*N,   1] = 0.9
    input_data[2*N:3*N, 0] = 0.9
    input_data[3*N:,    0] = 0.9
    input_data[3*N:,    1] = 0.9
    
    # target data for all the 6 boolean logic gates
    target_data = target_values[1]*torch.ones(6, 4*N)
    target_data[0, :3*N] = target_values[0]
    target_data[1, 3*N:] = target_values[0]
    target_data[2, :N] = target_values[0]
    target_data[3, N:] = target_values[0]
    target_data[4, :N] = target_values[0]
    target_data[4, 3*N:] = target_values[0]
    target_data[5, N:3*N] = target_values[0]
    
    if sigma is not None:
        gauss = torch.distributions.Normal(0.0, sigma)
        target_data += gauss.sample((6, 4*N))
    
    return gates, input_data, target_data

def get_control_problem(N, mean_I0=-0.3, mean_I1=-0.3, amp_I0=0.9, amp_I1=0.9):
    """
    function to generate input and target data for control problem
    Arguments:
        mean_I0&1:  mean of input I0 and I1
        amp_I0&1:   amplitude of input I0 and I1
            For example, mean of 0.3 and amp of 0.4 means input values are between -0.1 and 0.7
    Returns:
        input_data:     torch tensor (21*N, 2)
        target_data:    torch tensor (21*N, 1)
    """
    values_I0 = [mean_I0-amp_I0+amp_I0*2/2*(i//N//7) for i in range(21*N)]
    values_I1 = [mean_I1-amp_I1+amp_I1*2/6*(i//N%7) for i in range(21*N)]
    input_data = torch.tensor([values_I0, values_I1]).t()
    targets = [0,0,0,1,1,1,1,0,1,1,1,1,2,2,1,1,2,1,2,1,2]
    target_data = torch.tensor([targets]).view(-1,1).repeat(1,N).view(-1,1)
    return input_data, target_data