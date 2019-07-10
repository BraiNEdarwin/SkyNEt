#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:28:20 2019

Script that reads the results that have been stored by search_control_problem.py in workspace.
Once 'results' is in your workspace, plot_control_problem.py should be used to analyze the results.
These scripts have been separated because it is inconvenient to load the results from files every time.

@author: ljknoll
"""

import os
import re
import pickle
import torch

all_files = os.listdir()
r = re.compile("single_\d+.*.p")

datalist = list(filter(r.match, all_files))


# format of list results:
# history:initial_parameters, best:best parameters of epochs, error:error, class:classification
results = {'history':torch.tensor([]), 'best':[], 'error':[], 'class':[]}

for filename in datalist:
    with open(filename, 'rb') as f:
        temp = pickle.load(f)
        results['history'] = torch.cat((results['history'], temp[0]), dim=0)
        results['best'] += temp[1]
        results['error'] += temp[2]
        results['class'] += temp[3]