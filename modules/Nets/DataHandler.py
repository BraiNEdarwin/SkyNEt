#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:51:58 2018
DataHandlers contains functions to pre-process and load the data needed for the neural net training:
    1. DataLoader(data_dir, file_name, **kwargs)
    2. GetData(dir_file, syst = 'cuda')
    3. PrepData(main_dir, list_dirs, threshold = [-np.inf,np.inf])
@author: hruiz
"""
import sys
import os
import datetime
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math 

def loader(data_path):
    meta = {}
    with np.load(data_path) as data:
        for key in data:
            if key not in ['output', 'input']:
                meta[key] = data[key]
            else:
                exec(key+f' = data[\'{key}\']')
                if key in 'output':
                    meta['nr_raw_samples'] = len(data[key])
        ### Check if inputs available, otherwise generate them ###
        if 'input' not in list(data.keys()):
            print('Input generated as sine waves!')
            inputs = generate_inpsines(meta)
    data = locals()
    data.pop('key')
    data.pop('data')
    print('Data loaded with keys: \n',list(data.keys()))
    return data

def generate_inpsines(info):
    nr_raw_samples = info['nr_raw_samples']
    indices = np.arange(nr_raw_samples)[:,np.newaxis]
    freq = info['freq']
    amplitude = info['amplitude']
    offset = info['offset']
    fs = info['fs']
    phase = info['phase']
    sine_waves = amplitude*np.sin((2*np.pi*indices*freq + phase)/fs) + offset #Tel mark phase should be outside the brackets?

    return sine_waves
    
#%% 1. STEP: Clean data for further analysis
def PrepData(main_dir, data_filename = 'training_NN_data.npz',
             list_dirs=[], threshold = [-np.inf,np.inf]):
    '''Pre-process data, cleans clipping, generates input arrays if non existent (e.g. when sine-sampling
    was involved) and merges data sets if needed. The data arrays are merged into a single array and
    cropped given the thresholds.
    Arguments:
        - a string with path to main directory
    kwdargs:
        - data_filename: the name of the .npz file containing the data. It is assumend that this file 
            contains at least the key 'output'. Besides the keys 'output' and 'inputs', all other keys are 
            bundled into a dictionary called meta. If the key 'inputs' is non-existent, it generates the
            inputs assuming sine waves and using the information in meta. 
        - list_dirs: A list of strings indicating directories with training_NN_data.npz containing 'data'. 
        - threshold: A lower and upper threshold to crop data; default is [-np.inf,np.inf]
    
    NOTE:
        -The data is saved in main_dir+'/data4nn/ to a .npz file with keyes: inputs, outputs, 
        data_path and meta; the meta key has a dictionary as value containing metadata of the 
        sampling procedure, i.e sine, sawtooth, grid, random.
        -The inputs are on ALL electrodes in Volts and the output in nA.
        - Data does not undergo any transformation, this is left to the user.
        - Data structure of output and input are arrays of Nxd, where N is the number of voltage 
        configurations probed and d is the number of samples for each configuration or the input dimension.
    '''
    ### Load full data ###
    if list_dirs: # Merge data if list_dir is not empty
        raw_data = {}
        out_list = []
        inp_list = []
        meta_list = []
        datapth_list = []
        for dir_file in list_dirs:
            #loader loads data into a 
            _databuff = loader(main_dir+dir_file+data_filename)
            out_list.append(_databuff['output']) 
            meta_list.append(_databuff['meta'])
            datapth_list.append(_databuff['data_path'])
            ### Check if inputs available, otherwise generate them ###
            if 'input' in _databuff.keys():
                inp_list.append(_databuff['input'])
            else:
                inputs = generate_inpsines(_databuff['meta'])
                inp_list.append(inputs)
        #Generate numpy arrays out of the lists
        raw_data['output'] = np.concatenate(tuple(out_list))
        raw_data['input'] = np.concatenate(tuple(inp_list))
        raw_data['meta'] = meta_list
        raw_data['data_path'] = datapth_list
    else:
        raw_data = loader(main_dir+data_filename)
    
    ### Crop data ###
    nr_raw_samples = raw_data['meta']['nr_raw_samples']
    mean_output = np.mean(raw_data['output'],axis=1) 
    if type(threshold) is list:
        cropping_mask = (mean_output<threshold[1])*(mean_output>threshold[0])
    elif type(threshold) is float:
        cropping_mask = np.abs(mean_output)<threshold
    else:
        assert False, "Threshold not recognized! Must be list with lower and upper bound or float."
    
    output = raw_data['output'][cropping_mask]
    inputs = raw_data['inputs'][cropping_mask,:]
    print('Number of raw samples: ', nr_raw_samples)
    print('% of points cropped: ',(1-len(output)/nr_raw_samples)*100)
    
    plt.figure()
    plt.suptitle('Data for NN training')
    plt.subplot(211)
    plt.plot(inputs[:1000])
    plt.ylabel('inputs')
    plt.subplot(212)
    plt.plot(output[:1000])
    plt.ylabel('output')
    plt.show()
    
    # save with timestamp
    now = datetime.datetime.now()
    dirName = main_dir+'data4nn/'    
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    
    dirName += now.strftime("%Y_%m_%d")+'/'
    try:
        os.mkdir(dirName)
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
        
    save_to = dirName+'data_for_training'
    print(f'Cleaned data saved to \n {save_to}.npz')
    
    np.savez(save_to, inputs = inputs, output = output,
         meta=raw_data['meta'], data_path = raw_data['data_path'])
    
#%% STEP 2: Load Data, prepare for NN and return as torch tensors
def DataLoader(data_dir, file_name,
               val_size = 0.1, batch_size = 4*512, 
               syst = 'cuda', test_size = 0.0, steps=1):
    '''
    This function loads the data and returns it in a format suitable for the NN to handle:
        -Partitions the data into training, validation and test sets (if test_size is not None)
        -It defines the type of the tensor used by NN and defining if training is in CPU or GPU
        -Numpy arrays are converted to torch variables
    Arguments data_dir and file_name required are strings: a path to the directory containing the data and the name of the data file respectively.
    Default keyword arguments are:
                val_size = 0.1, 
                test_size = 0, 
                syst = 'cuda', 
                batch_size = 4*512
    Data structure loaded must be a .npz file with a directory having keys: 'inputs','outputs'.
    The inputs follow the convention that the first dimension is over CV configs and the second index is
    over input dimension, i.e. number of electrodes.
    '''
    assert isinstance(batch_size,int), 'Minibatch Size is not integer!!'
    print('Loading data from: \n'+data_dir+file_name)
    
    with np.load(data_dir+file_name) as data:
        meta = data['meta'].tolist()
#        print(type(meta))
        print('Metainfo about data:\n',meta)
        bf_inp = data['inputs'][::steps] # shape: Nx#electrodes
        bf_out = data['output'][::steps] #Outputs need dim Nx1
        print(f'Shape of outputs: {bf_out.shape}; shape of inputs: {bf_inp.shape}')
    #Shuffle data
    shuffler = np.random.permutation(len(bf_out))
    inputs = bf_inp[shuffler]
    outputs = bf_out[shuffler]
            
    assert len(outputs)==len(inputs), 'Inputs and Outpus have NOT the same length'
    nr_samples = len(outputs)
    pc = nr_samples/meta['nr_raw_samples']
    print(f'Nr. of samples is {nr_samples}; {math.ceil(pc*100)}% of raw data')
    
    n_test = int(test_size*nr_samples)
    if test_size:
        # Devide data in training and test set 
        print('Size of TEST set is ',n_test)
        outputs_test = outputs[:n_test] 
        inputs_test = inputs[:n_test]
        
        #Save test set
        test_file = data_dir+'test_set_from_trainbatch'
        np.savez(test_file, inputs=inputs_test, outputs=outputs_test)
        print(f'Test set saved in \n {test_file}.npz')
        
        outputs = outputs[n_test:] 
        inputs = inputs[n_test:]
    else:
        print('No TEST set partitioned')
    
    ### Partition into training and validation sets ###
    n_val = int(nr_samples*val_size)
    assert val_size+test_size<0.5, 'WARNING: Test and validation set over 50% of total data.'

    nr_minibatches = int((nr_samples-n_val)/batch_size)
    n_val += nr_samples - (nr_minibatches*batch_size + n_val)
    assert isinstance(n_val,int), 'n_val is not integer!!'
    print('Size of VALIDATION set is ',n_val)
    print('Size of TRAINING set is ',nr_samples-n_val)
    
    #Validation set
    inputs_val = inputs[:n_val]
    outputs_val = outputs[:n_val]
    # Training set
    inputs_train = inputs[n_val:]
    outputs_train = outputs[n_val:]
    ### Sanity Check ###
    assert nr_minibatches*batch_size + n_val + n_test == nr_samples, 'Data points not properly allocated!'
    if not outputs_train.shape[0] == inputs_train.shape[0]:
        raise ValueError('Input and Output Batch Sizes do not match!')
    
    ### Define Data Type for PyTorch ###
    if syst is 'cuda':
        print('Train with CUDA')
        dtype = torch.cuda.FloatTensor
#        itype = torch.cuda.LongTensor
    else: 
        print('Train with CPU')
        dtype = torch.FloatTensor
#        itype = torch.LongTensor
    
    # Create Tensors to hold inputs and outputs, and wrap them in Variables.
    # From Numpy to Pytorch b = torch.from_numpy(a)    
#    x = torch.from_numpy(inputs_train).type(dtype)
#    y = torch.from_numpy(outputs_train).type(dtype)
#    
#    x = Variable(x)
#    y = Variable(y)

#    permutation = torch.randperm(int(nr_samples-n_test)).type(itype) # Permute indices 
#    train_indices = permutation[n_val:]
#    val_indices = permutation[:n_val]
#    x_train = x[train_indices]
#    y_train = y[train_indices]
#    x_val = x[val_indices]
#    y_val = y[val_indices]
    
    x_train = torch.from_numpy(inputs_train).type(dtype)
    y_train = torch.from_numpy(outputs_train).type(dtype)
    x_val = torch.from_numpy(inputs_val).type(dtype)
    y_val = torch.from_numpy(outputs_val).type(dtype)

    return [(x_train,y_train),(x_val,y_val)]

#%% EXTRA: Just load data and return as torch.tensor
def GetData(dir_file, syst = 'cuda'):
    '''Get data from dir_file. Returns the inputs as torch Variables and targets as numpy-arrays. 
    dtype of inputs is defined with kwarg syst. Default is 'cuda'.
    NOTES:
        -The data must be in a .npz file with keys 'inputs' & 'outputs'
        -This function assumes that inputs are already scaled s.t. inputs[:,2:] are in [0,1]
            and inputs[:,:2] are in Volts (thay are the task defined inputs to the device), 
            so no scaling is performed in this function; to clean and scale the data use PrepData.
    '''
    
    targets = np.load(dir_file)['outputs']
    inputs = np.load(dir_file)['inputs']
    
    if syst is 'cuda':
        print('Inputs dtype defined for CUDA')
        dtype = torch.cuda.FloatTensor

    else: 
        print('Inputs dtype defined for CPU')
        dtype = torch.FloatTensor
           
    x = torch.tensor(inputs)
    inputs = Variable(x.type(dtype))
    
    return inputs, targets

#%%
####################################################################################################
####################################### MAIN #######################################################
####################################################################################################    
if __name__ == '__main__':
    main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/NN_data_Mark/7D_train_data/'
    if sys.argv[1] == '-dl':
        if len(sys.argv) > 2:
            data_dir = sys.argv[2]
        else:
            dir_data = 'data4nn/2019_04_03/'
            data_dir = main_dir+dir_data    
        file_name = 'data_for_training.npz'
        print('Loading data...')
        data = DataLoader(data_dir, file_name,steps=3)
    
    elif sys.argv[1] == '-pd': 
        print('Cleaning and preparing data...')
        PrepData(main_dir,data_filename = 'training_NN_dataT.npz', threshold = [-39.1,36.3])
