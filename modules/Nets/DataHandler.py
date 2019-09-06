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
import matplotlib.pyplot as plt
import math 
import pdb

def loader(data_path, index):
    print('Loading data from: \n',data_path)
    meta = {}
    data_dic = {} 
    with np.load(data_path) as data:
        print('with keys: \n',list(data.keys()))
        assert any('output' in s for s in list(data.keys())), 'Keyvalue \'output(s)\' is missing! Make sure you included the outputs in the data.'
        for key in data:
            if key in 'outputs':
                data_dic['outputs'] = data[key]
                meta['nr_raw_samples'] = len(data_dic['outputs'])
            elif key in 'inputs':
                data_dic['inputs'] = data[key]
            else:
                meta[key] = data[key]
                    
        ### Check if inputs available, otherwise generate them ###
        if 'inputs' not in list(data_dic.keys()):
            if index:
                print('Inputs are represented with their index, so lightNNet must be used.')
                data_dic['inputs'] = np.arange(0,data_dic['outputs'].shape[0])[:,np.newaxis]
            else:
                print('Input generated as sine waves!')
                data_dic['inputs'] = generate_inpsines(meta)

    data_dic['meta'] = meta        
    print('Data loaded with keys: \n',list(data_dic.keys()))
    print('meta key is dict with keys:\n', list(data_dic['meta'].keys()))
    return data_dic

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
             list_dirs=[], threshold = [-np.inf,np.inf], index=False, plot=False):

    '''Pre-process data, cleans clipping, generates input arrays if non existent (e.g. when sine-sampling
    was involved) and merges data sets if needed. The data arrays are merged into a single array and
    cropped given the thresholds.
    Arguments:
        - a string with path to main directory
    kwdargs:
        - data_filename: the name of the .npz file containing the data. It is assumend that this file 
            contains at least the key 'outputs'. Besides the keys 'outputs' and 'inputs', all other keys are 
            bundled into a dictionary called meta. If the key 'inputs' is non-existent, it generates the
            inputs assuming sine waves and using the information in meta. The data_filename is also used 
            to name the file in which the pre-processed data is saved; it assumes that the use of the data is specified
            in the first word of the file name and words are separated by underscore, e.g. training_data or test_data.
        - list_dirs: A list of strings indicating directories with training_NN_data.npz containing 'data'. 
        - threshold: A lower and upper threshold to crop data; default is [-np.inf,np.inf]
        - plot: if set to True, it plots the first 1000 samples of the inputs and outputs
    
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
            _databuff = loader(main_dir+dir_file+data_filename, index)
            out_list.append(_databuff['outputs']) 
            meta_list.append(_databuff['meta'])
            datapth_list.append(_databuff['data_path'])
            inp_list.append(_databuff['inputs'])
        #Generate numpy arrays out of the lists
        raw_data['outputs'] = np.concatenate(tuple(out_list))
        raw_data['inputs'] = np.concatenate(tuple(inp_list))
        raw_data['meta'] = meta_list
        raw_data['data_path'] = datapth_list
    else:
        raw_data = loader(main_dir+data_filename, index)
        for key in raw_data['meta'].keys():
            if 'file' in key:
                raw_data['data_path'] = raw_data['meta'][key]
    
    ### Crop data ###
    nr_raw_samples = raw_data['meta']['nr_raw_samples']
    try:
        mean_output = np.mean(raw_data['outputs'],axis=1) 
    except:
        mean_output = raw_data['outputs']
        
    if type(threshold) is list:
        cropping_mask = (mean_output<threshold[1])*(mean_output>threshold[0])
    elif type(threshold) is float:
        cropping_mask = np.abs(mean_output)<threshold
    else:
        assert False, "Threshold not recognized! Must be list with lower and upper bound or float."
    
    outputs = raw_data['outputs'][cropping_mask]
    inputs = raw_data['inputs'][cropping_mask,:]
    print('Number of raw samples: ', nr_raw_samples)
    print('% of points cropped: ',(1-len(outputs)/nr_raw_samples)*100)
    
    if plot:
        plt.figure()
        plt.suptitle('Data for NN training')
        plt.subplot(211)
        plt.plot(inputs[:1000])
        plt.ylabel('inputs')
        plt.subplot(212)
        plt.plot(outputs[:1000])
        plt.ylabel('outputs')
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
    
    dirName += now.strftime("%d_%m_%Y")+'/'
    try:
        os.mkdir(dirName)
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    target_file = data_filename.split('_')[0]
    save_to = dirName+f'data_for_{target_file}'
    print(f'Cleaned data saved to \n {save_to}.npz')
    
    np.savez(save_to, inputs = inputs, outputs = outputs,
         meta=raw_data['meta'], data_path = raw_data['data_path'])
    
#%% STEP 2: Load Data, prepare for NN and return as list with information dict
def DataLoader(data_dir, file_name,
               val_size = 0.1, batch_size = 4*512, 
               test_size = 0.0, steps=1):
    '''
    This function loads the data and returns it in a format suitable for the NN to handle.
    Partitions the data into training, validation and test sets (if test_size is not None)
    Arguments data_dir and file_name required are strings: a path to the directory containing the data and the name of the data file respectively.
    Default keyword arguments are:
                val_size = 0.1, 
                test_size = 0, 
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
        print('Metainfo about data:\n',meta.keys())
        bf_inp = data['inputs'][::steps] # shape: Nx#electrodes
        bf_out = data['outputs'][::steps] #Outputs need dim Nx1
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

    return [[inputs_train,outputs_train],[inputs_val,outputs_val],meta]

#%% EXTRA: Just load data and return as torch.tensor
def GetData(dir_file, device = 'cuda'):
    '''Get data from dir_file. Returns the inputs as torch.Tensor and targets/outputs as numpy-arrays. 
    dtype of inputs is defined with kwarg syst. Default is 'cuda'.
    NOTES:
        -The data must be in a .npz file with keys 'inputs' & 'outputs' and NxD-structure.
        -This function assumes that data is cleaned; to clean the data use PrepData.
    '''
    
    targets = np.load(dir_file)['outputs']
    inputs = np.load(dir_file)['inputs']
    
    if device is 'cuda':
        print('Inputs dtype defined for CUDA')
        dtype = torch.cuda.FloatTensor

    else: 
        print('Inputs dtype defined for CPU')
        dtype = torch.FloatTensor
           
    inputs = torch.from_numpy(inputs).type(dtype)
    
    return inputs, targets

#%%
####################################################################################################
####################################### MAIN #######################################################
####################################################################################################    
if __name__ == '__main__':
    main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/April_2019/train set/'
#    r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/April_2019/random_test_set/'

    if sys.argv[1] == '-dl':
        if len(sys.argv) > 2:
            data_dir = sys.argv[2]
        else:
            dir_data = 'data4nn/16_04_2019/'
            data_dir = main_dir+dir_data    
        file_name = 'data_for_training.npz'
        print('Loading data...')
        data = DataLoader(data_dir, file_name,steps=3)
        meta = data[-1]
        print(f'Data has meta-info {list(meta.keys())}')
    
    elif sys.argv[1] == '-pd': 
        print('Cleaning and preparing data...')
        PrepData(main_dir,data_filename = 'test_NN_data.npz', threshold = [-39.1,36.3])
