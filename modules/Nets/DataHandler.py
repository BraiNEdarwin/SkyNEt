#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:51:58 2018
DataHandlers contains functions to preprocess and load the data needed for the neural net training:
    1. DataLoader(data_dir, file_name, **kwargs)
    2. GetData(dir_file, syst = 'cuda')
    3. PrepData(main_dir, list_dirs, threshold = 1000)
@author: hruiz
"""
import sys
import os
import datetime
import numpy as np
import torch
from torch.autograd import Variable

#%%
def DataLoader(data_dir, file_name,
               val_size = 0.1, test_size = 0.1, 
               syst = 'cuda', batch_size = 4*512,
               test_set = False):
    '''
    This function loads the data and returns it in a format suitable for the NN to handle:
        -Partitions the data into training, validation and test sets
        -It defines the type of the tensor used by NN and defining if training is in CPU or GPU
        -Numpy arrays are converted to torch variables
    Argument data_dir required is a path to the directory containing the data
    Default keyword arguments are:
                file_name = 'combi_data4nn.npz',
                val_size = 0.1, 
                test_size = 0.1, 
                syst = 'cuda', 
                batch_size = 4*512
                test_set = False
    Data structure is a .npz file with a directory having keys: 'inputs','outputs','var_outputs'
    The inputs follow the convention that the first dimension is over CV configs and the second index is
    over [x_inp,CV], where x_inp are the inputs defined by the task. 
    NOTE: while the CV are scaled to the range [0,1], these x_inp values are not scaled!!
    '''
    assert isinstance(batch_size,int), 'Minibatch Size is not integer!!'
    print('Loading data from: \n'+data_dir)
    
    data = np.load(data_dir+file_name)
    ## DEFINE INPUTS ##
    x_inp = data['inputs'][:,:2]
    np.save(data_dir+'X_inputs',x_inp)
    #Shuffle indices
    shuffler = np.random.permutation(len(data['outputs']))
    inputs = data['inputs'][shuffler]
    # Nx(#electrodes): the first two are the inputs to the system, the rest are control voltages
    assert np.max(np.abs(inputs[:,2:]))<=1.0 and np.min(inputs[:,2:])>=0, 'Voltages are not scaled properly!!'
    ## DEFINE OUTPUTS ##
    outputs = data['outputs'][shuffler,np.newaxis] #Outputs need dim Nx1
    assert len(outputs)==len(inputs), 'Inputs and Outpus have NOT the same length'
    nr_samples = len(outputs)
    print('Nr. of samples is ', nr_samples)
    baseline_var = np.mean(data['var_outputs'])
    
    if test_set:
        # Devide data in training and test set 
        n_test = int(test_size*nr_samples)
        print('Size of TEST set is ',n_test)
        outputs_test = outputs[:n_test] 
        inputs_test = inputs[:n_test]
        data = np.concatenate((inputs_test,outputs_test),axis=1)
        
        #Save test set
        test_file = data_dir+'test_data_from_trainbatch'
        np.savez(test_file, data=data)
        print('Test set saved in \n'+test_file)
        
        outputs_train = outputs[n_test:] 
        inputs_train = inputs[n_test:]
    else:
        outputs_train = outputs
        inputs_train = inputs
        print('No TEST set partitioned')
        
    # Create Tensors to hold inputs and outputs, and wrap them in Variables.
    # From Numpy to Pytorch b = torch.from_numpy(a)
    # N is (mini-)batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    nr_samples = len(outputs_train) 
    n_val = int(nr_samples*val_size)

    nr_minibatches = int((nr_samples-n_val)/batch_size)
    n_val += nr_samples - (nr_minibatches*batch_size + n_val)
    assert isinstance(n_val,int), 'n_val is not integer!!'
    print('Size of VALIDATION set is ',n_val)
    print('Size of TRAINING set is ',nr_samples-n_val)
    
    ### Sanity Check ###
    assert nr_minibatches*batch_size + n_val == nr_samples, 'Data points not properly allocated!'
    if not outputs_train.shape[0] == inputs_train.shape[0]: raise NameError(
            'Input and Output Batch Sizes do not match!')
    
    ### Define Data Type for PyTorch ###
    if syst is 'cuda':
        print('Train with CUDA')
        dtype = torch.cuda.FloatTensor
        itype = torch.cuda.LongTensor
    else: 
        print('Train with CPU')
        dtype = torch.FloatTensor
        itype = torch.LongTensor
        
    x = torch.from_numpy(inputs_train).type(dtype)
    y = torch.from_numpy(outputs_train).type(dtype)
    
    x = Variable(x)
    y = Variable(y)
    
    ### Partition into training and validation sets ###
    permutation = torch.randperm(nr_samples).type(itype) # Permute indices 
    train_indices = permutation[n_val:]
    val_indices = permutation[:n_val]
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_val = x[val_indices]
    y_val = y[val_indices]

    return [(x_train,y_train),(x_val,y_val)], baseline_var

#%%
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
def PrepData(main_dir, list_dirs, threshold = 1000,nr_electrodes=8):
    '''Preprocess data to feed NN. It gets as arguments a string with path to main directory and
    a list of strings indicating directories with nparrays.npz containing 'data'. A kwarg threshold is given to crop data.
    The data arrays are merged into a single array, cropped given a threshold and the CVs are shifted & rescaled to be in [0,1]. This trafo is done with the [shift,range] of the data.
    NOTE:
        -The data is saved in main_dir+'/data4nn/ to a .npz file with keyes: inputs, cv_trafo, outputs, var_output,list_dirs
        -The convention of the inputs for the NN is [x_inp,CVs]. First dimension is the number of samples. Second dimension has length = # electrodes-1
        -The inputs are in range [-1,1] Volts, the CVs rescaled to [0,1] and the outputs in nA.
    '''
    data_list = []
    for dir_file in list_dirs:
        data_buff = np.load(main_dir+dir_file+'nparrays.npz')['data']
        data_list.append(data_buff)  
    data = np.concatenate(tuple(data_list))
    
    mean_output = np.mean(data[:,nr_electrodes-1:],axis=1) 
    cropping_mask = np.abs(mean_output)<threshold
    var_output = np.var(data[:,nr_electrodes-1:],axis=1)
    print('Max variance of output given inputs is ',np.max(var_output))
    
    cropped_data = data[cropping_mask,:]
    print('% of points cropped: ',(1-len(cropped_data)/len(data))*100)
    
    outputs = mean_output[cropping_mask]
    # Rescale to Volts
    inputs = cropped_data[:,:nr_electrodes-1]/1000 
    # Shift and rescale CVs to [0,1]
    shift = np.min(inputs[:,2:])
    cv_range = np.max(inputs[:,2:])-np.min(inputs[:,2:])
    inputs[:,2:] = (inputs[:,2:]-shift)/cv_range
    cv_trafo = [shift,cv_range]
#    # merge to data array
#    data = np.concatenate((inputs,outputs),axis=1)
#    assert data.shape[1]==nr_electrodes, 'Data has less than '+str(nr_electrodes)+' electrodes!'
    
    # save with timestamp
    now = datetime.datetime.now()
    dirName = main_dir+'data4nn/'    
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    
    save_to = dirName+'combined_'+now.strftime("%Y_%m_%d")
    print('Cleaned data saved to \n',save_to)
    np.savez(save_to, inputs = inputs, cv_trafo = cv_trafo,
         outputs = outputs, var_output = var_output, list_dirs = list_dirs)

#%%
####################################################################################################
####################################### MAIN #######################################################
####################################################################################################    
if __name__ == '__main__':

    if sys.argv[1] == '-dl':
        if len(sys.argv) > 2:
            data_dir = sys.argv[2]
        else:
            main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/'
            dir_data = '2018_08_07_164652_CP_FullSwipe/'
            data_dir = main_dir+dir_data    
        file_name = 'combi_data4nn.npz'
        print('Loading data...')
        data, baseline_var = DataLoader(data_dir, file_name, test_set = True)
    
    elif sys.argv[1] == '-pd': 
        print('Cleaning and preparing data...')
        main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/'
        list_dirs = ['25_07_2018_CP-full-search-77K/','2018_08_07_164652_CP_FullSwipe/']
        PrepData(main_dir, list_dirs, threshold = 3.57)