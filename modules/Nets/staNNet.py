#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:40:12 2018

@author: hruiz

Defines the standard Neural Network class using PyTorch nn-package. It defines 
the model and inplements the training procedure. 
INPUT: -data: a list of (input,output) pairs for training and validation 
            [(x_train,y_train),(x_val,y_val)] each with shape (Samples, Dim)
       -hidden_sizes: list with sizes of hidden layers
       -activation=nn.ReLU(): activation as nn-module

TODOs:
    - Better handling of device availability (use accelerator.py)
    - Implement more flexible training? (eg. choice of optimizer?)
    - amplitude conversion must be done in DataHandler.DataLoader
    - documentation
"""

import torch
import torch.nn as nn
import numpy as np 
#import pdb

class staNNet(nn.Module):

    def __init__(self,*args, activation=nn.ReLU(), device='cpu', conversion=100):
        super(staNNet, self).__init__()
        ### Define Data Type for PyTorch ###
        #Need to do it like this instead of with torch.cuda.is_available()
        # because DELL laptop has cuda but it is not supported (too old)
        if device is 'cuda':
            print('Train with GPU')
            self.dtype = torch.cuda.FloatTensor
        else: 
            print('Train with CPU')
            self.dtype = torch.FloatTensor
            
        if len(args)==1 and type(args[0]) is str:
            self._load_model(args[0])
            
        elif len(args) == 2: #data,hidden_sizes
            
           data, self.hidden_sizes = args
           
           #Define info dict
           self.info = {'activation':activation, 'conversion':conversion}
           for key, item in data[2].items():
               self.info[key] = item 
           #Prepare data
           self._data(data)
           
           self.D_in = self.load_data(self.x_train).size()[1]
           self.D_out = self.y_train.size()[1]
           
           self.info['D_in'] = self.D_in
           self.info['D_out'] = self.D_out
           self.info['hidden_sizes'] = self.hidden_sizes
           print(f'Meta-info: \n {self.info.keys()}')
           self.ttype = self.x_train.type()
           self._tests()
        
           ################### DEFINE MODEL ######################################
           self._contruct_model()

           if isinstance(self.x_train.data,torch.FloatTensor): 
               self.itype = torch.LongTensor
           else:
               self.itype = torch.cuda.LongTensor
               self.model.cuda()
               print('Sent to GPU')           
        
        else:
            assert False, \
            'Arguments must be either args=(data,hidden_sizes) or args=string to load the model!'      
    
    def train_nn(self,learning_rate,nr_epochs,batch_size,
                 betas = (0.9, 0.999), save_dir=None, save_interval=10,
                 data = None, seed=None):
        
        data_available = [key in net.__dict__.keys() for key in 
                          ['x_train', 'x_val', 'y_train', 'y_val']]
        assert all(data_available), 'Data missing! Please define as kwarg data.'
        
        if seed:
            torch.manual_seed(seed)
            print(f'The torch RNG is seeded with {seed}!')
        if not isinstance(data,type(None)):
            self._data(data)
            self._tests()
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas) # OR SGD?!
        print('Prediction using ADAM optimizer')
        self.L_val = np.zeros((nr_epochs,))
        self.L_train = np.zeros((nr_epochs,))
        for epoch in range(nr_epochs):
            self.model.train()
            permutation = torch.randperm(self.x_train.size()[0]).type(self.itype) # Permute indices 
            nr_minibatches = 0

            for i in range(0,len(permutation),batch_size):
                
                # Forward pass: compute predicted y by passing x to the model.
                indices = permutation[i:i+batch_size]
                x_train = self.load_data(self.x_train[indices])
                y_pred = self.model(x_train)
                
                # Compute and print loss.
                loss_training = self.loss_fn(y_pred, self.y_train[indices])

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()
                
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss_training.backward()
                
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()
                nr_minibatches += 1
                             
            self.model.eval()
            
            # Evaluate training error
            self.L_train[epoch] = self._errors(self.x_train,self.y_train)
            #Evaluate Validation error
            self.L_val[epoch] = self._errors(self.x_val,self.y_val)
            
            if save_dir and epoch % save_interval == 0:
                self.save_model(save_dir+f'/checkpoint_epoch{epoch}.pt')
            
            print('Epoch:', epoch, 'Val. Error:', self.L_val[epoch],
                  'Training Error:', self.L_train[epoch])
        self.info['L_train'] = self.L_train
        self.info['L_val'] = self.L_val
        print('Finished Training')

#%% ###########################################################################
    ################## Load and construct methods #############################
    ###########################################################################
    def _load_model(self,data_dir):
        print('Loading the model from '+data_dir)
        self.ttype = torch.torch.FloatTensor
        if torch.cuda.is_available():
            state_dic = torch.load(data_dir)
            self.ttype = torch.cuda.FloatTensor
        else:
            state_dic = torch.load(data_dir, map_location='cpu')
 
        # move info key from state_dic to self
        self.info = state_dic['info']
        print(f'Meta-info: \n {self.info.keys()}')
        state_dic.pop('info')

        self.D_in = self.info['D_in']
        self.D_out = self.info['D_out']
        self.hidden_sizes = self.info['hidden_sizes']

        self._contruct_model()
        
        self.model.load_state_dict(state_dic)

        if isinstance(list(net.model.parameters())[-1],torch.FloatTensor): 
            self.itype = torch.LongTensor
        else: 
            self.itype = torch.cuda.LongTensor
            self.model.cuda()   
        self.model.eval()
            
    def _contruct_model(self):

        self.l_in = nn.Linear(self.D_in, self.hidden_sizes[0])
        self.l_out = nn.Linear(self.hidden_sizes[-1], self.D_out)
        activ_func = self.info['activation']
        
        modules = [self.l_in,activ_func]
        
        h_layers = zip(self.hidden_sizes[:-1],self.hidden_sizes[1:])
        for h1,h2 in h_layers:
            hidden = nn.Linear(h1,h2)
            modules.append(hidden)
            modules.append(activ_func)
        
        modules.append(self.l_out)
        self.model = nn.Sequential(*modules)
        print('Model constructed with modules: \n' ,modules)
        
        self.loss_fn = nn.MSELoss()
        print(f'Loss is {self.loss_fn}')
        
#%% ###########################################################################
    ########################### Helper methods ################################
    ###########################################################################      

    def outputs(self,inputs,grad=False):
        data = self.load_data(inputs) 
        if grad: #why this?
          return self.model(data) * self.info['conversion']
        else:
          return self.model(data).data.cpu().numpy()[:,0] * self.info['conversion']
      
    def save_model(self, path):
        """
        Saves the model in given path, all other attributes are saved under
        the 'info' key as a new dictionary.
        """
        self.model.eval()
        state_dic = self.model.state_dict()
        state_dic['info'] = self.info
        torch.save(state_dic,path)

    def load_data(self, data): #why this?
        """
        Loads data that will be fed into the NN.
        """
        return data

    def _errors(self, x, y):
        get_indices = torch.randperm(len(x)).type(self.itype)[:len(self.x_val)]
        x = self.load_data(x[get_indices])
        prediction = self.model(x) * self.info['conversion']
        target = y[get_indices] * self.info['conversion']
        return self.loss_fn(prediction, target).item()
    
    def _data(self,data):
        self.x_train = self._to_torch(data[0][0])
        self.y_train = data[0][1]
        self.y_train = self.y_train/self.info['conversion']
        self.y_train = self._to_torch(self.y_train)
        self.x_val = self._to_torch(data[1][0])
        self.y_val = data[1][1]
        self.y_val = self.y_val/self.info['conversion']
        self.y_val = self._to_torch(self.y_val)

    def _tests(self):
        if not self.x_train.size()[0] == self.y_train.size()[0]:
            raise ValueError('Input and Output Batch Sizes do not match!')
        #Check if dimensions match
        if not self.D_in == self.x_train.size()[1]:
            raise ValueError(f'Dimensions do not match: D_in is {self.D_in}; input data has dimension {self.x_train.size()[1]}')
        if not self.D_out == self.y_train.size()[1]:
            raise ValueError(f'Dimensions do not match: D_out is {self.D_out}; output data has dimension {self.y_train.size()[1]}')

    def _to_torch(self,x):
        return torch.from_numpy(x).type(self.dtype)
        
if __name__ == '__main__':
    #%%
    ###############################################################################
    ########################### LOAD DATA  ########################################
    ###############################################################################
    
    from SkyNEt.modules.Nets.DataHandler import DataLoader as dl
    main_dir = r'../../test/NN_test/data4nn/Data_for_testing/' #r'D:\git_repos\SkyNEt\test\NN_test\data4nn\'
    file_name = 'data_for_training.npz'
    data = dl(main_dir, file_name, steps=3)
    
    #%%
    ###############################################################################
    ############################ DEFINE NN and RUN ################################
    ###############################################################################
    hidden_sizes = [128,64,32]
    learning_rate,nr_epochs,batch_size = 3e-4, 2, 64*32
    print('--------------- Test: construction and training ------------------')
    net = staNNet(data,hidden_sizes,device = 'cpu')
    net.train_nn(learning_rate,nr_epochs,batch_size, 
                 save_dir=main_dir, save_interval=nr_epochs-1)    
    #%%
    ###############################################################################
    ############################## SAVE NN ########################################
    ###############################################################################
    print('------------------ Test: saving and loading ----------------------')
    path = main_dir+f'TESTING_staNNet.pt'
    net.save_model(path)
    #Then later: net = staNNet(path)
    net2 = staNNet(path)
    print('--------------- Test: test cases ------------------')
    net2.train_nn(learning_rate, nr_epochs, batch_size, save_dir = main_dir, 
                 data = [[np.ones((23,7)), np.zeros((23,3))],
                        [np.ones(23), np.zeros((23,1))]])
