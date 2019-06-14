#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:40:12 2018

@author: hruiz

Defines the standard Neural Network class using PyTorch nn-package. It defines the model and inplements the training procedure. 
INPUT: ->data; a list of (input,output) pairs for training and validation [(x_train,y_train),(x_val,y_val)]. 
               The dimensions of x and y arrays must be (Samples,Dim) and they must be torch.FloatTensor
       ->depth: number of hidden layers
       ->width: can be list of number of nodes in each hidden layer
       ->kwarg: loss='MSE'; string name of loss function that defines the problem (only default implemented)
                activation='ReLU'; name of activation function defining the non-linearity in the network (only default implemented)
                betas=(0.9, 0.999) is a tuple with beta1 and beta2 for Adam
TO DO:
    - clean up the implementation of non linearity: it can be done by defining the non-lin at the object construction and so avoid the if-clauses
    - Apply BatchNorm
    - Implement a more flexible construction of the NN (with different widths for the hidden layers)
"""

import torch
import torch.nn as nn
import numpy as np 
#from matplotlib import pyplot as plt
import pdb

class staNNet(object):

    def __init__(self,*args,loss='MSE',activation='ReLU', dim_cv=5, BN=False):
        
        if len(args) == 3: #data,depth,width
           data,depth,width = args           
           self.info = {'activation':activation, 'loss':loss}
           for key, item in data[2].items():
               self.info[key] = item               
           self.x_train, self.y_train = data[0]
           self.y_train = self.y_train/self.info['amplification'].item()
           self.x_val, self.y_val = data[1]
           self.y_val = self.y_val/self.info['amplification'].item()                                     
           self.D_in = self.load_data(self.x_train[0:1]).size()[1]
           self.D_out = self.y_train.size()[1]
           self._BN = BN
           self.depth = depth
           self.width = width
           
           print(f'Meta-info: \n {list(self.info.keys())}')
           self.ttype = self.x_train.type()
           self._tests()
        
           ################### DEFINE MODEL ######################################
           self._contruct_model(loss)

           if isinstance(self.x_train.data,torch.FloatTensor): 
               self.itype = torch.LongTensor
           else:
               self.itype = torch.cuda.LongTensor
               self.model.cuda()
#               self.loss_fn.cuda() apparently this is not needed if both arguments are already on gpu
               print('Sent to GPU')           
        elif len(args)==1 and type(args[0]) is str:
            self._load_model(args[0])
        else:
            assert False, 'Arguments must be either 3 (data,depth,width) or a string to load the model!'      

                
    def _load_model(self,data_dir):
        print('Loading the model from '+data_dir)
        self.ttype = torch.torch.FloatTensor
        if torch.cuda.is_available():
            state_dic = torch.load(data_dir)
            self.ttype = torch.cuda.FloatTensor
        else:
            state_dic = torch.load(data_dir, map_location='cpu')
        if list(filter(lambda x: 'running_mean' in x,state_dic.keys())):
            print('BN active in loaded model')
            self._BN = True
        else:
            self._BN = False

 
        # move info key from state_dic to self
        if state_dic.get('info') is not None:
            self.info = state_dic['info']
            print(f'Model loaded with info dictionary containing: \n {self.info}')
            state_dic.pop('info')
        else:
            # for backwards compatibility with objects where information is stored directly in state_dic
            # TODO: this should be removed, because it assumes all parameters in network have either weight or bias in their name
            #       which might not always be the case in the future
            self.info = {}
            for key, item in list(state_dic.items()):
                # remove all keys in state_dic not containing bias or weight and store them as attributes of self
                if 'bias' not in key and 'weight' not in key:
                    self.info[key] = item
                    state_dic.pop(key)

        print('NN loaded with activation %s and loss %s' % (self.info['activation'], self.info['loss']))
        loss = self.info['loss']
        itms = list(state_dic.items())  
        layers = list(filter(lambda x: ('weight' in x[0]) and (len(x[1].shape)==2),itms))
        self.depth = len(layers)-2
        self.width = layers[0][1].shape[0]
        self.D_in = layers[0][1].shape[1]
        self.D_out = layers[-1][1].shape[0]

        self._contruct_model(loss)
        
        self.model.load_state_dict(state_dic)

        if isinstance(layers[-1][1],torch.FloatTensor): 
            self.itype = torch.LongTensor
        else: 
            self.itype = torch.cuda.LongTensor
            self.model.cuda()
#            self.loss_fn.cuda()    
        self.model.eval()
            
    def _contruct_model(self,loss):
        # Use the nn package to define our model and loss function.
#        self.model = nn.Sequential(
#            nn.Linear(self.D_in, self.width),
#            nn.ReLU(),
#            nn.Linear(self.width, self.D_out),
#        )
        
        self.l_in = nn.Linear(self.D_in, self.width)
        self.l_out = nn.Linear(self.width, self.D_out)
        self.l_hid = nn.Linear(self.width,self.width)

        if self._BN: 
            track_running_stats=False  
            print('BN tracking average: ',track_running_stats)
            self.bn_layer = nn.BatchNorm1d(self.width,track_running_stats=track_running_stats)
        
        activation = self.info['activation']
        if activation == 'tanh':
            activ_func = nn.Tanh()
            print('Activation is tanh')            
        elif activation == 'ReLU':
            activ_func = nn.ReLU()
            print('Activation is ReLU')
        elif activation is None:
            activ_func = None
        else:
            assert False, "Activation function ('%s') not recognized!" % activation
        

        if self._BN: 
            modules = [nn.BatchNorm1d(self.D_in,track_running_stats=track_running_stats),
                       self.l_in,activ_func]
        else:
            modules = [self.l_in,activ_func]
            
        for i in range(self.depth):
            if self._BN: 
                modules.append(self.bn_layer) 
                #BN before activation (like in paper) doesn't make much difference; 
                #before the linearity also not much difference vs non-BN model
            modules.append(self.l_hid)
            modules.append(activ_func)
        
        modules.append(self.l_out)
        modules = [x for x in modules if x !=None ]
        
        print('Model constructed with modules: \n',modules)
        self.model = nn.Sequential(*modules)
        print(f'Loss founction is defined to be {loss}')
        if loss == 'RMSE':
            self.a = torch.tensor([0.01900258860717661, 0.014385111570154395]).type(self.ttype)
            self.b = torch.tensor([0.21272562199413553, 0.0994027221336]).type(self.ttype)
        elif loss == 'MSE':
            self.loss_fn = nn.MSELoss()
        else:
            assert False, f'Loss function ERROR! {loss} is not recognized'
        
    def loss_fn(self, pred, targets):
        y = pred#targets
        sign = torch.sign(y)
        ay = (sign-1)/2 * (self.a[0] * torch.abs(y)) + (sign+1)/2 * (self.a[1] * torch.abs(y))
        b = (sign-1)/2 * self.b[0] + (sign+1)/2 * self.b[1]
        C = ((pred - targets) ** 2) / (ay**2 + b**2) #+ pred**2
        r = torch.mean(C)
#        r = torch.mean(torch.log(C))
        #pdb.set_trace()
        return r
    
    def train_nn(self,learning_rate,nr_epochs,batch_size,
                 betas = (0.9, 0.999),
                 data = None, seed=False):

        """TO DO: 
            check if x_train, x_val and y_train and y_val are defined, if not, raise an error asking to define
        """
        
        if seed:
            torch.manual_seed(22)
            print('The torch RNG is seeded!')
        if not isinstance(data,type(None)):
            self.x_train, self.y_train = data[0]
            self.x_val, self.y_val = data[1]
            #Check if dimensions match
            assert self.D_in == self.x_train.size()[1], f'Dimensions do not match: D_in is {self.D_in} while input has dimension {self.x_train.size()[1]}'
            assert self.D_out == self.y_train.size()[1], f'Dimensions do not match: D_out is {self.D_out} while y has dimension {self.y_train.size()[1]}'

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas) # OR SGD?!
        print('Prediction using ADAM optimizer')
        self.L_val = np.zeros((nr_epochs,))
        self.L_train = np.zeros((nr_epochs,))
        for epoch in range(nr_epochs):
    
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
            get_indices = torch.randperm(self.x_train.size()[0]).type(self.itype)[:10000]
            x = self.load_data(self.x_train[get_indices])
            y = self.model(x) *self.info['amplification'].item()
            y_subset = self.y_train[get_indices] * self.info['amplification'].item()
            loss = self.loss_fn(y,y_subset).item()
            self.L_train[epoch] = loss
            
            #Evaluate Validation error
            x_val = self.load_data(self.x_val)
            y = self.model(x_val) * self.info['amplification'].item()
            loss = self.loss_fn(y, self.y_val * self.info['amplification'].item()).item() 
            self.L_val[epoch] = loss
            
            print('Epoch:', epoch, 'Val. Error:', self.L_val[epoch],
                  'Training Error:', self.L_train[epoch])
            self.model.train()
        self.info['L_train'] = self.L_train
        self.info['L_val'] = self.L_val
        print('Finished Training')
#        plt.figure()
#        plt.plot(np.arange(nr_epochs),self.L_val)
#        plt.show()
        
    def _tests(self):
        if not self.x_train.size()[0] == self.y_train.size()[0]: 
            raise NameError('Input and Output Batch Sizes do not match!')
    
    def save_model(self, path):
        """
        Saves the model in given path, all other attributes are saved under the 'info' key as a new dictionary.
        """
        self.model.eval()
        state_dic = self.model.state_dict()
        state_dic['info'] = self.info
        torch.save(state_dic,path)

    def load_data(self, data):
        """
        Loads data that will be fed into the NN.
        """
        return data

    def outputs(self,inputs,grad=False):
        data = self.load_data(inputs)
        if grad:
          return self.model(data) * self.info['amplification'].item()
        else:
          return self.model(data).data.cpu().numpy()[:,0] * self.info['amplification'].item()
    
if __name__ == '__main__':
    #%%
    ###############################################################################
    ########################### LOAD DATA  ########################################
    ###############################################################################
    
    from SkyNEt.modules.Nets.DataHandler import DataLoader as dl
    main_dir = r'../../test/NN_test/'
    file_name = 'data_for_training.npz'
    data = dl(main_dir+r'data4nn/16_04_2019/', file_name, steps=3)
    
    #%%
    ###############################################################################
    ############################ DEFINE NN and RUN ################################
    ###############################################################################
    depth = 5
    width = 90
    learning_rate,nr_epochs,batch_size = 3e-4, 5, 64*32
    net = staNNet(data,depth,width)
    net.train_nn(learning_rate,nr_epochs,batch_size)    
    #%%
    ###############################################################################
    ############################## SAVE NN ########################################
    ###############################################################################
    path = main_dir+f'TESTING_staNNet.pt'
    net.save_model(path)
    #Then later: net = staNNet(path)
