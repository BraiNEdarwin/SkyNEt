#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:18:34 2019
Trains a neural network given data.
---------------
Arguments
data : List containing 2 tuples; the first with a training set (inputs,targets),
        the second with validation data. Both the inputs and targets must be 
        torch.Tensors (shape: nr_samplesXinput_dim, nr_samplesXoutput_dim).
network : The network to be trained
conf_dict : Configuration dictionary with hyper parameters for training
save_dir (kwarg, str)  : Path to save the results
---------------
Returns:
network (torch.nn.Module) : trained network
costs (np.array)    : array with the costs (training,validation) per epoch

Notes:
    1) The dopantNet is composed by a surrogate model of a dopant network device
    and bias learnable parameters that serve as control inputs to tune the 
    device for desired functionality. If you have this use case, you can get the
    control voltage parameters via network.parameters():
        params = [p.clone().detach() for p in network.parameters()]
        control_voltages = params[0]
    2) For training the surrogate model, the outputs must be scaled by the 
    amplification. Hence, the output of the model and the errors are  NOT in nA.
    To get the errors in nA, scale by the amplification**2.
    The dopant network already outputs the prediction in nA. To get the output 
    of the surrogate model in nA, use the method .outputs(inputs).

@author: hruiz
"""

from SkyNEt.config.acceleration import Accelerator
import torch

def trainer(data, network, loss_fn=torch.nn.MSELoss(), learning_rate=1e-4,
            nr_epochs=3000, batch_size=128, save_dir='../../test/NN_test/',
            save_interval=10, **kwargs):
    
    # set configurations
    if "seed" in kwargs.keys():
        torch.manual_seed(seed)
        print(f'The torch RNG is seeded with {seed}!')

    if "betas" in kwargs.keys():
        optimizer = torch.optim.Adam(network.parameters(), 
                                 lr=learning_rate, 
                                 betas=kwargs["betas"])
        print("Set betas to values: ",{kwargs["betas"]})
    else:
        optimizer = torch.optim.Adam(network.parameters(), 
                                 lr=learning_rate)    
    print('Prediction using ADAM optimizer')
    
    # Define variables
    x_train, y_train = data[0]
    x_val, y_val = data[1]
    costs = np.zeros((nr_epochs,2)) # training and validation costs per epoch
    samples = len(x_val)
    
    for epoch in range(nr_epochs):
        
        network.train()
        permutation = torch.randperm(x_train.size()[0]) # Permute indices 
        nr_minibatches = 0

        for i in range(0,len(permutation),batch_size):
            
            # Get prediction
            indices = permutation[i:i+batch_size]
            x = x_train[indices]
            y_pred = network(x)
            # GD step
            if 'regularizer' in dir(network):
                loss = loss_fn(y_pred, y_train[indices]) + network.regularizer()
            else:
                loss = loss_fn(y_pred, y_train[indices])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            nr_minibatches += 1
                         
        network.eval()       
        # Evaluate training error
        get_indices = torch.randperm(len(x_train))[:samples]
        x = x_train[get_indices]
        prediction = network(x)
        target = y_train[get_indices] 
        costs[epoch,0] = loss_fn(prediction, target).item()
        #Evaluate Validation error
        prediction = network(x_val)
        costs[epoch,1] = loss_fn(prediction, y_val).item()
        
        if save_dir and epoch % save_interval == 0:
            save_model(network, save_dir+f'checkpoint_epoch{epoch}.pt')
        
        if epoch % 10 == 0:
            print('Epoch:', epoch,
                  'Val. Error:', costs[epoch,1],
                  'Training Error:', costs[epoch,0])

    return costs

def save_model(model,path):
    """
    Saves the model in given path, all other attributes are saved under
    the 'info' key as a new dictionary.
    """
    model.eval()
    state_dic = model.state_dict()
    if 'info' in dir(model):
        state_dic['info'] = model.info
    torch.save(state_dic,path)

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import numpy as np
    from dopantNet import dopantNet
    
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
    in_list = [0,3]
    x = 0.5*np.random.randn(10,len(in_list)) 
    inp_train = torch.Tensor(x).to(device)
    t_train = torch.Tensor(5.*np.ones(10)).to(device)#torch.Tensor(np.random.randn(10,1)).to(device)
    x = 0.5*np.random.randn(4,len(in_list)) 
    inp_val = torch.Tensor(x).to(device)
    t_val = torch.Tensor(5.*np.ones(4)).to(device)#torch.Tensor(np.random.randn(4,1)).to(device)
    data = [(inp_train,t_train),(inp_val,t_val)]
    
    node = dopantNet(in_list)
    
    loss_array = []
    
    start_params = [p.clone().detach() for p in node.parameters()]
    
    costs = trainer(data, node, batch_size=len(t_train),
                                  learning_rate=3e-5,
                                  save_dir='../../test/NN_test/',
                                  save_interval=np.inf)
 
    out = node(inp_val)
    end_params = [p.clone().detach() for p in node.parameters()]
    print("CV params at the beginning: \n ",start_params[0])
    print("CV params at the end: \n",end_params[0])
    print("Example params at the beginning: \n",start_params[-1][:8])
    print("Example params at the end: \n",end_params[-1][:8])
    print("Length of elements in node.parameters(): \n",[len(p) for p in end_params])
    print("and their shape: \n",[p.shape for p in end_params])
    print(f'OUTPUT: {out.data.cpu()}')
    
    plt.figure()
    plt.plot(costs)
    plt.title("Loss per epoch")
    plt.legend(["Training","Validation"])
    plt.show()
