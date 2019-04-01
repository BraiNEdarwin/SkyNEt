"""
The function train() searches for the control voltages which matches the web output to the given target data.

@author: ljknoll

Arguments:
    train_data:
        shape (data size, # of inputs * # of vertices)
    target data:
        depends on loss function, but for MSEloss, target needs to be same shape as output of web
        shape f(data size, # of output networks)
    batch_size:     
        weight updates are done in minibatches, sets the size of one minibatch
    max_epochs (optional):
        maximum number of epochs one training can last
    verbose (optional):
        prints error at each iteration and other info
    beta (optional):
        scaling parameter for relu regularization outside [0,1] for cv
    optimizer (optional):
        pass a custom optimizer, note that extra keyword arguments are passed to this optimizer, default Adam
    loss_fn (optional):
        pass a custom loss function (without regularization), default MSE
    stop_fn (optional):
        function which determines when training should stop. With arguments:
            epoch (int): current epoch number
            error_list (list): list of previous errors
            best_error (float): best error value so far
        default is no function and training is done for max_epochs
    **kwargs (optional):
        any keyword arguments remaining are passed to the optimizer function

returns:
    error_list      (list) list of error values during training 
    best_params     (dict) web parameters which gave smallest error
"""

import torch

def train(self, 
          train_data,
          target_data,
          batch_size,
          max_epochs=100,
          verbose=False,
          beta=0.1,
          optimizer=None,
          loss_fn=None,
          stop_fn=None,
          **kwargs):
    train_data, target_data = self.check_cuda(train_data, target_data)
    
    self.check_graph(verbose=verbose)
    
    if optimizer is None:
        if verbose:
            print("INFO: Using Adam with: ", kwargs)
        optimizer = self.optimizer(self._params, **kwargs)
    else:
        if verbose:
            print("INFO: Using custom optimizer with, ", kwargs)
        optimizer = optimizer(self._params, **kwargs)
    
    if loss_fn is not None:
        del self.loss_fn
        self.loss_fn = loss_fn
    
    if stop_fn is None:
        if verbose:
            print("INFO: Not using stopping criterium")
        stop_fn = lambda *args: False
    
    error_list = []
    best_error = 1e5
    best_params = self.get_parameters()
    all_params = []
    for epoch in range(max_epochs):
        # train on complete data set in batches
        permutation = torch.randperm(len(train_data))
        for i in range(0,len(permutation), batch_size):
            indices = permutation[i:i+batch_size]
            y_pred = self.forward(train_data[indices])
            error = self.error_fn(y_pred, target_data[indices], beta)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
        
        # after training, calculate error of complete data set
        predictions = self.forward(train_data)
        error_value = self.error_fn(predictions, target_data, beta)
        
        all_params.append(self.get_parameters())
        
        if torch.isnan(error_value):
            print("WARN: Error is nan, stopping training.")
            return [10e5], best_params, all_params

        error_value = error_value.item()
        error_list.append(error_value)
        if verbose:
            print("INFO: error at epoch %s: %s" % (epoch, error_value))
        
        # if error improved, update best params and error
        if error_value < best_error:
            best_error = error_value
            best_params = self.get_parameters()
        
        # stopping criterium
        if stop_fn(epoch, error_list, best_error):
            break
    return error_list, best_params, all_params

def session_train(self, *args, nr_sessions=5, **kwargs):
    """
    Initialize random and train for nr_sessions, returns only the results of Train() with the lowest error
    """
    best_errors, error_list, best_params, all_params = [[],]*4
    for session in range(nr_sessions):
        self.reset_parameters('rand')
        temp_error_list, temp_best_params, temp_all_params = self.train(*args, **kwargs)
        best_error = min(temp_error_list)
        best_errors.append(best_error)
        error_list.append(temp_error_list)
        best_params.append(temp_best_params)
        all_params.append(temp_all_params)
        print("INFO: Session %i/%i, best error after %i iterations: %f" % (session+1, nr_sessions, len(temp_error_list), best_error))
    index_best = min(enumerate(best_errors), key=lambda x:x[1])[0]
    return error_list[index_best], best_params[index_best], all_params[index_best]