#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:46:46 2018
Class to create a web of connected neural networks.
The graph consists of vertices (networks) and arcs (connections between vertices).

Vertices:
Structure of self.graph (dictionary):
    keys: names of vertex, values: info of vertex
    vertex info (dictionary):
        'network'   : neural network object
        'input'     : input data which is added to control voltages (torch tensor)
        'isoutput'  : wether vertex is output vertex (boolean)
        'output'    : output data of vertex calculated by forward_vertex (torch tensor)

Arcs:
Each arc consist of a source and sink, where the data flows from source to sink.
Structure of self.arcs (dictionary):
    keys: tuple: (sink_name, sink_gate)
    values: source_name
    
See webNN_template.py for example use.

Shape of input data:
    (batch size, input_dimension * nr_output_networks)
Shape of target data:
    depends on loss function, but for MSEloss, target needs to be same shape as output of web:
    (batch size, nr_output_networks)


@author: ljknoll
"""

import torch
from collections import OrderedDict as odict

# imports for GA
import numpy as np
import SkyNEt.modules.Evolution as Evolution

# imports for plotting graph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

class webNNet(torch.nn.Module):
    def __init__(self):
        super(webNNet, self).__init__()
        self.graph = odict() # vertices of graph
        self.arcs = odict() # arcs of graph
        self.output_data = None # output data of graph
        self.nr_output_vertices = 0 # number of networks whose output data is used
        
        # setting defaults
        self.cuda = 'cpu'
        self.default_param = 0.8 # value to initialize control voltage parameters
        self.loss = torch.nn.MSELoss() # loss function (besides regularization)
        self.optimizer = torch.optim.Adam # optimizer function
        self.transfer = torch.sigmoid # function which maps output to input [0,1]
        
        # initialize empty bias and scale list, 
        # these store bias and scale for each output vertex
        self.register_parameter('bias', torch.nn.Parameter(torch.tensor([])))
        self.register_parameter('scale', torch.nn.Parameter(torch.tensor([])))
    
    ############################################################
    ### ---------------- Genetic Algorithm ----------------- ###
    ############################################################
    def prepare_config_obj(self, cf, loss_fn):
        """ prepares config object for GA use with the web class """
        # total number of genes: 5 control voltages for each vertex, one less for each arc
        cf.genes = len(self.graph)*5 - len(self.arcs)
        # number of genomes in each of the 5 partitions
        cf.partition = [5, cf.genes, 5, 5, cf.genes]
        # total number of genomes
        cf.genomes = sum(cf.partition)
        
        # set fitness functino of cf to default loss
        # loss function must return the error, not the fitness!
        if loss_fn is None:
            cf.Fitness = self.loss
        else:
            cf.Fitness = loss_fn
    
    def set_dict_indices_from_pool(self, pool):
        """ For each registered parameter in this web object, 
        store the indices of the parameters which are used.
        
        Example:
            when there is an arc from 'A' to 'B' at gate 5, 
            the GA does not need to update this control voltage as it is overwritten by the arc data.
            Therefore, these parameters are not included in the pool.
            The indices for A will then be [2,3,4,6] and gate 5 will not be passed to evolution
        """
        parameters = self.get_parameters()
        indices = {}
        # loop through parameters of network
        for par_name, par_params in parameters.items():
            indices[par_name] = []
            # check if parameter is a vertex or bias/scale
            if 'bias' in par_name or 'scale' in par_name:
                # TODO: implement checking for including bias and scale in pool,
                # for now don't include them
                indices[par_name] = []
            else:
                # loop through control voltages of vertex 'par_name'
                for j in range(len(par_params)):
                    # check if current control voltage is in use by an arc, thus no value is needed
                    # TODO: replace +2 by actual number of datainputs of network
                    if (par_name, j+2) not in self.arcs.keys():
                        indices[par_name].append(j)
        
        self.indices = indices

    def set_parameters_from_pool(self, pool):
        """ Uses the indices set by set_dict_indices_from_pool() to 
        update parameters with values from pool """
        pool_iter = iter(pool)
        with torch.no_grad(): # do not track gradients
            for par_name, indices in self.indices.items():
                # replace parameter par_name values with values from pool
                replacement = [next(pool_iter) for _ in range(len(indices))]
                getattr(self, par_name)[indices] = torch.tensor(replacement, dtype=torch.float32, device=self.cuda)
    
    def trainGA(self, 
                train_data,
                target_data,
                cf,
                loss_fn = None,
                verbose = False):
        """ Train web with Genetic Algorithm """
        
        train_data, target_data = self.check_cuda(train_data, target_data)
        
        self.check_graph()
        
        # prepare config object with information of web, # of genes, partitions, genomes, etc
        self.prepare_config_obj(cf, loss_fn)
        # initialize genepool
        genepool = Evolution.GenePool(cf)
        # stores which indices of self.parameters to change during training
        self.set_dict_indices_from_pool(genepool.pool[0])
        
        # np arrays to save genePools, outputs and fitness
        geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
        outputArray = np.zeros((cf.generations, cf.genomes, train_data.shape[0], self.nr_output_vertices))
        fitnessArray = np.zeros((cf.generations, cf.genomes))
        
        # Temporary arrays, overwritten each generation
        fitnessTemp = np.zeros((cf.genomes, cf.fitnessavg))
        outputAvg = torch.zeros(cf.fitnessavg, train_data.shape[0], self.nr_output_vertices, device=self.cuda)
        outputTemp = torch.zeros(cf.genomes, train_data.shape[0], self.nr_output_vertices, device=self.cuda)

        for i in range(cf.generations):
            for j in range(cf.genomes):
                for avgIndex in range(cf.fitnessavg):
                    # update parameters of each network
                    self.set_parameters_from_pool(genepool.pool[j])
                    self.forward(train_data)
                    outputAvg[avgIndex] = self.get_output()
                    # use negative loss as fitness for genepool.NextGen()
                    fitnessTemp[j, avgIndex] = -cf.Fitness(outputAvg[avgIndex], target_data).item()
                    
                outputTemp[j] = outputAvg[np.argmin(fitnessTemp[j])]
            
            genepool.fitness = fitnessTemp.min(1)  # Save best fitness

            # Save generation data
            geneArray[i, :, :] = genepool.pool
            outputArray[i, :, :] = outputTemp
            fitnessArray[i, :] = genepool.fitness

            if verbose:
                print("Generation nr. " + str(i + 1) + " completed")
                print("Best fitness: " + str(-max(genepool.fitness)))
            
            genepool.NextGen()

        return geneArray, outputArray, fitnessArray
    
    
    ############################################################
    ### ----------------- Gradient Descent ----------------- ###
    ############################################################
    def train(self, 
              train_data,
              target_data,
              batch_size,
              max_epochs=100,
              verbose=False,
              beta=0.1,
              optimizer=None,
              loss_fn=None,
              bias=False,
              scale=False,
              reg_scale=False,
              stop_func=None,
              **kwargs):
        """verbose: prints error at each iteration
        beta: scaling parameter for relu regularization outside [0,1] for cv
        maxiterations: the number of iterations after which training stops
        """
        train_data, target_data = self.check_cuda(train_data, target_data)
        
        self.check_graph()
        
        if stop_func is None:
            if verbose:
                print("INFO: Not using stopping criterium")
            stop_func = lambda *args: False
        
        if optimizer is None:
            if verbose:
                print("INFO: Using Adam with: ", kwargs)
            optimizer = self.optimizer(self.parameters(), **kwargs)
        else:
            if verbose:
                print("INFO: Using custom optimizer with, ", kwargs)
            optimizer = optimizer(self.parameters(), **kwargs)
        
        if not scale:
            reg_scale = False
        
        error_list = []
        best_error = 1e5
        best_params = self.get_parameters()
        for epoch in range(max_epochs):
            # train on complete data set in batches
            permutation = torch.randperm(len(train_data))
            for i in range(0,len(permutation), batch_size):
                indices = permutation[i:i+batch_size]
                y_pred = self.forward(train_data[indices], bias=bias, scale=scale)
                error = self.error_fn(y_pred, target_data[indices], beta, loss_fn, reg_scale=reg_scale)
                optimizer.zero_grad()
                error.backward()
                optimizer.step()
            
            # after training, calculate error of complete data set
            predictions = self.forward(train_data, bias=bias, scale=scale)
            error_value = self.error_fn(predictions, target_data, beta, loss_fn, reg_scale=reg_scale)
            if torch.isnan(error_value):
                print("WARN: Error is nan, stopping training.")
                return [0], best_params

            error_value = error_value.item()
            error_list.append(error_value)
            if verbose:
                print("INFO: error at epoch %s: %s" % (epoch, error_value))
            
            # if error improved, update best params and error
            if error_value < best_error:
                best_error = error_value
                best_params = self.get_parameters()
            
            # stopping criterium
            if stop_func(epoch, error_list, best_error):
                break
        return error_list, best_params
    
    def session_train(self, *args, nr_sessions=10, **kwargs):
        best_errors, error_list, best_params = [], [], []
        for session in range(nr_sessions):
            self.reset_parameters('rand')
            temp_error_list, temp_best_params = self.train(*args, **kwargs)
            best_error = min(temp_error_list)
            best_errors.append(best_error)
            error_list.append(temp_error_list)
            best_params.append(temp_best_params)
            print("Best error of session %i/%i: %f" % (session, nr_sessions, best_error))
        return error_list, best_params

    ##################################################
    ### ---------------- General ----------------- ###
    ##################################################
    def add_vertex(self, network, name, output=False, number_cv = 5):
        """Adds neural network as a vertex to the graph.
        Args:
            network: vertex (object with attribute D_in and method output())
            name: key of graph dictionary in which vertex is created (str)
            output: wheter of not this vertex' output is output of complete graph (boolean)
            number_cv: nr of control voltages, the parameters which are trained (can be unused when connected to arc)
        """
        
        assert not hasattr(self, name), "Name %s already in use, choose other name for vertex!" % name
        assert 'bias' not in name, "Name should not contain 'bias'"
        assert 'scale' not in name, "Name should not contain 'scale'"
        cv = self.default_param*torch.ones(number_cv)
        self.register_parameter(name, torch.nn.Parameter(cv))
        
        self.graph[name] = {  'network':network,
                              'isoutput':output}
        if output:
            self.bias = torch.nn.Parameter(torch.cat((self.bias, torch.tensor([0.0]))))
            self.scale = torch.nn.Parameter(torch.cat((self.scale, torch.tensor([0.0]))))
            self.nr_output_vertices  += 1
    
    def add_arc(self, source_name, sink_name, sink_gate):
        """Adds arc to graph, which connects an output of one vertex to the input of another.
        Args:
            source_name: name of vertex of source of data connection
            sink_name: name of vertex to which data will flow
            sink_gate: index of gate of sink vertex
        """
        # check if gate is already in use, combination of sink gate and sink name must be unique!
        assert (sink_name, sink_gate) not in self.arcs, "Sink gate (%s, %s), already in use!" % (sink_name, sink_gate)
        self.arcs[(sink_name, sink_gate)] = source_name
        
    def forward(self, x, bias=False, scale=False, verbose=False):
        """Evaluates the graph, returns output (torch.tensor)
        Start at network which is the output of the whole graph, 
        then recursively step through graph vertex by vertex
        """
        
        # reset output of the graph
        self.clear_output()
        
        # define input data for all networks
        self.set_input_data(x, verbose = verbose)
        
        tuple_scaled_data = ()
        for key,value in self.graph.items():
            # start at vertex which is defined as output
            if value['isoutput']:
                # recursively evaluate vertices
                self.forward_vertex(key)
                return_value = value['output']
                tuple_scaled_data += (return_value,)
        returned_data = torch.cat(tuple_scaled_data, dim=1)
        self.output_data = returned_data.data
        if scale:
            returned_data *= self.scale+1.
        if bias:
            returned_data += self.bias
        return returned_data

    def forward_vertex(self, vertex):
        """Calculates output of vertex"""
        v = self.graph[vertex]
        
        # skip if vertex is already evaluated
        if 'output' not in v:
            # control voltages, repeated to match batch size of train_data
            cv_data = getattr(self, vertex).repeat(len(v['train_data']), 1)
            # concatenate input with control voltage data
            data = torch.cat((v['train_data'], cv_data), dim=1)
            
            # check dependencies of vertex by looping through all arcs
            for sink,source_name in self.arcs.items():
                sink_name, sink_gate = sink
                # if current vertex is also vertex that the arc is pointing to,
                # that means we need the data through that arc
                if sink_name == vertex:
                    # first evaluate vertices on which this input depends
                    self.forward_vertex(source_name)
                    # insert data from arc into control voltage parameters
                    data[:, sink_gate] = self.transfer(self.graph[source_name]['output'][:,0])
            
            # feed through network
            v['output'] = v['network'].model(data)
    
    def error_fn(self, y_pred, y, beta, loss=None, reg_scale = 0.0):
        """Error function: loss function with added regularization"""
        # default loss function: MSE
        if loss is None:
            loss = self.loss
        
        # calculate regularization
        reg_loss = 0
        for name, x in self.named_parameters():
            if 'bias' in name:
                pass
            elif 'scale' in name:
                reg_loss += torch.sum(torch.abs(x))*reg_scale
            else:
                reg_loss += torch.sum(torch.relu(-x) + torch.relu(x-1.0))
        return loss(y_pred, y) + beta*reg_loss
    
    def set_input_data(self, x, verbose=False):
        """Store training data for each network, assumes the torch tensor has the same ordering as in the dictionary"""
        # assumes each vertex has same number of parameters (could change in future)
        dim = x.shape[1]
        # count number of parameters excluding biases and scales (which are first to parameters)
        nr_of_params = sum([len(i) for i in self.get_parameters().values()][2:])        
        # if input data is provided for each network
        if dim+nr_of_params is 7*len(self.graph):
            i = 0
            for key,v in self.graph.items():
                v['train_data'] = x[:,i:i+dim]
                i += dim
        # if input data is supposed to be reused for each network
        elif dim + int(nr_of_params/len(self.graph)) is 7:
            for v in self.graph.values():
                v['train_data'] = x
            if verbose:
                print("INFO: reusing input data for all networks")
        else:
            assert False, "Size of input data is incorrect, expected %i " % (7*len(self.graph) - nr_of_params)

    def get_parameters(self):
        """Returns a copy of all learnable parameters of object in dictionary"""
        params = {}
        for name, param in self.named_parameters():
            params[name] = torch.tensor(param.data, device=self.cuda)
        return params
    
    def reset_parameters(self, value = None):
        """Sets the control voltages of all networks to:
        None: default value
        'rand': randomly generated values
        can also be number, single tensor for each parameter or list of tensors 
        """
        if value is None:
            value = self.default_param
        # set parameters, control voltages of networks
        for name, param in self.named_parameters():
            with torch.no_grad():
                # 'rand' => random values except for bias and scale, bias and scale are zeroed
                if value is 'rand':
                    if 'bias' in name or 'scale' in name:
                        param.data = torch.zeros(len(param), device=self.cuda)
                    else:
                        param.data = torch.rand(len(param), device=self.cuda)
                # dictionary => dict containing all parameters of web structure
                elif isinstance(value, dict):
                    param.data = value[name]
                # single tensor => used for all vertices, bias and scale are zeroed
                elif isinstance(value, torch.Tensor):
                    if 'bias' in name or 'scale' in name:
                        param.data = torch.zeros(len(param), device=self.cuda)
                    else:
                        param.data = value
                # single value => same number copied and used for all vertices, bias and scale are zeroed
                else:
                    if 'bias' in name or 'scale' in name:
                        param.data = torch.zeros(len(param), device=self.cuda)
                    else:
                        param.data = value*torch.ones(len(param), device=self.cuda)
    
    def get_output(self, scale=True, bias=True):
        """Returns last computed output of web"""
        d = torch.tensor(self.output_data.data, device=self.cuda)
        if scale:
            d *= 1+self.scale.data
        if bias:
            d += self.bias.data
        return d
    
    def clear_output(self):
        """Reset output data of graph, NOT the parameters"""
        self.output_data = None
        for v in self.graph.values():
            # remove output data of vertex, return None if key does not exist
            v.pop('output', None)

    def check_cuda(self, *args):
        """Converts tensors that are going to be used to cuda"""
        if torch.cuda.is_available():
            self.cuda = torch.device('cuda')
            
            # move registered parameters (control voltages)
            self.to(self.cuda)
            
            buf_lst = []
            # move arguments
            for arg in args:
                buf_lst.append(arg.to(self.cuda))
        else:
            buf_lst = args
        return tuple(buf_lst)

    def check_graph(self, print_graph=False):
        """Checks if the build graph is valid, optional plotting of graph"""
        vertices = [*self.graph.keys()]
        arcs = self.arcs.copy()
        
        layers = []
        while len(vertices)>0:
            # find vertices which have no dependicies
            independent_vertices = []
            # loop through copy of list, because elements are deleted within the loop
            for i in list(vertices):
                if i not in [sink[0] for sink in arcs.keys()]:
                    independent_vertices.append(i)
                    vertices.remove(i)
            
            # if no independent indices where found, graph is cyclic
            assert len(independent_vertices), "Cyclic graph, please change graph structure"
            
            # add them as a new layer
            layers.append(independent_vertices)
            # remove arcs with these vertices as source
            # loop through copy of list, because elements get removed during loop
            for sink in list(arcs.keys()):
                if arcs[sink] in independent_vertices:
                    del arcs[sink]
        
        input_order = self.graph.keys()
        print('INFO: Assumed order of input data is: ' + ' '.join(input_order))
        output_order = [key if value['isoutput'] else '' for key,value in self.graph.items()]
        print('INFO: Order of output data is: ' + ' '.join(output_order))
        
        # ------------------- START plot graph ------------------- 
        if print_graph:
            height = len(layers)
            width =len(max(layers, key=len))
            boxw = 0.3
            boxh = 0.1
            fig, ax = plt.subplots()
            patches = []
            
            # returns index of element from 2d list
            def index_2d(myList, v):
                for i, x in enumerate(myList):
                    if v in x:
                        return (i, x.index(v))
            
            for sink, source_name in self.arcs.items():
                sink_name, sink_gate = sink
                nr_gates = self.graph[sink_name]['network'].D_in
                i_sink = index_2d(layers, sink_name)
                i_source = index_2d(layers, source_name)
                x1 = (i_source[1]+i_source[0]%2/2)/width + boxw/2
                x2 = (i_sink[1]+i_sink[0]%2/2)/width + sink_gate/(nr_gates-1)*boxw
                y1 = i_source[0]/height + boxh
                y2 = i_sink[0]/height
                patches.append(mpatches.Arrow(x1, y1, x2-x1, y2-y1, width=0.1))
            
            for j, layer in enumerate(layers):
                if j%2:
                    offset = 0.5
                else:
                    offset = 0.
                for i, vertex in enumerate(layer):
                    x = (i+offset)/width
                    y = j/height
                    patches.append(mpatches.Polygon(np.array(((x,y), (x+boxw,y), (x+boxw/2.,y+boxh))), ec="none"))
                    plt.text(x+boxw/2, y+boxh/2, vertex, ha="center", family='sans-serif', size=14)
            collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.4)
            ax.add_collection(collection)
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        # ------------------- END plot graph ------------------- 