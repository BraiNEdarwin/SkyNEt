#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:46:46 2018
Class to create a web of connected neural networks.
The graph consists of vertices (networks) and arcs (connections between vertices).
Each arc consist of a source and sink, where the data flows from source to sink.
Structure of graph (dictionary):
    keys: names of vertex, values: info of vertex
    vertex info (dictionary):
        'network'   : neural network object
        'input'     : input data which is added to control voltages (torch tensor)
        'isoutput'  : if output vertex (boolean)
        'output'    : output data calculated by forward (torch tensor)
Structure of arcs (dictionary):
    keys: tuple: (sink_name, sink_gate)
    values: source_name
    
See webNN_template.py for example use.
@author: ljknoll
"""

import torch

# imports for plotting graph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


class webNNet(torch.nn.Module):
    def __init__(self):
        super(webNNet, self).__init__()
        self.graph = {} # vertices of graph
        self.arcs = {} # arcs of graph
        self.output_data = None # output data of graph
        
        self.default_param = 0.8
        self.loss = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD
        
        self.register_parameter('bias', torch.nn.Parameter(torch.tensor([])))
        self.register_parameter('scale', torch.nn.Parameter(torch.tensor([])))
        
    def train(self, 
              train_data,
              targets,
              batch_size,
              nr_epochs=100,
              verbose=False,
              beta=0.01,
              optimizer=None,
              loss_fn=None,
              bias = False,
              scale=False,
              **kwargs):
        """verbose: prints error at each iteration
        beta: scaling parameter for relu regularization outside [0,1] for cv
        maxiterations: the number of iterations after which training stops
        """
        self.check_graph()
        
        self.set_input_data(train_data, verbose=True)
        
        if optimizer is None:
            print("INFO: Using SGD with, ", kwargs)
            optimizer = self.optimizer(self.parameters(), **kwargs)
        else:
            print("INFO: Using custom optimizer with, ", kwargs)
            optimizer = optimizer(self.parameters(), **kwargs)
        
        error_list = []
        best_error = 10e5
        best_params = self.get_parameters()
        for epoch in range(nr_epochs):
            permutation = torch.randperm(len(train_data))
            for i in range(0,len(permutation), batch_size):
                indices = permutation[i:i+batch_size]
                y_pred = self.forward(train_data[indices], bias=bias, scale=scale)
                error = self.error_fn(y_pred, targets[indices], beta, loss_fn)
                error_value = error.item()
                optimizer.zero_grad()
                error.backward()
                optimizer.step()
            
            predictions = self.forward(train_data, bias=bias, scale=scale)
            error_value = self.error_fn(predictions, targets, beta, loss_fn).item()
            error_list.append(error_value)
            if verbose:
                print("INFO: error at epoch %s: %s" % (epoch, error_value))
            if error_value < best_error:
                best_error = error_value
                best_params = self.get_parameters()
        return error_list, best_params

    def add_vertex(self, network, name, output=False):
        """Adds neural network as a vertex to the graph.
        Args:
            network: vertex (object with attribute D_in and method output())
            name: key of graph dictionary in which vertex is created (str)
            cv: control voltages, the parameters which are trained
        """
        
        assert not hasattr(self, name), "Name %s already in use, choose other name for vertex!" % name
        cv = self.default_param*torch.ones(5)
        self.register_parameter(name, torch.nn.Parameter(cv))
        
        self.graph[name] = {  'network':network,
                              'isoutput':output}
        if output:
            self.bias = torch.nn.Parameter(torch.cat((self.bias, torch.tensor([0.0]))))
            self.scale = torch.nn.Parameter(torch.cat((self.scale, torch.tensor([0.0]))))
    
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
        self.set_input_data(x)
        
        tuple_scaled_data = ()
        order = []
        for key,value in self.graph.items():
            # start at vertex which is defined as output
            if value['isoutput']:
                order.append(key)
                # recursively evaluate vertices
                self.forward_vertex(key)
                return_value = value['output']
                tuple_scaled_data += (return_value,)
        if verbose:
            print("Assumed order of output is: %s" % order)
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
                    data[:, sink_gate] = torch.sigmoid(self.graph[source_name]['output'])[0]
            
            # feed through network
            v['output'] = v['network'].model(data)
    
    def error_fn(self, y_pred, y, beta, loss=None):
        """Error function: loss function with added regularization"""
        # default loss function: MSE
        if loss is None:
            loss = self.loss
        
        # calculate regularization
        reg_loss = 0
        for name, x in self.named_parameters():
            if name not in ['bias', 'scale']:
                reg_loss += torch.sum(torch.relu(-x) + torch.relu(x-1.0))
            else:
                reg_loss += torch.sum(torch.abs(x))
        return loss(y_pred, y) + beta*reg_loss
    
    def set_input_data(self, x, verbose=False):
        """Store training data for each network, assumes the torch tensor has the same ordering as in the dictionary"""
        dim = x.shape[1]
        # if input data is provided for each network
        if int(dim/2) is len(self.graph):
            i = 0
            keys = []
            for key,v in self.graph.items():
                v['train_data'] = x[:,i:i+2]
                i += 2
                keys.append(key)
            if verbose:
                print("INFO: Got seperate input data for networks, assumed order is %s" % keys)
        # if input data is supposed to be reused for each network
        elif dim==2:
            for v in self.graph.values():
                v['train_data'] = x
            if verbose:
                print("INFO: reusing input data for all networks")
        else:
            assert False, "Number of input columns/2 (%s) should match number of vertices in graph (%s)" % (dim/2, len(self.graph))

    def get_parameters(self):
        """Returns a copy of all learnable parameters of object in dictionary"""
        params = {}
        for name, param in self.named_parameters():
            params[name] = torch.tensor(param.data)
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
                if value is 'rand':
                    if name in ['bias', 'scale']:
                        param.data = torch.zeros(len(param))
                    else:
                        param.data = torch.rand(len(param))
                elif isinstance(value, dict):
                    param.data = value[name]
                elif isinstance(value, torch.Tensor):
                    if name in ['bias', 'scale']:
                        param.data = torch.zeros(len(param))
                    else:
                        param.data = value
                else:
                    if name in ['bias', 'scale']:
                        param.data = torch.zeros(len(param))
                    else:
                        param.data = value*torch.ones(len(param))
    
    def get_output(self, scale=True, bias=True):
        """Returns last computed output of web"""
        d = self.output_data
        if scale:
            d *= 1+self.scale
        if bias:
            d += self.bias
        return d
    
    def clear_output(self):
        """Reset output data of graph, NOT the parameters"""
        self.output_data = None
        for v in self.graph.values():
            # remove output data of vertex, return None if key does not exist
            v.pop('output', None)

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
                    patches.append(mpatches.Rectangle((x,y), boxw, boxh, ec="none"))
                    plt.text(x+boxw/2, y+boxh/2, vertex, ha="center", family='sans-serif', size=14)
            collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.4)
            ax.add_collection(collection)
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        # ------------------- END plot graph ------------------- 