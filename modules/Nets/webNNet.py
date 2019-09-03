#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:46:46 2018

IMPORTANT: 
    - for documentation on train function, see webNNetTrain.py
    - functions starting with _ are meant for internal use only


Class to create a web of connected neural networks (device simulations),
 this web can be seen as a non-cyclic directed graph.
The graph consists of vertices (networks) and arcs (connections between vertices).

See Skynet/experiments/Web/webNN_template.py for example use.

The structure of the web is layed out below.

Vertices:
Structure of self.graph (dictionary):
    keys: names of vertex, values: info of vertex
    vertex info (dictionary):
        'network'       : neural network object which simulates a device
        'train_data'    : input data which is automatically added to control voltages (torch tensor)
        'isoutput'      : wether vertex is output vertex (boolean)
        'output'        : output data of vertex calculated by _forward_vertex (torch tensor)
        'swapindices'   : indices which are used to swap columns to correct gate positions before a vertex is evaluated
        'voltage_bounds': first row contains minimum voltage values a gate can handle, second row are maxima
        'transfer'      : list of transfer functions used to map the output of a network to correct voltage range of gate

Arcs:
Each arc consist of a source and sink, where the data flows from source to sink.
Structure of self.arcs (dictionary):
    keys: tuple: (sink_name, sink_gate)
    values: source_name

@author: ljknoll
"""

import torch
from collections import OrderedDict as odict

import SkyNEt.modules.Nets.webNNetTrain as webNNetTrain

# import additional methods (see 'optional#1' below)
import SkyNEt.modules.Nets.webNNetTrainGA as GA


# imports for plotting graph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

class webNNet(torch.nn.Module):
    def __init__(self):
        super(webNNet, self).__init__()
        self.graph = odict()            # vertices of graph
        self.arcs = odict()             # arcs of graph
        self.output_data = None         # output data of graph
        self.nr_output_vertices = 0     # number of networks whose output data is used
        self.nr_of_params = 0           # number of parameters of network which need to be trained (excluding those set by add_parameters)
        self._params = [{'params':[]}]  # attribute which builds the parameter groups, passed to optimizer
        
        # setting defaults
        self.cuda = 'cpu'
        self.loss_fn = torch.nn.MSELoss()                   # loss function (besides regularization)
        self.custom_par = {}                                # keeps track of  custom parameters added
        self.nr_of_custom_params = 0
        self.custom_reg = lambda : torch.FloatTensor([0])   # function which returns the regularization of custom parameters
        self.optimizer = torch.optim.Adam                   # optimizer function
        self.transfer = torch.sigmoid                       # function which maps output to input [0,1]
    
    #attach training function to class
    train, session_train = webNNetTrain.train, webNNetTrain.session_train
    
    # attach GA functions to class (optional#1)
    prepare_config_obj, set_dict_indices_from_pool, set_parameters_from_pool, trainGA = GA.prepare_config_obj, GA.set_dict_indices_from_pool, GA.set_parameters_from_pool, GA.trainGA
    noveltyGA = GA.noveltyGA

    
    def add_vertex(self, network, name, output=False, input_gates=None, voltage_bounds=None):
        """Adds neural network as a vertex to the graph.
        Args:
            network:        (object with attribute D_in and method output()) vertex
            name:           (str) key of graph dictionary in which vertex is created (str)
            output:         (bool) wheter of not this vertex' output is output of complete graph (boolean)
            input_gates:    (list) numbers of gates which should be used as inputs
            voltage_bounds: (2 by D_in tensor) first row are lower bounds of all control voltages, second row are upper bounds
            evaluated:      (bool) flag indicating if vertex has already been evaluated
        """
        
        assert not hasattr(self, name), "Name %s already in use, choose other name for vertex!" % name
        
        D_in = network.D_in
        
        # set default input_gates
        if input_gates is None:
            input_gates = [0,1]
        assert max(input_gates, default=0)<D_in, "Gate number (%i) exceeds range (0-%i)" % (max(input_gates), D_in-1)
        
        # generate control_gates with remaining gates
        control_gates = []
        for i in range(D_in):
            if i not in input_gates:
                control_gates.append(i)
        
        
        if voltage_bounds is None:
            if hasattr(network, 'info') and 'offset' in network.info and 'amplitude' in network.info:
                # collect voltage bounds from info of network
                info = network.info
                voltage_bounds = torch.cat((torch.FloatTensor(info['offset']-info['amplitude']),
                                            torch.FloatTensor(info['offset']+info['amplitude']))).view(-1,D_in)
            else:
                # default to [[zeros], [ones]]
                voltage_bounds = torch.cat((torch.zeros(D_in), torch.ones(D_in))).view(-1,D_in)
        else:
            voltage_bounds = torch.tensor(voltage_bounds)
        
        assert voltage_bounds.shape == (2, D_in), "shape of voltage bounds (%s) does not match expected shape (%s)"

        # define full transfer functions for each gate (may not be used)
        transfer = [(lambda y: (lambda x: self.transfer(x)*(y[1]-y[0])+y[0]))(ii) for ii in voltage_bounds.t()]
        
        # keep only the values of control gates, input gate values are not used
        voltage_bounds = voltage_bounds[:, control_gates]

        # add parameter to model
        number_cv = len(control_gates)                          # nr of control voltages
        cv = torch.rand(number_cv)*(voltage_bounds[1]-voltage_bounds[0]) + voltage_bounds[0] # initialize randomly
        self.register_parameter(name, torch.nn.Parameter(cv))   # register parameter to object
        self.nr_of_params += number_cv                          # update number of parameters in graph
        self._params[0]['params'].append(getattr(self, name))   # add vertex parameter to first group
        
        # In forward_vertex inputs and controls are concatenated to e.g. [I,I,c,c,c,c] with gate numbers [2,3,0,1,4,5,6]
        # here, I generate a list of indices such that can be swapped in correct order [0-7]
        swapindices = [i for i in range(len(input_gates), D_in)]
        for c,i in enumerate(input_gates):
            swapindices.insert(i,c)
        
        self.graph[name] = {  'network':network,
                              'isoutput':output,
                              'swapindices':swapindices,
                              'voltage_bounds':voltage_bounds,
                              'transfer':transfer,
                              'evaluated':False}
        
        if output:
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
    
    def add_parameters(self, parameter_names, parameters, custom_reg = None, **kwargs):
        """Adds custom parameters to be trained to web
        parameter_names     list of strings, names of parameters
        parameters          list of torch tensors (these values are also used when resetting parameters)
        custom_reg          (optional) function which regularizes the custom parameters
        **kwargs            (optional) keyword arguments for this group of parameters passed to optimizer, 
                            for example, lr=0.2, which means the learning rate for these parameters are overwritten,
                            see https://pytorch.org/docs/stable/optim.html#per-parameter-options
        """
        assert len(parameter_names) == len(parameters), 'Parameters and names have different length!'
        for i, name in enumerate(parameter_names):
            self.register_parameter(name, torch.nn.Parameter(parameters[i]))
            self.custom_par[name] = parameters[i].clone()
            self.nr_of_custom_params += len(parameters[i])
        
        # add parameters as new group to be passed in optimizer
        self._params.append({'params':[getattr(self, name) for name in parameter_names], **kwargs})
        
        if custom_reg is not None:
            self.custom_reg = custom_reg
    
    def forward(self, x):
        """Evaluates the graph, returns output (torch.tensor)
        Start at network which is the output of the whole graph, 
        then recursively step through graph vertex by vertex
        """
        
        # reset output of the graph
        self._clear_output()
        
        # define input data for all networks
        self._set_input_data(x)
        
        tuple_data = ()
        for key,value in self.graph.items():
            # start at vertex which is defined as output
            if value['isoutput']:
                # recursively evaluate vertices
                self._forward_vertex(key)
                return_value = value['output']
                tuple_data += (return_value,)
        returned_data = torch.cat(tuple_data, dim=1)
        self.output_data = returned_data.data
        return returned_data

    def _forward_vertex(self, vertex):
        """Calculates output of vertex"""
        v = self.graph[vertex]
        
        # skip if vertex is already evaluated
        if not v['evaluated']:
            # control voltages, repeated to match batch size of train_data
            cv_data = getattr(self, vertex).repeat(self._batch_size, 1)
            
            # concatenate input with control voltage data
            try:
                data = torch.cat((v['train_data']*self.scale + self.bias, cv_data), dim=1) # If scale and bias are parameters
            except AttributeError: 
                data = torch.cat((v['train_data'], cv_data), dim=1)
            
            # swap columns according to input/control indices
            data = data[:,self.graph[vertex]['swapindices']]
            
            # check dependencies of vertex by looping through all arcs
            for sink,source_name in self.arcs.items():
                sink_name, sink_gate = sink
                # if current vertex is also vertex that the arc is pointing to,
                # that means we need the data through that arc
                if sink_name == vertex:
                    # first evaluate vertices on which this input depends
                    self._forward_vertex(source_name)
                    # insert data from arc into control voltage parameters with correct transfer function
                    data[:, sink_gate] = v['transfer'][sink_gate](self.graph[source_name]['output'][:,0])
            # feed through network

            v['output'] = v['network'].outputs(data,grad=True)
            v['evaluated'] = True

    
    def error_fn(self, y_pred, y, beta):
        """Error function: loss function with added regularization"""        
        # calculate regularization of control voltage parameters
        reg_loss = 0

        for name in self.graph.keys():
            x = getattr(self, name)
            voltage_bounds = self.graph[name]['voltage_bounds']
            reg_loss += torch.sum(torch.relu(voltage_bounds[0] - x) + torch.relu(-voltage_bounds[1] + x))
        return self.loss_fn(y_pred, y) + beta*reg_loss + self.custom_reg()

    
    def _set_input_data(self, x):
        """Store training data for each network, assumes the torch tensor has the same ordering as in the dictionary"""
        # assumes each vertex has same number of parameters (could change in future)
        dim = x.shape[1]
        # if input data is provided for each network
        if dim+self.nr_of_params is 7*len(self.graph):
            i = 0
            for key,v in self.graph.items():
                nr_columns = v['network'].D_in-v['voltage_bounds'].shape[1] # nr of columns of x to be used for this vertex
                v['train_data'] = x[:,i:i+nr_columns]
                i += nr_columns
        else:
            assert False, "Size of input data is incorrect, expected (%i), got (%i)" % (7*len(self.graph) - self.nr_of_params, dim)
        self._batch_size = x.shape[0]

    def get_parameters(self):
        """Returns a copy of all learnable parameters of object in dictionary"""
        params = {}
        for name, param in self.named_parameters():
            params[name] = torch.tensor(param.data, device=self.cuda)
        return params
    
    def reset_parameters(self, value = None):
        """Sets the control voltage parameters of all devices.
        (custom parameters are set to their initial values)
        options for value are:
            None or 'rand'  randomly generated values
            dict            use values from dictionary, must include custom parametes
            iterable        replace values one by one from single iterable (e.g. from param_history)
        """
        if value is None or (isinstance(value, str) and value == 'rand'):
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in self.graph.keys():
                        voltage_bounds = self.graph[name]['voltage_bounds']
                        diff = (voltage_bounds[1]-voltage_bounds[0])
                        param.data = torch.rand(len(param), device=self.cuda)*diff + voltage_bounds[0]
                    else:

                        param.data = torch.tensor(self.custom_par[name].clone(), device=self.cuda)
        elif isinstance(value, dict):
            assert self.get_parameters().keys() == value.keys(), "Different keys in given dict"
            with torch.no_grad():
                for name, param in self.named_parameters():
                     param.data = value[name]
        else:
            try:
                assert len(value) == self.nr_of_params+self.nr_of_custom_params, "length of given list of parameters (%i) does not match nr of parameters (%i)"% (len(value), self.nr_of_params)
                with torch.no_grad():
                    c = 0
                    for name, param in self.named_parameters():
                        for i in range(len(param)):
                            param[i] = value[c]
                            c += 1
            except TypeError:
                print("type of given parameters (%s) is not available to set parameters of web" % type(value).__name__)
    
    def get_output(self):
        """Returns last computed output of web"""
        return torch.tensor(self.output_data.data, device=self.cuda)
    
    def _clear_output(self):
        """Reset output data of graph, NOT the parameters"""
        self.output_data = None
        for v in self.graph.values():
            # reset evaluated flag
            v['evaluated'] = False

    def check_cuda(self, *args):
        """Converts tensors that are going to be used to cuda"""
        if False: #torch.cuda.is_available():
            self.cuda = torch.device('cuda')
            
            # move registered parameters (control voltages)
            self.to(self.cuda)
            
            buf_lst = []
            # move arguments
            for arg in args:
                buf_lst.append(arg.to(self.cuda))
        else:
            buf_lst = args
        if len(buf_lst) == 1:
            return buf_lst[0]
        else:
            return tuple(buf_lst)


    def check_graph(self, print_graph=False, verbose=True):
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
        output_order = [key if value['isoutput'] else '' for key,value in self.graph.items()]
        if verbose:
            print('INFO: Assumed order of input data is: ' + ' '.join(input_order))
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
                    patches.append(mpatches.Rectangle((x,y), boxw, boxh, ec="none"))
                    plt.text(x+boxw/2, y+boxh/2, vertex, ha="center", family='sans-serif', size=14)
            collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.4)
            ax.add_collection(collection)
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        # ------------------- END plot graph ------------------- 