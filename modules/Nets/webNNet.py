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
        
    def add_vertex(self, network, name, output=False):
        """Adds neural network as a vertex to the graph.
        Args:
            network: vertex (object with attribute D_in and method output())
            name: key of graph dictionary in which vertex is created (str)
            cv: control voltages, the parameters which are trained
        """
        
        assert not hasattr(self, name), "Name %s already in use, choose other name for vertex!" % name
#        cv = torch.rand(5)
        cv = 0.9*torch.ones(5)
        self.register_parameter(name, torch.nn.Parameter(cv))
        
        self.graph[name] = {  'network':network,
                              'isoutput':output}
    
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

    def forward(self):
        """Evaluates the graph, returns output (torch.tensor)
        Start at network which is the output of the whole graph, 
        then recursively step through graph vertex by vertex
        """
        
        # reset output of the graph
        self.clear_output()
        
        for key,value in self.graph.items():
            # start at vertex which is defined as output
            if value['isoutput']:
                # recursively evaluate vertices
                self.forward_vertex(key)
                self.output_data = value['output']
                return value['output']
    
    def train(self, x, y, verbose=False):
        self.check_graph()
        self.set_train_data(x)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        
        loss_list = []
        for i in range(100):
            y_pred = self.forward()
            loss = criterion(y_pred, y)
            loss_list.append(loss.item())
            if verbose:
                print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        plt.plot(loss_list)
        plt.xlabel('epochs')
        plt.ylabel('MSE loss')
        return loss_list
    
    def set_train_data(self, x):
        """Store training data for each network, assumes the torch tensor is the same ordering as the dictionary"""
        dim = x.shape[1]
        if int(dim/2) is len(self.graph):
            i = 0
            keys = []
            for key,v in self.graph.items():
                v['train_data'] = x[:,i:i+2]
                i += 2
                keys.append(key)
            print("INFO: Assumed order of input data is %s" % keys)
        elif dim==2:
            for v in self.graph.values():
                v['train_data'] = x
            print("INFO: reusing input data for all networks")
        else:
            assert False, "Number of input columns/2 (%s) should match number of vertices in graph (%s)" % (dim/2, len(self.graph))
    
    def get_output(self):
        return self.output_data
    
    def clear_output(self):
        """Reset output data of graph, NOT the parameters"""
        self.output_data = None
        for v in self.graph.values():
            # remove output data of vertex, return None if key does not exist
            v.pop('output', None)

    def check_graph(self, print_graph=False):
        """Checks if the build graph is valid"""
        vertices = [*self.graph.keys()]
        arcs = self.arcs.copy()
        
        layers = []
        while len(vertices)>0:
            # find vertices which have no dependicies
            independent_vertices = []
            # loop through copy of list, because elements are deleted in loop
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
        
        # plot graph 
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