# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:13:26 2019

@author: Jardi
"""

import torch
import numpy as np
from collections import OrderedDict as odict
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet

class resNNet(webNNet):
    
    def __init__(self, *args, loss = 'MSE', activation = 'ReLU', dim_cv = 5, BN = False):
        super(resNNet,self).__init__()
        self.feedback_arcs = odict()
    
    def add_feedback(self, source_name, sink_name, sink_gate):
        assert (sink_name, sink_gate) not in self.arcs, "Sink gate (%s, %s), already in use!" % (sink_name, sink_gate)
        self.feedback_arcs[(sink_name, sink_gate)] = source_name
        
    def forward(self, x):
        if self.feedback_arcs:
            #evaluate the graph once before applying input to set the initial output values
            super(resNNet, self).forward(torch.zeros(1, x.shape[1]))
            for sink, source_name in self.feedback_arcs.items():
                sink_name, sink_gate = sink
                v_sink, v_source = self.graph[sink_name], self.graph[source_name]
                # this assumes one input; could change later
                if v_sink['swapindices'].index(0) == sink_gate:
                    #Sink of feedback is an input
                    pass
                else:
                    #Sink of feedback is a control voltage
                    cv_index = v_sink['swapindices'].index(sink_gate + 1)
                    print(cv_index)
                    cv_init = getattr(self, sink_name)[cv_index]
                    
                    output = torch.full_like(x[:,0], np.nan)
                    for i, val in enumerate(x[:,0]):
                        #Update cv
                        
                        getattr(self, sink_name)[cv_index] = v_sink['transfer'][sink_gate](cv_init + v_source['output'][:,0])[0]
                        
                        #forward pass
                        output[i] = super(resNNet, self).forward(torch.tensor(val).view(1, x.shape[1]))
                    
                    return output
        else:
            #no feedback
            return super(resNNet, self).forward(x)
        