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
        #self.feedback_transfer = Feedbacktransfer
        
    def add_vertex(self, network, name, output=False, input_gates=None, voltage_bounds=None):
        super(resNNet, self).add_vertex(network, name, output, input_gates, voltage_bounds)
    
    def add_feedback(self, source_name, sink_name, sink_gate, input_bounds, input_gain = 1, feedback_gain = 1):
        assert (sink_name, sink_gate) not in self.arcs, "Sink gate (%s, %s), already in use!" % (sink_name, sink_gate)
        self.feedback_arcs[(sink_name, sink_gate)] = source_name
        self.graph[sink_name]['input_bounds'] = input_bounds
        self.graph[sink_name]['input_gain'] = input_gain
        self.graph[sink_name]['feedback_gain'] = feedback_gain
        #self.graph[sink_name]['feedback_transfer'] = lambda x, a: self.feedback_transfer(x, a, input_bounds, input_gain, feedback_gain)*(input_bounds[1]-input_bounds[0])+input_bounds[0]
        
    def init_chain(self, network, no_nodes, input_gate, voltage_bounds, feedback_gate, input_gain = 1, feedback_gain = 0.5):
        self.add_vertex(network, '0', input_gates = [input_gate], voltage_bounds = voltage_bounds)
        input_bounds = torch.tensor(voltage_bounds[:, feedback_gate])
        for i in range(no_nodes-2):
            self.add_vertex(network, str(i+1), input_gates = [], voltage_bounds = voltage_bounds)
            self.add_arc(str(i), str(i+1), input_gate)
            #for recurrent
            #self.add_feedback(str(i+1), str(i), feedback_gate, input_bounds, input_gain, feedback_gain)
        self.add_vertex(network, str(no_nodes-1), output = True, input_gates = [], voltage_bounds = voltage_bounds)
        self.add_arc(str(no_nodes-2), str(no_nodes-1), input_gate)
        self.add_feedback(str(no_nodes-1), '0', feedback_gate, input_bounds, input_gain, feedback_gain)
        #for recurrent
        #self.add_feedback(str(no_nodes-1), str(no_nodes-2), feedback_gate, input_bounds, input_gain, feedback_gain)
        #self.add_feedback('0', str(no_nodes-1), feedback_gate, input_bounds, input_gain, feedback_gain)
        
    def forward(self, x, delay):
        if self.feedback_arcs:
            #evaluate the graph once before applying input to set the initial output values
            #super(resNNet, self).forward(torch.zeros(1, x.shape[1]))
            output = torch.full((len(x[:,0]), len(self.graph)), np.nan)
            output[0:delay,:] = super(resNNet, self).forward(x[0:delay,0].view(delay, x.shape[1]))
            for i, val in enumerate(x[delay:,0], delay):
                print(i)
                new_x = None
                for sink, source_name in self.feedback_arcs.items():
                    sink_name, sink_gate = sink
                    v_sink, v_source = self.graph[sink_name], self.graph[source_name]
                    # this assumes one input; could change later
                    if sink_name == '0' and v_sink['swapindices'].index(0) == sink_gate:
                        #Sink of feedback is an input
                        new_x = self._Feedbacktransfer(output[i-delay,:], val, v_sink, sink_gate).view(1, x.shape[1])
                    else:
                        #Sink of feedback is a control voltage
                        cv_index = v_sink['swapindices'].index(sink_gate + 1)
                        cv_init = getattr(self, sink_name)[cv_index]
                        
                        #Update cv
                        getattr(self, sink_name)[cv_index] = self._Feedbacktransfer(v_source['output'][:,0], cv_init, v_sink, sink_gate)
                        
                #forward pass
                if new_x is None:
                    new_x = val.view(1, x.shape[1])
                super(resNNet, self).forward(new_x)
                output[i,:] = self._collect_outputs()
            
            self.output = output
            return output
        else:
            #no feedback
            return super(resNNet, self).forward(x)
        
    def forward_delay(self, x, delay):
        #evaluate the graph once before applying input to set the initial output values
        super(resNNet, self).forward(torch.zeros(delay, x.shape[1]))
        output = torch.full((len(x[:,0]), len(self.graph)), np.nan)
        for i in range(0, len(x[delay:,0])+1, delay):
            print(i)
            new_x = None
            for sink, source_name in self.feedback_arcs.items():
                sink_name, sink_gate = sink
                v_sink, v_source = self.graph[sink_name], self.graph[source_name]
                #this assumes one input and first node named '0'; could change later
                if sink_name == '0' and v_sink['swapindices'].index(0) == sink_gate:
                    #Sink of feedback is an input
                    new_x = self._Feedbacktransfer(v_source['output'][:,0], x[i:i+delay], v_sink, sink_gate).view(delay, x.shape[1])
                else:
                    #Sink of feedback is a control voltage
                    cv_index = v_sink['swapindices'].index(sink_gate + 1)
                    cv_init = getattr(self, sink_name)[cv_index]
                    
                    out = self._Feedbacktransfer(v_source['output'][:,0], cv_init, v_sink, sink_gate)
                    getattr(self, sink_name)[cv_index] = torch.mean(out)
            
            #forward pass
            if new_x is None:
                new_x = x[i:i+delay].view(delay, x.shape[1])
            
            super(resNNet, self).forward(new_x)
            output[i:i+delay,:] = self._collect_outputs(delay)
        
        self.output = output
        return output
    
    def _collect_outputs(self, delay = 1):
        output = torch.full((delay, len(self.graph)), np.nan)
        for i, val in enumerate(self.graph):
            output[:,i:i+delay] = self.graph[val]['output']
        return output
    
    def get_virtual_outputs(self, tau):
        out = torch.full((int(len(self.output)/tau), tau), np.nan)
        for i in range(tau):
            out[:,i] = self.output[i::tau].view((len(out),))
        self.output = out
        return out
    
    def train_weights(self, x, L, skip):
        x = np.delete(x, np.s_[:skip])
        target = np.full((len(x), L), np.nan)
        for i in range(L):
            target[:,i] = np.roll(x, i+1).reshape((len(x),))
        #target = np.transpose(target)
        out = self.output.detach().numpy()
        self.trained_weights = np.transpose(np.dot(np.linalg.pinv(out[skip+L:]), target[L:]))
        return self.trained_weights, target[L:]
            
    def _delay(self, x, delay):
        return torch.cat((x[-delay:], x[:-delay]))
    
    def _Feedbacktransfer(self, x, init, sink, sink_gate):
        result = sink['input_gain'] * init + sink['feedback_gain'] * sink['transfer'][sink_gate](x).view(init.shape)
        for i, val in enumerate(result):
            result[i] = self._clamp(val, sink['input_bounds'])
        return result
    
    def _clamp(self, v, bounds):
        if v > bounds[1]:
            v = bounds[1]
        elif v < bounds[0]:
            v = bounds[0]
        return v
        
        
def Transferfunction(x):
    return torch.sigmoid(x/10)
    #return (torch.clamp(x + 30, 0, 60))/60

#def Feedbacktransfer(x, init, bounds, input_gain = 1, feedback_gain = 0.5):
#    result = input_gain * init + Transferfunction(feedback_gain * x)
#    print(feedback_gain * x, bounds, (result < bounds[1]).item())
#    if (result > bounds[1]).item():
#        result = bounds[1]
#    elif (result < bounds[0]).item():
#        result = bounds[0]
#    return result