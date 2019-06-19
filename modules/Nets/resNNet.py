# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:13:26 2019

@author: Jardi
"""

import torch
import numpy as np
from collections import OrderedDict as odict
from SkyNEt.modules.Nets.webNNet import webNNet
import SkyNEt.modules.Evolution as Evolution
import time

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
        
    def forward_delay(self, x, delay, theta, gain):
        #evaluate the graph once before applying input to set the initial output values
        super(resNNet, self).forward(torch.zeros(delay, x.shape[1]))
        node_range = np.linspace(1,delay, delay)
        omega = torch.from_numpy(np.exp(-node_range*theta)).float().view(delay, 1)
        delta = torch.from_numpy((1-np.exp(-theta))*np.triu(np.exp(-theta*(node_range-np.meshgrid(node_range,node_range)[1])))).float()
        output = torch.full((len(x[:,0]), len(self.graph)), np.nan)
        for i in range(0, len(x[delay:,0])+1, delay):
            #print(i)
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
            
            x_old = v_source['output'][delay-1].item()  
            super(resNNet, self).forward(new_x)
            f = self._collect_outputs(delay)
            output[i:i+delay,:] = omega*x_old + gain*torch.sum(delta*f, 0).view(delay, 1)
            #output[i:i+delay,:] = self._collect_outputs(delay)
#            if i == 0:
#                output[0,:] = f[0]
#                for ii in range(1,delay):
#                    output[ii,:] = output[ii-1]/np.e**theta + (1-1/np.e**theta)*f[ii]
#            else:
#                for ii in range(delay):
#                    output[i+ii,:] = output[i+ii-1]/np.e + (1-1/np.e)*f[ii]
            self.graph['0']['output'] = output[i:i+delay,:]
        
        self.output = output
        return output
    
    def _collect_outputs(self, delay = 1):
        output = torch.full((delay, len(self.graph)), np.nan)
        for i, val in enumerate(self.graph):
            output[:,i:i+delay] = self.graph[val]['output']
        return output
    
    def get_virtual_outputs(self, tau, output = None):
        if output is None:
            output = self.output
        else:
            output = torch.from_numpy(output).t()
        out = torch.full((int(len(output)/tau), tau), np.nan)
        for i in range(tau):
            out[:,i] = output[i::tau].view((len(out),))
        self.output = out
        return out
    
    def train_weights(self, x, L, skip, out = None):
        x = np.delete(x, np.s_[:skip])
        target = np.full((len(x), L), np.nan)
        for i in range(L):
            target[:,i] = np.roll(x, i+1).reshape((len(x),))
        #target = np.transpose(target)
        if out is None:
            out = self.output.detach().numpy()
        self.trained_weights = np.transpose(np.dot(np.linalg.pinv(out[skip+L:]), target[L:]))
        return self.trained_weights, target[L:]
    
    def train_weights_RR(self, x, L, skip, alpha, out = None):
        x = np.delete(x, np.s_[:skip])
        target = np.full((len(x), L), np.nan)
        for i in range(L):
            target[:,i] = np.roll(x, i+1).reshape((len(x),))
        #target = np.transpose(target)
        if out is None:
            out = self.output.detach().numpy()
        R = np.dot(out[skip+L:].transpose(), out[skip+L:])
        P = np.dot(out[skip+L:].transpose(), target[L:])
        #Id = np.identity(R.size)
        self.trained_weights = np.transpose(np.dot(np.linalg.inv(R + alpha ** 2), P))
        return self.trained_weights, target[L:]
            
    def _delay(self, x, delay):
        return torch.cat((x[-delay:], x[:-delay]))
    
    def _Feedbacktransfer(self, x, init, sink, sink_gate):
        out = sink['transfer'][sink_gate](x)/2
        if sink['input_gain'] + sink['feedback_gain'] > 2:
            scaling = 2/(sink['input_gain'] + sink['feedback_gain'])
        else:
            scaling = 1
        result = scaling * (sink['input_gain'] * init + sink['feedback_gain'] * out.view(init.shape))
        result = torch.clamp(result, sink['input_bounds'][0], sink['input_bounds'][1])
        return result

    def trainGA(self, train_data, inpt, cf, verbose = False, output = True):
        # initialize genepool
        genepool = Evolution.GenePool(cf)
        
        # np arrays to save genePools, outputs and fitness
        geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
        if output:
            outputArray = np.zeros((cf.generations, cf.genomes, inpt.shape[0]))
        fitnessArray = np.zeros((cf.generations, cf.genomes))
        memoryArray = np.zeros((cf.generations, cf.genomes))
        
        # Temporary arrays, overwritten each generation
        fitnessTemp = np.zeros((cf.genomes, cf.fitnessavg))
        outputAvg = torch.zeros(cf.fitnessavg, int(inpt.shape[0]/cf.fitnessavg), self.nr_output_vertices, device=self.cuda)
        outputTemp = torch.zeros(cf.genomes, int(inpt.shape[0]/cf.fitnessavg), self.nr_output_vertices, device=self.cuda)
        memoryTemp = np.zeros((cf.genomes, cf.fitnessavg))
    
        for i in range(cf.generations):
            #Time multiplex data
            start = time.time()
            N = train_data.shape[0]/cf.fitnessavg
            mask = torch.rand(cf.vir_nodes).repeat(int(N*cf.fitnessavg)).view(inpt.shape) - 0.5
            inpt_masked = inpt * mask
            for j in range(cf.genomes):
                for avgIndex in range(cf.fitnessavg):
                    # update parameters of each network
                    self.set_parameters_from_pool(genepool.pool[j], cf, genepool)
                    with torch.no_grad():
                        outputAvg[avgIndex] = self.forward_delay(inpt_masked[int(avgIndex*N*cf.vir_nodes):int((avgIndex+1)*N*cf.vir_nodes)], cf.vir_nodes, cf.theta, cf.gain)
                    virout = self.get_virtual_outputs(cf.vir_nodes).detach().numpy()
                    
                    weights, targets = self.train_weights(train_data[int(avgIndex*N):int((avgIndex+1)*N)], cf.output_nodes, cf.skip)
                    fitnessTemp[j, avgIndex], memoryTemp[j, avgIndex] = cf.Fitness(virout, weights, targets)
                    
                outputTemp[j] = outputAvg[np.argmax(fitnessTemp[j])]
            
            genepool.fitness = fitnessTemp.mean(1)  # Save best fitness
    
            # Save generation data
            geneArray[i, :, :] = genepool.pool
            if output:
                outputArray[i, :, :] = outputTemp.detach().numpy().reshape(cf.genomes, int(inpt.shape[0]/cf.fitnessavg))
            fitnessArray[i, :] = genepool.fitness
            memoryArray[i, :] = memoryTemp.mean(1)
    
            if verbose:
                print("Generation nr. " + str(i + 1) + " completed in " + str((time.time() - start)/60) + " minutes")
                print("Best fitness: " + str(np.amax(genepool.fitness)))
                print("Memory Capacity: " + str(memoryTemp[np.where(genepool.fitness == np.amax(genepool.fitness))][0].mean()))
            
            genepool.NextGen()
    
        if output:
            return geneArray, fitnessArray, memoryArray, outputArray
        else:
            return geneArray, fitnessArray, memoryArray
    
    def set_parameters_from_pool(self, pool, cf, genepool):
        genes = np.full(pool.shape, np.nan)
        for i, gene in enumerate(pool):
            genes[i] = genepool.MapGenes(cf.generange[i], pool[i])
        for i in range(6):
            getattr(self, '0')[i] = genes[i]
        self.graph['0']['input_gain'] = genes[6]
        self.graph['0']['feedback_gain'] = genes[7]
        cf.theta = genes[8]
        cf.gain = genes[9]
        
        
def Transferfunction(x):
    #out = (x+50)/100
    out = (x+1)/11
    return torch.clamp(out, 0, 1)

#def Feedbacktransfer(x, init, bounds, input_gain = 1, feedback_gain = 0.5):
#    result = input_gain * init + Transferfunction(feedback_gain * x)
#    print(feedback_gain * x, bounds, (result < bounds[1]).item())
#    if (result > bounds[1]).item():
#        result = bounds[1]
#    elif (result < bounds[0]).item():
#        result = bounds[0]
#    return result