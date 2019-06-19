# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:32:47 2019

@author: Jardi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.resNNet import resNNet, Transferfunction
import time

start = time.time()

## Parameters
vir_nodes = 100
theta = 0.37668741
tau = int(vir_nodes * theta)
N = 5300
vlow1, vhigh1 = -1, 1
vlow2, vhigh2 = -1.2, 1.2
voltage_bounds = np.repeat([[vlow2, vlow1], [vhigh2, vhigh1]], [5, 2, 5, 2]).reshape(-1, 7).astype(np.float32)
input_electrode = 4
feedback_electrode = input_electrode
input_bounds = torch.tensor(voltage_bounds[:, feedback_electrode])
skip = 200
nodes = 100
input_gain = -0.12342481
feedback_gain = 4.86875437
nonlinear_gain = 3.93680934

## Input Signal
u = torch.FloatTensor(int(N), 1).uniform_(vlow2, vhigh2)
#u = torch.load(r'C:\Users\Jardi\Desktop\BachelorOpdracht\Resultaten\Delay line\input.pt')
u_np = u.numpy()
inpt = torch.repeat_interleave(u, vir_nodes).view(vir_nodes*N, 1)
mask = torch.rand(vir_nodes).repeat(N).view(inpt.shape) - 0.5
#mask = torch.load(r'C:\Users\Jardi\Desktop\BachelorOpdracht\Resultaten\Delay line\mask1000.pt')
inpt_mask = inpt * mask
inpt_np = inpt.numpy()
inpt_mask_np = inpt_mask.numpy()

## Load neural net
main_dir = r'D:/Jardi/NNModels/'
data_dir = 'MSE_n_d5w90_500ep_lr3e-3_b2048.pt'
net = staNNet(main_dir+data_dir)

## Initialise reservoir
res = resNNet()

#d = res.graph

## Set transfer function
#res.transfer = torch.nn.Hardtanh(0)
res.transfer = Transferfunction

## Add devices
res.add_vertex(net, '0', output = True, input_gates = [input_electrode], voltage_bounds = voltage_bounds)
res.add_feedback('0', '0', feedback_electrode, input_bounds, input_gain, feedback_gain)

#cvs = np.array([-0.62019978, -0.22513563, 0.85174519, 0.0061291, 0.96584976, -0.24035166]) #10 nodes
cvs = np.array([-0.75814799, -0.52293706,  0.50786779,  0.95651859,  0.84188576, 0.50251109]) #100 nodes

for i, val in enumerate(cvs):
    getattr(res, '0')[i] = val

## forward pass
with torch.no_grad():
    output = res.forward_delay(inpt_mask, vir_nodes, theta, nonlinear_gain)
output_np = output.detach().numpy()
virout = res.get_virtual_outputs(vir_nodes)
virout_np = virout.detach().numpy()
#output = np.full_like(inpt_mask_np, np.nan)
#out_init = np.tanh(0)
#out_init = net.outputs(torch.zeros((1, 1)))*3/25000 - 0.6
#f = np.full((vir_nodes, 1), np.nan)
##f = np.tanh(input_gain * inpt_mask_np[0:vir_nodes] + feedback_gain * out_init)
#f = net.outputs(input_gain * inpt_mask_np[0:vir_nodes] + feedback_gain * out_init)*3/25000 - 0.6
#output[0] = out_init/np.e + (1-1/np.e)*f[0]
#for i in range(1, vir_nodes):
#    output[i] = output[i-1]/np.e + (1-1/np.e)*f[i]
#for i in range(vir_nodes, len(inpt_mask_np[vir_nodes:,0])+1, vir_nodes):
#    #f = np.tanh(input_gain * inpt_mask_np[i:i+vir_nodes] + feedback_gain * output[i-vir_nodes:i])
#    f = net.outputs(input_gain * inpt_mask_np[i:i+vir_nodes] + feedback_gain * output[i-vir_nodes:i])*3/25000 - 0.6
#    for ii in range(vir_nodes):
#        output[i+ii] = output[i+ii-1]/np.e + (1-1/np.e)*f[ii]
#    print(i)
#virout_np = np.full((N, vir_nodes), np.nan)
#for i in range(vir_nodes):
#    virout_np[:,i] = output[i::vir_nodes].reshape(N,)

#u_np = np.load(r"..\..\..\Resultaten\u.npy")
#virout_np = np.load(r"..\..\..\Resultaten\virout.npy")
#bias = np.ones((virout_np.shape[0], 1))
#virout_np = np.append(virout_np, bias, axis=1)

weights, target = res.train_weights(u_np, nodes, skip, virout_np)

prediction = np.dot(weights, np.transpose(virout_np[skip+nodes:,:]))

MCk = np.full(nodes, np.nan)
for i in range(nodes):
    MCk[i] = np.corrcoef(target[:,i], prediction[i,:])[0,1]**2
    if np.isnan(MCk[i]):
        MCk[i] = 0
MC = sum(MCk)

print(time.time()-start)

def validateMC(weights, it):
    MCk_val = np.zeros(nodes)
    MC_it = np.full(it, np.nan)
    for i in range(it):
        u_val = torch.FloatTensor(int(N), 1).uniform_(vlow2, vhigh2)
        inpt_val = torch.repeat_interleave(u_val, vir_nodes).view(vir_nodes*N, 1)
        mask_val = torch.rand(vir_nodes).repeat(N).view(inpt_val.shape) - 0.5
        inpt_mask_val = inpt_val * mask_val
        
        ## forward pass
        with torch.no_grad():
            res.forward_delay(inpt_mask_val, vir_nodes, theta)
        virout_val = res.get_virtual_outputs(vir_nodes)
        virout_val_np = virout_val.numpy()
        virout_val_np = np.append(virout_val_np, bias, axis=1)
        
        _, target_val = res.train_weights(u_val.numpy(), nodes, skip, virout_val_np)
        prediction = np.dot(weights, np.transpose(virout_val_np[skip+nodes:,:]))
        
        MCk_it = np.full(nodes, np.nan)
        for ii in range(nodes):
            MCk_it[ii] = np.corrcoef(target_val[:,ii], prediction[ii,:])[0,1]**2
        MCk_val += MCk_it
        MC_it[i] = sum(MCk_it)
        
        plt.figure()
        x = np.linspace(1, nodes, nodes)
        plt.plot(x, MCk_it)
        plt.ylim([0,1.05])
        plt.title('Forgetting curve (D = ' + str(vir_nodes) + ', n_max = ' + str(N - nodes) + ', gain = ' + str(feedback_gain) + ')')
        plt.xlabel('i')
        plt.ylabel('Memory function m(i)')
        plt.grid(True)
        plt.tight_layout
    
    MCk_val /= it
    MC_val = sum(MCk_val)
    return MC_val, MCk_val, MC_it
    
## plot stuff
plt.figure()
x = np.linspace(1, nodes, nodes)
plt.plot(x, MCk)
plt.ylim([0,1.05])
plt.title('Forgetting curve (D = ' + str(vir_nodes) + ', n_max = ' + str(N - nodes - skip) + ', gain = ' + str(feedback_gain) + ')')
plt.xlabel('i')
plt.ylabel('Memory function m(i)')
plt.grid(True)
plt.tight_layout

#MC_val, MCk_val, MC_it = validateMC(weights, 1)

#plt.savefig('../../../Resultaten/MC/MC_D' + str(vir_nodes) + 'N_' + str(N - nodes) + 'gain_' + str(feedback_gain) + '.svg')

plt.figure()
x2 = np.linspace(-5, 20, 1000)
y = res.graph['0']['transfer'][0](torch.from_numpy(x2)).numpy()
plt.plot(x2, y)
plt.grid(True)
plt.tight_layout


#plt.figure()
#ax1 = plt.subplot(3, 1, 1)
#ax1.plot(inpt_np[skip:skip+20*N])
#plt.xlabel('Time step')
#plt.ylabel('Input voltage (V)')
#plt.title('Input (electrode ' + str(input_electrode) + ')')
#plt.grid(True)
#
#ax2 = plt.subplot(3, 1, 2)
#ax2.plot(inpt_mask_np[skip:skip+20*N])
#plt.xlabel('Time step')
#plt.ylabel('Masked input voltage (V)')
#plt.title('Masked input')
#plt.grid(True)
#plt.tight_layout()
#
#ax3 = plt.subplot(3, 1, 3)
#ax3.plot(output[skip:skip+20*N])
#plt.xlabel('Time step')
#plt.ylabel('Output current (nA)')
#plt.title('Output current')
#plt.grid(True)
#plt.tight_layout()
#
#plt.figure()
#plt.plot(virout_np)
#plt.title('Virtual outputs')
#plt.grid(True)
#plt.tight_layout()
##
#x = np.linspace(skip+nodes+1, vir_nodes*N, vir_nodes*(N-skip-nodes))
##
##plt.figure()
##plt.plot(x, inpt_np)
##plt.plot(x, np.transpose(np.repeat(prediction, vir_nodes, axis=1)))
##plt.title('Delayed outputs')
##plt.grid(True)
##plt.tight_layout()
#
#plt.figure()
#plt.plot(x, inpt_np[vir_nodes*(skip+nodes):])
#plt.plot(x, np.repeat(prediction[0,:], vir_nodes))
#plt.plot(x, np.repeat(target[:,0], vir_nodes))