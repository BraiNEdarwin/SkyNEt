# -*- coding: utf-8 -*-
"""
Spyder Editor

Generating synthetic data sets with nonlinear boundaries to feed SkyNEt
"""

import numpy as np
from matplotlib import pyplot as plt
import data_generators as dg
import pdb
#Create N samples of 2-dim uniformely distributed features
N = 10000
eps = 0.4
sample_0, sample_1 = dg.ring(N,R_out=0.8, R_in=0.3,epsilon=eps)
#Subsample the largest class
nr_samples = min(len(sample_0),len(sample_1))
max_array = max(len(sample_0),len(sample_1))
indices = np.random.permutation(max_array)[:nr_samples]
if len(sample_0) == max_array:
    sample_0 = sample_0[indices]
else: 
    sample_1 = sample_1[indices]

#Sort samples within each class wrt the values of input x (i.e. index 0)
sorting_indx0 = np.argsort(sample_0,axis=0)[:,0]
sorting_indx1 = np.argsort(sample_1,axis=0)[:,0]
xsorted_smpl0 = sample_0[sorting_indx0]
xsorted_smpl1 = sample_1[sorting_indx1]

#Filter by positive and negative values of y-axis
nonnyxsrtd_smpl0 = xsorted_smpl0[xsorted_smpl0[:,1]>=0]
nyxsrtd_smpl0 = xsorted_smpl0[xsorted_smpl0[:,1]<0]
nonnyxsrtd_smpl1 = xsorted_smpl1[xsorted_smpl1[:,1]>=0]
nyxsrtd_smpl1 = xsorted_smpl1[xsorted_smpl1[:,1]<0]

#Reverse neg. y inputs
nyxsrtd_smpl0 = nyxsrtd_smpl0[::-1]
nyxsrtd_smpl1 = nyxsrtd_smpl1[::-1]

# Define input variables and their target
inp0 = np.concatenate((nonnyxsrtd_smpl0,nyxsrtd_smpl0))
inp1 = np.concatenate((nonnyxsrtd_smpl1,nyxsrtd_smpl1))
inp = np.concatenate((inp0,inp1))
target = np.concatenate((np.zeros_like(inp0[:,0]),np.ones_like(inp1[:,0])))

plt.figure()
plt.subplot(1,2,1)
plt.plot(inp[:,0],'.-')
plt.plot(inp[:,1],'.-r')
plt.plot(target,'k',label='target')
plt.title('Waveform of inputs')
plt.legend()
plt.subplot(1,2,2)
plt.plot(inp0[:,0],inp0[:,1],'.',label='class 0')
plt.plot(inp1[:,0],inp1[:,1],'.r',label='class 1')
plt.title('2D representation of classes')
plt.legend()
plt.show()

#np.savez(r'../experiments/2D_binary_classification/Ring/Class_data_%1.2f' % eps ,
#         inp_wvfrm = inp, target = target,
#         inp_cl0 = inp0, inp_cl1 = inp1)

#plt.figure()
#plt.subplot(2,1,1)
#plt.plot(nonnyxsrtd_smpl0[:,0], label = 'x-input',color = 'b')
#plt.plot(nyxsrtd_smpl0[:,0], color = 'b')
#plt.plot(nonnyxsrtd_smpl0[:,1], '.-', label = '+y input')
#plt.plot(nyxsrtd_smpl0[:,1], '.-', label = '-y input')
#plt.title('class 0')
#plt.legend()
#plt.subplot(2,1,2)
#plt.plot(nonnyxsrtd_smpl1[:,0], label = 'x-input',color = 'b')
#plt.plot(nyxsrtd_smpl1[:,0], color = 'b')
#plt.plot(nonnyxsrtd_smpl1[:,1], '.-',label = '+y input')
#plt.plot(nyxsrtd_smpl1[:,1], '.-',label = '-y input')
#plt.title('class 1')
#plt.legend()
#plt.show()