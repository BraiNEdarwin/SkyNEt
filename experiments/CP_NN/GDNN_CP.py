#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:43:16 2019

@author: hruiz
"""
import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt

from SkyNEt.modules.Nets.CPNet import CPNet
#
#np.random.seed(333)
#torch.manual_seed(23)
########################### LOAD NN & DATA ########################################
main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/'
data_dir = main_dir+'2018_08_07_164652_CP_FullSwipe/'
#assert False, '!!'
## Load Inputs and Targets
data_dir2 = '/home/hruiz/Documents/PROJECTS/DARWIN/Code/Archive/Evolution_NN_CP_2InpV/2018_11_02_142235_CP_inputs_and_targets/'
data = np.load(data_dir2+'2018_11_02_142235_CP_inputs_and_targets.npz')
inputs = data['inputs']
targets = data['target'] + 1

plt.figure(figsize=(8,8))
plt.subplot(211)
plt.plot(inputs,linewidth=4)
plt.legend(['Input 1','Input 2'])
plt.title('Input and target values of Control Problem',fontsize=18)
plt.subplot(212)
plt.plot(targets,'ko-')
plt.legend(['Targets'])
plt.xlabel('Ordered data',fontsize=18)
plt.tight_layout()
plt.show()

#Load net
net = CPNet( data_dir + 'lr2e-4_eps400_mb512_20180807CP.pt')
# Return everything to cpu
net.model.cpu()
w=torch.tensor([1./4.,1./12.,1./5.])
net.loss_fn = torch.nn.CrossEntropyLoss(weight=w)
net.loss_fn.cpu()
# Prepare data for PyTorch
dtype = torch.FloatTensor
inputs = torch.from_numpy(inputs).type(dtype)
targets = torch.from_numpy(targets).type(torch.LongTensor)
x = Variable(inputs)
y = Variable(targets)
# Permute time point indices 
permutation = torch.randperm(y.data.shape[0])
Np_test = 50
test_indices = permutation[:Np_test]
train_indices = permutation[Np_test:]

x_train = x[train_indices]
y_train = y[train_indices]
x_test = x[test_indices]
y_test = y[test_indices]

plt.figure()
plt.plot(train_indices.numpy(),y_train.numpy(),'o')
plt.plot(test_indices.numpy(),y_test.numpy(),'ro')
plt.plot(targets.numpy(),'k')

data = [(x_train,y_train),(x_test,y_test)]
lr_cv, lr_scr, nr_epochs, batch_size = 1e-3, 0.4, 1000, 512 #for 0.22 val err w. seed: lr_scr=0.3,lr_cv=7e-3, mb=256
nruns = 4000
pred_voltages = np.zeros((nruns,5))
valErr_pred = np.zeros((nruns,nr_epochs))
classes_run = np.zeros((nruns,3150))
dboundary = np.zeros((nruns,2,int(0.1*nr_epochs)))
min_valerr = np.inf
for i in range(nruns):
    print('Run #',i)
    pred_voltages[i], valErr_pred[i] = net.predict(data, lr_cv, lr_scr,
                                         nr_epochs, batch_size,lambda_scr=1e-6, seed=False)
    y = net.predictor(x).cpu().detach().numpy()
    labels = np.arange(y.shape[1])
    classes = [labels[yn==np.max(yn)] for yn in y]
    classes_run[i] = np.array(classes).T
    dboundary[i,0] = -net.score_params[1,-int(0.1*nr_epochs):,0]/net.score_params[0,-int(0.1*nr_epochs):,0]
    dboundary[i,1] = -net.score_params[1,-int(0.1*nr_epochs):,2]/net.score_params[0,-int(0.1*nr_epochs):,2]
    print('Std of decision boundary :', np.std(dboundary[i,:,:],axis=-1))
    if valErr_pred[i,-1]<min_valerr:
        min_valerr = valErr_pred[i,-1]
    print('Min. Val. Error: ',min_valerr)


best_idx = np.arange(nruns)[valErr_pred[:,-1]==valErr_pred[:,-1].min()]
plt.figure()
plt.hist(valErr_pred[:,-1],100)
plt.figure()
plt.plot(classes_run[best_idx,:].T)
plt.plot(targets.cpu().data.numpy(),'k:')

best_set = np.arange(nruns)[valErr_pred[:,-1]<=0.13]
plt.figure()
plt.subplot(2,1,1)
plt.plot(valErr_pred[best_set].T)
plt.subplot(2,1,2)
plt.plot(classes_run[best_set,:].T)
plt.plot(targets.cpu().data.numpy(),'k:')

plt.figure()
plt.subplot(2,1,1)
plt.plot(dboundary[best_set,0,:].T)
plt.title('Decision boundary of p0 while learning')
plt.subplot(2,1,2)
plt.plot(dboundary[best_set,1,:].T)
plt.title('Decision boundary of p2 while learning')
                                                
print('Score weights \n', net.score_weights)
print('Score bias', net.score_bias)
#print('Predicted voltages: \n',pred_voltages)
plt.figure()
plt.plot(valErr_pred.T)
plt.figure()
plt.plot(classes_run.T)
plt.figure()
plt.plot(np.arange(nr_epochs),net.cv_epoch)

plt.figure()
plt.plot(targets.cpu().data.numpy(),'ko')
plt.plot(classes,'r')
plt.plot(y)
plt.legend(['targets','classes','p0','p1','p2'])
plt.show()

t = targets.cpu().data.numpy()
for i in range(3):
    count=np.zeros_like(t)
    count[t[:,0]==i]=1
    print('nr examples for class ', i,':',count.sum())

print('length of each case:',150)

plt.figure()
plt.subplot(121)
plt.plot(net.score_params[0])
plt.title('weights')
plt.subplot(122)
plt.plot(net.score_params[1])
plt.title('bias')

best_cv = np.array([0.00635616, 0.12560761, 0.86242163, 0.9589921, 0.9429861])
x = x.data.numpy()
cv = best_cv[:,np.newaxis].T*np.ones((x.shape[0],len(best_cv)))
ipts = np.concatenate((x,cv),axis=1)
ipts = torch.from_numpy(ipts).type(dtype)
out = net.outputs(ipts)

plt.figure()
#plt.plot(out,label='output')
plt.plot(targets.cpu().data.numpy(),'ko',label='targets')
plt.plot(targets.cpu().data.numpy(),'r-',label='classifier')
plt.plot(-3*out+2.5,label='trafo. output')
plt.plot(1.31*np.ones_like(out),'k:')
plt.plot(0.03*np.ones_like(out),'k:')
plt.legend()
plt.show()

file_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Code/packages/SkyNEt/experiments/CP_NN/'
name = 'Results_{}Runs_{}lrcv-{}lrscr-{}mb.npz'.format(nruns,lr_cv,lr_scr,batch_size)
np.savez(file_dir+name,lr_cv=lr_cv, lr_scr=lr_scr, nr_epochs=nr_epochs, batch_size=batch_size,nruns=nruns, 
         pred_voltages=pred_voltages,
         valErr_pred=valErr_pred,
         classes_run=classes_run,
         dboundary=dboundary)