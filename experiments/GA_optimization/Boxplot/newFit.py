#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:03:17 2019

@author: annefleur
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import time 
from matplotlib import pyplot as plt


no_measurements = 100
dat_list = []
for i in range(no_measurements):
    dat_list.append('meas'+str(i)+'/')

accuracy = []
binlabels = []
capacity = []
fitness = []
found = []
genes =[]
output = []
target = []
found_gen = []
threshold = []


#directory1 = r'../../../test/evolution_test/VCdim_testing5/Allgates/Current/'
#rand = False 
mutation = 1
show_acc = True
if mutation ==1: 
    directory1 =  r'../../../test/evolution_test/VCdim_testing5/Verification_data_ramp_new2/'
    
elif mutation == 2:
    directory1 = r'../../../../test/evolution_test/VCdim_testing5/Round9/'
    meas_list = ['Found_gen-4/']
    
    


#for i in range(0,len(dat_list)):



for i in range(len(dat_list)):
    file = dat_list[i]
    with np.load(directory1 + file + 'Results.npz') as data:
        accuracy.append(data['accuracy_classifier'])
        binlabels.append(data['binary_labels'])
        capacity.append(data['capacity'])
        fitness.append(data['fitness_classifier'])
        found.append(data['found_classifier'])
        genes.append(data['genes_classifier'])
        output.append(data['output_classifier'])
        target.append(data['target_classifier'])
        found_gen.append(data['end_classifier'])
        threshold.append(data['threshold'])
     
          
if show_acc: 
    
    acc = np.array(accuracy) 
else: 
    acc= np.array(found_gen)
acc[acc>80] = 100
acc[acc==0.992]=1      
subplot_no = 231
plt.figure()
if mutation == 1:
    for i in range(len(binlabels[0])):
        ax = plt.subplot(subplot_no)
    #    temp = np.vstack((acc[:20,i],acc[60:80,i],acc[120:140,i],acc[180:200,i],acc[240:260,i],acc[300:320,i],acc[360:380,i],acc[420:440,i],acc[480:500,i] )).T  
    #        temp = np.vstack((acc[:30,i],acc[60:90,i],acc[120:150,i],acc[180:210,i],acc[240:270,i],acc[300:330,i],acc[360:390,i],acc[420:450,i],acc[480:510,i] )).T
    #        temp = np.vstack((acc[:40,i],acc[60:100,i],acc[120:160,i],acc[180:220,i],acc[240:280,i],acc[300:340,i],acc[360:400,i],acc[420:460,i],acc[480:520,i] )).T
    #        temp = np.vstack((acc[:50,i],acc[60:110,i],acc[120:170,i],acc[180:230,i],acc[240:290,i],acc[300:350,i],acc[360:410i],acc[420:470,i],acc[480:530,i] )).T
        temp = np.vstack((acc[1:,i]))
        frame = pd.DataFrame(temp, columns=[1])
        medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
        meanlineprops = dict(linestyle='--', linewidth=1, color='black')
        if show_acc: 
            
            ax = sns.boxplot( data=frame, color="red",medianprops=medianprops,meanprops=meanlineprops, meanline=True,showmeans=True)
        else: 
            ax = sns.boxplot( data=frame, color="skyblue",medianprops=medianprops,meanprops=meanlineprops, meanline=True,showmeans=True)
        ax.set_xticklabels([-4,-3,-2,-1,0,1])
        if show_acc:
            ax.set(ylim=(0, 1.01))
            plt.ylabel('Accuracy',fontsize=20)
        else: 
            ax.set(ylim=(0, 101))
            plt.ylabel('Generation found',fontsize=20)
        plt.xlabel('A',fontsize=20)
        plt.show()
        plt.title(binlabels[0][i],fontsize=20)
        subplot_no = subplot_no +1 
elif mutation ==2:
    for i in range(len(binlabels[0])):
        ax = plt.subplot(subplot_no)
    #    temp = np.vstack((acc[:20,i],acc[60:80,i],acc[120:140,i],acc[180:200,i],acc[240:260,i],acc[300:320,i],acc[360:380,i],acc[420:440,i],acc[480:500,i] )).T  
    #        temp = np.vstack((acc[:30,i],acc[60:90,i],acc[120:150,i],acc[180:210,i],acc[240:270,i],acc[300:330,i],acc[360:390,i],acc[420:450,i],acc[480:510,i] )).T
    #        temp = np.vstack((acc[:40,i],acc[60:100,i],acc[120:160,i],acc[180:220,i],acc[240:280,i],acc[300:340,i],acc[360:400,i],acc[420:460,i],acc[480:520,i] )).T
    #        temp = np.vstack((acc[:50,i],acc[60:110,i],acc[120:170,i],acc[180:230,i],acc[240:290,i],acc[300:350,i],acc[360:410i],acc[420:470,i],acc[480:530,i] )).T
        temp = np.vstack((acc[:,i]))
        frame = pd.DataFrame(temp)
        medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
        meanlineprops = dict(linestyle='--', linewidth=1, color='black')
        ax = sns.boxplot( data=frame, color="red",medianprops=medianprops,meanprops=meanlineprops, meanline=True,showmeans=True)
        ax.set_xticklabels([-4])
        ax.set(ylim=(0, 1.01))
        plt.xlabel('A',fontsize=20)
        plt.ylabel('Accuracy',fontsize=20)
        plt.show()
        plt.title(binlabels[0][i],fontsize=20)
        subplot_no = subplot_no +1 


print(len(acc[:100,:][acc[:100,:]<0.9]))
print(len(acc[100:200,:][acc[100:200,:]<0.9]))
print(len(acc[200:300,:][acc[200:300,:]<0.9]))
print(len(acc[300:400,:][acc[300:400,:]<0.9]))
print(len(acc[400:500,:][acc[400:500,:]<0.9]))
print(len(acc[500:600,:][acc[500:600,:]<0.9]))
#print(len(acc[600:700,:][acc[600:700,:]<0.9]))
print(len(acc[700:800,:][acc[700:800,:]<0.9]))


print(len(acc[:100,:3][acc[:100,:3]<0.9])+len(acc[:100,4:][acc[:100,4:]<0.9]))
print(len(acc[100:200,:3][acc[100:200,:3]<0.9])+len(+acc[100:200,4:][acc[100:200,4:]<0.9]))
print(len(acc[200:300,:3][acc[200:300,:3]<0.9])+len(+acc[200:300,4:][acc[200:300,4:]<0.9]))
print(len(acc[300:400,:3][acc[300:400,:3]<0.9])+len(+acc[300:400,4:][acc[300:400,4:]<0.9]))
print(len(acc[400:500,:3][acc[400:500,:3]<0.9])+len(+acc[400:500,4:][acc[400:500,4:]<0.9]))
print(len(acc[500:600,:3][acc[500:600,:3]<0.9])+len(+acc[500:600,4:][acc[500:600,4:]<0.9]))
#print(len(acc[600:700,:3][acc[600:700,:3]<0.9])+len(+acc[600:700,4:][acc[600:700,4:]<0.9]))
#print(len(acc[700:800,:3][acc[700:800,:3]<0.9])+len(+acc[700:800,4:][acc[700:800,4:]<0.9]))


plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

  
