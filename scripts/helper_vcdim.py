#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:13:13 2019

@author: hruiz
"""
from helper_vcdim import *
import numpy as np
from matplotlib import pyplot as plt
import pprint as pp
import os
from cycler import cycler
import pdb

def get_badcases(data, threshold_fit = 0.9, threshold_acc = 0.8):
    binary_labels = data['binary_labels']
    fitness_classifier =  data['fitness_classifier']
    accuracy_classifier = data['accuracy_classifier']
    
    low_fitness = np.arange(len(binary_labels))[fitness_classifier<threshold_fit]   
    z=zip(binary_labels[low_fitness].tolist(),fitness_classifier[low_fitness].tolist())
    lowfit_list = [x for x in z]
    print('Low fitness cases:')
    pp.pprint(lowfit_list)
    
    low_accuracy = np.arange(len(binary_labels))[accuracy_classifier<threshold_acc]
    z=zip(binary_labels[low_accuracy].tolist(),accuracy_classifier[low_accuracy].tolist())
    lowacc_list = [x for x in z]
    print('Low accuracy cases:')
    pp.pprint(lowacc_list)
    
    lowfit_hiacc = [n for n in low_fitness if n not in low_accuracy]
    z=zip(binary_labels[lowfit_hiacc].tolist(),fitness_classifier[lowfit_hiacc].tolist())
    lfha_list = [x for x in z]
    print('Low fit, high acc:')
    pp.pprint(lfha_list)
    
    d = {'low acc': lowacc_list, 'low acc index': low_accuracy, 
         'low fit' : lowfit_list,'low fit index': low_fitness, 
         'lfha' : lfha_list, 'lfha index' : lowfit_hiacc}
    return d 

def get_profile(dir_file, top=5):
    learning_profile = {}
    var_genes = []
    max_fitness = []
    var_output = []
    top_genes = []
    for dirpath, dirnames, filenames in os.walk(dir_file):
        if dirnames:
            labels = [dirs.split('-')[-1] for dirs in dirnames]
            print('Labels extracted: \n', labels)
        for file in filenames:
            if file in ['data.npz']:
                path = os.path.join(dirpath, file)
                data = np.load(path)
                max_fitness.append(np.max(data['fitness'],axis=1))
                # Get sorting indices of top 5 genomes
                indices = [np.argsort(x) for x in data['fitness']]
                top_genomes = [x[-top:] for x in indices]
                # Get top genes and their outputs
                topg = [data['genes'][i,x] for i,x in enumerate(top_genomes)]
                top_genes.append(topg)
                var_genes.append(np.var(topg,axis=1))
                top_out = [data['output'][i,x] for i,x in enumerate(top_genomes)]
                varo = np.var(top_out,axis=1)
                var_output.append(np.sum(varo,axis=-1))
#                pdb.set_trace()                
                data.close()
    learning_profile['top genes'] = np.asarray(top_genes)
    learning_profile['var top_genes'] = np.asarray(var_genes).T
    learning_profile['max fitness'] = np.asarray(max_fitness).T
    learning_profile['var top_output'] = np.asarray(var_output).T
    learning_profile['labels'] = labels
    return learning_profile

def plot_top_genes(lr_profile, labels, list_gen=[0]):
    assert labels in lr_profile['labels'], 'No cases assigned!'
    case = [x==labels for x in lr_profile['labels']]
    nr_top = lr_profile['top genes'].shape[2]
    assert len(list_gen)<nr_top, 'Length of list of genes must be at most %s' % nr_top
    plt.figure()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    c = prop_cycle.by_key()['color']
    linestyle_cycler = cycler('linestyle',['-',':','-.','--'])
    plt.suptitle('Learning profile of best %s genome for %s' % (str(list_gen),str(labels)))
    ax5 = plt.subplot2grid((3,2),(0,0))
    ax5.set_prop_cycle(linestyle_cycler)
    plt.title('Top Gene #1')
    ax6 = plt.subplot2grid((3,2),(0,1))
    ax6.set_prop_cycle(linestyle_cycler)
    plt.title('Top Gene #2')
    ax7 = plt.subplot2grid((3,2),(1,0))
    ax7.set_prop_cycle(linestyle_cycler)
    plt.title('Top Gene #3')
    ax8 = plt.subplot2grid((3,2),(1,1))
    ax8.set_prop_cycle(linestyle_cycler)
    plt.title('Top Gene #4')
    ax9 = plt.subplot2grid((3,2),(2,0))
    ax9.set_prop_cycle(linestyle_cycler)
    plt.title('Top Gene #5')
    ax10 = plt.subplot2grid((3,2),(2,1))  
    plt.title('Variance of Top %s Genes' % nr_top)
#    pdb.set_trace()                
    for n in list_gen:
        ax5.plot(lr_profile['top genes'][case,:,n,0].T,color=c[0])
        ax6.plot(lr_profile['top genes'][case,:,n,1].T,color=c[1])
        ax7.plot(lr_profile['top genes'][case,:,n,2].T,color=c[2])
        ax8.plot(lr_profile['top genes'][case,:,n,3].T,color=c[3])
        ax9.plot(lr_profile['top genes'][case,:,n,4].T,color=c[4])
    ax10.plot(lr_profile['var top_genes'][:,:,case].T[0],alpha=0.75)
    
def load_and_viz(dir_file, coi=None):
    data = np.load(dir_file+'Summary_Results.npz')
    lr_profile = get_profile(dir_file)
    bad_dict = get_badcases(data)
    
    gcl = data['genes_classifier']
    accl = data['accuracy_classifier']
    outcl = data['output_classifier']
    fitcl = data['fitness_classifier']
    binary_labels = data['binary_labels']
    
    
    plt.figure()
    ax = plt.subplot2grid((3,2),(0,0))
    plt.title('Fitness of fittest')
    ax1 = plt.subplot2grid((3,2),(0,1))
    plt.title('Final Output')
    ax2 = plt.subplot2grid((3,2),(1,0),rowspan=2)
    plt.ylabel('Gene value')
    plt.xticks([0,1,2,3,4],['#1','#2','#3','#4','#5'],rotation=90)
    plt.xlabel('Control Voltages')
    ax3 = plt.subplot2grid((3,2),(1,1))
    plt.xlabel('Fitness')
    plt.ylabel('Accuracy')
    ax4 = plt.subplot2grid((3,2),(2,1))
    plt.title('Low accuracy examples')
          
    for i,gene in enumerate(gcl[1:-1]):
        a = accl[i+1]
        c = tuple(gene)[:3]+(1,)
        lb = str(binary_labels[i+1]).lstrip('[').rstrip(']')
        lb = ''.join(lb.split())
        indx = [x==lb for x in lr_profile['labels']]
        ax.plot(lr_profile['max fitness'][:,indx],color=c)
        ax1.plot(outcl[i+1].T, color=c)
        if i+1 in bad_dict['low acc index']:
            ax2.plot(gene,'o', color=c)
            labels = np.asarray(bad_dict['low acc'])[i+1==bad_dict['low acc index']][0][0]
            ax3.plot(fitcl[i+1],accl[i+1],'o', color=c,label=str(lb))
            ax4.plot(outcl[i+1],color=c)
        else:
            ax2.plot(gene,'d', color=c)
            ax3.plot(fitcl[i+1],accl[i+1],'d', color=c) 
        
        if lb in coi: plot_top_genes(lr_profile,lb)
        
    ax3.legend(fontsize='x-small')
    plt.tight_layout()
    plt.show()
    
    data.close()