# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:43:13 2018
Fit a distribution to the data and create a histogram of an estimated 
distribution sample and the real data.


@author: Mark Boon
"""

# Imports:
import scipy
import scipy.stats
import matplotlib.pyplot as plt



class Distribution(object):
    
    def __init__(self,dist_names_list = []):
        self.dist_names = ['norm','lognorm','expon','exponpow','cauchy','maxwell','laplace']
        self.dist_results = []
        self.params = {}
        
        self.DistributionName = ""
        self.PValue = 0
        self.Param = None
        
        self.isFitted = False
        
        
    def Fit(self, y):
        """ Determines how good distributions fit the data using a 
        Kolmogorov-Smirnov test"""
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y) # Uses MLE to find the fit for the parameters
            
            self.params[dist_name] = param
            #Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args = param);
            self.dist_results.append((dist_name,p))

        #select the best fitted distribution
        sel_dist,p = (max(self.dist_results, key=lambda item:item[1]))
        #store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p
        
        self.isFitted = True
        return self.DistributionName, self.PValue
    
    
    def Random(self, n = 1):
        """ Generates random numbers according to the fitted distribution"""
        if self.isFitted:
            dist_name = self.DistributionName
            param = self.params[dist_name]
            #initiate the scipy distribution
            dist = getattr(scipy.stats, dist_name)
            return dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
        else:
            raise ValueError('Must first run the Fit method.')
            
            
    def Plot(self, y, bins = 20):
        """Visualize the results"""
        x = self.Random(n=len(y))
        plt.hist(x, bins, alpha = 0.5, label='Fitted', normed=True)
        plt.hist(y, bins, alpha = 0.5, label='Actual', normed=True)
        plt.legend(loc='upper right')
