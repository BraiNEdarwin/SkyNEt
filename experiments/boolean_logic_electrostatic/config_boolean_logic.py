import numpy as np
import os
from SkyNEt.config.config_class import config_class

class experiment_config(config_class):
    '''This is the experiment configuration file for the boolean logic experiment.
    It also serves as a template for other experiments, so please model your work
    after this when you make new experiments.
    This experiment_config class inherits from config_class default values that are known to work well with boolean logic.
    You can define user-specific parameters in the construction of the object in __init__() or define
    methods that you might need after, e.g. a new fitness function or input and output generators.
    Remember if you define a new fitness function or generator, you have to redefine the self.Fitness,
    self.Target_gen and self.Input_gen in __init__()

    ----------------------------------------------------------------------------
    Description of general parameters
    ----------------------------------------------------------------------------
    comport; the COM port to which the ivvi rack is connected.
    amplification; specify the amount of nA/V. E.g. if you set the IVVI to 100M,
        then amplification = 10
    generations; the amount of generations for the GA
    generange; the range that each gene ([0, 1]) is mapped to. E.g. in the Boolean
        experiment the genes for the control voltages are mapped to the desired
        control voltage range.
    partition; this tells the GA how it will evolve the next generation.
        In order, this will make the GA evolve the specified number with
        - promoting fittest partition[0] genomes unchanged
        - adding Gaussian noise to the fittest partition[1] genomes
        - crossover between fittest partition[2] genomes
        - crossover between fittest partition[3] and randomly selected genes
        - randomly adding parition[4] genomes
    genomes; the amount of genomes in the genepool, speficy this parameter instead
        of partition if you don't care about the specific partition.
    genes; the amount of genes per genome
    mutationrate; the probability of mutation for each gene (between 0 and 1)
    fitnessavg; the amount of times the same genome is tested to obtain the fitness
        value.
    fitnessparameters; the parameters for FitnessEvolution (see its doc for
        specifics)
    filepath; the path used for saving your experiment data
    name; name used for experiment data file (date/time will be appended)

    ----------------------------------------------------------------------------
    Description of method parameters
    ----------------------------------------------------------------------------
    signallength; the length in s of the Boolean P and Q signals
    edgelength; the length in s of the edge between 0 and 1 in P and Q
    fs; sample frequency for niDAQ or ADwin

    ----------------------------------------------------------------------------
    Description of methods
    ----------------------------------------------------------------------------
    TargetGen; specify the target function you wish to evolve, options are:
        - OR
        - AND
        - NOR
        - NAND
        - XOR
        - XNOR
    Fitness; specify the fitness function, standard options are:
        - FitnessEvolution; standard fitness used for boolean logic
        - FitnessNMSE; normalised mean squared error
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM5'  # COM port for the ivvi rack
        self.device = 'adwin'  # Either nidaq or adwin
        
        # Total signal is 5*edgelength + 4*signallength
        self.edgelength = 0.1
        self.signallength = 0.2 
        self.extraweight = 0.1  # Add this amount of s of weight 0 to the signal bits
        
        self.Fitness = self.FitnessCorr

        # Define experiment
        self.postgain = 100
        self.amplification = 10
        self.InputGen = self.BetterInputGen
        self.TargetGen = self.XOR
        self.generations = 100
        self.fitnessavg = 1
        baseVoltage = 10
        # self.generange = [[-1000, 400], 
                           # [-baseVoltage*1000/5, baseVoltage*1000/5],
                           # [-baseVoltage*1000/5, baseVoltage*1000/5],
                           # [-baseVoltage*1000/5, baseVoltage*1000/5],
                           # [-baseVoltage*1000/5, baseVoltage*1000/5],                          
                           # [0, 20/5]]
        self.generange = [[-1000, 1000], 
                   [-1000, 1000],
                   [-1000, 1000],
                   [-1000, 1000],
                   [-1000, 1000],                          
                   [0, 1]]


        # Specify either partition or genomes
        #self.partition = [5, 5, 5, 5, 5]
        self.genomes = 25

        # Documentation
        self.genelabels = ['CV1/T1','CV2/T3','CV3/T11','CV4/T13','CV5/T15', 'Input scaling']

        ################################################
        ######### USER-SPECIFIC PARAMETERS #############
        ################################################

        ################# Save settings ################
        self.filepath = r'D:\data\BramdW\DNB8_BdW1\D9\regular_boolean\\'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))

        #                       Summing module S2d              Matrix module       on chip
        self.electrodeSetup = [[1,2,'ao0',3,'ao1',4,5,'out'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        self.name = 'XOR_illustration'

        ################################################
        ################# OFF-LIMITS ###################
        ################################################
        # Check if genomes parameter has been changed
        if(self.genomes != sum(self.default_partition)):
            if(self.genomes%5 == 0):
                self.partition = [int(self.genomes/5)]*5  # Construct equally partitioned genomes
            else:
                print('WARNING: The specified number of genomes is not divisible by 5.'
                      + ' The remaining genomes are generated randomly each generation. '
                      + ' Specify partition in the config instead of genomes if you do not want this.')
                self.partition = [self.genomes//5]*5  # Construct equally partitioned genomes
                self.partition[-1] += self.genomes%5  # Add remainder to last entry of partition

        self.genomes = int(sum(self.partition))  # Make sure genomes parameter is correct
        self.genes = int(len(self.generange))  # Make sure genes parameter is correct

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
    
    def FitnessCorr(self, x, target, W):
        '''
        This implements the fitness function
        F = self.fitnessparameters[0] * m / (sqrt(r) + self.fitnessparameters[3] * abs(c)) + self.fitnessparameters[1] / r + self.fitnessparameters[2] * Q
        where m,c,r follow from fitting x = m*target + c to minimize r
        and Q is the fitness quality as defined by Celestine in his thesis
        appendix 9
        W is a weight array consisting of 0s and 1s with equal length to x and
        target. Points where W = 0 are not included in fitting.
        '''

        #extract fit data with weights W
        indices = np.argwhere(W)  #indices where W is nonzero (i.e. 1)

        x_weighed = np.empty(len(indices))
        target_weighed = np.empty(len(indices))
        for i in range(len(indices)):
            x_weighed[i] = x[indices[i]]
            target_weighed[i] = target[indices[i]]
            
        # Get separation between 0 and 1
        separation = np.min(x_weighed[target_weighed==1]) - np.max(x_weighed[target_weighed==0])

        F = np.corrcoef(x_weighed, target_weighed)[0, 1]
        
        # Add separation if gate is valid gate
        #if(separation > 0):
        #F = separation
        clipcounter = 0
        for i in range(len(x_weighed)):
            if(abs(x_weighed[i]) > 3.1*10):
                clipcounter = clipcounter + 1
                F = -100
        return F
        
    def AND(self, final_edge = True):
        x = np.array([])
        t = np.array([])
        
        x_bin = [0, 0, 0, 1]
        
        n_edge = round(self.fs * self.edgelength)  # Amount of datapoints per edge
        n_signal = round(self.fs * self.signallength)  # Amount of datapoints per signal
        
        for i in range(4):
            # Add edge
            if(i == 0):
                x = np.append(x, np.linspace(0, x_bin[0], n_edge))
            else:
                x = np.append(x, np.linspace(x_bin[i-1], x_bin[i], n_edge))
                
            # Add signal
            x = np.append(x, np.ones(n_signal)*x_bin[i])
            
        if final_edge:
            x = np.append(x, np.linspace(x_bin[-1], 0, n_edge))
            t = np.linspace(0, (5*n_edge+4*n_signal)/self.fs, 5*n_edge+4*n_signal)
        else:
            t = np.linspace(0, (4*n_edge+4*n_signal)/self.fs, 4*n_edge+4*n_signal)
        
        
        return t, x
        
    def OR(self, final_edge = True):
        x = np.array([])
        t = np.array([])
        
        x_bin = [0, 1, 1, 1]
        
        n_edge = round(self.fs * self.edgelength)  # Amount of datapoints per edge
        n_signal = round(self.fs * self.signallength)  # Amount of datapoints per signal
        
        for i in range(4):
            # Add edge
            if(i == 0):
                x = np.append(x, np.linspace(0, x_bin[0], n_edge))
            else:
                x = np.append(x, np.linspace(x_bin[i-1], x_bin[i], n_edge))
                
            # Add signal
            x = np.append(x, np.ones(n_signal)*x_bin[i])
            
        if final_edge:
            x = np.append(x, np.linspace(x_bin[-1], 0, n_edge))
            t = np.linspace(0, (5*n_edge+4*n_signal)/self.fs, 5*n_edge+4*n_signal)
        else:
            t = np.linspace(0, (4*n_edge+4*n_signal)/self.fs, 4*n_edge+4*n_signal)
        
        
        return t, x

    def NAND(self, final_edge = True):
        x = np.array([])
        t = np.array([])
        
        x_bin = [1, 1, 1, 0]
        
        n_edge = round(self.fs * self.edgelength)  # Amount of datapoints per edge
        n_signal = round(self.fs * self.signallength)  # Amount of datapoints per signal
        
        for i in range(4):
            # Add edge
            if(i == 0):
                x = np.append(x, np.linspace(0, x_bin[0], n_edge))
            else:
                x = np.append(x, np.linspace(x_bin[i-1], x_bin[i], n_edge))
                
            # Add signal
            x = np.append(x, np.ones(n_signal)*x_bin[i])
            
        if final_edge:
            x = np.append(x, np.linspace(x_bin[-1], 0, n_edge))
            t = np.linspace(0, (5*n_edge+4*n_signal)/self.fs, 5*n_edge+4*n_signal)
        else:
            t = np.linspace(0, (4*n_edge+4*n_signal)/self.fs, 4*n_edge+4*n_signal)
        
        
        return t, x
        
    def NOR(self, final_edge = True):
        x = np.array([])
        t = np.array([])
        
        x_bin = [1, 0, 0, 0]
        
        n_edge = round(self.fs * self.edgelength)  # Amount of datapoints per edge
        n_signal = round(self.fs * self.signallength)  # Amount of datapoints per signal
        
        for i in range(4):
            # Add edge
            if(i == 0):
                x = np.append(x, np.linspace(0, x_bin[0], n_edge))
            else:
                x = np.append(x, np.linspace(x_bin[i-1], x_bin[i], n_edge))
                
            # Add signal
            x = np.append(x, np.ones(n_signal)*x_bin[i])
            
        if final_edge:
            x = np.append(x, np.linspace(x_bin[-1], 0, n_edge))
            t = np.linspace(0, (5*n_edge+4*n_signal)/self.fs, 5*n_edge+4*n_signal)
        else:
            t = np.linspace(0, (4*n_edge+4*n_signal)/self.fs, 4*n_edge+4*n_signal)
        
        
        return t, x

    def XOR(self, final_edge = True):
        x = np.array([])
        t = np.array([])
        
        x_bin = [0, 1, 1, 0]
        
        n_edge = round(self.fs * self.edgelength)  # Amount of datapoints per edge
        n_signal = round(self.fs * self.signallength)  # Amount of datapoints per signal
        
        for i in range(4):
            # Add edge
            if(i == 0):
                x = np.append(x, np.linspace(0, x_bin[0], n_edge))
            else:
                x = np.append(x, np.linspace(x_bin[i-1], x_bin[i], n_edge))
                
            # Add signal
            x = np.append(x, np.ones(n_signal)*x_bin[i])
            
        if final_edge:
            x = np.append(x, np.linspace(x_bin[-1], 0, n_edge))
            t = np.linspace(0, (5*n_edge+4*n_signal)/self.fs, 5*n_edge+4*n_signal)
        else:
            t = np.linspace(0, (4*n_edge+4*n_signal)/self.fs, 4*n_edge+4*n_signal)
        
        
        return t, x
                
    def XNOR(self, final_edge = True):
        x = np.array([])
        t = np.array([])
        
        x_bin = [1, 0, 0, 1]
        
        n_edge = round(self.fs * self.edgelength)  # Amount of datapoints per edge
        n_signal = round(self.fs * self.signallength)  # Amount of datapoints per signal
        
        for i in range(4):
            # Add edge
            if(i == 0):
                x = np.append(x, np.linspace(0, x_bin[0], n_edge))
            else:
                x = np.append(x, np.linspace(x_bin[i-1], x_bin[i], n_edge))
                
            # Add signal
            x = np.append(x, np.ones(n_signal)*x_bin[i])
            
        if final_edge:
            x = np.append(x, np.linspace(x_bin[-1], 0, n_edge))
            t = np.linspace(0, (5*n_edge+4*n_signal)/self.fs, 5*n_edge+4*n_signal)
        else:
            t = np.linspace(0, (4*n_edge+4*n_signal)/self.fs, 4*n_edge+4*n_signal)
        
        
        return t, x
        
    def BetterInputGen(self, final_edge = True):
        x = np.array([])
        y = np.array([])
        t = np.array([])
        w = np.array([])
        
        x_bin = [0, 1, 0, 1]
        y_bin = [0, 0, 1, 1]
        
        n_edge = round(self.fs * self.edgelength)  # Amount of datapoints per edge
        n_signal = round(self.fs * self.signallength)  # Amount of datapoints per signal
        n_extraweight = round(self.fs * self.extraweight)  # Amount of datapoints per signal
        
        w_signal = np.ones(n_signal)
        w_signal[:n_extraweight] = 0
        
        for i in range(4):
            # Add edge
            if(i == 0):
                x = np.append(x, np.linspace(0, x_bin[0], n_edge))
                y = np.append(y, np.linspace(0, y_bin[0], n_edge))
                w = np.append(w, np.zeros(n_edge))
            else:
                x = np.append(x, np.linspace(x_bin[i-1], x_bin[i], n_edge))
                y = np.append(y, np.linspace(y_bin[i-1], y_bin[i], n_edge))
                w = np.append(w, np.zeros(n_edge))
                
            # Add signal
            x = np.append(x, np.ones(n_signal)*x_bin[i])
            y = np.append(y, np.ones(n_signal)*y_bin[i])
            w = np.append(w, w_signal)
            
        if final_edge:
            x = np.append(x, np.linspace(x_bin[-1], 0, n_edge))
            y = np.append(y, np.linspace(y_bin[-1], 0, n_edge))
            w = np.append(w, np.zeros(n_edge))
            t = np.linspace(0, (5*n_edge+4*n_signal)/self.fs, 5*n_edge+4*n_signal)
        else:
            t = np.linspace(0, (4*n_edge+4*n_signal)/self.fs, 4*n_edge+4*n_signal)
        
        
        return t, x, y, w

