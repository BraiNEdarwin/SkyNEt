import numpy as np
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
#        self.comport = 'COM3'  # COM port for the ivvi rack
#        self.device = 'nidaq'  # Either nidaq or adwin
        
        # switch network
        self.switch_comport = 'COM3'
        self.nr_channels = 7
        self.switch_device = 6

        # Define experiment
        self.amplification = 1
        self.TargetGen = self.XOR
        self.generations = 5
        self.generange = [[-2000,2000], [-2000, 2000], [-2000, 2000], [-2000, 2000], [-2000,2000], [-500, 500]]
        self.Fitness = self.lennart_fit
        # Specify either partition or genomes
        self.partition = [5, 5, 5, 5, 5]
#        self.genomes = 10

        # Documentation
        self.genelabels = ['CV0/E2','CV1/E3','CV2/E5','CV3/E6', 'Input scaling']

        # Save settings
        self.filepath = r'D:\Rik\web_boolean\\'  #Important: end path with double backslash
        self.name = 'XOR'

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
    def BoolInput(self):
        t = np.arange(4)
        x = np.zeros((2,4))
        x[0, [1,3]] = np.ones(2)
        x[1, [2,3]] = np.ones(2)
        w = np.ones(4,dtype=bool)
        target = np.zeros(4)
        target[3] = 1.0
        return [t, x[0], x[1], w]
    
    def AND(self):
        t = np.arange(4)
        x = np.zeros(4)
        x[-1] = 0.
        return t, x
    def OR(self):
        t = np.arange(4)
        x = np.ones(4)
        x[0] = 0.
        return t, x
    def XOR(self):
        t = np.arange(4)
        x = np.ones(4)
        x[0] = 0.
        x[-1] = 0.
        return t, x
    def NAND(self):
        t,x = self.AND()
        return t, 1-x
    def NOR(self):
        t,x = self.OR()
        return t, 1-x
    def XNOR(self):
        t,x = self.XOR()
        return t, 1-x
    
    
    def corr_fit(self, output, target,clpval=3.55):
        # if np.any(np.abs(output)>clpval*self.amplification):
        #     #print(f'Clipping value set at {clpval}')
        #     corr = -1
        # else:
        x = output[:,np.newaxis]
        y = target[:,np.newaxis]
        X = np.stack((x, y), axis=0)[:,:,0]
        corr = np.corrcoef(X)[0,1]
#        print('corr_fit')
        return corr

    def lennart_fit(self, output, target, w, *args, **kwargs):
        # this function ignores w, and should only be used with 4 points
        corr = self.corr_fit(output, target)
        indices_high = target==1
        indices_low = target==0

        # abs_diff = max(output)-min(output)
        lowest_ones = np.min(output[indices_high])
        highest_zeros = np.max(output[indices_low])
        diff = lowest_ones-highest_zeros
        # return (np.log(1+diff))/(1e-6+np.mean((output_scaled-target)**2))
        fitness = (corr+1)/2/(1+np.exp(-diff+1))
        # print(corr, diff, 1/(1+np.exp(-diff+1)), fitness)
        return fitness
    
    def marx_fit(self, output, target, w, clpval = 3.55):
        if np.any(np.abs(output)>clpval*self.amplification):
            #print(f'Clipping value set at {clpval}')
            return -1
            
        corr = self.corr_fit(output, target)
         # Apply weights w
        indices = np.argwhere(w)  #indices where w is nonzero (i.e. 1)
        target_weighed = np.zeros(len(indices))
        output_weighed = np.zeros(len(indices))
        for i in range(len(indices)):
            target_weighed[i] = target[indices[i][0]]
            output_weighed[i] = output[indices[i][0]]
        # Determine normalized separation
        indices1 = np.argwhere(target_weighed)  #all indices where target is nonzero
        x0 = np.empty(0)  #list of values where x should be 0
        x1 = np.empty(0)  #list of values where x should be 1
        for i in range(len(target_weighed)):
            if(i in indices1):
                x1 = np.append(x1, output_weighed[i])
            else:
                x0 = np.append(x0, output_weighed[ i])
        Q = np.abs(min(x1) - max(x0))
        return corr*(1/(1+np.exp(-Q+1)))/(1-corr)