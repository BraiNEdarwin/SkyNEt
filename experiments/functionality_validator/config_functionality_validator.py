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
        self.comport = 'COM3'  # COM port for the ivvi rack
        self.device = 'cDAQ'  # Either nidaq or adwin

        self.cv_amplification = 1
        #self.controlVoltages = np.load(r'D:\data\Mark\wave_search\paper_chip\2019_04_27_115357_train_data_2d_f_0_05\NN\gates\results_MSE_n_more_XNOR.npz')['CV'] # given in volts
        #self.controlVoltages = np.load(r'D:\data\Mark\wave_search\paper_chip\2019_04_27_115357_train_data_2d_f_0_05\NN\ring\results_MSE_n_tanh_ring_inverse.npz')['CV'] # given in volts
        self.controlVoltages = np.array([-1.1714, 0.3669, 0.4937, 0.0138, -0.6853])[np.newaxis,:]
        self.inputScaling = 1 #-1.0583    
        self.inputOffset = -0.5 #np.array([0., 0.])

        self.x = np.array([[0,0,1,1],[0,1,0,1]])* self.inputScaling + self.inputOffset
        #self.x = np.load(r'D:\data\Mark\ring_data\Ring_class_data_0.40.npz')['inp_wvfrm'].T 
        #self.x = self.x/np.max(np.abs(self.x)) * self.inputScaling + self.inputOffset[:,np.newaxis]
        # Define experiment
        self.postgain = 1
        self.amplification = 100  # nA/V

        self.fs = 1000
        self.pointlength = 100   # Amount of datapoints for a single sample
        self.rampT = int(self.fs/100)    # datapoints to ramp from one datapoint to the next

        ################################################
        ######### USER-SPECIFIC PARAMETERS #############
        ################################################

        ################# Save settings ################
        self.filepath = r'D:\data\Mark\predict\\'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))

        #                       Summing module S2d              Matrix module       on chip
        self.input_electrodes = [1,2]
        self.electrodeSetup = [['ao5','inp0','ao1''ao0','a02','inp4','ao6','out'],[1,3,5,6,11,13,15,17],[5,6,7,8,1,2,3,4]]
        self.name = 'MSE_n_tanh_XNOR'

	
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

        F = np.corrcoef(x_weighed, target_weighed)[0, 1]
        clipcounter = 0
        for i in range(len(x_weighed)):
            if(abs(x_weighed[i]) > 3.1*10):
                clipcounter = clipcounter + 1
                F = -100
        return F
