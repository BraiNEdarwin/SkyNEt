import numpy as np
import itertools
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
        self.device = 'nidaq'  # Either nidaq or adwin

        # Define experiment
        self.amplification = 10
        self.TargetGen = 13 # defines which feature needs to be extraction from 0-15
        self.generations = 100
        self.generange = [[-1500,1500], [-1500, 1500], [-1500, 1500]]

        # Specify either partition or genomes
        #self.partition = [5, 5, 5, 5, 5]
        self.genomes = 25
        self.ntest = 1
        self.measurelength = 100

        # Documentation
        self.genelabels = ['CV1','CV2','CV3']

        # Save settings
        self.filepath = r'D:\Data\Bram\Feature_extracion_2\\'  #Important: end path with double backslash
        self.name = '1101'

        self.Fitness = self.Fitness_extractiondiff
        self.InputGen = self.Fe_input

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

    def Fitness_extractiondiff(self, output, marker):
        I_std = np.zeros([16])
        I_average = np.zeros([16])

        #find the average current and standard deviation of all outputs and without the desired feature. 
        for i in range(16):
            I_std[i] = np.std(output[i])
            I_average[i] = np.average(output[i])
        I_other = np.delete(I_average, marker)
        I_otherstd = np.delete(I_std, marker)

        indexh = np.argmax(I_other+I_otherstd)
        indexl = np.argmin(I_other-I_otherstd)

        #distinguis if the feature will have to go for a positive or negative extractor and check how far it is away from this feature. 
        #done for higher but negative and lower but positive ass well.
        if I_average[marker]-I_std[marker]<I_other[indexh]+I_otherstd[indexh] and I_average[marker]+I_std[marker]>I_other[indexl]-I_otherstd[indexl]:
            sign = -1
        else:
            sign = 1


        if abs((I_average[marker]-I_std[marker])-(I_other[indexh]+I_otherstd[indexh]))<abs((I_average[marker]+I_std[marker])-(I_other[indexl]-I_otherstd[indexl])):
            F = abs((I_average[marker]-I_std[marker])-(I_other[indexh]+I_otherstd[indexh]))*sign/abs(I_other[indexh]-I_other[indexl]) 
        else: 
            F = abs((I_average[marker]+I_std[marker])-(I_other[indexl]-I_otherstd[indexl]))*sign/abs(I_other[indexh]-I_other[indexl])


        #Fitness for when positive needs to be highest and negative needs to be lowest.
        # if I_average[marker] > 0 
        #     F = I_average[marker]+I_std[marker]-I_other[indexh]-I_otherstd[indexh]
        # else 
        #     F = I_average[marker]-I_std[marker]-I_other[indexl]+I_otherstd[indexl]


        for i in range(len(output)): 
            for j in range(len(output[0])):
                if(abs(output[i][j])>3.1*10):
                    return -100

        return F


    def Fitness_extractionvar(self, output, marker):
        I_std = np.zeros([16])
        I_average = np.zeros([16])

        #find the average current and standard deviation of all outputs and without the desired feature. 
        for i in range(16):
            I_std[i] = np.std(output[i])
            I_average[i] = np.average(output[i])
        I_other = np.delete(I_average, marker)
        I_otherstd = np.delete(I_std, marker)

        indexh = np.argmax(I_other+I_otherstd)
        indexl = np.argmin(I_other-I_otherstd)

        #distinguis if the feature will have to go for a positive or negative extractor and check how far it is away from this feature. 
        #done for higher but negative and lower but positive ass well.
        if I_average[marker]-I_std[marker]<I_other[indexh]+I_otherstd[indexh] and I_average[marker]+I_std[marker]>I_other[indexl]-I_otherstd[indexl]:
            sign = -1
        else:
            sign = 1


        if abs((I_average[marker]-I_std[marker])-(I_other[indexh]+I_otherstd[indexh]))<abs((I_average[marker]+I_std[marker])-(I_other[indexl]-I_otherstd[indexl])):
            F = abs(I_average[marker]-I_std[marker])**2/(np.average(I_other)+np.var(I_other))
        else: 
            F = abs(I_average[marker]+I_std[marker])**2/(np.average(I_other)+np.var(I_other))


        #Fitness for when positive needs to be highest and negative needs to be lowest.
        # if I_average[marker] > 0 
        #     F = I_average[marker]+I_std[marker]-I_other[indexh]-I_otherstd[indexh]
        # else 
        #     F = I_average[marker]-I_std[marker]-I_other[indexl]+I_otherstd[indexl]


        for i in range(len(output)): 
            for j in range(len(output[0])):
                if(abs(output[i][j])>3.1*10):
                    return -100

        return F

    def Fe_input(self):
        a = [0, 1]
        inp = np.array(list(itertools.product(*[a,a,a,a])))
        return inp