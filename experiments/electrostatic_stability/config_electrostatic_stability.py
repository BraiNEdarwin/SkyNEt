import numpy as np
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):
    '''
    In1 and In2 are directly adjacent to the output.
    C1 and C2 are adjacent to these, voltages on these (electrostatic)
    electrodes is given by self.voltages.
    C3, C4, C5 are constantly biased throughout the experiment at
    self.controlVoltages.
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################

        # Specify CVs as list of lists.
        #self.controlVoltages = [[-750, -450, -150, 0, 150, 450, 750]]*5
        #self.input2 = [-900, -600, -300, 0, 300, 600, 900]
        #self.input1 = [-900,0,900]

        self.fieldHigh = 10000  # mV
        self.fieldLow = 1000  # mV
        self.fieldPoints =  20  # Amount of voltages between high and low
        self.fieldRepeat = 5  # Amount of measurements per field value
        self.fieldWait = 5  # s, waiting time after setting field value
        self.fieldVoltages = []
        fieldSpace = np.linspace(self.fieldHigh,
                                 self.fieldLow,
                                 self.fieldPoints)
        for i in range(self.fieldPoints):
            for j in range(self.fieldRepeat):
                self.fieldVoltages.append(fieldSpace[i])
        self.controlVoltages = [900, 900, 900]  # C3, C4, C5 
        # Assemble voltage list
        self.voltages = np.zeros(len(self.fieldVoltages), 2)
        self.voltages[:, 0] = self.fieldVoltages
        self.voltages[:, 1] = -self.voltages[:, 0]
        self.voltages = self.voltages/5  # compensate for gain
        self.gridElectrodes = len(self.controlVoltages)
        self.waveElectrodes = 2

        self.factor = 2
        self.freq2 = np.array([5,7,13,17,19]) # 2,np.pi,
        self.freq = np.sqrt(self.freq2[:self.waveElectrodes])*self.factor
        self.sampleTime = 5 # Sample time of the sine waves for one grid point (in seconds)

        self.fs = 1000
        self.transientTest = True
        self.n = 50

        self.amplification = 10  # nA/V
        self.Vmax = 0.9 # Maximum amount of voltage for the inputs

        self.keithley_address = 'GPIB0::17::INSTR'
        self.device = 'nidaq'
        #                               Summing module S2d      Matrix module           device
        self.electrodeSetup = [['ao0',1,2,3,4,5,'ao1','out'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        # Save settings
        self.filepath = r'D:\\data\\Mark\\wave_grid\\'
        self.name = 'sine_freq_test_factor_'+str(self.factor)
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        
        
        self.t = np.arange(0, self.sampleTime, 1/self.fs)
        self.phase = 2*np.pi*np.random.rand(self.waveElectrodes,1)

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
