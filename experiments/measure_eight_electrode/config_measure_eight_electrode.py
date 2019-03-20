import numpy as np
from SkyNEt.config.config_class import config_class

class experiment_config(config_class):
    '''
    This config file is for:
    This script applies control+input voltage configurations defined in the
    config file. One of these voltages is applied with the Keithley2400,
    such that the current through this electrode can be measured.
    By performing this measurement 7 times (all electrodes but the output),
    all flowing currents in a particular configuration can be indexed.

    IMPORTANT: All electrode voltages (except the output) are assumed
    to be supplied by the IVVI DACs in the following order:
    Input1   - DAC1 
    Input2   - DAC2
    Control1 - DAC3
    Control2 - DAC4
    Control3 - DAC5
    Control4 - DAC6
    Control5 - DAC7

    Then, say you wish to measure the current through Control1:
    - Unplug Control1 from DAC3
    - Plug Control1 into the ISO input
    - Set measure_electrode to 2 below
    - Finally give the full control sequence as:
        [max(in1), max(in2), C1, C2, ..., C5] in mV

    Optionally, you can supply an arbitrary Nx7 matrix of control 
    sequences and an N length list names to run this script in batch
    mode.
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM3'  # COM port for the ivvi rack
        self.measure_electrode = 6  # 0-6
        self.control_sequence = np.zeros((6, 8))
        self.control_sequence[0] = [500,500,-11.3,308.4,470.4,-777.9,385.7, 0]
        self.control_sequence[1] = [500,500,-724.6,-771.3,-141.7,-508.1,-284.7, 0]
        self.control_sequence[2] = [500,500,-826.9,-838.8,77.9,423.4,-46.6, 0]
        self.control_sequence[3] = [500,500,-891.9,-236.8,556.9,-49.6,244.0, 0]
        self.control_sequence[4] = [500,500,-802.9,-730.0,-363.4,511.7,-459.1, 0]
        self.control_sequence[5] = [500,500,56.4,347.6,-649.1,552.3,-804.5, 0]
        
        self.control_sequence = np.zeros((1, 8))
        self.control_sequence[0] = [500,500,-11.3,308.4,470.4,-777.9,385.7, 0]
        # self.control_sequence[1] = [500,500,-724.6,-771.3,-141.7,-508.1,-284.7, 0]
        # self.control_sequence[2] = [500,500,-826.9,-838.8,77.9,423.4,-46.6, 0]
        # self.control_sequence[3] = [500,500,-891.9,-236.8,556.9,-49.6,244.0, 0]
        # self.control_sequence[4] = [500,500,-802.9,-730.0,-363.4,511.7,-459.1, 0]
        # self.control_sequence[5] = [500,500,56.4,347.6,-649.1,552.3,-804.5, 0]
        
        self.resistance = 1E3  # Resistors in resistor box
        self.amplification = 10  # nA/V
        
        # Measure N points with interval wait_time
        self.N = 100
        self.fs = 1000

        # Save settings
        self.filepath = r'D:\Data\BramdW\measure_eight_electrode_test\\'  #Important: end path with double backslash
        self.gates = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
        self.name = 'FullBoolean'

        ################################################
        ################# OFF-LIMITS ###################
        ################################################

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
