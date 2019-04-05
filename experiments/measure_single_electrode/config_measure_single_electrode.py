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
        self.measure_electrode = 7  # 0-7
        self.control_sequence = np.zeros((6, 8))
        self.control_sequence[0] = [173, 173, -165, 195, -446, -665, -608, 0]
        self.control_sequence[1] = [400, 400, -626, -72, 287, 146, -9, 0]
        self.control_sequence[2] = [501, 501, -566, 103, 74, 591, 144, 0]
        self.control_sequence[3] = [368,368,267,-46,-365,97,204, 0]
        self.control_sequence[4] = [736, 736, -73, 567, 521, 277, 203, 0]
        self.control_sequence[5] = [979, 979, -347, -1044, 713, 726, -22, 0]

        # Measure N points with interval wait_time
        self.N = 100
        self.wait_time = 1E-3

        # Save settings
        self.filepath = r'D:\Data\BramdW\D9\measure_single_electrode\all_gates_powermin_newfitness\\'  #Important: end path with double backslash
        self.name = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
        suffix = f'_electrode{self.measure_electrode}'
        for i in range(len(self.name)):
            self.name[i] = self.name[i] + suffix

        ################################################
        ################# OFF-LIMITS ###################
        ################################################

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
