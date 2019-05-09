from SkyNEt.config.config_class import config_class

class experiment_config(config_class):
    '''
    This is the config file for switching the switch network setup
    to a particular device. There are actually only two parameters:
    comport; string, e.g. 'COM3' the comport to which the arduino is 
                connected
    device; int, 0-7 indicating the device to which you wish to switch
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!

        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        
        # Arduino switch network
        self.comport = 'COM3'  # COM port of the arduino
        self.device = 1 # single device to be measured
        
#        self.set_frequency = 0.5