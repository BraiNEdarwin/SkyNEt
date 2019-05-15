

from SkyNEt.instruments import InstrumentImporter
import SkyNEt.modules.SaveLib as SaveLib

import config_test_cdaq as config

import time
import numpy as np
import matplotlib.pyplot as plt

nr_channels = 7
cdaq = InstrumentImporter.nidaqIO.IO_cDAQ(nr_channels=nr_channels)

#ramp_to_array = np.zeros(nr_channels)
#ramp_to_array[4] = 1.0
#cdaq.ramp(ramp_to_array, ramp_speed = 0.5)
#cdaq.ramp_zero(ramp_speed = 0.5)

# Initialize config object
#cf = config.experiment_config()
#set_frequency = 10

# create directory in which to save everything
#saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# arduino switch network
## Initialize serial object
#ser = InstrumentImporter.switch_utils.init_serial(cf.comport)
## Switch to device
#InstrumentImporter.switch_utils.connect_single_device(ser, cf.device)
## Status print
#print(f'Connected device {cf.device}')



ser = InstrumentImporter.switch_utils.init_serial('COM3')
# Switch to device
InstrumentImporter.switch_utils.connect_single_device(ser,7)

#state = np.zeros(nr_channels)
#state[0] = 0
#cdaq.set_state(state)
InstrumentImporter.switch_utils.close(ser)




#n=50
#data = np.zeros((7, n))
#data[0,:] = np.sin(np.linspace(0,np.pi/2,n))
#ramp = np.linspace(0,0.5, 50)
#data[0,:] = np.concatenate((ramp, 0.5-ramp))
#dataRamped =(data, set_frequency)

#dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(data, set_frequency)


#plt.plot(data[0])