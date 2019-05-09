

from SkyNEt.instruments import InstrumentImporter
import SkyNEt.modules.SaveLib as SaveLib

import config_test_cdaq as config

import time
import numpy as np
import matplotlib.pyplot as plt


# Initialize config object
cf = config.experiment_config()

# create directory in which to save everything
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# arduino switch network
# Initialize serial object
ser = InstrumentImporter.switch_utils.init_serial(cf.comport)
# Switch to device
InstrumentImporter.switch_utils.connect_single_device(ser, cf.device)
# Status print
print(f'Connected device {cf.device}')


data = np.zeros((7, 100))
ramp = np.linspace(0,0.5, 50)
data[0,:] = np.concatenate((ramp, 0.5-ramp))
dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(data, cf.set_frequency)


plt.plot(data[0])