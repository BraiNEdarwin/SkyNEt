import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments.ADwin import adwinIO
from SkyNEt.instruments.niDAQ import nidaqIO
import numpy as np
import os
import config_IV as config

#load the information from the config class.
config = config.experiment_config()

# Initialize save directory.
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# degine the device input using the function in the config class.
Input = config.Sweepgen( config.v_high, config.v_low, config.n_points, config.direction)

#measure using the device specified in the config class.
if config.device == 'nidaq':
    Output = nidaqIO.IO(Input, config.fs)
elif config.device == 'adwin':
    adwin = adwinIO.initInstrument()
    Output = adwinIO.IO(adwin, Input, config.fs)
else:
    print('specify measurement device')

#save the Input and Output.
np.savez(os.path.join(saveDirectory, 'config.name'),Input=Input, Output=Output)

#plot the IV curve.
plt.figure()
plt.plot(Input[0:len(Output)], Output)
plt.show()




