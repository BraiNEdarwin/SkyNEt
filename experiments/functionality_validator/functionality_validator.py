'''
In this script the control voltages corresponding to different functionalities is set and a measurement is performed.
'''

# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import config_functionality_validator as config
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter

# Other imports
import time
import numpy as np
import matplotlib.pyplot as plt

# Initialize config object
cf = config.experiment_config()

# Initialize input and target
t = cf.InputGen()[0]  # Time array
x = cf.x

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize instruments
try:
    ivvi = InstrumentImporter.IVVIrack.initInstrument(comport = cf.comport)
except:
    pass

# Initialize input and output arrays
allInputs = np.zeros((cf.controlVoltages.shape[0], 7, x.shape[1]*cf.pointlength + x.shape[1]*(cf.rampT+1)))
outputs = np.zeros((cf.controlVoltages.shape[0], x.shape[1]*cf.pointlength + x.shape[1]*(cf.rampT+1)))

# Set the ramping on the input signal
x_ramp = np.zeros((x.shape[0], x.shape[1]*cf.pointlength + x.shape[1]*(cf.rampT+1))) # +1 because of ramp at start and end.
for i in range(x.shape[0]):
    x_ramp[i,0:cf.rampT] = np.linspace(0,x[i,0],cf.rampT)
    x_ramp[i,cf.rampT: cf.rampT + cf.pointlength] = x[i,0]*np.ones(cf.pointlength)
    for j in range(1,x.shape[1]):
        x_ramp[i, (cf.rampT + cf.pointlength)*j: cf.pointlength*j + cf.rampT*(j+1)] = np.linspace(x[i,j-1],x[i,j],cf.rampT)
        x_ramp[i, cf.pointlength*j + cf.rampT*(j+1): (cf.pointlength + cf.rampT)*(j+1)] = x[i,j]*np.ones(cf.pointlength)

#plt.plot(x_ramp[0],x_ramp[1])
#plt.show()

#%% Measurement loop
# Set the DAC voltages
for i in range(0,cf.controlVoltages.shape[0]):
    print('Measuring case ' + str(i+1))
    
    # This part is not used to measure for the nidaq and adwin but saves all the voltages set on the electrodes per validation
    inputs = cf.controlVoltages[i,:][:,np.newaxis] * np.ones((cf.controlVoltages.shape[1], x_ramp.shape[1]))
    for j in range(x_ramp.shape[0]):
        inputs = np.insert(inputs, cf.input_electrodes[j], x_ramp[j], axis=0)
    allInputs[i,:,:] = inputs
        
    # Feed input to measurement devices
    if(cf.device == 'nidaq'):
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, cf.controlVoltages[i,:]*1000) # *1000 since it was saved in volts
        time.sleep(3)  # Wait after setting DACs
        outputs[i] = InstrumentImporter.nidaqIO.IO(x_ramp, cf.fs)     
                        
    elif(cf.device == 'adwin'):
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, cf.controlVoltages[i,:]*1000) # *1000 since it was saved in volts
        time.sleep(3)  # Wait after setting DACs
        adw = InstrumentImporter.adwinIO.initInstrument()
        outputs[i] = InstrumentImporter.adwinIO.IO(adw, x_ramp, cf.fs)        
        
    elif(cf.device == 'cDAQ'):
        print('cDAQ will be used to validate')
        outputs[i] = InstrumentImporter.nidaqIO.IO_cDAQ(inputs, cf.fs)
        
    else:
        print('Specify measurement device as either adwin, nidaq or cDAQ')

  # Save generation
SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                         t = t,
                         inputs = cf.controlVoltages,
                         x = x,
                         x_ramp = x_ramp,
                         cDAQ_inputs = allInputs,
                         outputs = outputs*cf.amplification)

# Plot output
plt.figure()
plt.plot(outputs.T*cf.amplification)
plt.show()

InstrumentImporter.reset(0, 0)
