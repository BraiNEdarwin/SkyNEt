'''
In this script the control voltages corresponding to different functionalities is set and a measurement is performed.
This is done withig 3 for loops:
1. for i in range generations for every generation rankes the fitnsessed of the genomes (control voltage sets) of the generation and used the GA to define the genomes of the next generation.
   After this the output with the best fitness as well as this fitness compared to the generation are plotted.
2. for j in range genomes for ever genome in a generation it sets the desisred control voltages on the DAC's of the IVVI-rack and scales the boulean logic input.
   Next, it uses either the adwin or nidaq to measure the boulean logic input output relation after which the coresponding fitness is calculated.
   Finally, the genomes and output of each genome is plotted.
3. for k in range genes is use to map the 0-1 generated genome to the generange at put it in the desired orded in the coltrolvoltage array. 
'''

# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.PlotBuilder as PlotBuilder
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
#outputs = np.zeros((6,530))
outputs = np.zeros((len(cf.controlVoltages), x.shape[1]*cf.pointlength + x.shape[1]*(cf.rampT+1)))
#%% Measurement loop

# Set the DAC voltages
for gates in range(0,1):
    print('Measuring gate ' + str(gates+1))
    InstrumentImporter.IVVIrack.setControlVoltages(ivvi, cf.controlVoltages*1000) # *1000 since it was saved in volts
    time.sleep(5)  # Wait after setting DACs

    # Set the ramping on the input signal

    x_ramp = np.zeros((x.shape[0], x.shape[1]*cf.pointlength + x.shape[1]*(cf.rampT+1))) # +1 because of ramp at start and end.
    for i in range(x.shape[0]):
        x_ramp[i,0:cf.rampT] = np.linspace(0,x[i,0],cf.rampT)
        x_ramp[i,cf.rampT: cf.rampT + cf.pointlength] = x[i,0]*np.ones(cf.pointlength)

    for i in range(x.shape[0]):
        for j in range(1,x.shape[1]):
            x_ramp[i, (cf.rampT + cf.pointlength)*j: cf.pointlength*j + cf.rampT*(j+1)] = np.linspace(x[i,j-1],x[i,j],cf.rampT)
            x_ramp[i, cf.pointlength*j + cf.rampT*(j+1): (cf.pointlength + cf.rampT)*(j+1)] = x[i,j]*np.ones(cf.pointlength)

    #plt.plot(x_ramp[0,:],x_ramp[1,:])
    #plt.show()


    # Feed input to measurement devices
    if(cf.device == 'nidaq'):
        outputs[gates] = InstrumentImporter.nidaqIO.IO(x_ramp, cf.fs)
    elif(cf.device == 'adwin'):
        adw = InstrumentImporter.adwinIO.initInstrument()
        outputs[gates] = InstrumentImporter.adwinIO.IO(adw, x_ramp, cf.fs)
    elif(cf.device == 'cDAQ'):
        outputs[gates] = InstrumentImporter.nidaqIO.IO_cDAQ(x_ramp, cf.fs)

    else:
        print('Specify measurement device as either adwin, nidaq or cDAQ')

  # Save generation
SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                         t = t,
                         inputs = cf.controlVoltages,
                         x = x,
                         x_ramp = x_ramp,
                         outputs = outputs*cf.amplification)

# Plot output
plt.figure()
plt.plot(outputs.T*cf.amplification)
plt.show()

InstrumentImporter.reset(0, 0)
