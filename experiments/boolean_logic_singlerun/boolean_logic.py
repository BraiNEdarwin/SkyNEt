'''
In this script the controlvoltages corresponding to different genomes are set and a boulean logic measurement is performed.
This is done withing 3 for loops:
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
import config_boolean_logic as config
from SkyNEt.instruments import InstrumentImporter

# Other imports
import time
import numpy as np
import matplotlib.pyplot as plt

# Initialize config object
cf = config.experiment_config()

# Initialize input and target
t = cf.InputGen()[0]  # Time array
x = np.asarray(cf.InputGen()[1:3])  # Array with P and Q signal

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize instruments
try:
    ivvi = InstrumentImporter.IVVIrack.initInstrument(comport = cf.comport)
except:
    pass
	
#%% Measurement loop

# Set the DAC voltages
InstrumentImporter.IVVIrack.setControlVoltages(ivvi, cf.controlVoltages)
time.sleep(5)  # Wait after setting DACs

# Set the input scaling
x_scaled = x*cf.inputScaling

# Feed input to measurement device
if(cf.device == 'nidaq'):
    output = InstrumentImporter.nidaqIO.IO(x_scaled, cf.fs)
elif(cf.device == 'adwin'):
    adw = InstrumentImporter.adwinIO.initInstrument()
    output = InstrumentImporter.adwinIO.IO(adw, x_scaled, cf.fs)
else:
    print('Specify measurement device as either adwin or nidaq')

# Save generation
SaveLib.saveExperiment(saveDirectory,
                         t = t,
                         x = x_scaled,
                         output = output*cf.amplification)

# Plot output
plt.figure()
plt.plot(t, output[0]*cf.amplification)
plt.show()

InstrumentImporter.reset(0, 0)


