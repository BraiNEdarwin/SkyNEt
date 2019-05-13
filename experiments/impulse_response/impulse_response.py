import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400
import numpy as np
import os
import config_impulse_response as config
import time

# Load the information from the config class.
config = config.experiment_config()
# Initialize save directory.
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)
# Initialize serial object
ser = InstrumentImporter.switch_utils.init_serial(cf.comport)
# Switch to device
InstrumentImporter.switch_utils.connect_single_device(ser, cf.device)
# Define the device input using the function in the config class.
Input = config.impulsegen(config.v_off, config.v_pulse, config.n_points)
Output = np.zeros(config.n_points)
feedback = np.zeros(config.n_points+1)
# TODO Measure in a loop using the device specified in the config class.
if config.device == 'nidaq':
    for ii in range(len(Input)):
	    nidaqin=[[Input[ii]],[feedback[ii]]]
	    Output[ii] = InstrumentImporter.nidaqIO.IO(nidaqin, config.fs,inputPorts = [1, 1, 0, 0, 0, 0, 0])
		feedback[ii+1]= Output[ii]*config.W
elif config.device == 'adwin':
    adwin = InstrumentImporter.adwinIO.initInstrument()
	for ii in range(len(Input)):
	    adwinin=[[Input[ii]],[feedback[ii]]]
        Output[ii] = InstrumentImporter.adwinIO.IO(adwin, adwinin, config.fs, inputPorts = [1, 1, 0, 0, 0, 0, 0])
        feedback[ii+1]= Output[ii]*config.W

elif config.device == 'keithley':
    keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')

    # Set compliances
    keithley.compliancei.set(1E-6)
    keithley.compliancev.set(4)

    # Turn keithley output on
    keithley.output.set(1)

    for ii in range(len(Output)):
        # Set voltage
        keithley.volt.set(Input[ii])

        # Record current
        time.sleep(0.05)
        Output[ii] = keithley.curr()

    # Turn keithley output off
    keithley.output.set(0)
else:
    print('specify measurement device')

	
# Save the Input and Output
SaveLib.saveExperiment(saveDirectory, input = Input, output = Output*config.amplification)

# Convert to current
#R = 1E6  # Ohm
#V = grounded_input[0] - Output[0]
#I = Output[0]/R
# Plot the IV curve
for n in range(1):
    plt.figure()
    plt.plot(Output)#,'b',grounded_input[1], Output[n],'g',grounded_input[2], Output[n],'r',grounded_input[3], Output[n],'k')
    plt.xlabel('timesteps')
    plt.ylabel('current (nA)')
    plt.title('impulse response')
# plt.figure()
# plt.plot(grounded_input[0], Output[0],'b',grounded_input[0], Output[1],'g',grounded_input[0], Output[2],'k',grounded_input[0], Output[3],'r')
plt.show()
# Final reset
InstrumentImporter.reset(0, 0)