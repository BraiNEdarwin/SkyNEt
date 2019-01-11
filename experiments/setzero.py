''''
Measurement script to perform an evolution experiment of a selected
gate. This will initially be tested on the Heliox (with nidaq) setup.
'''

# Import packages



from instruments.niDAQ import nidaqIO
from instruments.DAC import IVVIrack
import time

# temporary imports
import numpy as np
ivvi = IVVIrack.initInstrument()
inp = np.zeros((2,20))

controlVoltages = np.zeros(16)
    
IVVIrack.setControlVoltages(ivvi, controlVoltages)

    # feed 0 to nidaq
nidaqIO.IO(inp, 1000)
# print (list(iter.product(a, repeat=6)))

