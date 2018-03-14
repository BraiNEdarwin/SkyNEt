'''
quick test script to test connection to the IVVI module.
'''
import time
import matplotlib.pyplot as plt
#import qcodes
#from qcodes.instrument_drivers.stanford_research.SR860 import SR860
from qcodes.instrument_drivers.QuTech.IVVI import IVVI

#ivvi = IVVI("ivvi", "COM5")

#ivvi.set_dacs_zero()

# test setting the first dac to 50 mV value
# ivvi.dac1(50)