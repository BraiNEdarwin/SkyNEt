'''
This holds some functions to perform basic operations on the DACs of the IVVI rack
'''

from qcodes.instrument_drivers.QuTech.IVVI import IVVI
import numpy as np


def initInstrument(dac_step = 500, dac_delay = 0.01):
	ivvi = IVVI("ivvi", "COM3", dac_step = 500, dac_delay=.01)
	ivvi.set_dacs_zero()  #safety
	return ivvi


def setControlVoltages(ivvi, controlVoltages):
	for i in range(len(controlVoltages)):
		command = 'ivvi.dac{}({})'.format(i+1,controlVoltages[i])
		exec(command)

def setControlVoltage(ivvi, controlVoltage, dacNo):
	
	command = 'ivvi.dac{}({})'.format(dacNo + 1,controlVoltage)
	exec(command)