'''
This holds some functions to perform basic operations on the DACs of the IVVI rack
'''

from SkyNEt.qcodes.instrument_drivers.QuTech.IVVI import IVVI
import numpy as np


def initInstrument(dac_step = 500, dac_delay = 0.01, comport = 'COM4', name = 'ivvi'):
	'''
	Initializes the ivvi rack.
	List of arguments:
	dac_step; amount of mV/step with which dac voltages are updated
	dac_delay; time in seconds between dac steps
	comport; COM port to which ivvi rack is connected
	'''
	ivvi = IVVI(name, comport, dac_step = 500, dac_delay=.01)
	ivvi.set_dacs_zero()  # Safety
	return ivvi


def setControlVoltages(ivvi, controlVoltages):
	'''
	Sets voltages on the ivvi rack DACs. controlVoltages is a 1D array or list with
	values. They will be set sequentially to the DACs, starting at DAC 1.
	So if len(controlVoltages) = 5, then DAC1 through DAC5 will be used.
	'''
	for i in range(len(controlVoltages)):
		command = 'ivvi.dac{}({})'.format(i+1,controlVoltages[i])
		exec(command)

def setControlVoltage(ivvi, controlVoltage, dacNo):
	'''
	Sets the voltage controlVoltage on DAC number dacNo.
	'''
	command = 'ivvi.dac{}({})'.format(dacNo + 1,controlVoltage)
	exec(command)
