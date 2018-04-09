'''
This holds some functions to perform basic operations on the DACs of the IVVI rack
'''

from qcodes.instrument_drivers.QuTech.IVVI import IVVI


def initInstrument():
	ivvi = IVVI("ivvi", "COM5", dac_step = 500, dac_delay=.01)
	ivvi.set_dacs_zero()  #safety
	return ivvi


def setControlVoltages(ivvi, controlVoltages):
	for i in range(len(controlVoltages)):
		command = 'ivvi.dac{}({})'.format(i+1,controlVoltages[i])
		exec(command)