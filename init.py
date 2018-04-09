from instruments.DAC import IVVIrack

# initialize instruments
ivvi = IVVIrack.initInstrument()

# set the DAC voltages
for k in range(genes):
	IVVIrack.setControlVoltages(ivvi, controlVoltages)