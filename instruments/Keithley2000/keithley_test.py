# Test script
import Keithley_2000
import SkyNEt.instruments.DAC.IVVIrack as IVVIrack
import time

address = 'GPIB0::17::INSTR'

ivvi = IVVIrack.initInstrument()
keithley = Keithley_2000.initInstrument(address)

test_voltage = 100 # in mV

IVVIrack.setControlVoltage(ivvi, test_voltage, 0)

tic = time.time()

V = Keithley_2000.readValues(keithley, 50, 5000)

print(time.time() - tic)