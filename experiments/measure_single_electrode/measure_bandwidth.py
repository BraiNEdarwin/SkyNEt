'''
This script applies control+input voltage configurations defined in the
config file. One of these voltages is applied with the Keithley2400,
such that the current through this electrode can be measured.
By performing this measurement 7 times (all electrodes but the output),
all flowing currents in a particular configuration can be indexed.
'''
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import config_measure_single_electrode as config
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400
# Other imports
import time
import numpy as np

filepath = r'D:\\data\\BramdW\\measure_bandwidth\\NAND_0_0p5sin0p1\\'
ivvi = InstrumentImporter.IVVIrack.initInstrument()
adw = InstrumentImporter.adwinIO.initInstrument()
fs = 20000
siglen = 10 # seconds
freq = np.linspace(10,1000,100)
InstrumentImporter.IVVIrack.setControlVoltages(ivvi, [-826.9,-838.8,77.9,423.4,-46.6,0])
time.sleep(1)
for ii in freq:
    x = np.zeros(2,siglen*fs)
    x = 0.01*np.sin(2*np.pi*ii*np.arange(siglen*fs)/fs)+0.5
    output = InstrumentImporter.adwinIO.IO(adw, x, fs)
    datetime = time.strftime("%d_%m_%Y_%H%M%S")
    fp = filepath + '/' + datetime + ii + 'Hz'
 #   np.savetxt(fp,output)
    
InstrumentImporter.reset(0, 0)

