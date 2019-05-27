from SkyNEt.instruments import InstrumentImporter
import numpy as np
import time

ivvi = InstrumentImporter.IVVIrack.initInstrument()


controlVoltages = np.array([1000,1000,1000,1000,1000])

InstrumentImporter.IVVIrack.setControlVoltages(ivvi, controlVoltages)

time.sleep(10)
