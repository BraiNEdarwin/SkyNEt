# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments.ADwin import adwinIO
import SkyNEt.modules.Evolution as Evolution
from SkyNEt.instruments.DAC import IVVIrack
from SkyNEt.modules.PlotBuilder import PlotBuilder

# Other imports
import time
import numpy as np
import signal
import sys

controlvoltages = [1, 0.5, 0.2, 0.6, 0.7, 0.8, 0.3, 0.4]

adw = adwinIO.initInstrument()

adwinIO.setControlVoltages(adw, controlvoltages, 1000)