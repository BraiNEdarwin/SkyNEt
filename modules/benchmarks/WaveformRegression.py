'''
This generates input data for a waveform regression benchmark test. Basically 
various function definitions for various waveforms.
'''

import numpy as np
from scipy import signal


#benchmark parameters
frequency = 1e3
amplitude = 1
nPoints = 1e10

def sineWave(Fs):
    t = np.linspace(0, nPoints * Fs, nPoints)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def squareWave(Fs):
    t = np.linspace(0, nPoints * Fs, nPoints)
    return amplitude * signal.square(2 * np.pi * frequency * t)


def sawTooth(Fs):
    t = np.linspace(0, nPoints * Fs, nPoints)
    return amplitude * signal.sawtooth(2 * np.pi * frequency * t)


