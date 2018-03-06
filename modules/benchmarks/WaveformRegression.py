'''
This generates input data for a waveform regression benchmark test. Basically
various function definitions for various waveforms.
'''

import numpy as np
from scipy import signal


# benchmark parameters
amplitude = 1


def sineWave(Fs, periods, frequency):
    t = np.linspace(0, periods / frequency, (Fs / frequency) * periods)
    return [t, amplitude * np.sin(2 * np.pi * frequency * t)]


def squareWave(Fs, periods, frequency):
    t = np.linspace(0, periods / frequency, (Fs / frequency) * periods)
    return [t, amplitude * signal.square(2 * np.pi * frequency * t)]


def sawTooth(Fs, periods, frequency):
    t = np.linspace(0, periods / frequency, (Fs / frequency) * periods)
    return [t, amplitude * signal.sawtooth(2 * np.pi * frequency * t)]


def doubleFrequency(Fs, periods, frequency):
    t = np.linspace(0, periods / frequency, (Fs / frequency) * periods)
    return [t, amplitude * np.sin(2 * np.pi * frequency * t)]
