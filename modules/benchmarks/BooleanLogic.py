'''
This generates input data for a boolean logic evolution experiment.
'''

import numpy as np


# parameters
signalLength = 0.5  #in seconds 
edgeLength = 0.01  #in seconds


def InputSignals(Fs, signalLength = signalLength, edgeLength = edgeLength):
    samples = 4 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)
    x = np.empty(samples)
    y = np.empty(samples)
    t = np.linspace(0, samples/Fs, samples)

    x[0:round(Fs * signalLength / 4)] = 0
    x[round(Fs * signalLength / 4) : round(Fs * signalLength / 4) + round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    x[round(Fs * signalLength / 4) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 4) + round(Fs * edgeLength)] = 1
    x[2 * round(Fs * signalLength / 4) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength)] = np.linspace(1, 0, round(Fs * edgeLength))
    x[2 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength)] = 0
    x[3 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    x[3 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)] = 1

    y[0:round(Fs * signalLength / 4)] = 0
    y[round(Fs * signalLength / 4) : round(Fs * signalLength / 4) + round(Fs * edgeLength)] = 0
    y[round(Fs * signalLength / 4) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 4) + round(Fs * edgeLength)] = 0
    y[2 * round(Fs * signalLength / 4) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    y[2 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength)] = 1
    y[3 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)] = 1
    y[3 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)] = 1

    return t, x, y

def AND(Fs, signalLength = signalLength, edgeLength = edgeLength):
    samples = 4 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)
    x = np.empty(samples)
    t = np.linspace(0, samples/Fs, samples)

    x[0:round(Fs * signalLength / 4)] = 0
    x[round(Fs * signalLength / 4) : round(Fs * signalLength / 4) + round(Fs * edgeLength)] = 0
    x[round(Fs * signalLength / 4) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 4) + round(Fs * edgeLength)] = 0
    x[2 * round(Fs * signalLength / 4) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength)] = 0
    x[2 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength)] = 0
    x[3 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    x[3 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)] = 1

    return t, x

def OR(Fs, signalLength = signalLength, edgeLength = edgeLength):
    samples = 4 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)
    x = np.empty(samples)
    t = np.linspace(0, samples/Fs, samples)

    x[0:round(Fs * signalLength / 4)] = 0
    x[round(Fs * signalLength / 4) : round(Fs * signalLength / 4) + round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    x[round(Fs * signalLength / 4) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 4) + round(Fs * edgeLength)] = 1
    x[2 * round(Fs * signalLength / 4) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength)] = 1
    x[2 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength)] = 1
    x[3 * round(Fs * signalLength / 4) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)] = 1
    x[3 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 4) + 3 * round(Fs * edgeLength)] = 1

    return t, x