
import numpy as np


# parameters
signalLength = 0.5  #in seconds 
edgeLength = 0.01  #in seconds

def InputSignals(Fs, signalLength = signalLength, edgeLength = edgeLength):
    samples = 6 * round(Fs * signalLength / 6) + 6 * round(Fs * edgeLength)
    x = np.empty(samples)
    y = np.empty(samples)
    z = np.empty(samples)
    v = np.empty(samples)
    w = np.empty(samples)
    t = np.linspace(0, samples/Fs, samples)

    x[0:round(Fs * signalLength / 6)] = 1
    x[round(Fs * signalLength / 6) : round(Fs * signalLength / 6) + round(Fs * edgeLength)] = 1
    x[round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + round(Fs * edgeLength)] = 1
    x[2 * round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = 1
    x[2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = 1
    x[3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = np.linspace(1, 0, round(Fs * edgeLength))
    x[3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = 0
    x[4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength)] = 0
    x[4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = 0
    x[5 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = 0
    x[5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength) : 6 * round(Fs * signalLength / 6) + 6 * round(Fs * edgeLength)] = 0

    y[0:round(Fs * signalLength / 6)] = 1
    y[round(Fs * signalLength / 6) : round(Fs * signalLength / 6) + round(Fs * edgeLength)] = np.linspace(1, 0, round(Fs * edgeLength))
    y[round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + round(Fs * edgeLength)] = 0
    y[2 * round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = 0
    y[2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = 0
    y[3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    y[3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = 1
    y[4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength)] = 1
    y[4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = 1
    y[5 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = np.linspace(1, 0, round(Fs * edgeLength))
    y[5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength) : 6 * round(Fs * signalLength / 6) + 6 * round(Fs * edgeLength)] = 0

    z[0:round(Fs * signalLength / 6)] = 0
    z[round(Fs * signalLength / 6) : round(Fs * signalLength / 6) + round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    z[round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + round(Fs * edgeLength)] = 1
    z[2 * round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = np.linspace(1, 0, round(Fs * edgeLength))
    z[2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = 0
    z[3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    z[3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = 1
    z[4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength)] = np.linspace(1, 0, round(Fs * edgeLength))
    z[4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = 0
    z[5 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    z[5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength) : 6 * round(Fs * signalLength / 6) + 6 * round(Fs * edgeLength)] = 1

    v[0:round(Fs * signalLength / 6)] = 0
    v[round(Fs * signalLength / 6) : round(Fs * signalLength / 6) + round(Fs * edgeLength)] = 0
    v[round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + round(Fs * edgeLength)] = 0
    v[2 * round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    v[2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = 1
    v[3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = np.linspace(1, 0, round(Fs * edgeLength))
    v[3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = 0
    v[4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    v[4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = 1
    v[5 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = 1
    v[5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength) : 6 * round(Fs * signalLength / 6) + 6 * round(Fs * edgeLength)] = 1

    w[0:round(Fs * signalLength / 6)] = 1
    w[round(Fs * signalLength / 6) : round(Fs * signalLength / 6) + round(Fs * edgeLength)] = 0
    w[round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + round(Fs * edgeLength)] = 1
    w[2 * round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = 0
    w[2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = 1
    w[3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = 0
    w[3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = 1
    w[4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength)] = 0
    w[4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = 1
    w[5 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = 0
    w[5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength) : 6 * round(Fs * signalLength / 6) + 6 * round(Fs * edgeLength)] = 1


    return t, x, y, z, v, w


def TargetSignal(Fs, signalLength = signalLength, edgeLength = edgeLength):
    samples = 6 * round(Fs * signalLength / 6) + 6 * round(Fs * edgeLength)
    x = np.empty(samples)
    t = np.linspace(0, samples/Fs, samples)

    x[0:round(Fs * signalLength / 6)] = -1
    x[round(Fs * signalLength / 6) : round(Fs * signalLength / 6) + round(Fs * edgeLength)] = np.linspace(-1, 1, round(Fs * edgeLength))
    x[round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + round(Fs * edgeLength)] = 1
    x[2 * round(Fs * signalLength / 6) + round(Fs * edgeLength) : 2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = np.linspace(1, 0, round(Fs * edgeLength))
    x[2 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength)] = 0
    x[3 * round(Fs * signalLength / 6) + 2 * round(Fs * edgeLength) : 3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = 0
    x[3 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength)] = 0
    x[4 * round(Fs * signalLength / 6) + 3 * round(Fs * edgeLength) : 4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength)] = np.linspace(0, 1, round(Fs * edgeLength))
    x[4 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = 1
    x[5 * round(Fs * signalLength / 6) + 4 * round(Fs * edgeLength) : 5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength)] = np.linspace(1, -1, round(Fs * edgeLength))
    x[5 * round(Fs * signalLength / 6) + 5 * round(Fs * edgeLength) : 6 * round(Fs * signalLength / 6) + 6 * round(Fs * edgeLength)] = -1

    return t, x