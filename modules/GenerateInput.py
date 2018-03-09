'''
Generates integer (0 to 65535) array of the benchmark data.
Also passes to be trained output (ideal data) to the main script
'''

import modules.benchmarks.WaveformRegression as wr


def hardwareInput(benchmark, Fs, periods, frequency):
    if(benchmark == 'wr'):
        [t, x] = wr.sineWave(Fs, periods, frequency)
        return [t, float_to_int(x)]


def softwareInput(benchmark, Fs, periods, frequency):
    if(benchmark == 'wr'):
        return wr.sineWave(Fs, periods, frequency)


def targetOutput(benchmark, Fs, periods, frequency):
    if(benchmark == 'wr'):
        #return wr.doubleFrequency(Fs, 2 * periods, 2 * frequency)
        return wr.squareWave(Fs, periods, frequency)


def float_to_int(x):
    x = (x + 10) / 20 * 65536
