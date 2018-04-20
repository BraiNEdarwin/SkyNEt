'''
Generates integer (0 to 65535) array of the benchmark data.
Also passes to be trained output (ideal data) to the main script
'''

import modules.benchmarks.WaveformRegression as wr
import modules.benchmarks.BooleanLogic as bl


def hardwareInput(benchmark, Fs, periods, frequency):
    if(benchmark[0] == 'wr'):
        [t, x] = wr.sineWave(Fs, periods, frequency)
        return [t, float_to_int(x)]


def softwareInput(benchmark, Fs, periods, frequency):
    if(benchmark[0] == 'wr'):
        return wr.sineWave(Fs, periods, frequency)
    if(benchmark[0] == 'bl'):
        return bl.InputSignals(Fs)


def targetOutput(benchmark, Fs, periods, frequency):
    if(benchmark[0] == 'wr'):
        #return wr.doubleFrequency(Fs, 2 * periods, 2 * frequency)
        return wr.squareWave(Fs, periods, frequency)
    if(benchmark[0] == 'bl'):
        if(benchmark[1] == 'OR'):
            return bl.OR(Fs)
        if(benchmark[1] == 'AND'):
            return bl.AND(Fs)
        if(benchmark[1] == 'NAND'):
            return bl.NAND(Fs)
        if(benchmark[1] == 'NOR'):
            return bl.NOR(Fs)
        if(benchmark[1] == 'XOR'):
            return bl.XOR(Fs)
        if(benchmark[1] == 'XNOR'):
            return bl.XNOR(Fs)

def float_to_int(x):
    x = (x + 10) / 20 * 65536
