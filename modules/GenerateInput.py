'''
Generates integer (0 to 65535) array of the benchmark data.
Also passes to be trained output (ideal data) to the main script
'''

from benchmarks import WaveformRegression as wr


def hardwareInput(benchmark, Fs):
    if(benchmark == 'wr'):
        x = wr.sineWave(Fs)
        return float_to_int(x)


def softwareInput(benchmark, Fs):
    if(benchmark == 'wr'):
    	return wr.sineWave(Fs)


def targetOutput(benchmark, Fs):
    if(benchmark == 'wr'):
        return wr.squareWave(Fs)


def float_to_int(x):
    x = (x + 10) / 20 * 65536
