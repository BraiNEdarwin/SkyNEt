'''
Generates integer (0 to 65535) array of the benchmark data.
Also passes to be trained output (ideal data) to the main script
'''
import numpy as np
import modules.benchmarks.WaveformRegression as wr
import modules.benchmarks.BooleanLogic as bl
import modules.benchmarks.Manifold as mf


def hardwareInput(benchmark, Fs, periods, frequency):
    if(benchmark[0] == 'wr'):
        [t, x] = wr.sineWave(Fs, periods, frequency)
        return [t, float_to_int(x)]


def softwareInput(benchmark, Fs, periods, frequency):
    if(benchmark[0] == 'wr'):
        return wr.sineWave(Fs, periods, frequency)

    if(benchmark[0] == 'bl'):
    	return bl.InputSignals(Fs)

    if(benchmark[0] == 'mf'):
        return mf.InputSignals(Fs)





def targetOutput(benchmark, Fs, periods, frequency):
    if(benchmark[0] == 'wr'):
        #return wr.doubleFrequency(Fs, 2 * periods, 2 * frequency)
        return wr.squareWave(Fs, periods, frequency)

    if(benchmark[0] == 'bl'):
    	if(benchmark[1] == 'AND'):
    		return bl.AND(Fs)
    	if(benchmark[1] == 'NAND'):
    		return bl.NAND(Fs)
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
            
    if(benchmark[0] == 'mf'):
        return mf.TargetSignal(Fs)

def float_to_int(x):
    x = (x + 10) / 20 * 65536

def SpiralInput(n_points, sp_offset):

    x = np.linspace(0, 4*np.pi, n_points)
    
    x_spiral1 = x*np.sin(x)+sp_offset
    x_spiral1 = x_spiral1/np.min(x_spiral1)
    y_spiral1 = x*np.cos(x)
    y_spiral1 = y_spiral1 / np.max(y_spiral1)

    x_spiral2 = -x*np.sin(x)-sp_offset
    x_spiral2 = x_spiral2/np.max(x_spiral2)
    y_spiral2 = -x*np.cos(x)
    y_spiral2 = y_spiral2/np.min(y_spiral2)

    return [x_spiral1, y_spiral1, x_spiral2, y_spiral2]


