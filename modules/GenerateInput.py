'''
Generates integer (0 to 65535) array of the benchmark data.
Also passes to be trained output (ideal data) to the main script
'''
import numpy as np
from matplotlib import pyplot as plt
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

def ControlProblem():
    
    min_laps = 150
    nr_laps = 7
    I4 = np.zeros((nr_laps*min_laps,1))
    for i in range(nr_laps):
        I4[(2*i+1)*min_laps:2*(i+1)*min_laps] = 1.0
    
    I3 = np.ones_like(I4)
    I2 = np.copy(I3)
    I2[:4*min_laps] = 0.0
    
    for i in range(int(nr_laps/2)):
        I3[(2*i)*(2*min_laps):(2*i+1)*(2*min_laps)] = 0.0
        
    env_states = np.concatenate((I2,I3,I4),axis=1)
    env_states = np.tile(env_states,(3,1))
    
    agent = np.zeros((env_states.shape[0],1))
    agent[len(I2):2*len(I2)] = 0.4
    agent[2*len(I2):] = 0.8
    
    inputs = np.concatenate((agent,env_states),axis=1)
    
    target = np.zeros((inputs.shape[0],1))
    target[:len(I2)] = np.copy(I4)
    target[agent[:,0]==0.4,:] = np.copy(I3)
    target[-(len(I2)+150):-len(I2)] = -1.0
    target[-3*min_laps:] = -1.0 
    
    plt.figure()
    plt.subplot(5,1,1)
    plt.plot(inputs[:,0])
    plt.ylabel('agent state')
    for i in range(1,inputs.shape[1]):
        plt.subplot(5,1,i+1)
        plt.plot(inputs[:,i])
        plt.ylabel('env. state '+str(i+1))
    plt.subplot(5,1,5)
    plt.plot(target,'k')
    plt.ylabel('target')

    W = np.ones_like(target)
    t = np.arange(len(target))
    return inputs, target, W, t
