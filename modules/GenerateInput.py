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

def ControlProblem_4Iv():
    
    min_laps = 150
    nr_laps = 7
    main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/'
    dir_data = '2018_08_07_164652_CP_FullSwipe/'
    data_dir = main_dir+dir_data
    x_inp = np.load(data_dir+'CP_inputs.npy')
    max_gene_val = 0.83333333
    min_gene_val = 0.16666667
    
    I4 = np.ones((nr_laps*min_laps,1))*min_gene_val
    for i in range(nr_laps):
        I4[(2*i+1)*min_laps:2*(i+1)*min_laps] = max_gene_val  #Trained NN as CV (gene) values
    
    I3 = max_gene_val*np.ones_like(I4)  #Trained NN as CV (gene) values
    for i in range(int(nr_laps/2)):
        I3[(2*i)*(2*min_laps):(2*i+1)*(2*min_laps)] = min_gene_val
    
    I2 = np.max(x_inp[:,1])*np.ones_like(I4)
    I2[:4*min_laps] = np.min(x_inp[:,1]) #Trained NN as input values
        
    env_states = np.concatenate((I2,I3,I4),axis=1)
    env_states = np.tile(env_states,(3,1))
    
    agent = np.zeros((env_states.shape[0],1))
    agent[:len(I2)] = np.min(x_inp[:,0])
    agent[2*len(I2):] = np.max(x_inp[:,0])
    
    inputs = np.concatenate((agent,env_states),axis=1)
    
    target = np.zeros((inputs.shape[0],1))
    target[:len(I2)] = (np.copy(I4)-min_gene_val)/(np.max(I4)-min_gene_val)
    target[agent[:,0]==0.0,:] = (np.copy(I3)-min_gene_val)/(np.max(I3)-min_gene_val)
    target[-(len(I2)+150):-len(I2)] = -1.0
    target[-3*min_laps:] = -1.0 
    target = -target
    
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

def ControlProblem_2Iv():
    
    min_laps = 150
    nr_laps = 7
    main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/'
    dir_data = '2018_08_07_164652_CP_FullSwipe/'
    data_dir = main_dir+dir_data
    x_inp = np.load(data_dir+'CP_inputs.npy')
    block = nr_laps*min_laps
    env_states = np.zeros((block,))
    env_values = np.unique(x_inp[:,1])
    for i in range(nr_laps):
        env_states[i*min_laps:(i+1)*min_laps] = env_values[i]  #Trained NN as CV (gene) values
    
    env_states = np.tile(env_states,3)[:,np.newaxis]
    agent = np.zeros((env_states.shape[0],1))
    agent[:block] = np.min(x_inp[:,0])
    agent[2*block:] = np.max(x_inp[:,0])
    
    inputs = np.concatenate((agent,env_states),axis=1)
    
    target = np.zeros((inputs.shape[0],1))
    target[:3*min_laps] = -1.0
    target[7*min_laps:8*min_laps] = -1.0
    target[12*min_laps:14*min_laps] = 1.0
    target[16*min_laps:17*min_laps] = 1.0
    target[18*min_laps:19*min_laps] = 1.0
    target[20*min_laps:] = 1.0
#    assert 0==1, 'WARNING!! TARGET TO BE DEFINED'
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(inputs[:,0])
    plt.ylabel('agent state')
    for i in range(1,inputs.shape[1]):
        plt.subplot(3,1,i+1)
        plt.plot(inputs[:,i])
        plt.ylabel('env. state '+str(i+1))
    plt.subplot(3,1,3)
    plt.plot(target,'k')
    plt.ylabel('target')

    W = np.ones_like(target)
    t = np.arange(len(target))
    return inputs, target, W, t
