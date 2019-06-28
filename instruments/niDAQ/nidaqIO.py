'''
This module provides input/output functions for communicating with the
NI USB 6216. (Note that nidaqmx has to be installed seperately first.)
'''
import SkyNEt.instruments.niDAQ.nidaqmx as nidaqmx
import SkyNEt.instruments.niDAQ.nidaqmx.constants as constants
import SkyNEt.instruments.niDAQ.nidaqmx.system.device as device
from SkyNEt.instruments import InstrumentImporter
import numpy as np
import math
import time

def reset_device():    
    dev = device.Device(name='Dev1')
    dev2 = device.Device(name='cDAQ1Mod1')
    dev.reset_device()
    dev2.reset_device()
	

def IO(y, Fs, inputPorts = [1, 0, 0, 0, 0, 0, 0], highRange=False):
    '''
    Input/output function for communicating with the NI USB 6216 when measuring
    with one input and one output.

    Input arugments
    ---------------
    y: N x M array, N output ports, M datapoints
    Fs: sample frequency
    n_ai: number of input ports

    Return
    -------
    data: P x M array, P input ports, M datapoints
    '''

    # Sanity check on input voltages
    if not highRange:
        if max(abs(y) > 2):  
            print('WARNING: input voltages exceed threshold of 2V: highest absolute voltage is ' + str(max(abs(y))))
            print('If you want to use high range voltages, set highRange to True.')
            print('Aborting measurement...')
            InstrumentImporter.reset(0, 0)
            exit() 

    if len(y.shape) == 1:
        n_ao = 1
    else:
        n_ao = y.shape[0]

    if n_ao == 1:
        if (len(y.shape) == 2): # This means that the input is a 2D array
            y = y[0]
        N = len(y)
    if n_ao == 2:
        N = y.shape[1]


    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
      # Define ao/ai channels
        for i in range(n_ao):
            output_task.ao_channels.add_ao_voltage_chan('Dev1/ao'+str(i)+'', 'ao'+str(i)+'', -5, 5)
        for i in range(len(inputPorts)):
            if(inputPorts[i] == 1):
                input_task.ai_channels.add_ai_voltage_chan('Dev1/ai'+str(i)+'') 

        
        # Configure sample rate and set acquisition mode to finite
        output_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = N+1)
        input_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = N+1)

        # Output triggers on the read operation
        output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/Dev1/ai/StartTrigger')

        # Fill the output buffer
    
        output_task.write(y, auto_start=False)

        # Start tasks
        output_task.start()
        input_task.start()
        
        #read data

        read_data = input_task.read(N + 1, math.ceil(N/Fs)+1)


        read_data = np.asarray(read_data)
        if len(read_data.shape) == 1:
            read_data = read_data[np.newaxis,:]

        data = np.delete(read_data,(0), axis=1) #trim off the first datapoint, read lags one sample behind write

        # Stop and close the tasks
        input_task.stop()
        output_task.stop()

    return data

def IO_cDAQ(y, Fs, inputPorts = [1, 0, 0, 0, 0, 0, 0]):
    '''
    Input/output function for sampling with the NI 9264 in the NI 9171 chassis and measuring with NI USB 6216.
    Warning: The synchronization is far from ideal.
    Writing is first initialized and directly afterwards read is initialized.
    An output signal is used that spikes 50ms after start writing and on the actual input signal
    50 datapoints of zeros are added. After the measurement the first X datapoints are cut from the read data,
    where X is the amount of datapoints that are read before reading the spike.
    This way the read/write is synchronized, but ao wil always first write zeros to the system which might be
    undesirable for certain experiments.
    To use this script, always connect ao7 of the cDAQ to ai7 of the NI 6216 for the reference spike.
    Input arugments
    ---------------
    y: N x M array, N output ports, M datapoints
    Fs: sample frequency
    n_ai: number of input ports
    Return
    -------
    data: P x M array, P input ports, M datapoints
    '''
    if len(y.shape) == 1:
        n_ao = 1
    else:
        n_ao = y.shape[0]

    if n_ao == 1:
        if (len(y.shape) == 2): # This means that the input is a 2D array
            y = y[0]
        N = len(y)
    if n_ao >= 2:
        N = y.shape[1]

    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
      # Define ao/ai channels
        for i in range(n_ao):
            output_task.ao_channels.add_ao_voltage_chan('cDAQ1Mod1/ao'+str(i)+'', 'ao'+str(i)+'', -5, 5)
        for i in range(len(inputPorts)):
            if(inputPorts[i] == 1):
                input_task.ai_channels.add_ai_voltage_chan('Dev1/ai'+str(i)+'') #, min_val=-1, max_val=1 

        y = np.asarray(y)
        if len(y.shape) == 1:
            y = y[np.newaxis,:]
        # Define ao7 as sync signal for the NI 6216 ai0
        output_task.ao_channels.add_ao_voltage_chan('cDAQ1Mod1/ao7', 'ao7', -5, 5)
        input_task.ai_channels.add_ai_voltage_chan('Dev1/ai7') 

        # Append some zeros to the initial signal such that no input data is lost
        # This should be handled with proper synchronization
        y_corr = np.zeros((y.shape[0], y.shape[1] + int(Fs*0.2))) # Add 200ms of reaction in terms of zeros
        y_corr[:,int(Fs*0.2):] = y[:]
        if len(y_corr.shape) == 1:
            y_corr = np.concatenate((y_corr[np.newaxis], np.zeros((1,y_corr.shape[1]))))   # Set the trigger
        else:
            y_corr = np.concatenate((y_corr, np.zeros((1,y_corr.shape[1]))))   # Set the trigger
        y_corr[-1,int(Fs*0.2)] = 1 # Start input data

        # Configure sample rate and set acquisition mode to finite
        output_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan =y_corr.shape[1])
        input_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = y_corr.shape[1])

        output_task.write(y_corr)

        # Start tasks
        output_task.start()

        read_data = input_task.read(y_corr.shape[1], math.ceil(y_corr.shape[1]/Fs)+1) 

        read_data = np.asarray(read_data)
        cut_value = 0
        if len(read_data.shape) == 1:
            read_data = read_data[np.newaxis,:]
        for i in range(0,int(Fs*0.2)+1):
            if read_data[-1,i] >= 0.5:
                cut_value = i
                break
        if cut_value == 0:
            print('Warning: initialize spike not recognized')
        #trim off the first datapoints, read lags some samples behind write
        data = read_data[:-1,cut_value:N+cut_value] 
        if data.shape[1] != y.shape[1]:
            print('Warning: output data not same size as input data. Output: ' + str(data.shape[1]) + ' points, input: ' + str(y.shape[1]) + ' points.')
        # Stop and close the tasks
        input_task.stop()     
        output_task.stop()

    return data