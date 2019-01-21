'''
This module provides input/output functions for communicating with the
NI USB 6216. (Note that nidaqmx has to be installed seperately first.)
'''
import SkyNEt.instruments.niDAQ.nidaqmx as nidaqmx
import SkyNEt.instruments.niDAQ.nidaqmx.constants as constants
import SkyNEt.instruments.niDAQ.nidaqmx.system.device as device
import numpy as np
import math
import time

def reset_device():    
    dev = device.Device(name='Dev1')
    dev2 = device.Device(name='cDAQ1Mod1')
    dev.reset_device()
    dev2.reset_device()
	

def IO(y, Fs, inputPorts = [1, 0, 0, 0, 0, 0, 0]):
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

    np.append(y, 0)  # Finish by setting dacs to 0
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
    if n_ao == 2:
        N = y.shape[1]

    np.append(y, 0)  # Finish by setting dacs to 0
    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
      # Define ao/ai channels
        for i in range(n_ao):
            output_task.ao_channels.add_ao_voltage_chan('cDAQ1Mod1/ao'+str(i)+'', 'ao'+str(i)+'', -5, 5)
        for i in range(len(inputPorts)):
            if(inputPorts[i] == 1):
                input_task.ai_channels.add_ai_voltage_chan('Dev1/ai'+str(i)+'') 


        # Define ao7 as trigger signal for the NI 6216 to start reading
        output_task.ao_channels.add_ao_voltage_chan('cDAQ1Mod1/ao7', 'ao7', -5, 5)

        if len(y.shape) == 1:
            y = np.concatenate((y[np.newaxis], np.ones((1,N))))   # Set the trigger
        else:
            y = np.concatenate((y, np.ones((1,N+1))))   # Set the trigger
        
        # Configure sample rate and set acquisition mode to finite
        output_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = N+1)
        input_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = N+1)

        # Output triggers on the read operation
        #input_task.in_stream.InStream(auto_start = False)
        #input_task.triggers.start_trigger.cfg_dig_edge_start_trig('/Dev1/PFI0')        

        ### Somehow if you start the input task it already starts reading before obtaining
        ### the 'read command'. ~50ms before it starts writing
        input_task.start()

        # Fill the output buffer
        #output_task.write(y, auto_start=False)
        output_task.write(y)

        # Start tasks
        output_task.start()
        #
        read_data = input_task.read(N + 1, math.ceil(N/Fs)+1)
        #read_data = input_task.in_stream.readall()
        

        read_data = np.asarray(read_data)
        if len(read_data.shape) == 1:
            read_data = read_data[np.newaxis,:]

        data = np.delete(read_data,(0), axis=1) #trim off the first datapoint, read lags one sample behind write

        # Stop and close the tasks
        input_task.stop()
        
        output_task.stop()

    return data