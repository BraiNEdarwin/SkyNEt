'''
This module provides input/output functions for communicating with the
NI USB 6216. (Note that nidaqmx has to be installed seperately first.)
'''
import SkyNEt.instruments.niDAQ.nidaqmx as nidaqmx
import SkyNEt.instruments.niDAQ.nidaqmx.constants as constants
import SkyNEt.instruments.niDAQ.nidaqmx.system.device as device
import numpy as np
import math

def reset_device():
	dev = device.Device(name='Dev1')
	dev.reset_device()
	

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

    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
      # Define ao/ai channels
        for i in range(n_ao):
            output_task.ao_channels.add_ao_voltage_chan('Dev1/ao'+str(i)+'', 'ao'+str(i)+'', -10, 10)
        for i in range(len(inputPorts)):
            if(inputPorts[i] == 1 and i!=7):
                input_task.ai_channels.add_ai_voltage_chan('Dev1/ai'+str(i)+'', 'ai'+str(i)+'', min_val=-5, max_val=5)
                input_task.ai_channels[f'ai{i}'].ai_rng_low = -5
                input_task.ai_channels[f'ai{i}'].ai_rng_high = 5
                input_task.ai_channels[f'ai{i}'].ai_term_cfg = nidaqmx.constants.TerminalConfiguration.RSE
            elif(inputPorts[i] == 1):
                input_task.ai_channels.add_ai_voltage_chan('Dev1/ai'+str(i)+'', 'ai'+str(i)+'', min_val=-5, max_val=5)



        
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
        read_data = input_task.read(N + 1, math.ceil(N/Fs))

        read_data = np.asarray(read_data)
        if len(read_data.shape) == 1:
            read_data = read_data[np.newaxis,:]

        data = np.delete(read_data,(0), axis=1) #trim off the first datapoint, read lags one sample behind write

        # Stop and close the tasks
        input_task.stop()
        output_task.stop()

    return data