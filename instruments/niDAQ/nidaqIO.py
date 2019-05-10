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
        read_data = input_task.read(N + 1, math.ceil(N/Fs))

        read_data = np.asarray(read_data)
        if len(read_data.shape) == 1:
            read_data = read_data[np.newaxis,:]

        data = np.delete(read_data,(0), axis=1) #trim off the first datapoint, read lags one sample behind write

        # Stop and close the tasks
        input_task.stop()
        output_task.stop()

    return data


class IO_cDAQ:
    '''
    Edited by Lennart:
        class to set and ramp up to voltages on the cdaq
        voltages are kept at their values indefinitely if you don't ramp down to zero by yourself
    
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
    def __init__(self, nr_channels=7):
        assert nr_channels>0, 'There must be at least one channel'
        self.nr_channels = nr_channels
        self.state = np.zeros(self.nr_channels)
        self.__set_state(self.state)
    
    def __set_state(self, state):
        """ 
        WARNING, DONT USE THIS TO FUNCTION DIRECTLY!
        Always ramp to values if your devices to not break.
        
        Sets single state of channels instantly.
        """
        assert state.shape == (self.nr_channels,), "State shape %s incorrect, expecting (%i,)" % (state.shape, self.nr_channels)
        
        with nidaqmx.Task() as output_task:
            for i in range(self.nr_channels):
                # Define ao/ai channel
                output_task.ao_channels.add_ao_voltage_chan('cDAQ1Mod1/ao'+str(i)+'', 'ao'+str(i)+'', -5, 5)
            
            output_task.write(np.expand_dims(state, axis=1), auto_start=True)
            output_task.stop()
    
    def ramp(self, target_state, ramp_speed=1., set_frequency=1000):
        """
        slowly ramp to target_state with ramp speed and set frequency and will make script wait untill the ramping is complete
        ramp_speed:     Maximum V/s which is used to ramp to target values
        """
        target_state = np.array(target_state, dtype='float32')
        assert target_state.shape == (self.nr_channels,), 'ERROR: Wrong size of voltages %s, expected (%i)' % (str(target_state.shape), self.nr_channels)
        assert ramp_speed<=5., 'ERROR: For the safety of the device, ramp speed should probably be smaller than 5 V/s' 
        
        maximum_difference = max(abs(target_state-self.state))             # in Volt
        time_length = max(2, int(maximum_difference/ramp_speed*set_frequency))  # number of data points used to ramp with set_frequency (minimum is 2)
        ramp_data = np.zeros((self.nr_channels, time_length))                   # initialize ramp data array
        
        with nidaqmx.Task() as output_task:
            for i in range(self.nr_channels):
                # Define ao/ai channel
                output_task.ao_channels.add_ao_voltage_chan('cDAQ1Mod1/ao'+str(i)+'', 'ao'+str(i)+'', -5, 5)
                # calculate ramp voltages for this channel
                ramp_data[i,:] = np.linspace(self.state[i], target_state[i], time_length)
    
            # Configure sample rate and set acquisition mode to finite
            output_task.timing.cfg_samp_clk_timing(set_frequency, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = time_length)
            output_task.write(ramp_data, auto_start=True)
            output_task.wait_until_done(timeout=10) 
            # Stop and close the tasks
            output_task.stop()
        self.state = target_state
    
    def ramp_zero(self, **kwargs):
        """ Ramps back down to zero """
        self.ramp(np.zeros(self.nr_channels), **kwargs)
           

