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
	
def IO(y, Fs):
    '''
    Input/output function for communicating with the NI USB 6216 when measuring
    with one input and one output.
    Input argument:
    y; 1D numpy array with data you wish to send on port ao0
    Fs; sample frequency at which data will be sent
    Returns:
    data; 1D numpy array with data measured on ai0
    '''
    N = len(y)
    np.append(y, 0)  # Finish by setting dacs to 0
    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
        # Define ao/ai channels
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao0', 'ao0', -10, 10)
        input_task.ai_channels.add_ai_voltage_chan('Dev1/ai0')

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
        data = read_data[1:] #trim off the first datapoint, read lags one sample behind write

        # Stop and close the tasks
        input_task.stop()
        output_task.stop()

    return data


def IO_2D(x, Fs):
    '''
    Input/output function for communicating with the NI USB 6216 when measuring
    with one input and two outputs.
    Input argument:
    y; 2D numpy array (2 rows!) with data you wish to send on port ao0 and ao1
    Fs; sample frequency at which data will be sent
    Returns:
    data; 1D numpy array with data measured on ai0
    '''
    N = x.shape[1]
    np.append(x, [[0], [0]], axis=1)  # Finish by setting dacs to 0
    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
        # Define ao/ai channels
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao0', 'ao0', -5, 5)
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao1', 'ao1', -5, 5)
        input_task.ai_channels.add_ai_voltage_chan('Dev1/ai0')

        # Configure sample rate and set acquisition mode to finite
        output_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = N + 1)
        input_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = N + 1)

        # Output triggers on the read operation
        output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/Dev1/ai/StartTrigger')

        # Fill the output buffer
        output_task.write(x, auto_start=False)

        # Start tasks
        output_task.start()
        input_task.start()

        # Read data
        read_data = input_task.read(N + 1, math.ceil(N/Fs))
        data = read_data[1:] #trim off the first datapoint, read lags one sample behind write

        # Stop and close the tasks
        input_task.stop()
        output_task.stop()

    return data

def IO_2D2I(x, Fs):
    '''
    Input/output function for communicating with the NI USB 6216 when measuring
    with two inputs and two outputs.
    Input argument:
    y; 2D numpy array (2 rows!) with data you wish to send on port ao0 and ao1
    Fs; sample frequency at which data will be sent
    Returns:
    data; 2D numpy array with data measured on ai0 and ai1
    '''
    N = x.shape[1]
    np.append(x, [[0], [0]], axis=1)  # Finish by setting dacs to 0
    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
        # Define ao/ai channels
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao0', 'ao0', -5, 5)
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao1', 'ao1', -5, 5)
        input_task.ai_channels.add_ai_voltage_chan('Dev1/ai0')
        input_task.ai_channels.add_ai_voltage_chan('Dev1/ai1')

        # Configure sample rate and set acquisition mode to finite
        output_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = N + 1)
        input_task.timing.cfg_samp_clk_timing(Fs, sample_mode=constants.AcquisitionType.FINITE, samps_per_chan = N + 1)

        # Output triggers on the read operation
        output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/Dev1/ai/StartTrigger')

        # Fill the output buffer
        output_task.write(x, auto_start=False)

        # Start tasks
        output_task.start()
        input_task.start()

        # Read data
        read_data = input_task.read(N + 1, math.ceil(N/Fs))
        data = read_data[1:] #trim off the first datapoint, read lags one sample behind write

        # Stop and close the tasks
        input_task.stop()
        output_task.stop()

    return data
